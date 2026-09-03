#!/bin/bash
#
# Startup script for a SpinDoctor cloud_tasks worker instance.
#
# cloud_tasks runs this as root on every compute instance it creates, after
# inserting its own "export RMS_CLOUD_TASKS_..." lines directly below the
# shebang.  That insertion is why the first line must be exactly "#!/bin/bash":
# cloud_tasks refuses a startup script whose shebang does not end in "/bash",
# so "#!/usr/bin/env bash" is rejected.
#
# Name it from the job configuration file:
#
#     gcp:
#       startup_script_file: cloud_support/startup_script.sh
#
# What it does, in order: attaches and mounts the read-only data disk holding
# the SPICE kernels and star catalogs, installs SpinDoctor from git into a
# virtual environment, points the environment at the mounted data and at the
# results bucket, and runs the worker, which then pulls tasks from the queue
# until the queue is empty.
#
# Every setting below can be overridden by exporting it before this script
# runs, but the ordinary way to change one is to edit its default here.

set -uo pipefail

################################################################################
# Settings
################################################################################

# Where the code comes from.
SPINDOCTOR_GIT_URL="${SPINDOCTOR_GIT_URL:-https://github.com/SETI/rms-nav.git}"
SPINDOCTOR_GIT_REF="${SPINDOCTOR_GIT_REF:-main}"
SPINDOCTOR_DIR="${SPINDOCTOR_DIR:-/opt/spindoctor}"

# Which worker to run, and any arguments it takes.  One of
# sd_offset_cloud_tasks, sd_backplanes_cloud_tasks, sd_mosaic_cloud_tasks,
# sd_create_bundle_cloud_tasks or sd_results_index_cloud_tasks.
SPINDOCTOR_WORKER="${SPINDOCTOR_WORKER:-sd_offset_cloud_tasks}"
SPINDOCTOR_WORKER_ARGS="${SPINDOCTOR_WORKER_ARGS:-}"

# The data disk carrying SPICE/ and star-catalogs/.  It is attached read-only,
# so one disk serves every instance in the pool; it must live in the same zone
# as the instances, which means the job configuration has to pin "zone:"
# rather than let cloud_tasks choose one.
DATA_DISK_NAME="${DATA_DISK_NAME:-CHANGE-ME}"
DATA_DISK_DEVICE_NAME="${DATA_DISK_DEVICE_NAME:-spindoctor-data}"
DATA_DISK_MOUNT="${DATA_DISK_MOUNT:-/mnt/pd1}"

# Where results go, and where the images the tasks name are read from.  The
# results root must be writable by the instance service account.  The holdings
# root is only for programs that enumerate images themselves; the URLs in a
# task file are absolute and are used as they stand.
NAV_RESULTS_ROOT="${NAV_RESULTS_ROOT:-gs://CHANGE-ME/nav-offset-results}"
PDS3_HOLDINGS_DIR="${PDS3_HOLDINGS_DIR:-https://pds-rings.seti.org/holdings}"

# Scratch space for downloaded images and kernels.  It lives on the boot disk,
# so size the boot disk in the job configuration for the number of tasks an
# instance runs at once.
FILECACHE_CACHE_ROOT="${FILECACHE_CACHE_ROOT:-/var/tmp/filecache}"

STARTUP_LOG="${STARTUP_LOG:-/var/log/spindoctor-startup.log}"

################################################################################
# Logging
################################################################################

# Everything below goes to a file on the instance and to the serial console,
# which is where a startup failure has to be read from: an instance that fails
# here never registers with the queue, so no other record of it exists.
exec > >(tee -a "${STARTUP_LOG}") 2>&1

say() {
    echo "[spindoctor-startup $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

die() {
    say "FATAL: $*"
    exit 1
}

# A git URL may carry a credential, and everything said here reaches the serial
# console, which is readable by anyone who can describe the instance.
SPINDOCTOR_GIT_URL_SAFE="$(sed -E 's#://[^/@]*@#://#' <<<"${SPINDOCTOR_GIT_URL}")"

say "Starting on $(hostname), job ${RMS_CLOUD_TASKS_JOB_ID:-unknown}"

################################################################################
# Instance identity
################################################################################

metadata() {
    curl --silent --fail --header 'Metadata-Flavor: Google' \
        "http://metadata.google.internal/computeMetadata/v1/$1"
}

PROJECT_ID="$(metadata project/project-id)" || die 'Cannot read the project id'
INSTANCE_NAME="$(metadata instance/name)" || die 'Cannot read the instance name'
ZONE="$(basename "$(metadata instance/zone)")" || die 'Cannot read the instance zone'
say "Instance ${INSTANCE_NAME} in ${ZONE} of project ${PROJECT_ID}"

################################################################################
# Data disk
################################################################################

DEVICE_LINK="/dev/disk/by-id/google-${DATA_DISK_DEVICE_NAME}"

attach_data_disk() {
    # Attaching is best-effort: an instance that already has the disk (a retried
    # startup, or a disk attached at creation) gets an error back that means
    # "nothing to do".  Whether the disk is really there is decided below, by
    # waiting for its device node.
    local token payload attempt
    token="$(metadata instance/service-accounts/default/token |
        python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])')" ||
        die 'Cannot read an access token; the instance has no service account'
    payload="$(printf '{"source":"projects/%s/zones/%s/disks/%s","deviceName":"%s","mode":"READ_ONLY","type":"PERSISTENT","autoDelete":false}' \
        "${PROJECT_ID}" "${ZONE}" "${DATA_DISK_NAME}" "${DATA_DISK_DEVICE_NAME}")"
    for attempt in 1 2 3 4 5; do
        say "Attaching disk ${DATA_DISK_NAME} read-only (attempt ${attempt})"
        curl --silent --show-error --request POST \
            --header "Authorization: Bearer ${token}" \
            --header 'Content-Type: application/json' \
            --data "${payload}" \
            "https://compute.googleapis.com/compute/v1/projects/${PROJECT_ID}/zones/${ZONE}/instances/${INSTANCE_NAME}/attachDisk" |
            sed 's/^/    /'
        # The API call returns an operation, not a device; the device node is
        # what the mount needs, so wait for that either way.
        for _ in $(seq 30); do
            [[ -e ${DEVICE_LINK} ]] && return 0
            sleep 2
        done
        say "Disk ${DATA_DISK_NAME} has not appeared as ${DEVICE_LINK}"
    done
    return 1
}

if [[ ${DATA_DISK_NAME} == 'CHANGE-ME' ]]; then
    die 'DATA_DISK_NAME is unset; edit the settings block of this script'
fi
if [[ ${NAV_RESULTS_ROOT} == *CHANGE-ME* ]]; then
    die "NAV_RESULTS_ROOT is still the template's placeholder (${NAV_RESULTS_ROOT});
    a worker that cannot write its results is worth stopping before it starts"
fi

attach_data_disk || die "Could not attach ${DATA_DISK_NAME}: check that the disk
    is in zone ${ZONE} (pin \"zone:\" in the job configuration), that no
    instance holds it read-write, and that the instance service account may
    compute.instances.attachDisk and compute.disks.use"

# A disk made with a partition table presents its filesystem as -part1; one
# formatted whole presents the device itself.  The partition node is created by
# udev a moment after the disk node, so give it that moment before concluding
# there is no partition table.
PARTITION="${DEVICE_LINK}-part1"
for _ in $(seq 5); do
    [[ -e ${PARTITION} ]] && break
    sleep 2
done
[[ -e ${PARTITION} ]] || PARTITION="${DEVICE_LINK}"
say "Filesystem device is ${PARTITION}"

mkdir -p "${DATA_DISK_MOUNT}"
if mountpoint -q "${DATA_DISK_MOUNT}"; then
    say "${DATA_DISK_MOUNT} is already mounted"
else
    # noload leaves the journal alone, which is what every reader of a disk
    # shared read-only must do; a plain ro mount of ext4 still wants to replay
    # it.  A filesystem that does not know the option is mounted without it.
    mount -o ro,noload "${PARTITION}" "${DATA_DISK_MOUNT}" ||
        mount -o ro "${PARTITION}" "${DATA_DISK_MOUNT}" ||
        die "Cannot mount ${PARTITION} on ${DATA_DISK_MOUNT}"
    say "Mounted ${PARTITION} read-only on ${DATA_DISK_MOUNT}"
fi

for required in SPICE star-catalogs/UCAC4 star-catalogs/YBSC; do
    [[ -d ${DATA_DISK_MOUNT}/${required} ]] ||
        die "${DATA_DISK_MOUNT} holds no ${required} directory; wrong disk or wrong layout"
done

################################################################################
# Install
################################################################################

export DEBIAN_FRONTEND=noninteractive

# Unattended upgrades hold the dpkg lock through the first minutes of a boot,
# which is exactly when this runs, so a first apt failure means "wait", not
# "give up".
apt_get() {
    local attempt
    for attempt in 1 2 3 4 5 6; do
        apt-get "$@" && return 0
        say "apt-get $1 failed (attempt ${attempt}); the dpkg lock is probably held"
        sleep 20
    done
    return 1
}

apt_get update -y || die 'apt-get update failed'
apt_get install -y git python3 python3-pip python3-venv || die 'apt-get install failed'

if [[ -d ${SPINDOCTOR_DIR}/.git ]]; then
    say "Reusing the checkout in ${SPINDOCTOR_DIR}"
else
    # Checked out detached rather than cloned at a branch, so that
    # SPINDOCTOR_GIT_REF may name a commit and a pool can be pinned to the
    # revision it was meant to run rather than to wherever a branch has moved.
    say "Cloning ${SPINDOCTOR_GIT_URL_SAFE} at ${SPINDOCTOR_GIT_REF} into ${SPINDOCTOR_DIR}"
    git clone --filter=blob:none "${SPINDOCTOR_GIT_URL}" "${SPINDOCTOR_DIR}" ||
        die 'git clone failed'
    git -C "${SPINDOCTOR_DIR}" checkout --detach "${SPINDOCTOR_GIT_REF}" ||
        die "No such revision in ${SPINDOCTOR_GIT_URL_SAFE}: ${SPINDOCTOR_GIT_REF}"
fi

INSTALLED_COMMIT="$(git -C "${SPINDOCTOR_DIR}" rev-parse HEAD)" ||
    die 'Cannot read the checked-out commit'
say "Installing commit ${INSTALLED_COMMIT}"

cd "${SPINDOCTOR_DIR}" || die "Cannot enter ${SPINDOCTOR_DIR}"
python3 -m venv venv || die 'Cannot create the virtual environment'
source venv/bin/activate
pip install --upgrade pip || die 'Cannot upgrade pip'
pip install -e . || die 'Cannot install SpinDoctor'
say "Installed $(python3 -c 'import spindoctor; print(spindoctor.__version__)' 2>/dev/null ||
    echo 'SpinDoctor') at ${INSTALLED_COMMIT}"

################################################################################
# Environment for the worker
################################################################################

# The data the navigation reads, all of it on the read-only disk.
export OOPS_RESOURCES="${DATA_DISK_MOUNT}"
export SPICE_PATH="${DATA_DISK_MOUNT}/SPICE"
export UCAC4_PATH="${DATA_DISK_MOUNT}/star-catalogs/UCAC4"
export YBSC_PATH="${DATA_DISK_MOUNT}/star-catalogs/YBSC"

export PDS3_HOLDINGS_DIR
export NAV_RESULTS_ROOT
export FILECACHE_CACHE_ROOT
mkdir -p "${FILECACHE_CACHE_ROOT}" || die "Cannot create ${FILECACHE_CACHE_ROOT}"

# One task already occupies one core: cloud_tasks runs
# RMS_CLOUD_TASKS_NUM_TASKS_PER_INSTANCE of them at once.  Left unpinned, each
# task's BLAS would open a thread per core on top of that and the instance
# would spend its time in contention.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

say "Data disk    ${DATA_DISK_MOUNT}"
say "Holdings     ${PDS3_HOLDINGS_DIR}"
say "Results      ${NAV_RESULTS_ROOT}"
say "File cache   ${FILECACHE_CACHE_ROOT}"
say "Tasks at once ${RMS_CLOUD_TASKS_NUM_TASKS_PER_INSTANCE:-unset}"

################################################################################
# Run the worker
################################################################################

say "Running ${SPINDOCTOR_WORKER} ${SPINDOCTOR_WORKER_ARGS}"
# shellcheck disable=SC2086
"${SPINDOCTOR_WORKER}" ${SPINDOCTOR_WORKER_ARGS}
status=$?
say "${SPINDOCTOR_WORKER} exited with status ${status}"

# cloud_tasks terminates the pool once the queue drains, so the instance is
# deliberately left running here.  Uncomment to have it stop itself instead,
# which costs less when a worker exits early but hides the instance from
# anyone wanting to look at it.
# shutdown -h now

exit ${status}
