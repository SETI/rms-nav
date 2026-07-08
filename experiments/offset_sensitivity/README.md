# Offset Sensitivity Analysis

This directory contains tools for analyzing navigation sensitivity to offset variations by generating a grid of simulated navigation tasks and analyzing the results.

**Note**: When specifying negative values for command-line arguments (e.g., `--u-min`, `--u-max`, `--v-min`, `--v-max`, or slice notation like `--u`), use the `=` syntax (e.g., `--u-min=-0.5`) to avoid the negative sign being interpreted as a separate argument.

## Purpose

These programs enable systematic testing of navigation accuracy across a range of U/V offset values. The workflow consists of:

1. **Task Generation**: Create a grid of navigation tasks, each using the same simulated model template but with different `offset_u` and `offset_v` values embedded in the task parameters.

2. **Result Analysis**: Collect navigation results from the generated tasks, extract the computed offsets, and visualize the error between the original grid offsets and the found offsets as heatmaps.

This approach allows identification of systematic biases, non-linearities, or regions of degraded performance in the navigation system.

## Task Generator: `generate_offset_tasks.py`

Generates a JSON file containing navigation tasks for testing offset sensitivity. Each task uses the same simulated model template but with different `offset_u` and `offset_v` values specified in the model JSON.

### Usage

```bash
python experiments/offset_sensitivity/generate_offset_tasks.py \
    --model-template <template.json> \
    --task-file <output_tasks.json> \
    [offset range options]
```

### Options

#### Required Arguments

- `--model-template <path>`: Path to the template simulated model JSON file (as created by `nav_create_simulated_model.py`). This file defines the base simulated image parameters (bodies, stars, sizes, etc.). The generator will create tasks with identical parameters except for the `offset_u` and `offset_v` values.

- `--task-file <path>`: Path to write the output task JSON file. The file will contain an array of task objects compatible with `nav_offset_cloud_tasks`.

#### Offset Range Options

Offset ranges can be specified either using slice notation or individual min/max/stride arguments. Slice notation takes precedence if provided.

**Slice Notation (recommended)**:
- `--u <min:max:stride>`: U offset range in the form `min:max:stride` (e.g., `--u "-1:1:0.1"` for -1.0 to 1.0 in steps of 0.1).
- `--v <min:max:stride>`: V offset range in the form `min:max:stride` (e.g., `--v "-1:1:0.1"`).

**Individual Arguments** (defaults: min=0.0, max=1.0, stride=0.1):
- `--u-min <float>`: Minimum U offset.
- `--u-max <float>`: Maximum U offset.
- `--u-stride <float>`: Stride for U offset.
- `--v-min <float>`: Minimum V offset.
- `--v-max <float>`: Maximum V offset.
- `--v-stride <float>`: Stride for V offset.

### Example

Generate tasks for a U offset range from -0.5 to 0.5 (step 0.1) and V offset range from -0.3 to 0.3 (step 0.05):

```bash
python experiments/offset_sensitivity/generate_offset_tasks.py \
    --model-template model.json \
    --task-file offset_tasks.json \
    --u "-0.5:0.5:0.1" \
    --v "-0.3:0.3:0.05"
```

This creates a grid of 11 x 13 = 143 tasks, each with a unique combination of U and V offsets.

### Output Format

Each task in the output JSON file contains:
- `task_id`: Unique identifier for the task.
- `data.dataset_name`: Set to `"sim"` for simulated observations.
- `data.files[0].image_file_url`: Points to the template JSON file.
- `data.files[0].label_file_url`: Points to the template JSON file.
- `data.files[0].results_path_stub`: Format `{template_name}_{u_offset}_{v_offset}` (e.g., `model_0.1_-0.2`).
- `data.files[0].extra_params.sim_params`: Complete simulated model parameters including the grid-specific `offset_u` and `offset_v` values.

## Running Navigation Tasks

After generating the task file, the tasks must be executed to produce navigation results. Tasks can be run either locally or in the cloud.

### Local Execution

To run tasks locally using multiple CPU cores:

```bash
nav_offset_cloud_tasks \
    --task-file offset_tasks.json \
    --nav-results-root /path/to/nav_results \
    --num-cpus 16
```

The `--nav-results-root` directory will contain the navigation results, with each task producing a metadata file named `{results_path_stub}_metadata.json`.

### Cloud Execution

Tasks can also be executed in the cloud using the cloud tasks infrastructure. For details on cloud execution, see the [rms-cloud-tasks documentation](https://rms-cloud-tasks.readthedocs.io).

## Result Analyzer: `analyze_offset_results.py`

Scans a navigation results directory for metadata files, extracts original grid offsets from filenames, compares them to the computed offsets in the metadata, and creates heatmaps showing the U and V offset errors.

### Usage

```bash
python experiments/offset_sensitivity/analyze_offset_results.py \
    --nav-results-root <results_directory> \
    [filtering and display options]
```

### Options

#### Required Arguments

- `--nav-results-root <path>`: Root directory containing the navigation results. The program scans for files matching the pattern `*_metadata.json` and extracts U/V offsets from the filenames (format: `{template_name}_{u_offset}_{v_offset}_metadata.json`).

#### Filtering Options

- `--template-name <prefix>`: Optional template name prefix to filter by. Only metadata files whose template name starts with this prefix will be processed.

- `--u-min <float>`: Minimum U offset to include in the analysis (inclusive). If not specified, uses the entire range of loaded data.

- `--u-max <float>`: Maximum U offset to include in the analysis (inclusive). If not specified, uses the entire range of loaded data.

- `--v-min <float>`: Minimum V offset to include in the analysis (inclusive). If not specified, uses the entire range of loaded data.

- `--v-max <float>`: Maximum V offset to include in the analysis (inclusive). If not specified, uses the entire range of loaded data.

#### Display Options

- `--show <choice>`: Which heatmaps to display. Choices: `both` (default), `u`, or `v`. When `both` is selected, two side-by-side heatmaps are shown. When `u` or `v` is selected, only the corresponding error heatmap is displayed.

- `--output-file <path>`: Optional path to save the plot. The suffix determines the file format (`eps`, `jpeg`, `jpg`, `pdf`, `pgf`, `png`, `ps`, `raw`, `rgba`, `svg`, `svgz`, `tif`, `tiff`, `webp`). If not provided, the plot is displayed interactively using matplotlib's `show()`.

### Example

Analyze all results and display both U and V error heatmaps interactively:

```bash
python experiments/offset_sensitivity/analyze_offset_results.py \
    --nav-results-root /path/to/nav_results
```

Analyze only results with U offsets between -0.2 and 0.2, show only the U error heatmap, and save to a file:

```bash
python experiments/offset_sensitivity/analyze_offset_results.py \
    --nav-results-root /path/to/nav_results \
    --u-min -0.2 \
    --u-max 0.2 \
    --show u \
    --output-file u_error_heatmap.png
```

Filter by template name and V offset range:

```bash
python experiments/offset_sensitivity/analyze_offset_results.py \
    --nav-results-root /path/to/nav_results \
    --template-name model \
    --v-min -0.1 \
    --v-max 0.1
```

### Output

The program creates heatmaps where:
- **X-axis**: Original U offset (from filename)
- **Y-axis**: Original V offset (from filename)
- **Color/intensity**: Error value (found offset - original offset)

Two separate heatmaps are generated:
- **U Offset Error**: Shows `found_u - original_u` at each grid point
- **V Offset Error**: Shows `found_v - original_v` at each grid point

The heatmaps use a diverging colormap (RdBu_r) where:
- Red indicates positive error (found offset > original offset)
- Blue indicates negative error (found offset < original offset)
- White/neutral indicates zero error (perfect recovery)

Files with `status != 'success'` or `offset == None` in the metadata are skipped with warning messages.
