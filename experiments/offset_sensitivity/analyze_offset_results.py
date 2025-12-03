#!/usr/bin/env python3
"""Analyze navigation results and create heatmaps showing offset accuracy.

This program scans a results directory for metadata files, extracts the
original grid offsets from filenames, compares them to the found offsets
in the metadata, and creates heatmaps showing the U and V offset errors.
"""

import argparse
import json
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from filecache import FCPath


def parse_filename(filename: str) -> Optional[tuple[str, float, float]]:
    """Parse a metadata filename to extract template name and offsets.

    Expected format: {template_name}_{u_offset}_{v_offset}_metadata.json

    Parameters:
        filename: The filename to parse.

    Returns:
        Tuple of (template_name, u_offset, v_offset) or None if parsing fails.
    """
    if not filename.endswith('_metadata.json'):
        print(f'Warning: Filename does not end with "_metadata.json": {filename}')
        return None

    base = filename[:-len('_metadata.json')]

    # Try to find the last two numbers (offsets) in the filename
    # Pattern: name followed by _number_number
    # We need to handle negative numbers too
    pattern = r'^(.+)_(-?\d+\.?\d*)_(-?\d+\.?\d*)$'
    match = re.match(pattern, base)

    if match:
        template_name = match.group(1)
        try:
            u_offset = float(match.group(2))
            v_offset = float(match.group(3))
            return (template_name, u_offset, v_offset)
        except ValueError as e:
            print(f'Warning: Could not convert offsets to float in {filename}: {e}')
            return None

    print(f'Warning: Filename does not match expected pattern: {filename}')
    return None


def collect_results(
    nav_results_root: FCPath,
    template_name_filter: Optional[str] = None,
) -> dict[tuple[float, float], tuple[float, float]]:
    """Collect results from metadata files.

    Parameters:
        nav_results_root: Root directory containing metadata files.
        template_name_filter: Optional template name prefix to filter by.

    Returns:
        Dictionary mapping (original_u, original_v) to (found_u, found_v).
    """
    results: dict[tuple[float, float], tuple[float, float]] = {}

    nav_results_root = nav_results_root.absolute()
    if not nav_results_root.exists():
        print(f'Error: Results directory does not exist: {nav_results_root}')
        return results

    # Get all files in the directory
    try:
        files = list(nav_results_root.iterdir())
    except Exception as e:
        print(f'Error scanning directory {nav_results_root}: {e}')
        return results

    metadata_files = [f for f in files if f.name.endswith('_metadata.json')]

    print(f'Found {len(metadata_files)} metadata files')

    for metadata_file in metadata_files:
        # Parse filename
        parsed = parse_filename(metadata_file.name)
        if parsed is None:
            continue

        template_name, original_u, original_v = parsed

        # Apply template name filter if provided
        if template_name_filter is not None:
            if not template_name.startswith(template_name_filter):
                continue

        # Read metadata
        try:
            with metadata_file.open() as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            print(f'Warning: Invalid JSON in {metadata_file.name}: {e}')
            continue
        except OSError as e:
            print(f'Warning: Could not read metadata file {metadata_file.name}: {e}')
            continue

        # Check status
        if metadata.get('status') != 'success':
            print(f'Skipping {metadata_file.name}: status is {metadata.get("status")}')
            continue

        # Get offset
        offset = metadata.get('offset')
        if offset is None:
            print(f'Skipping {metadata_file.name}: offset is None')
            continue

        # Offset format is [dv, du] according to nav_master.py
        # Explicitly validate type and length
        if not isinstance(offset, (list, tuple)):
            print(f'Warning: Invalid offset type in {metadata_file.name}: {type(offset)}')
            continue
        if len(offset) != 2:
            print(f'Warning: Invalid offset length in {metadata_file.name}: {len(offset)}')
            continue

        try:
            found_v, found_u = float(offset[0]), float(offset[1])
        except (ValueError, TypeError) as e:
            print(f'Warning: Could not convert offset values to float in {metadata_file.name}: {e}')
            continue

        # Store result
        results[(original_u, original_v)] = (found_u, found_v)

    print(f'Successfully processed {len(results)} metadata files')
    return results


def filter_results(
    results: dict[tuple[float, float], tuple[float, float]],
    u_min: Optional[float] = None,
    u_max: Optional[float] = None,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
) -> dict[tuple[float, float], tuple[float, float]]:
    """Filter results by U and V offset ranges.

    Parameters:
        results: Dictionary mapping (original_u, original_v) to (found_u, found_v).
        u_min: Minimum U offset to include (inclusive). If None, no lower bound.
        u_max: Maximum U offset to include (inclusive). If None, no upper bound.
        v_min: Minimum V offset to include (inclusive). If None, no lower bound.
        v_max: Maximum V offset to include (inclusive). If None, no upper bound.

    Returns:
        Filtered dictionary mapping (original_u, original_v) to (found_u, found_v).
    """
    filtered = {}
    for (orig_u, orig_v), (found_u, found_v) in results.items():
        if u_min is not None and orig_u < u_min:
            continue
        if u_max is not None and orig_u > u_max:
            continue
        if v_min is not None and orig_v < v_min:
            continue
        if v_max is not None and orig_v > v_max:
            continue
        filtered[(orig_u, orig_v)] = (found_u, found_v)
    return filtered


def create_heatmaps(
    results: dict[tuple[float, float], tuple[float, float]],
    output_path: Optional[FCPath] = None,
    show: str = 'both',
) -> None:
    """Create heatmaps showing U and V offset errors.

    Parameters:
        results: Dictionary mapping (original_u, original_v) to (found_u, found_v).
        output_path: Optional path to save plots. If None, show interactively.
        show: Which heatmaps to display: 'both', 'u', or 'v'.
    """
    if not results:
        print('No results to plot')
        return

    # Extract all unique U and V values
    u_values = sorted({u for u, v in results.keys()})
    v_values = sorted({v for u, v in results.keys()})

    # Handle edge case: pcolormesh needs at least 2 points in each dimension
    # Add a small padding if there's only one value, keeping the original value
    if len(u_values) == 1:
        u_padding = abs(u_values[0]) * 0.01 if u_values[0] != 0 else 0.01
        u_values = [u_values[0] - u_padding, u_values[0], u_values[0] + u_padding]
    if len(v_values) == 1:
        v_padding = abs(v_values[0]) * 0.01 if v_values[0] != 0 else 0.01
        v_values = [v_values[0] - v_padding, v_values[0], v_values[0] + v_padding]

    # Create grid
    u_grid, v_grid = np.meshgrid(u_values, v_values)

    # Initialize error arrays
    u_error_grid = np.full_like(u_grid, np.nan, dtype=float)
    v_error_grid = np.full_like(v_grid, np.nan, dtype=float)

    # Fill in error values
    for (orig_u, orig_v), (found_u, found_v) in results.items():
        # Find indices (handle case where we padded single values)
        u_idx = u_values.index(orig_u) if orig_u in u_values else min(
            range(len(u_values)), key=lambda i: abs(u_values[i] - orig_u)
        )
        v_idx = v_values.index(orig_v) if orig_v in v_values else min(
            range(len(v_values)), key=lambda i: abs(v_values[i] - orig_v)
        )

        # Compute errors
        u_error = found_u - orig_u
        v_error = found_v - orig_v

        # Note: meshgrid creates arrays where first dimension is v, second is u
        u_error_grid[v_idx, u_idx] = u_error
        v_error_grid[v_idx, u_idx] = v_error

    # Determine which plots to show
    show_u = show in ('both', 'u')
    show_v = show in ('both', 'v')

    # Create subplots based on what to show
    if show_u and show_v:
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        axes = [ax1, ax2]
    else:
        _fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        axes = [ax]

    # Plot U error heatmap
    if show_u:
        ax = axes[0]
        im1 = ax.pcolormesh(u_grid, v_grid, u_error_grid, shading='auto', cmap='RdBu_r')
        ax.set_xlabel('Original U Offset')
        ax.set_ylabel('Original V Offset')
        ax.set_title('U Offset Error (Found - Original)')
        ax.set_aspect('equal')
        plt.colorbar(im1, ax=ax, label='U Error')

    # Plot V error heatmap
    if show_v:
        ax = axes[1] if show_u and show_v else axes[0]
        im2 = ax.pcolormesh(u_grid, v_grid, v_error_grid, shading='auto', cmap='RdBu_r')
        ax.set_xlabel('Original U Offset')
        ax.set_ylabel('Original V Offset')
        ax.set_title('V Offset Error (Found - Original)')
        ax.set_aspect('equal')
        plt.colorbar(im2, ax=ax, label='V Error')

    if show_u and show_v:
        _fig.tight_layout()
    else:
        # For single plot, use tight_layout then adjust to remove extra margins
        _fig.tight_layout()
        _fig.subplots_adjust(left=0.12, right=0.92, top=0.95, bottom=0.1)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        local_path = output_path.get_local_path()
        plt.savefig(local_path, dpi=150, bbox_inches='tight')
        output_path.upload()
        print(f'Plots saved to: {output_path}')
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Analyze navigation results and create offset error heatmaps'
    )

    parser.add_argument(
        '--nav-results-root',
        type=str,
        required=True,
        help='Root directory containing metadata files',
    )

    parser.add_argument(
        '--template-name',
        type=str,
        default=None,
        help='Optional template name prefix to filter by',
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output plot filename (optional - if not provided, show interactively)',
    )

    parser.add_argument(
        '--show',
        type=str,
        choices=['both', 'u', 'v'],
        default='both',
        help='Which heatmaps to display: "both", "u", or "v" (default: both).',
    )

    parser.add_argument(
        '--u-min',
        type=float,
        default=None,
        help='Minimum U offset to include (inclusive). If not specified, use entire range.',
    )

    parser.add_argument(
        '--u-max',
        type=float,
        default=None,
        help='Maximum U offset to include (inclusive). If not specified, use entire range.',
    )

    parser.add_argument(
        '--v-min',
        type=float,
        default=None,
        help='Minimum V offset to include (inclusive). If not specified, use entire range.',
    )

    parser.add_argument(
        '--v-max',
        type=float,
        default=None,
        help='Maximum V offset to include (inclusive). If not specified, use entire range.',
    )

    args = parser.parse_args()

    nav_results_root = FCPath(args.nav_results_root)
    output_path = FCPath(args.output_file) if args.output_file else None

    results = collect_results(nav_results_root, args.template_name)

    # Filter results by U/V ranges if specified
    if (args.u_min is not None or args.u_max is not None or
            args.v_min is not None or args.v_max is not None):
        results = filter_results(
            results,
            u_min=args.u_min,
            u_max=args.u_max,
            v_min=args.v_min,
            v_max=args.v_max,
        )
        print(f'After filtering: {len(results)} results')

    create_heatmaps(results, output_path, show=args.show)


if __name__ == '__main__':
    main()
