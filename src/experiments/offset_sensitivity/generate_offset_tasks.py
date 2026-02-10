#!/usr/bin/env python3
"""Generate a grid of navigation tasks with varying U/V offsets.

This program creates a JSON file containing tasks for testing navigation
sensitivity to offset variations. Each task uses the same simulated model
template but with different offset_u and offset_v values.
"""

import argparse
import json

from filecache import FCPath
import numpy as np


def positive_float(value: str) -> float:
    """Type function for argparse that ensures a positive float value.

    Parameters:
        value: String value to convert to float.

    Returns:
        Positive float value.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive float.
    """
    try:
        fval = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f'Invalid float value: {value}') from e
    if fval <= 0:
        raise argparse.ArgumentTypeError(f'Stride must be positive, got: {value}')
    return fval


def parse_slice(slice_str: str) -> tuple[float, float, float]:
    """Parse a slice string of the form min:max:stride.

    Parameters:
        slice_str: String in the format "min:max:stride".

    Returns:
        Tuple of (min, max, stride).

    Raises:
        ValueError: If the slice string is invalid or stride is not positive.
    """
    parts = slice_str.split(':')
    if len(parts) != 3:
        raise ValueError(f'Invalid slice format: {slice_str}. Expected min:max:stride')
    try:
        min_val = float(parts[0])
        max_val = float(parts[1])
        stride = float(parts[2])
    except ValueError as e:
        raise ValueError(f'Invalid slice format: {slice_str}. {e}') from e
    if stride <= 0:
        raise ValueError(
            f'Stride must be positive, got: {stride} in slice "{slice_str}"'
        )
    return (min_val, max_val, stride)


def format_offset(offset: float) -> str:
    """Format an offset value for use in filenames.

    Handles negative signs and ensures consistent formatting.
    """
    # Use a format that handles negatives well (e.g., -0.010 becomes -0.01)
    return f'{offset:.6f}'.rstrip('0').rstrip('.')


def generate_tasks(
    template_path: FCPath,
    u_min: float,
    u_max: float,
    u_stride: float,
    v_min: float,
    v_max: float,
    v_stride: float,
    output_path: FCPath,
) -> None:
    """Generate tasks for a grid of U/V offsets.

    Parameters:
        template_path: Path to the template JSON file.
        u_min: Minimum U offset.
        u_max: Maximum U offset.
        u_stride: Stride for U offset.
        v_min: Minimum V offset.
        v_max: Maximum V offset.
        v_stride: Stride for V offset.
        output_path: Path to write the output tasks JSON file.
    """
    # Print offset information before generating tasks
    print(f'U offsets: {u_min} to {u_max} (stride {u_stride})')
    print(f'V offsets: {v_min} to {v_max} (stride {v_stride})')

    with template_path.open() as f:
        template_data = json.load(f)

    template_name = template_path.stem

    u_offsets_arr = np.arange(u_min, u_max + u_stride / 2, u_stride)
    v_offsets_arr = np.arange(v_min, v_max + v_stride / 2, v_stride)

    u_offsets: list[float] = [float(x) for x in u_offsets_arr]
    v_offsets: list[float] = [float(x) for x in v_offsets_arr]

    print(f'U offsets: {len(u_offsets)} values')
    print(f'V offsets: {len(v_offsets)} values')
    print(f'Total tasks: {len(u_offsets) * len(v_offsets)}')
    print()

    tasks = []
    task_index = 0

    # Generate a task for each U/V combination
    for u_offset in u_offsets:
        for v_offset in v_offsets:
            # Create sim_params with updated offsets
            sim_params = template_data.copy()
            sim_params['offset_u'] = u_offset
            sim_params['offset_v'] = v_offset

            u_str = format_offset(float(u_offset))
            v_str = format_offset(float(v_offset))

            results_path_stub = f'{template_name}_{u_str}_{v_str}'

            task_id = f'sim-{template_name}-{u_str}_{v_str}-{task_index}'

            task = {
                'task_id': task_id,
                'data': {
                    'arguments': {
                        'nav_models': None,
                        'nav_techniques': None,
                    },
                    'dataset_name': 'sim',
                    'files': [
                        {
                            'image_file_url': str(template_path),
                            'label_file_url': str(template_path),
                            'results_path_stub': results_path_stub,
                            'index_file_row': {},
                            'extra_params': {
                                'sim_params': sim_params,
                            },
                        }
                    ],
                },
            }

            tasks.append(task)
            task_index += 1

    with output_path.open('w') as f:
        json.dump(tasks, f, indent=2)

    print(f'Generated {len(tasks)} tasks')
    print(f'Output written to: {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate a grid of navigation tasks with varying U/V offsets'
    )

    parser.add_argument(
        '--model-template',
        type=str,
        required=True,
        help='Path to template simulated model JSON file',
    )

    parser.add_argument(
        '--u',
        type=str,
        default=None,
        help='U offset slice in the form min:max:stride (e.g., "0:1:0.1")',
    )

    parser.add_argument(
        '--u-min',
        type=float,
        default=0.0,
        help='Minimum U offset (default: 0.0)',
    )

    parser.add_argument(
        '--u-max',
        type=float,
        default=1.0,
        help='Maximum U offset (default: 1.0)',
    )

    parser.add_argument(
        '--u-stride',
        type=positive_float,
        default=0.1,
        help='Stride for U offset (default: 0.1)',
    )

    parser.add_argument(
        '--v',
        type=str,
        default=None,
        help='V offset slice in the form min:max:stride (e.g., "0:1:0.1")',
    )

    parser.add_argument(
        '--v-min',
        type=float,
        default=0.0,
        help='Minimum V offset (default: 0.0)',
    )

    parser.add_argument(
        '--v-max',
        type=float,
        default=1.0,
        help='Maximum V offset (default: 1.0)',
    )

    parser.add_argument(
        '--v-stride',
        type=positive_float,
        default=0.1,
        help='Stride for V offset (default: 0.1)',
    )

    parser.add_argument(
        '--task-file',
        type=str,
        required=True,
        help='Output task JSON file path',
    )

    args = parser.parse_args()

    # Parse slice notation if provided, otherwise use individual arguments
    if args.u is not None:
        try:
            u_min, u_max, u_stride = parse_slice(args.u)
        except ValueError as e:
            parser.error(f'Invalid --u argument: {e}')
    else:
        u_min = args.u_min
        u_max = args.u_max
        u_stride = args.u_stride

    if args.v is not None:
        try:
            v_min, v_max, v_stride = parse_slice(args.v)
        except ValueError as e:
            parser.error(f'Invalid --v argument: {e}')
    else:
        v_min = args.v_min
        v_max = args.v_max
        v_stride = args.v_stride

    template_path = FCPath(args.model_template)
    output_path = FCPath(args.task_file)

    generate_tasks(
        template_path=template_path,
        u_min=u_min,
        u_max=u_max,
        u_stride=u_stride,
        v_min=v_min,
        v_max=v_max,
        v_stride=v_stride,
        output_path=output_path,
    )


if __name__ == '__main__':
    main()
