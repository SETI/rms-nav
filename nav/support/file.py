import json
import sys
from typing import Any

import numpy as np
from ruamel.yaml import YAML


def _clean_val(v: Any) -> Any:
    """Converts NumPy scalar types to Python native types.

    Parameters:
        v: The value to convert.

    Returns:
        The converted value as a Python native type.
    """

    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return _clean_list(list(v))
    return v


def _clean_obj(obj: Any) -> Any:
    """Recursively converts NumPy types in any object to Python native types.

    Parameters:
        obj: The object to clean, can be a dict, list, tuple or scalar value.

    Returns:
        The object with all NumPy types converted to Python native types.
    """

    if isinstance(obj, dict):
        obj = _clean_dict(obj)
    elif isinstance(obj, (list, tuple)):
        obj = _clean_list(obj)
    else:
        obj = _clean_val(obj)
    return obj


def _clean_dict(obj: dict[Any, Any]) -> dict[Any, Any]:
    """Recursively converts NumPy types in a dictionary to Python native types.

    Parameters:
        obj: The dictionary to clean.

    Returns:
        The dictionary with all NumPy types converted to Python native types.
    """

    for k, v in obj.items():
        obj[k] = _clean_obj(v)
    return obj


def _clean_list(obj: list[Any] | tuple[Any, ...]) -> list[Any]:
    """Recursively converts NumPy types in a list or tuple to Python native types.

    Parameters:
        obj: The list or tuple to clean.

    Returns:
        A list with all NumPy types converted to Python native types. Note that a list
        is returned even if a tuple was provided.
    """

    obj = list(obj)
    for i, v in enumerate(obj):
        obj[i] = _clean_obj(v)
    return obj


def dump_yaml(data: Any, stream: Any = sys.stdout) -> None:
    """Dumps data as YAML output after converting NumPy types to Python types.

    Parameters:
        data: The data to dump as YAML.
    """

    data = _clean_obj(data)
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.dump(data, stream)


def json_as_string(data: Any) -> str:
    """Dumps data as a JSON string after converting NumPy types to Python types.

    Parameters:
        data: The data to dump as JSON.
    """

    data = _clean_obj(data)
    return json.dumps(data, indent=2)
