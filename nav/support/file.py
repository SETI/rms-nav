import sys
from typing import Any

import numpy as np
from ruamel.yaml import YAML


def _clean_val(v: Any) -> Any:
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.int_, np.int8, np.int16, np.int32, np.int64)):
        return int(v)
    if isinstance(v, (np.float32, np.float64)):
        return float(v)
    if isinstance(v, np.ndarray):
        return _clean_list(list(v))
    return v


def _clean_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        _clean_dict(obj)
    elif isinstance(obj, (list, tuple)):
        _clean_list(obj)
    else:
        obj = _clean_val(obj)
    return obj


def _clean_dict(obj: dict[Any, Any]) -> dict[Any, Any]:
    for k, v in obj.items():
        obj[k] = _clean_obj(v)
    return obj


def _clean_list(obj: list[Any] | tuple[Any, ...]) -> list[Any]:
    obj = list(obj)
    for i, v in enumerate(obj):
        obj[i] = _clean_obj(v)
    return obj


def dump_yaml(data: Any) -> None:
    data = _clean_obj(data)
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.dump(data, sys.stdout)
