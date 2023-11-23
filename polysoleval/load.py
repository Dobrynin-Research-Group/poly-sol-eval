import json
from pathlib import Path
from typing import Any, Optional

from ruamel.yaml import YAML
import torch

from .responses import ModelResponse, RangeResponse


__all__ = [
    "MaybeModel",
    "load_models_from_yaml",
    "load_range_from_yaml",
    "load_model_file",
]

_yaml_load = YAML(typ="safe", pure=True).load
MaybeModel = Optional[torch.nn.Module]


def load_models_from_yaml(filepath: Path) -> list[ModelResponse]:
    with open(filepath, "r") as fp:
        model_list: list[dict[str, Any]] = _yaml_load(fp)

    responses: list[ModelResponse] = list()
    for m in model_list:
        responses.append(ModelResponse.model_validate_json(json.dumps(m)))
    return responses


def load_range_from_yaml(filepath: Path) -> RangeResponse:
    with open(filepath, "r") as fp:
        range_dict: dict[str, Any] = _yaml_load(fp)

    range_dict["name"] = filepath.stem
    range_dict["phi_res"] = range_dict["phi_range"].pop("shape")
    range_dict["nw_res"] = range_dict["nw_range"].pop("shape")

    return RangeResponse.model_validate_json(json.dumps(range_dict))


def load_model_file(filepath: Path) -> MaybeModel:
    if not filepath.is_file():
        return None

    return torch.jit.load(filepath)
