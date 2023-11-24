import json
from pathlib import Path
from typing import Any, Optional

from ruamel.yaml import YAML
import torch

from .response_models import ModelResponse, RangeResponse


__all__ = [
    "MaybeModel",
    "load_models_from_yaml",
    "load_range_from_yaml",
    "load_model_file",
]

_yaml_load = YAML(typ="safe", pure=True).load
MaybeModel = Optional[torch.nn.Module]


def load_models_from_yaml(filepath: Path) -> list[ModelResponse]:
    """Load model descriptions from a YAML file.

    Contents of the file should be a list of mappings. The mappings should have the
    following keys: ``name``, ``description``, and ``link``, with only ``name`` being
    required.

    Args:
        filepath (Path): Filepath of the YAML file.

    Returns:
        list[ModelResponse]: A list of validated ``ModelResponse``s successfully read
          from the file.
    """
    with open(filepath, "r") as fp:
        model_list: list[dict[str, Any]] = _yaml_load(fp)

    responses: list[ModelResponse] = list()
    for m in model_list:
        responses.append(ModelResponse.model_validate_json(json.dumps(m)))
    return responses


def load_range_from_yaml(filepath: Path) -> RangeResponse:
    """Load a set of ranges from a YAML file.

    Contents of the file should a mapping of mappings with the following keys:
    ``phi_range``, ``nw_range``, ``visc_range``, ``bg_range``, ``bth_range``,
    ``pe_range``. Each mapping should have the keys ``min_value`` and ``max_value``
    and optionally ``log_scale``. Omitting ``log_scale`` is the same as
    ``log_scale: False``. ``phi_range`` and ``nw_range`` should each contain the key
    ``shape``, an integer describing the resolution of the data in that dimension.

    Args:
        filepath (Path): Filepath of the YAML file.

    Returns:
        RangeResponse: The validated ``RangeResponse``, with attributes ``phi_res``
          and ``nw_res`` taken as the respective values of ``shape``.
    """
    with open(filepath, "r") as fp:
        range_dict: dict[str, Any] = _yaml_load(fp)

    range_dict["name"] = filepath.stem
    range_dict["phi_res"] = range_dict["phi_range"].pop("shape")
    range_dict["nw_res"] = range_dict["nw_range"].pop("shape")

    return RangeResponse.model_validate_json(json.dumps(range_dict))


def load_model_file(filepath: Path) -> MaybeModel:
    """Load a pre-trained model from the given file.

    The file should contain a single TorchScript model stored using ``torch.jit.save``.

    Args:
        filepath (Path): Filepath of the TorchScript model.

    Returns:
        MaybeModel: Either ``None`` (if the file does not exist) or the stored model.
    """
    if not filepath.is_file():
        return None

    return torch.jit.load(filepath)
