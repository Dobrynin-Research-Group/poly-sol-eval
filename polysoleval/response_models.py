from dataclasses import field
import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, PositiveFloat, PositiveInt, FilePath
from pydantic.dataclasses import dataclass
from ruamel.yaml import YAML
import torch


__all__ = [
    "MaybeModel",
    "load_range_from_yaml",
    "BasicRange",
    "RangeSet",
    "ModelType",
    "ModelInstance",
    "ModelTypesResponse",
    "ModelInstancesResponse",
    # "RepeatUnitRequest",
    "EvaluationCase",
    "EvaluationResponse",
]

yaml_parser = YAML(typ="safe", pure=True)
MaybeModel = Optional[torch.nn.Module]


class BasicRange(BaseModel):
    min_value: float
    max_value: float
    log_scale: bool = False


class RangeSet(BaseModel):
    name: str

    phi_res: PositiveInt
    nw_res: PositiveInt

    phi_range: BasicRange
    nw_range: BasicRange
    visc_range: BasicRange
    bg_range: BasicRange
    bth_range: BasicRange
    pe_range: BasicRange


class ModelType(BaseModel):
    name: str
    description: Optional[str] = None
    link: Optional[str] = None

    @classmethod
    def all_from_yaml(cls, yaml_file: str | Path):
        yaml_file = Path(yaml_file)
        if not yaml_file.is_file():
            raise FileNotFoundError(str(yaml_file))

        with open(yaml_file, "r") as f:
            model_dict: dict[str, Any] = yaml_parser.load(f)

        return [
            cls(name=k, description=v["description"], link=v["link"])
            for k, v in model_dict.items()
        ]


class ModelTypesResponse(BaseModel):
    model_types: list[ModelType]


class ModelInstancesResponse(BaseModel):
    model_instances: list[RangeSet]


@dataclass
class ModelInstance:
    model_type: ModelType
    range_path: FilePath
    bg_model_path: FilePath
    bth_model_path: FilePath
    bg_model: torch.nn.Module = field(init=False, repr=False, compare=False)
    bth_model: torch.nn.Module = field(init=False, repr=False, compare=False)
    range_set: RangeSet = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self.bg_model = torch.jit.load(self.bg_model_path)
        self.bth_model = torch.jit.load(self.bth_model_path)
        self.range_set = load_range_from_yaml(self.range_path)


# @dataclass
# class RepeatUnitRequest(BaseModel):
#     length: PositiveFloat
#     mass: PositiveFloat


class EvaluationCase(BaseModel):
    bg: Optional[PositiveFloat] = None
    bth: Optional[PositiveFloat] = None
    bg_plateau: Optional[PositiveFloat] = None
    bth_plateau: Optional[PositiveFloat] = None
    pe: PositiveFloat = 1e-9
    pe_variance: Optional[PositiveFloat] = None
    kuhn_length: Optional[PositiveFloat] = None
    thermal_blob_size: Optional[PositiveFloat] = None
    dp_of_thermal_blob: Optional[PositiveFloat] = None
    excluded_volume: Optional[PositiveFloat] = None
    thermal_blob_conc: Optional[PositiveFloat] = None
    concentrated_conc: Optional[PositiveFloat] = None


class EvaluationResponse(BaseModel):
    bg_only: EvaluationCase
    bth_only: EvaluationCase
    both_bg_and_bth: EvaluationCase
    token: str


def load_range_from_yaml(filepath: Path) -> RangeSet:
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
        range_dict: dict[str, Any] = yaml_parser.load(fp)

    range_dict["name"] = filepath.stem
    range_dict["phi_res"] = range_dict["phi_range"].pop("shape")
    range_dict["nw_res"] = range_dict["nw_range"].pop("shape")

    return RangeSet.model_validate_json(json.dumps(range_dict))
