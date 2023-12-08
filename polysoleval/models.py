from math import log10
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator, PositiveFloat
from ruamel.yaml import YAML
import torch


__all__ = [
    "Range",
    "RangeSet",
    "NeuralNetType",
    "NeuralNetPair",
    "RepeatUnit",
    "PeResult",
    "ComboCase",
    "BgCase",
    "BthCase",
]

_yaml_parser = YAML(typ="safe", pure=True)


class Range(BaseModel):
    min_value: float
    max_value: float
    log_scale: bool = False
    _scale_min: float = 1.0
    _diff: float = 1.0

    @model_validator(mode="after")
    def _validate_all(self):
        if self.max_value <= self.min_value:
            raise ValueError("min_value must be less than max_value")

        if self.log_scale:
            if self.min_value <= 0.0 or self.max_value <= 0.0:
                raise ValueError(
                    "min_value and max_value must be positive with log_scale=True"
                )
            self._diff = log10(self.max_value / self.min_value)
            self._scale_min = log10(self.min_value)
        else:
            self._diff = self.max_value - self.min_value
            self._scale_min = self.min_value

        return self

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor in-place according to the range.

        If the tensor only contains values between ``Range.min_value`` and
        ``Range.max_value`` before, then the tensor will only contain values between 0
        and 1 after.

        Args:
            tensor (torch.Tensor): The data to be normalized. This will occur in-place
              and on whatever device holds the tensor.
        """
        if self.log_scale:
            tensor.log10_()
        tensor -= self._scale_min
        tensor /= self._diff
        return tensor

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Unnormalize a tensor in-place according to the range.

        If the tensor only contains values between 0 and 1 before, then the tensor will
        only contain values between ``Range.min_value`` and ``Range.max_value`` after.

        Args:
            tensor (torch.Tensor): The data to be unnormalized. This will occur
              in-place and on whatever device holds the tensor.
        """
        tensor *= self._diff
        tensor += self._scale_min
        if self.log_scale:
            torch.pow(10.0, tensor, out=tensor)
        return tensor


class RangeSet(BaseModel):
    name: str

    phi_res: int = Field(gt=4)
    nw_res: int = Field(gt=4)

    phi_range: Range
    nw_range: Range
    visc_range: Range
    bg_range: Range
    bth_range: Range
    pe_range: Range

    @classmethod
    def from_yaml(cls, filepath: Path):
        with open(filepath, "r") as fp:
            range_dict: dict[str, Any] = _yaml_parser.load(fp)

        range_dict["name"] = filepath.stem
        range_dict["phi_res"] = range_dict["phi_range"].pop("shape")
        range_dict["nw_res"] = range_dict["nw_range"].pop("shape")

        # for key in range_dict:
        #     if key.endswith("_range"):
        #         value = range_dict[key]
        #         r = Range.model_construct(**value)
        #         range_dict[key] = Range.model_validate(r)
        # tmp_range = cls.model_construct(**range_dict)
        return cls(**range_dict)


class NeuralNetType(BaseModel):
    name: str
    description: Optional[str] = None
    link: Optional[str] = None

    @classmethod
    def all_from_yaml(cls, yaml_file: str | Path):
        yaml_file = Path(yaml_file)
        if not yaml_file.is_file():
            raise FileNotFoundError(str(yaml_file))

        with open(yaml_file, "r") as f:
            neuralnet_dict: dict[str, Any] = _yaml_parser.load(f)

        return [
            cls(name=k, description=v["description"], link=v["link"])
            for k, v in neuralnet_dict.items()
        ]


class NeuralNetPair:
    def __init__(
        self,
        neuralnet_type: NeuralNetType,
        range_path: Path,
        bg_net_path: Path,
        bth_net_path: Path,
    ):
        self.neuralnet_type = neuralnet_type
        self.range_set = RangeSet.from_yaml(range_path)
        self.bg_net: torch.nn.Module = torch.jit.load(bg_net_path)
        self.bth_net: torch.nn.Module = torch.jit.load(bth_net_path)


class RepeatUnit(BaseModel):
    length: PositiveFloat
    mass: PositiveFloat


class PeResult(BaseModel):
    value: PositiveFloat
    error: PositiveFloat


class EvaluationCase(BaseModel):
    pe_value: PositiveFloat
    pe_error: PositiveFloat


class ComboCase(EvaluationCase):
    bg: PositiveFloat
    bth: PositiveFloat
    bg_plateau: PositiveFloat
    bth_plateau: PositiveFloat
    kuhn_length: PositiveFloat
    thermal_blob_size: PositiveFloat
    dp_of_thermal_blob: PositiveFloat
    excluded_volume: PositiveFloat
    thermal_blob_conc: PositiveFloat
    concentrated_conc: PositiveFloat


class BgCase(EvaluationCase):
    bg: PositiveFloat
    bg_plateau: PositiveFloat


class BthCase(EvaluationCase):
    bth: PositiveFloat
    bth_plateau: PositiveFloat
    kuhn_length: PositiveFloat
    concentrated_conc: PositiveFloat
