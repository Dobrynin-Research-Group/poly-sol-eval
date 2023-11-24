from dataclasses import dataclass, field
from math import log10

import torch


@dataclass(slots=True)
class Range:
    min_value: float
    max_value: float
    log_scale: bool = False
    scale_min: float = field(init=False)
    diff: float = field(init=False)

    def __post_init__(self):
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        if self.log_scale and self.min_value <= 0.0:
            raise ValueError(
                "min_value and max_value must be positive if log_scale is True"
            )
        if self.log_scale:
            self.scale_min = log10(self.min_value)
            self.diff = log10(self.max_value / self.min_value)
        else:
            self.scale_min = self.min_value
            self.diff = self.max_value - self.min_value

    def normalize(self, tensor: torch.Tensor) -> None:
        if self.log_scale:
            tensor.log10_()
        tensor -= self.scale_min
        tensor /= self.diff

    def unnormalize(self, tensor: torch.Tensor) -> None:
        tensor *= self.diff
        tensor += self.scale_min
        if self.log_scale:
            torch.pow(10.0, tensor, out=tensor)
