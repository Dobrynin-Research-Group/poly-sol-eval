from dataclasses import dataclass, field
from math import log10

import torch


@dataclass(slots=True)
class Range:
    min_value: float
    max_value: float
    log_scale: bool = False
    _scale_min: float = field(init=False, repr=False)
    _diff: float = field(init=False, repr=False)

    def __post_init__(self):
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        if self.log_scale and self.min_value <= 0.0:
            raise ValueError(
                "min_value and max_value must be positive if log_scale is True"
            )
        if self.log_scale:
            self._scale_min = log10(self.min_value)
            self._diff = log10(self.max_value / self.min_value)
        else:
            self._scale_min = self.min_value
            self._diff = self.max_value - self.min_value

    def normalize(self, tensor: torch.Tensor) -> None:
        if self.log_scale:
            tensor.log10_()
        tensor -= self._scale_min
        tensor /= self._diff

    def unnormalize(self, tensor: torch.Tensor) -> None:
        tensor *= self._diff
        tensor += self._scale_min
        if self.log_scale:
            torch.pow(10.0, tensor, out=tensor)
