from dataclasses import dataclass, field
from math import log10

import torch


@dataclass(slots=True)
class Range:
    """
    ``Range(min_value: float, max_value: float, log_scale: bool = False)``

    Represent a range of values with methods for normalizing tensors.

    Attributes:
        min_value (float): The minimum value.
        max_value (float): The maximum value.
        log_scale (bool): If True, denotes a logarithmic/geometric scale for the range.
          If False, denotes a linear scale. Defaults to False.

    Raises:
        ValueError: If ``min_value >= max_value`` or if ``log_scale`` is True and
          either ``min_value`` or ``max_value`` are non-positive.
    """

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
