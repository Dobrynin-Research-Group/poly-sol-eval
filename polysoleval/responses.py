from typing import Optional

from pydantic import BaseModel


__all__ = [
    "ModelResponse",
    "BasicRange",
    "RangeResponse",
    "InputParameters",
    "EvaluationResult",
    "EvaluationResponse",
]


class ModelResponse(BaseModel):
    name: str
    description: Optional[str] = None
    link: Optional[str] = None


class BasicRange(BaseModel):
    min_value: float
    max_value: float
    log_scale: bool = False


class RangeResponse(BaseModel):
    name: str

    phi_res: int
    nw_res: int

    phi_range: BasicRange
    nw_range: BasicRange
    visc_range: BasicRange
    bg_range: BasicRange
    bth_range: BasicRange
    pe_range: BasicRange


class InputParameters(BaseModel):
    bg: Optional[float]
    bth: Optional[float]
    pe: float
    pe_variance: Optional[float]
    rep_unit_length: Optional[float]
    rep_unit_mass: Optional[float]


class EvaluationResult(BaseModel):
    bg: Optional[float] = None
    bth: Optional[float] = None
    pe: float = 0.0
    pe_variance: Optional[float] = None
    kuhn_length: float = 0.0
    thermal_blob_size: Optional[float] = None
    dp_of_thermal_blob: Optional[float] = None
    excluded_volume: Optional[float] = None
    thermal_blob_conc: Optional[float] = None
    concentrated_conc: float = 0.0


class EvaluationResponse(BaseModel):
    bg_only: EvaluationResult
    bth_only: EvaluationResult
    both_bg_and_bth: EvaluationResult
