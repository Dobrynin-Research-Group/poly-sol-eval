from typing import Optional

from pydantic import BaseModel

from polysoleval.evaluate import AVOGADRO_CONSTANT


__all__ = [
    "ModelResponse",
    "BasicRange",
    "RangeResponse",
    "InputParameters",
    "EvaluationResult",
    "EvaluationResponse",
]


def phi_to_conc(phi: float, rep_unit_len: float):
    return phi / rep_unit_len**3 / AVOGADRO_CONSTANT * 1e24


def conc_to_phi(conc: float, rep_unit_len: float):
    return conc * rep_unit_len**3 * AVOGADRO_CONSTANT / 1e24


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

    @classmethod
    def create(
        cls,
        *,
        bg: Optional[float],
        bth: Optional[float],
        pe: float,
        pe_variance: Optional[float],
        rep_unit_length: float,
    ):
        thermal_blob_size = None
        dp_of_thermal_blob = None
        excluded_volume = None
        thermal_blob_conc = None
        concentrated_conc = None

        if bg and bth:
            kuhn_length = rep_unit_length / bth**2
            phi_th = bth**3 * (bth / bg) ** (1 / (2 * 0.588 - 1))
            thermal_blob_size = rep_unit_length * bth**2 / phi_th
            dp_of_thermal_blob = (bth**3 / phi_th) ** 2
            thermal_blob_conc = phi_to_conc(phi_th, rep_unit_length)
            excluded_volume = phi_th * kuhn_length**3
        elif bg:
            kuhn_length = rep_unit_length / bg ** (1 / 0.412)
        elif bth:
            kuhn_length = rep_unit_length / bth**2
        else:
            raise ValueError("must supply at least one of bg or bth")
        concentrated_conc = phi_to_conc(
            1 / kuhn_length**2 / rep_unit_length, rep_unit_length
        )

        return cls(
            bg=bg,
            bth=bth,
            pe=pe,
            pe_variance=pe_variance,
            kuhn_length=kuhn_length,
            thermal_blob_size=thermal_blob_size,
            excluded_volume=excluded_volume,
            dp_of_thermal_blob=dp_of_thermal_blob,
            thermal_blob_conc=thermal_blob_conc,
            concentrated_conc=concentrated_conc,
        )


class EvaluationResponse(BaseModel):
    bg_only: EvaluationResult
    bth_only: EvaluationResult
    both_bg_and_bth: EvaluationResult
