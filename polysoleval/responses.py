from pydantic import BaseModel

from polysoleval.conversions import phi_to_conc
from polysoleval.globals import *
from polysoleval.models import *


class NeuralNetTypes(BaseModel):
    neuralnet_types: list[NeuralNetType]


class ValidRangeSets(BaseModel):
    range_sets: list[RangeSet]


class Evaluation(BaseModel):
    bg_only: BgCase
    bth_only: BthCase
    bg_and_bth: ComboCase
    token: str = ""
    warnings: list[str]

    @classmethod
    def from_params(
        cls,
        bg: float,
        bth: float,
        pe_combo: PeResult,
        pe_bg: PeResult,
        pe_bth: PeResult,
        rep_unit: RepeatUnit,
    ) -> "Evaluation":
        bg_case = BgCase(
            bg=bg, bg_plateau=bg**NU3_1, pe_value=pe_bg.value, pe_error=pe_bg.error
        )

        kuhn_length = rep_unit.length / bth**2
        phi_xx = bth**4
        bth_case = BthCase(
            bth=bth,
            bth_plateau=bth ** (1 / 6),
            pe_value=pe_bth.value,
            pe_error=pe_bth.error,
            kuhn_length=kuhn_length,
            concentrated_conc=phi_to_conc(phi_xx, rep_unit),
        )

        phi_th = bth**3 * (bth / bg) ** (1 / NU2_1)
        thermal_blob_size = rep_unit.length * bth**2 / phi_th
        dp_of_thermal_blob = (bth**3 / phi_th) ** 2
        thermal_blob_conc = phi_to_conc(phi_th, rep_unit)
        excluded_volume = phi_th * kuhn_length**3

        combo_case = ComboCase(
            bg=bg,
            bth=bth,
            bg_plateau=bg ** (-NU3_1 / 3),
            bth_plateau=bth ** (-1 / 6),
            pe_value=pe_combo.value,
            pe_error=pe_combo.error,
            kuhn_length=kuhn_length,
            thermal_blob_size=thermal_blob_size,
            dp_of_thermal_blob=dp_of_thermal_blob,
            excluded_volume=excluded_volume,
            thermal_blob_conc=thermal_blob_conc,
            concentrated_conc=phi_xx,
        )

        return cls(
            bg_only=bg_case, bth_only=bth_case, bg_and_bth=combo_case, warnings=list()
        )
