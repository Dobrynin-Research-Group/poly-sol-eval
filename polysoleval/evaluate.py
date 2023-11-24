from typing import Optional

import numpy.typing as npt
import torch

from polysoleval.responses import RangeResponse, EvaluationResponse, EvaluationResult
from polysoleval.analysis.fitting import do_fits
from polysoleval.analysis.inference import do_inferences
from polysoleval.analysis.preprocess import *


def create_result(
    *,
    bg: Optional[float],
    bth: Optional[float],
    pe: float,
    pe_variance: Optional[float],
    rep_unit: RepeatUnit,
) -> EvaluationResult:
    """Compute a complete ``EvaluationResult`` from the given parameters.

    The repeat unit project length (``rep_unit_length``) is needed to convert
    concentrations and length scales from reduced units to real units. At least one of
    ``bg`` or ``bth`` must not be ``None``. All given parameters must be positive.

    Args:
        bg (Optional[float]): The good solvent parameter.
        bth (Optional[float]): The theta solvent/thermal blob parameter.
        pe (float): The entanglement packing number.
        pe_variance (Optional[float]): The standard error of ``pe``.
        rep_unit_length (float): The projection length of the polymer's repeat unit.

    Raises:
        ValueError: If both ``bg`` and ``bth`` are ``None`` or not given, or if any
          given parameter is non-positive.

    Returns:
        EvaluationResult: The values computed from the given independent parameters.
    """
    if (
        (bg is not None and bg <= 0.0)
        or (bg is not None and bg <= 0.0)
        or pe <= 0.0
        or (pe_variance is not None and pe_variance <= 0)
        or rep_unit.length <= 0.0
        or rep_unit.mass <= 0.0
    ):
        raise ValueError(
            "all given parameters must be positive"
            f"\n{bg = }, {bth = }, {pe = }, {pe_variance = }, {rep_unit = }"
        )

    thermal_blob_size = None
    dp_of_thermal_blob = None
    excluded_volume = None
    thermal_blob_conc = None
    concentrated_conc = None

    if bg and bth:
        kuhn_length = rep_unit.length / bth**2
        phi_th = bth**3 * (bth / bg) ** (1 / (2 * GOOD_EXP - 1))
        thermal_blob_size = rep_unit.length * bth**2 / phi_th
        dp_of_thermal_blob = (bth**3 / phi_th) ** 2
        thermal_blob_conc = phi_to_conc(phi_th, rep_unit)
        excluded_volume = phi_th * kuhn_length**3
    elif bg:
        kuhn_length = rep_unit.length / bg ** (1 / (1 - GOOD_EXP))
    elif bth:
        kuhn_length = rep_unit.length / bth**2
    else:
        raise ValueError("must supply at least one of bg or bth")

    concentrated_conc = phi_to_conc(1 / kuhn_length**2 / rep_unit.length, rep_unit)

    return EvaluationResult(
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


async def evaluate_dataset(
    concentration_gpL: npt.NDArray,
    mol_weight_kgpmol: npt.NDArray,
    specific_viscosity: npt.NDArray,
    repeat_unit: RepeatUnit,
    bg_model: torch.nn.Module,
    bth_model: torch.nn.Module,
    range_config: RangeResponse,
) -> EvaluationResponse:
    """Perform an evaluation of experimental data given one previously trained PyTorch
    model for each of the :math:`B_g` and :math:`B_{th}` parameters.


    Args:
        concentration_gpL (npt.NDArray): Experimental concentration data in units of
          grams per mole (1D numpy array).
        mol_weight_kgpmol (npt.NDArray): Experimental molecular weight data in units of
          kilograms per mole (1D numpy array).
        specific_viscosity (npt.NDArray): Experimental specific viscosity data in
          dimensionless units (1D numpy array).
        repeat_unit (RepeatUnit): The projection length and molar mass of the polymer
          repeat unit.
        bg_model (torch.nn.Module): A pretrained model for evaluating the :math:`B_g`
          parameter.
        bth_model (torch.nn.Module): A pretrained model for evaluating the :math:`B_th`
          parameter.
        range_config (RangeResponse): Used to homogenize inferencing of the models by
          normalizing the experimental data in the same manner as the procedural data
          that the models were trained on.

    Returns:
        EvaluationResponse: Three separate ``EvaluationResult``s, one for each of the
          following cases: (1) both :math:`B_g` and :math:`B_{th}` are valid, (2) only
          :math:`B_g` is valid, and (3) only :math:`B_{th}` is valid.
    """

    reduced_conc, degree_polym = reduce_data(
        concentration_gpL,
        mol_weight_kgpmol,
        repeat_unit,
    )
    visc_normed_bg, visc_normed_bth = transform_data_to_grid(
        reduced_conc,
        degree_polym,
        specific_viscosity,
        Resolution(range_config.phi_res, range_config.nw_res),
        range_config.phi_range,
        range_config.nw_range,
        range_config.visc_range,
    )

    bg_range = Range(
        min_value=range_config.bg_range.min_value,
        max_value=range_config.bg_range.max_value,
        log_scale=range_config.bg_range.log_scale,
    )
    bth_range = Range(
        min_value=range_config.bth_range.min_value,
        max_value=range_config.bth_range.max_value,
        log_scale=range_config.bth_range.log_scale,
    )
    bg, bth = await do_inferences(
        bg_model, visc_normed_bg, bg_range, bth_model, visc_normed_bth, bth_range
    )

    pe_combo, pe_bg_only, pe_bth_only = await do_fits(
        bg, bth, reduced_conc, degree_polym, specific_viscosity
    )

    combo_result = create_result(
        bg=bg,
        bth=bth,
        pe=pe_combo.opt,
        pe_variance=pe_combo.var,
        rep_unit=repeat_unit,
    )
    bg_only_result = create_result(
        bg=bg,
        bth=None,
        pe=pe_bg_only.opt,
        pe_variance=pe_bg_only.var,
        rep_unit=repeat_unit,
    )
    bth_only_result = create_result(
        bg=None,
        bth=bth,
        pe=pe_bth_only.opt,
        pe_variance=pe_bth_only.var,
        rep_unit=repeat_unit,
    )

    return EvaluationResponse(
        both_bg_and_bth=combo_result,
        bg_only=bg_only_result,
        bth_only=bth_only_result,
    )
