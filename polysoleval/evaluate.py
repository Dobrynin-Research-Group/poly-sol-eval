import numpy.typing as npt
import torch

from polysoleval.analysis.fitting import do_fits
from polysoleval.analysis.inference import do_inferences
from polysoleval.analysis.preprocess import *
from polysoleval import responses


def lamda(bg, bth, phi):
    if bth:
        return np.minimum(1, bth**4 / phi)
    elif bg:
        return np.minimum(1, phi ** (-0.236 / 0.764) * bg ** (2 / (0.412 * 0.764)))
    else:
        raise ValueError()


def g_lamdag(bg, bth, phi):
    if bth and bg:
        return np.minimum(
            bg ** (0.056 / 0.528 / 0.764)
            * bth ** (0.944 / 0.528)
            * phi ** (-1 / 0.764),
            np.minimum(bth**2 / phi ** (4 / 3), bth**6 / phi**2),
        )
    elif bg:
        return (bg**3 / phi) ** (1 / 0.764)
    elif bth:
        return np.minimum(bth**2 / phi ** (4 / 3), bth**6 / phi**2)
    else:
        raise ValueError()


def create_response(
    *,
    pe_bg: PeResult,
    pe_bth: PeResult,
    pe_combo: PeResult,
    bg: float,
    bth: float,
    rep_unit: RepeatUnit,
) -> responses.Evaluation:
    bg_case = EvaluationCase(
        bg=bg,
        bg_plateau=bg ** (1 / 3 - GOOD_EXP),
        pe=pe_bg.opt,
        pe_variance=pe_bg.var,
    )

    kuhn_length = rep_unit.length / bth**2
    phi_xx = bth**4
    bth_case = EvaluationCase(
        bth=bth,
        bth_plateau=bth ** (1 / 6),
        pe=pe_bth.opt,
        pe_variance=pe_bth.var,
        kuhn_length=kuhn_length,
        concentrated_conc=phi_to_conc(phi_xx, rep_unit),
    )

    phi_th = bth**3 * (bth / bg) ** (1 / (2 * GOOD_EXP - 1))
    thermal_blob_size = rep_unit.length * bth**2 / phi_th
    dp_of_thermal_blob = (bth**3 / phi_th) ** 2
    thermal_blob_conc = phi_to_conc(phi_th, rep_unit)
    excluded_volume = phi_th * kuhn_length**3

    combo_case = EvaluationCase(
        bg=bg,
        bth=bth,
        bg_plateau=bg ** (1 / 3 - GOOD_EXP),
        bth_plateau=bth ** (-1 / 6),
        pe=pe_combo.opt,
        pe_variance=pe_combo.var,
        kuhn_length=kuhn_length,
        thermal_blob_size=thermal_blob_size,
        dp_of_thermal_blob=dp_of_thermal_blob,
        excluded_volume=excluded_volume,
        thermal_blob_conc=thermal_blob_conc,
        concentrated_conc=phi_xx,
    )

    return responses.Evaluation(
        bg_only=bg_case, bth_only=bth_case, bg_and_bth=combo_case
    )


async def evaluate_dataset(
    concentration_gpL: npt.NDArray,
    mol_weight_kgpmol: npt.NDArray,
    specific_viscosity: npt.NDArray,
    repeat_unit: RepeatUnit,
    bg_model: torch.nn.Module,
    bth_model: torch.nn.Module,
    range_config: RangeSet,
) -> tuple[responses.Evaluation, npt.NDArray]:
    """Perform an evaluation of experimental data given one previously trained PyTorch
    model for each of the :math:`B_g` and :math:`B_{th}` parameters.

    Args:
        concentration_gpL (numpy.ndarray, 1D): Experimental concentration data in units
          of grams per mole.
        mol_weight_kgpmol (numpy.ndarray, 1D): Experimental molecular weight data in
          units of kilograms per mole.
        specific_viscosity (numpy.ndarray, 1D): Experimental specific viscosity data in
          dimensionless units.
        repeat_unit (RepeatUnit): The projection length and molar mass of the polymer
          repeat unit.
        bg_model (torch.nn.Module): A pretrained model for evaluating the :math:`B_g`
          parameter.
        bth_model (torch.nn.Module): A pretrained model for evaluating the :math:`B_th`
          parameter.
        range_config (RangeSet): Used to homogenize inferencing of the models by
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

    arr = np.stack(
        [
            reduced_conc,
            degree_polym,
            specific_viscosity,
            (
                specific_viscosity
                / degree_polym
                / reduced_conc ** (1 / (3 * GOOD_EXP - 1))
            ),
            specific_viscosity / degree_polym / reduced_conc**2,
            degree_polym / g_lamdag(bg, None, reduced_conc),
            specific_viscosity * lamda(bg, None, reduced_conc),
            degree_polym / g_lamdag(None, bth, reduced_conc),
            specific_viscosity * lamda(None, bth, reduced_conc),
            degree_polym / g_lamdag(bg, bth, reduced_conc),
            specific_viscosity * lamda(bg, bth, reduced_conc),
        ],
        axis=0,
    )

    pe_combo, pe_bg, pe_bth = await do_fits(arr)

    return response, arr
