from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from polysoleval.globals import *
from polysoleval.models import PeResult, Range, RangeSet, RepeatUnit
from polysoleval.analysis.fitting import do_fits
from polysoleval.analysis.inference import do_inferences
from polysoleval.analysis.preprocess import *


class Results(NamedTuple):
    bg: float
    bth: float
    pe_combo: PeResult
    pe_bg_only: PeResult
    pe_bth_only: PeResult
    array: npt.NDArray


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


async def evaluate_dataset(
    concentration_gpL: npt.NDArray,
    mol_weight_kgpmol: npt.NDArray,
    specific_viscosity: npt.NDArray,
    repeat_unit: RepeatUnit,
    bg_model: torch.nn.Module,
    bth_model: torch.nn.Module,
    range_set: RangeSet,
) -> Results:
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
        range_set.phi_res,
        range_set.nw_res,
        range_set.phi_range,
        range_set.nw_range,
        range_set.visc_range,
    )

    bg, bth = await do_inferences(
        bg_model,
        visc_normed_bg,
        range_set.bg_range,
        bth_model,
        visc_normed_bth,
        range_set.bth_range,
    )

    arr = np.stack(
        [
            reduced_conc,
            degree_polym,
            specific_viscosity,
            specific_viscosity / degree_polym / reduced_conc ** (1 / (3 * NU - 1)),
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

    return Results(bg, bth, pe_combo, pe_bg, pe_bth, arr)
