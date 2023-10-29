import json
from pathlib import Path
from typing import NamedTuple
import numpy as np
from scipy.optimize import curve_fit
import torch

import psst
from psst import models


AVOGADRO_CONSTANT = 6.0221408e23
GOOD_EXP = 0.588
MODELS: dict[str, type[torch.nn.Module]] = {
    "vgg13": models.Vgg13,
    "inception3": models.Inception3,
}


class RepeatUnit(NamedTuple):
    """Details of the repeat unit. Mass in units of g/mol, projection length (along
    fully extended axis) in nm (:math:`10^{-9}` m).
    """

    mass: float
    length: float


class PeResult(NamedTuple):
    """The optimized value of :math:`P_e` and the variance of that value from the
    fitting function.
    """

    opt: float
    var: float


class Result(NamedTuple):
    bg: float
    bth: float
    pe_combo: PeResult
    pe_bg_only: PeResult
    pe_bth_only: PeResult


def reduce_data(
    conc: np.ndarray,
    mol_weight: np.ndarray,
    repeat_unit: RepeatUnit,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the concentration and molecular weight measurements from concentration
    in g/L to :math:`(c*l^3)` and weight-average molecular weight in kg/mol to weight-
    average degree of polymerization (number of repeat units per chain).

    Args:
        conc (np.ndarray): Concentration in g/L.
        mol_weight (np.ndarray): Weight-average molecular weight in kg/mol.
        repeat_unit (RepeatUnit): The mass in g/mol and length in nm of a repeat unit.

    Returns:
        tuple[np.ndarray, np.ndarray]: The reduced concentration :math:`cl^3` and
          degree of polymerization :math:`N_w`.
    """

    reduced_conc = AVOGADRO_CONSTANT * conc * (repeat_unit.length / 1e8) ** 3
    degree_polym = mol_weight / repeat_unit.mass * 1e3

    return reduced_conc, degree_polym


def process_data_to_grid(
    phi: np.ndarray,
    nw: np.ndarray,
    visc: np.ndarray,
    phi_range: psst.Range,
    nw_range: psst.Range,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a set of data ``(phi, nw, visc)`` into an "image", where each "pixel"
    along axis 0 represents a bin of values in the log-scale of ``phi_range`` and those
    along axis 1 represent bins of values in the log-scale of ``nw_range``.

    Args:
        phi (np.ndarray): Experimental concentration data in reduced form :math:`cl^3`.
        nw (np.ndarray): Experimental data for weight-average degree of polymerization.
        visc (np.ndarray): Experimental specific viscosity data. Data at index ``i``
          should correspond to a solution state with reduced concentration ``phi[i]``
          and weight-average DP of polymer chains ``nw[i]``.
        phi_range (psst.Range): The minimum, maximum, and number of values in the range
          of reduced concentration values (``phi_range.log_scale`` should be True).
        nw_range (psst.Range): The minimum, maximum, and number of values in the range
          of weight-average DP values (``nw_range.log_scale`` should be True).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Three arrays representing the
          concentration value of each "pixel" along axis 0 (1D), the DP value of each
          "pixel" along axis 1 (1D), and the average values of specific viscosity at
          the given concentrations and DPs (2D). Simply put, the value of viscosity at
          index ``(i, j)`` approximately corresponds to the reduced concentration at
          index ``i`` and the DP at index ``j``.
    """
    assert phi_range.num is not None
    assert nw_range.num is not None

    visc_out = np.zeros((phi_range.num, nw_range.num))
    counts = np.zeros((phi_range.num, nw_range.num), dtype=np.uint32)

    log_phi_bins: np.ndarray = np.linspace(
        np.log10(phi_range.min_value),
        np.log10(phi_range.max_value),
        phi_range.num,
        endpoint=True,
    )
    phi_bin_edges = np.zeros(log_phi_bins.shape[0] + 1)
    phi_bin_edges[(0, -1)] = 10 ** log_phi_bins[(0, -1)]
    phi_bin_edges[1:-1] = 10 ** ((log_phi_bins[1:] + log_phi_bins[:-1]) / 2)
    phi_indices = np.digitize(phi, phi_bin_edges)

    log_nw_bins: np.ndarray = np.linspace(
        np.log10(nw_range.min_value),
        np.log10(nw_range.max_value),
        nw_range.num,
        endpoint=True,
    )
    nw_bin_edges = np.zeros(log_nw_bins.shape[0] + 1)
    nw_bin_edges[(0, -1)] = 10 ** log_nw_bins[(0, -1)]
    nw_bin_edges[1:-1] = 10 ** ((log_nw_bins[1:] + log_nw_bins[:-1]) / 2)
    nw_indices = np.digitize(nw, nw_bin_edges)

    data = np.stack((phi_indices, nw_indices, visc), axis=1)
    for p, n, v in data:
        visc_out[p, n] += v
        counts[p, n] += 1

    counts = np.maximum(counts, np.ones_like(counts))
    visc_out /= counts

    return 10**log_phi_bins, 10**log_nw_bins, visc_out


def init_parameters(
    bg_range: psst.Range,
    bth_range: psst.Range,
    pe_range: psst.Range,
) -> tuple[psst.NormedTensor, psst.NormedTensor, psst.NormedTensor]:
    """Create ``psst.NormedTensors`` of the three parameters (each of a single value)
    for simple normalization.

    Args:
        bg_range (psst.Range): Range of values used for normalization of :math:`B_g`.
        bth_range (psst.Range): As for ``bg_range``, but for :math:`B_{th}`
        pe_range (psst.Range): As for ``bg_range``, but for :math:`P_e`

    Returns:
        tuple[psst.NormedTensor, psst.NormedTensor, psst.NormedTensor]: Three
          one-element ``psst.NormedTensor``s for ``bg``, ``bth``, and ``pe``
          (initially set to 0.0).
    """
    assert bg_range.num is not None
    assert bth_range.num is not None
    assert pe_range.num is not None

    bg = psst.NormedTensor.create(
        1,
        min_value=bg_range.min_value,
        max_value=bg_range.max_value,
        log_scale=bg_range.log_scale,
    )
    bth = psst.NormedTensor.create(
        1,
        min_value=bth_range.min_value,
        max_value=bth_range.max_value,
        log_scale=bth_range.log_scale,
    )
    pe = psst.NormedTensor.create(
        1,
        min_value=pe_range.min_value,
        max_value=pe_range.max_value,
        log_scale=pe_range.log_scale,
    )
    return bg, bth, pe


def transform_data(
    reduced_conc: np.ndarray,
    degree_polym: np.ndarray,
    spec_visc: np.ndarray,
    phi_range: psst.Range,
    nw_range: psst.Range,
    visc_range: psst.Range,
) -> tuple[psst.NormedTensor, psst.NormedTensor]:
    r"""Transform the raw, reduced data into two 2D arrays of reduced, normalized
    viscosity data ready for use in a neural network.

    Args:
        reduced_conc (np.ndarray): Reduced concentration data :math:`\varphi=cl^3`.
        degree_polym (np.ndarray): Weight-average DP data :math:`N_w`.
        spec_visc (np.ndarray): Specific viscosity raw data
        phi_range (psst.Range): Range of values specifying the value of reduced
          concentration for each grid index along axis 0.
        nw_range (psst.Range): Range of values specifying the value of weight-average
          DP for each grid index along axis 1.
        visc_range (psst.Range): Range of values specifying the maximum and minimum
          values of specific viscosity.

    Returns:
        tuple[psst.NormedTensor, psst.NormedTensor]: The reduced specific viscosities
          :math:`\eta_{sp}/N_w \phi^{1.31}` and :math:`\eta_{sp}/N_w \phi^2`.
    """
    phi_arr, nw_arr, visc_arr = process_data_to_grid(
        reduced_conc,
        degree_polym,
        spec_visc,
        phi_range,
        nw_range,
    )

    bg_denom = nw_arr.reshape(1, -1) * phi_arr.reshape(-1, 1) ** (
        1 / (3 * GOOD_EXP - 1)
    )
    bth_denom = nw_arr.reshape(1, -1) * phi_arr.reshape(-1, 1) ** 2

    visc_normed_bg = (
        psst.NormedTensor.create_from_numpy(
            visc_arr / bg_denom,
            min_value=visc_range.min_value / bg_denom.max(),
            max_value=visc_range.max_value / bg_denom.min(),
            log_scale=visc_range.log_scale,
        )
        .normalize()
        .clamp_(0, 1)
    )
    visc_normed_bth = (
        psst.NormedTensor.create_from_numpy(
            visc_arr / bth_denom,
            min_value=visc_range.min_value / bth_denom.max(),
            max_value=visc_range.max_value / bth_denom.min(),
            log_scale=visc_range.log_scale,
        )
        .normalize()
        .clamp_(0, 1)
    )

    return visc_normed_bg, visc_normed_bth


def inference_models(
    model_type: str,
    bg_state_dict: dict,
    bth_state_dict: dict,
    visc_normed_bg: psst.NormedTensor,
    visc_normed_bth: psst.NormedTensor,
    bg_range: psst.Range,
    bth_range: psst.Range,
) -> tuple[float, float]:
    """Run the processed viscosity data through the models to gain the inference
    results for :math:`B_g` and :math:`B_{th}`.

    Args:
        model_type (str): One of the pre-trained models from the keys of MODELS.
        bg_state_dict (dict): The state dictionary of the bg model.
        bth_state_dict (dict): The state dictionary of the bth model.
        visc_normed_bg (psst.NormedTensor): The reduced and normalized viscosity data
          for the bg model.
        visc_normed_bth (psst.NormedTensor): The reduced and normalized viscosity data
          for the bth model.
        bg_range (psst.Range): The minimum and maximum of the bg parameter (used for
          normalization and unnormalization).
        bth_range (psst.Range): The minimum and maximum of the bg parameter (used for
          normalization and unnormalization).

    Returns:
        tuple[float, float]: The inferred values of :math:`B_g` and :math:`B_{th}`,
          respectively.
    """
    Model = MODELS[model_type]
    bg_model = Model()
    bth_model = Model()
    bg_model.load_state_dict(bg_state_dict)
    bth_model.load_state_dict(bth_state_dict)

    assert bg_range.num is not None
    assert bth_range.num is not None

    bg = psst.NormedTensor.create(
        1,
        min_value=bg_range.min_value,
        max_value=bg_range.max_value,
        log_scale=bg_range.log_scale,
    ).normalize()
    bth = psst.NormedTensor.create(
        1,
        min_value=bth_range.min_value,
        max_value=bth_range.max_value,
        log_scale=bth_range.log_scale,
    ).normalize()
    bg[:] = bg_model(visc_normed_bg)
    bth[:] = bth_model(visc_normed_bth)

    return bg.unnormalize().item(), bth.unnormalize().item()


def fit_func(nw_over_g_lamda_g: np.ndarray, pe: float) -> np.ndarray:
    return nw_over_g_lamda_g * (1 + (nw_over_g_lamda_g / pe**2)) ** 2


def fit_func_jac(nw_over_g_lamda_g: np.ndarray, pe: float) -> np.ndarray:
    return -2 * nw_over_g_lamda_g**2 / pe**3 - 4 * nw_over_g_lamda_g**3 / pe**5


def combo_case(
    bg: float,
    bth: float,
    phi: np.ndarray,
    nw: np.ndarray,
    spec_visc: np.ndarray,
) -> PeResult:
    # ne/pe**2 == g*lam_g
    ne_over_pe2 = np.minimum(
        bg ** (0.056 / 0.528 / 0.764) * bth ** (0.944 / 0.528) * phi ** (-1 / 0.764),
        np.minimum(bth**2 / phi ** (4 / 3), bth**6 / phi**2),
    )
    lamda = np.minimum(1, bth**4 / phi)

    popt, pcov = curve_fit(
        fit_func,
        nw / ne_over_pe2,
        lamda * spec_visc,
        p0=(8.0,),
        bounds=(2.0, 40.0),
        jac=fit_func_jac,
    )

    return PeResult(popt[0], pcov[0][0])


def bg_only_case(
    bg: float,
    phi: np.ndarray,
    nw: np.ndarray,
    spec_visc: np.ndarray,
) -> PeResult:
    g = (bg**3 / phi) ** (1 / 0.764)
    lamda = np.minimum(1, phi ** (-0.236 / 0.764) * bg ** (2 / (0.412 * 0.764)))
    popt, pcov = curve_fit(
        fit_func,
        nw / g,
        lamda * spec_visc,
        p0=(8.0,),
        bounds=(2.0, 40.0),
        jac=fit_func_jac,
    )

    return PeResult(popt[0], pcov[0][0])


def bth_only_case(
    bth: float,
    phi: np.ndarray,
    nw: np.ndarray,
    spec_visc: np.ndarray,
) -> PeResult:
    # ne/pe**2 == g*lam_g
    ne_over_pe2 = np.minimum(bth**2 / phi ** (4 / 3), bth**6 / phi**2)
    lamda = np.minimum(1, bth**4 / phi)

    popt, pcov = curve_fit(
        fit_func,
        nw / ne_over_pe2,
        lamda * spec_visc,
        p0=(8.0,),
        bounds=(2.0, 40.0),
        jac=fit_func_jac,
    )

    return PeResult(popt[0], pcov[0][0])


def evaluate_data(
    model_path: Path,
    conc: np.ndarray,
    mol_weight: np.ndarray,
    spec_visc: np.ndarray,
    repeat_unit: RepeatUnit,
) -> Result:
    """Perform an evaluation of the PyTorch Module designated by `model_path` using the
    given data to obtain.

    Args:
        model_path (Path): Path to a saved instance of a pre-trained model.
        conc (np.ndarray): Concentration of solute in solution (units of g/L).
        mw (np.ndarray): Weight-average molecular weight of polymer chains in solution
          (units of kg/mol).
        spec_visc (np.ndarray): Specific viscosity of solution (solution viscosity
          divided by solvent viscosity minus 1).
        repeat_unit (RepeatUnit): The molar mass and projection length of the polymer
          repeat unit.

    Returns:
        tuple[float, float, float]: The estimated values of :math:`B_g`,
          :math:`B_{th}`, and :math:`P_e`.
    """

    with open(model_path, "r") as f:
        model_dict: dict = json.load(f)

    if not isinstance(model_dict, dict):
        raise RuntimeError("Expected file at {model_path} to be of simple JSON format")

    model_type: str = model_dict["model_type"]
    phi_range = psst.Range(**model_dict["phi_range"])
    nw_range = psst.Range(**model_dict["nw_range"])
    visc_range = psst.Range(**model_dict["visc_range"])
    bg_range = psst.Range(**model_dict["bg_range"])
    bth_range = psst.Range(**model_dict["bth_range"])
    bg_state_dict: dict = model_dict["bg_state_dict"]
    bth_state_dict: dict = model_dict["bth_state_dict"]

    reduced_conc, degree_polym = reduce_data(conc, mol_weight, repeat_unit)
    visc_normed_bg, visc_normed_bth = transform_data(
        reduced_conc, degree_polym, spec_visc, phi_range, nw_range, visc_range
    )

    bg, bth = inference_models(
        model_type,
        bg_state_dict,
        bth_state_dict,
        visc_normed_bg,
        visc_normed_bth,
        bg_range,
        bth_range,
    )

    pe_combo = combo_case(bg, bth, reduced_conc, degree_polym, spec_visc)
    pe_bg_only = bg_only_case(bg, reduced_conc, degree_polym, spec_visc)
    pe_bth_only = bth_only_case(bth, reduced_conc, degree_polym, spec_visc)

    return Result(bg, bth, pe_combo, pe_bg_only, pe_bth_only)
