from __future__ import annotations
import multiprocessing as mpr
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from scipy.optimize import curve_fit

import psst


__all__ = [
    "AVOGADRO_CONSTANT",
    "GOOD_EXP",
    "process_data_to_grid",
    "transform_data",
    "fit_func",
    "evaluate_dataset",
    "RepeatUnit",
    "PeResult",
    "InferenceResult",
]


class RepeatUnit(NamedTuple):
    """Details of the repeat unit. Mass in units of g/mol, projection length (along
    fully extended axis) in nm (:math:`10^{-9}` m).
    """

    length: float
    mass: float


class PeResult(NamedTuple):
    """The optimized value of :math:`P_e` and the variance of that value from the
    fitting function.
    """

    opt: float
    var: float


class InferenceResult(NamedTuple):
    bg: float
    bth: float
    pe_combo: PeResult
    pe_bg_only: PeResult
    pe_bth_only: PeResult
    reduced_conc: np.ndarray
    degree_polym: np.ndarray
    specific_visc: np.ndarray


AVOGADRO_CONSTANT = 6.0221408e23
GOOD_EXP = 0.588


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


# TODO: allow for linear spacing on phi and nw
def process_data_to_grid(
    phi_data: np.ndarray,
    nw_data: np.ndarray,
    visc_data: np.ndarray,
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
    assert phi_range.shape is not None and nw_range.shape is not None
    shape = (phi_range.shape, nw_range.shape)
    visc_out = np.zeros(shape)
    counts = np.zeros(shape, dtype=np.uint32)

    log_phi_bins: np.ndarray = np.linspace(
        np.log10(phi_range.min_value),
        np.log10(phi_range.max_value),
        shape[0],
        endpoint=True,
    )
    phi_bin_edges = np.zeros(log_phi_bins.shape[0] + 1)
    phi_bin_edges[[0, -1]] = 10 ** log_phi_bins[[0, -1]]
    phi_bin_edges[1:-1] = 10 ** ((log_phi_bins[1:] + log_phi_bins[:-1]) / 2)
    phi_indices = np.digitize(phi_data, phi_bin_edges)
    phi_indices = np.minimum(
        phi_indices, np.zeros_like(phi_indices) + phi_indices.shape[0] - 1
    )

    log_nw_bins: np.ndarray = np.linspace(
        np.log10(nw_range.min_value),
        np.log10(nw_range.max_value),
        shape[1],
        endpoint=True,
    )
    nw_bin_edges = np.zeros(log_nw_bins.shape[0] + 1)
    nw_bin_edges[[0, -1]] = 10 ** log_nw_bins[[0, -1]]
    nw_bin_edges[1:-1] = 10 ** ((log_nw_bins[1:] + log_nw_bins[:-1]) / 2)
    nw_indices = np.digitize(nw_data, nw_bin_edges)
    nw_indices = np.minimum(
        nw_indices, np.zeros_like(nw_indices) + nw_indices.shape[0] - 1
    )

    for p, n, v in zip(phi_indices, nw_indices, visc_data):
        visc_out[p, n] += v
        counts[p, n] += 1

    counts = np.maximum(counts, np.ones_like(counts))
    visc_out /= counts

    return 10**log_phi_bins, 10**log_nw_bins, visc_out


def transform_data(
    reduced_conc: np.ndarray,
    degree_polym: np.ndarray,
    spec_visc: np.ndarray,
    phi_range: psst.Range,
    nw_range: psst.Range,
    visc_range: psst.Range,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Transform the raw, reduced data into two 2D tensors of reduced, normalized
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

    visc_normed_bg_range = psst.Range(
        min_value=visc_range.min_value / bg_denom.max(),
        max_value=visc_range.max_value / bg_denom.min(),
        log_scale=visc_range.log_scale,
    )
    visc_normed_bth_range = psst.Range(
        min_value=visc_range.min_value / bth_denom.max(),
        max_value=visc_range.max_value / bth_denom.min(),
        log_scale=visc_range.log_scale,
    )

    visc_normed_bg = torch.as_tensor(visc_arr / bg_denom, dtype=torch.float32)
    visc_normed_bg_range.normalize(visc_normed_bg)
    visc_normed_bg.clamp_(0, 1)

    visc_normed_bth = torch.as_tensor(visc_arr / bth_denom, dtype=torch.float32)
    visc_normed_bth_range.normalize(visc_normed_bth)
    visc_normed_bth.clamp_(0, 1)

    return visc_normed_bg, visc_normed_bth


def inference_model(
    model: torch.nn.Module,
    visc_normed: torch.Tensor,
    b_range: psst.Range,
) -> float:
    """Run the processed viscosity data through a pre-trained model to gain the
    inference results for either :math:`B_g` or :math:`B_{th}`.

    Args:
        model (torch.nn.Module): The pre-trained deep learning model.
        visc_normed (torch.Tensor): The normalized viscosity, appropriate to the
          parameter to be evaluated.
        b_range (psst.Range): The Range of the parameter to be evaluated.
        device (torch.device): The device on which to run the inference.

    Returns:
        float: The inferred value of the parameter.
    """

    pred: torch.Tensor = model(visc_normed.unsqueeze_(0))

    b_range.unnormalize(pred)

    return pred.squeeze().item()


def fit_func(nw_over_g_lamda_g: np.ndarray, pe: float) -> np.ndarray:
    return nw_over_g_lamda_g * (1 + (nw_over_g_lamda_g / pe**2)) ** 2


def fit_func_jac(nw_over_g_lamda_g: np.ndarray, pe: float) -> np.ndarray:
    pe_arr = np.array([pe]).reshape(1, 1)
    nw_over_g_lamda_g = nw_over_g_lamda_g.reshape(-1, 1)
    return -2 * nw_over_g_lamda_g**2 / pe**3 - 4 * nw_over_g_lamda_g**3 / pe**5


def fit_pe_combo(
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


def fit_pe_bg_only(
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


def fit_pe_bth_only(
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


def evaluate_dataset(
    concentration_gpL: npt.NDArray,
    mol_weight_kgpmol: npt.NDArray,
    specific_viscosity: npt.NDArray,
    repeat_unit: RepeatUnit,
    bg_model: torch.nn.Module,
    bth_model: torch.nn.Module,
    range_config: psst.RangeConfig,
) -> InferenceResult:
    """Perform an evaluation of experimental data given one previously trained PyTorch
    model for each of the :math:`B_g` and :math:`B_{th}` parameters.

    Args:
        concentration_gpL (np.ndarray): Experimental concentration data in units of
          grams per mole (1D numpy array).
        mol_weight_kgpmol (np.ndarray): Experimental molecular weight data in units of
          kilograms per mole (1D numpy array).
        specific_viscosity (np.ndarray): Experimental specific viscosity data in
          dimensionless units (1D numpy array).
        repeat_unit (RepeatUnit): The projection length and molar mass of the polymer
          repeat unit.
        bg_model (torch.nn.Module): A pretrained model for evaluating the :math:`B_g`
          parameter.
        bth_model (torch.nn.Module): A pretrained model for evaluating the :math:`B_th`
          parameter.
        range_config (psst.RangeConfig): A set of ``psst.Range``s. Used to homogenize
          inferencing of the models by normalizing the experimental data in the same
          manner as the procedural data that the models were trained on.

    Returns:
        InferenceResult: The results of the model inferences, complete with estimates
          for :math:`B_g` and :math:`B_{th}`; three estimates of :math:`P_e` with
          fitting uncertainties, one each for the case where both :math:`B_g` and
          :math:`B_{th}` are valid, the case where only :math:`B_g` is valid (athermal
          solvent), and the case where only :math:`B_{th}` is valid (theta solvent);
          the reduced concentration :math:`\\varphi=cl^3`; the weight-average degree of
          polymerization; and the unaltered specific viscosity.
    """

    reduced_conc, degree_polym = reduce_data(
        concentration_gpL,
        mol_weight_kgpmol,
        repeat_unit,
    )
    visc_normed_bg, visc_normed_bth = transform_data(
        reduced_conc,
        degree_polym,
        specific_viscosity,
        range_config.phi_range,
        range_config.nw_range,
        range_config.visc_range,
    )

    with mpr.Pool(3) as pool:
        bg_result = pool.apply_async(
            inference_model, (bg_model, visc_normed_bg, range_config.bg_range)
        )
        bth_result = pool.apply_async(
            inference_model, (bth_model, visc_normed_bth, range_config.bth_range)
        )

        bg = bg_result.get()
        bth = bth_result.get()

        pe1 = pool.apply_async(
            fit_pe_combo, (bg, bth, reduced_conc, degree_polym, specific_viscosity)
        )
        pe2 = pool.apply_async(
            fit_pe_bg_only, (bg, reduced_conc, degree_polym, specific_viscosity)
        )
        pe3 = pool.apply_async(
            fit_pe_bth_only, (bth, reduced_conc, degree_polym, specific_viscosity)
        )

        pe_combo = pe1.get()
        pe_bg_only = pe2.get()
        pe_bth_only = pe3.get()

    return InferenceResult(
        bg,
        bth,
        pe_combo,
        pe_bg_only,
        pe_bth_only,
        reduced_conc,
        degree_polym,
        specific_viscosity,
    )
