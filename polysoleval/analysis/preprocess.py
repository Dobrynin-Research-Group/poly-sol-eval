from typing import NamedTuple, overload

import numpy as np
import numpy.typing as npt
import torch

from polysoleval.responses import BasicRange
from polysoleval.range import Range

AVOGADRO_CONSTANT = 6.0221408e23
GOOD_EXP = 0.588


class Resolution(NamedTuple):
    phi: int
    nw: int


class RepeatUnit(NamedTuple):
    """Details of the repeat unit. Mass in units of g/mol, projection length (along
    fully extended axis) in nm (:math:`10^{-9}` m).
    """

    length: float
    mass: float


@overload
def phi_to_conc(phi: float, rep_unit: RepeatUnit) -> float:
    ...


@overload
def phi_to_conc(phi: npt.NDArray, rep_unit: RepeatUnit) -> npt.NDArray:

def phi_to_conc(phi, rep_unit: RepeatUnit):
    return phi / rep_unit.length**3 / (AVOGADRO_CONSTANT / 1e24 / rep_unit.mass)


@overload
def conc_to_phi(conc: float, rep_unit: RepeatUnit) -> float:
    ...


@overload
def conc_to_phi(conc: npt.NDArray, rep_unit: RepeatUnit) -> npt.NDArray:

def conc_to_phi(conc, rep_unit: RepeatUnit):
    return conc * rep_unit.length**3 * AVOGADRO_CONSTANT / 1e24 / rep_unit.mass


def reduce_data(
    conc: npt.NDArray,
    mol_weight: npt.NDArray,
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

    reduced_conc: npt.NDArray = conc_to_phi(conc, repeat_unit)
    degree_polym = mol_weight / repeat_unit.mass * 1e3

    return reduced_conc, degree_polym


# TODO: allow for linear spacing on phi and nw
def process_data_to_grid(
    phi_data: npt.NDArray,
    nw_data: npt.NDArray,
    visc_data: npt.NDArray,
    res: Resolution,
    phi_range: BasicRange,
    nw_range: BasicRange,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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
    shape = (res.phi, res.nw)
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


def transform_data_to_grid(
    reduced_conc: npt.NDArray,
    degree_polym: npt.NDArray,
    spec_visc: npt.NDArray,
    res: Resolution,
    phi_range: BasicRange,
    nw_range: BasicRange,
    visc_range: BasicRange,
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
        res,
        phi_range,
        nw_range,
    )

    bg_denom = nw_arr.reshape(1, -1) * phi_arr.reshape(-1, 1) ** (
        1 / (3 * GOOD_EXP - 1)
    )
    bth_denom = nw_arr.reshape(1, -1) * phi_arr.reshape(-1, 1) ** 2

    visc_normed_bg_range = Range(
        min_value=visc_range.min_value / bg_denom.max(),
        max_value=visc_range.max_value / bg_denom.min(),
        log_scale=visc_range.log_scale,
    )
    visc_normed_bth_range = Range(
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
