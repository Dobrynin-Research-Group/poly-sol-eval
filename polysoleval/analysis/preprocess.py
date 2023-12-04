import numpy as np
import numpy.typing as npt
import torch

from polysoleval.globals import NU3_1
from polysoleval.models import Range, RepeatUnit
from polysoleval.conversions import conc_to_phi


__all__ = ["reduce_data", "process_data_to_grid", "transform_data_to_grid"]


def reduce_data(
    conc: npt.NDArray,
    mol_weight: npt.NDArray,
    repeat_unit: RepeatUnit,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Reduce the concentration and molecular weight measurements.

    The measurements are reduced from concentration in g/L to :math:`\varphi=cl^3` and
    weight-average molecular weight in kg/mol to weight-average degree of
    polymerization (number of repeat units per chain).

    Args:
        conc (npt.NDArray): Concentration in g/L.
        mol_weight (npt.NDArray): Weight-average molecular weight in kg/mol.
        repeat_unit (RepeatUnit): The length in nm and mass in g/mol of a repeat unit.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: The reduced concentration :math:`\varphi=cl^3`
          and degree of polymerization :math:`N_w`.
    """

    reduced_conc = conc_to_phi(conc, repeat_unit)
    degree_polym = mol_weight / repeat_unit.mass * 1e3

    return reduced_conc, degree_polym


def _range_to_bins(in_range: Range, num_bins: int) -> npt.NDArray:
    bin_func = np.geomspace if in_range.log_scale else np.linspace
    return bin_func(in_range.min_value, in_range.max_value, num_bins, endpoint=True)


def _bins_to_bin_edges(bins: npt.NDArray, log_scale: bool) -> npt.NDArray:
    bin_edges = np.zeros(bins.shape[0] + 1)
    bin_edges[[0, -1]] = bins[[0, -1]]
    if log_scale:
        bin_edges[1:-1] = np.sqrt(bins[:-1] * bins[1:])
    else:
        bin_edges[1:-1] = (bins[:-1] + bins[1:]) / 2
    return bin_edges


def _bin_data(data: npt.NDArray, bin_edges: npt.NDArray) -> npt.NDArray:
    indices = np.digitize(data, bin_edges)
    indices = np.minimum(indices, np.zeros_like(indices) + indices.shape[0] - 1)
    return indices


def process_data_to_grid(
    phi_data: npt.NDArray,
    nw_data: npt.NDArray,
    visc_data: npt.NDArray,
    phi_res: int,
    nw_res: int,
    phi_range: Range,
    nw_range: Range,
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
    phi_bins = _range_to_bins(phi_range, phi_res)
    phi_bin_edges = _bins_to_bin_edges(phi_bins, phi_range.log_scale)
    phi_indices = _bin_data(phi_data, phi_bin_edges)

    nw_bins = _range_to_bins(nw_range, nw_res)
    nw_bin_edges = _bins_to_bin_edges(nw_bins, nw_range.log_scale)
    nw_indices = _bin_data(nw_data, nw_bin_edges)

    visc_out = np.zeros((phi_res, nw_res))
    counts = np.zeros((phi_res, nw_res), dtype=np.uint32)
    for p, n, v in zip(phi_indices, nw_indices, visc_data):
        visc_out[p, n] += v
        counts[p, n] += 1

    counts = np.maximum(counts, np.ones_like(counts))
    visc_out /= counts

    return phi_bins, nw_bins, visc_out


def transform_data_to_grid(
    reduced_conc: npt.NDArray,
    degree_polym: npt.NDArray,
    spec_visc: npt.NDArray,
    phi_res: int,
    nw_res: int,
    phi_range: Range,
    nw_range: Range,
    visc_range: Range,
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
        phi_res,
        nw_res,
        phi_range,
        nw_range,
    )

    bg_denom = nw_arr.reshape(1, -1) * phi_arr.reshape(-1, 1) ** (1 / NU3_1)
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
