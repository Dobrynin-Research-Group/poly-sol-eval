import asyncio
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


class PeResult(NamedTuple):
    """The optimized value of :math:`P_e` and the variance of that value from the
    fitting function.
    """

    opt: float
    var: float


def fit_func(nw_over_g_lamda_g: npt.NDArray, pe: float) -> npt.NDArray:
    return nw_over_g_lamda_g * (1 + (nw_over_g_lamda_g / pe**2)) ** 2


def fit_func_jac(nw_over_g_lamda_g: npt.NDArray, pe: float) -> npt.NDArray:
    # pe_arr = np.array([pe]).reshape(1, 1)
    nw_over_g_lamda_g = nw_over_g_lamda_g.reshape(-1, 1)
    return -2 * nw_over_g_lamda_g**2 / pe**3 - 4 * nw_over_g_lamda_g**3 / pe**5


async def fit_pe_combo(
    bg: float,
    bth: float,
    phi: npt.NDArray,
    nw: npt.NDArray,
    spec_visc: npt.NDArray,
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


async def fit_pe_bg_only(
    bg: float,
    phi: npt.NDArray,
    nw: npt.NDArray,
    spec_visc: npt.NDArray,
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


async def fit_pe_bth_only(
    bth: float,
    phi: npt.NDArray,
    nw: npt.NDArray,
    spec_visc: npt.NDArray,
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


async def do_fits(
    bg: float,
    bth: float,
    reduced_conc: npt.NDArray,
    degree_polym: npt.NDArray,
    specific_viscosity: npt.NDArray,
) -> tuple[PeResult, PeResult, PeResult]:
    """Perform three least squares fits for estimating Pe in three different cases:
    (1) both Bg and Bth are valid, (2) only Bg is valid, (3) only Bth is valid.

    Args:
        bg (float): The estimated value of Bg.
        bth (float): The estimated value of Bg.
        reduced_conc (npt.NDArray): The reduced concentration :math:`cl^3`.
        degree_polym (npt.NDArray): The weight-average degree of polymerization
          :math:`N_w`.
        specific_viscosity (npt.NDArray): The specific viscosity.

    Returns:
        tuple[PeResult, PeResult, PeResult]: The estimated values and standard errors
          for the following three cases, respectively: (1) both Bg and Bth are valid,
          (2) only Bg is valid, (3) only Bth is valid.
    """
    combo_task = asyncio.create_task(
        fit_pe_combo(bg, bth, reduced_conc, degree_polym, specific_viscosity)
    )
    bg_only_task = asyncio.create_task(
        fit_pe_bg_only(bg, reduced_conc, degree_polym, specific_viscosity)
    )
    bth_only_task = asyncio.create_task(
        fit_pe_bth_only(bth, reduced_conc, degree_polym, specific_viscosity)
    )
    pe_combo = await combo_task
    pe_bg_only = await bg_only_task
    pe_bth_only = await bth_only_task
    return pe_combo, pe_bg_only, pe_bth_only
