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
    r"""The generalized function for fitting the specific viscosity data.

    Args:
        nw_over_g_lamda_g (npt.NDArray): The quantity
          :math:`N_w/g\lambda_g`, where :math:`N_w` is the weight-average DP of the
          polymers, :math:`g` is the DP of the chain segment within a correlation blob,
          and :math:`\lambda_g` is the correction factor accounting for the
          Rubinstein-Colby conjecture.
        pe (float): The entanglement packing number :math:`P_e` to be fit.

    Returns:
        npt.NDArray: The theoretical specific viscosity data based on the given inputs.

    Note:
        The specific viscosity can be written in a universal form as
        :math:`\frac{\lambda\eta_{sp}}{P_e^2} = \frac{N_w}{N_e} \left(1 + \left(\frac{N_w}{N_e}\right)\right)^2`
        where :math:`N_e = P_e^2 g \lambda_g` and :math:`\lambda` is the correction
        factor accounting for the different concentration scaling in the concentrated
        solution regime.
    """
    return nw_over_g_lamda_g * (1 + (nw_over_g_lamda_g / pe**2)) ** 2


def fit_func_jac(nw_over_g_lamda_g: npt.NDArray, pe: float) -> npt.NDArray:
    nw_over_g_lamda_g = nw_over_g_lamda_g.reshape(-1, 1)
    return -2 * nw_over_g_lamda_g**2 / pe**3 - 4 * nw_over_g_lamda_g**3 / pe**5


async def fit_pe_combo(
    bg: float,
    bth: float,
    phi: npt.NDArray,
    nw: npt.NDArray,
    spec_visc: npt.NDArray,
) -> PeResult:
    """Convenience function for fitting :math:`P_e` in the case that both :math:`B_g`
    and :math:`B_{th}` exist.

    Args:
        bg (float): The value of :math:`B_g`.
        bth (float): The value of :math:`B_{th}`.
        phi (npt.NDArray): The reduced concentration data.
        nw (npt.NDArray): The weight-average DP data.
        spec_visc (npt.NDArray): The specific viscosity data.

    Returns:
        PeResult: The optimized value of the entanglement packing number :math:`P_e`
          and standard error thereof.
    """
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
    """Convenience function for fitting :math:`P_e` in the case that only :math:`B_g`
    exists.

    Args:
        bg (float): The value of :math:`B_g`.
        phi (npt.NDArray): The reduced concentration data.
        nw (npt.NDArray): The weight-average DP data.
        spec_visc (npt.NDArray): The specific viscosity data.

    Returns:
        PeResult: The optimized value of the entanglement packing number :math:`P_e`
          and standard error thereof.
    """
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
    """Convenience function for fitting :math:`P_e` in the case that only
    :math:`B_{th}` exists.

    Args:
        bth (float): The value of :math:`B_{th}`.
        phi (npt.NDArray): The reduced concentration data.
        nw (npt.NDArray): The weight-average DP data.
        spec_visc (npt.NDArray): The specific viscosity data.

    Returns:
        PeResult: The optimized value of the entanglement packing number :math:`P_e`
          and standard error thereof.
    """
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
    """Perform three least squares fits for estimating :math:`P_e`.

    The fits are performed for three different cases: (1) both :math:`B_g` and
    :math:`B_{th}` are valid, (2) only :math:`B_g` is valid, (3) only :math:`B_{th}`
    is valid.

    Args:
        bg (float): The estimated value of :math:`B_g`.
        bth (float): The estimated value of :math:`B_{th}`.
        reduced_conc (npt.NDArray): The reduced concentration :math:`cl^3`.
        degree_polym (npt.NDArray): The weight-average degree of polymerization
          :math:`N_w`.
        specific_viscosity (npt.NDArray): The specific viscosity :math:`\\eta_{sp}`.

    Returns:
        tuple[PeResult, PeResult, PeResult]: The estimated values and standard errors
          for the three cases describted above.
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
