import asyncio

import numpy.typing as npt
from scipy.optimize import curve_fit
from polysoleval.exceptions import PSSTException
from polysoleval.logging import get_logger

from polysoleval.models import PeResult


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


async def fit_pe(
    xdata: npt.NDArray,
    ydata: npt.NDArray,
    init_guess: float = 8.0,
    bounds: tuple[float, float] = (2.0, 40.0),
) -> PeResult:
    """Convenience function for fitting :math:`P_e`.

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
    log = get_logger()
    log.debug(f"fit_pe({xdata = }, {ydata = }, {init_guess = }, {bounds = })")
    log.debug("%(taskName)s")

    try:
        res: tuple[npt.NDArray, npt.NDArray] = curve_fit(
            fit_func,
            xdata,
            ydata,
            p0=(init_guess,),
            bounds=bounds,
            jac=fit_func_jac,
        )
        popt, pcov = res[0].item(), res[1].item()
    except ValueError as ve:
        raise PSSTException.FittingValueError from ve
    except RuntimeError:
        popt, pcov = 1.0, 1.0

    return PeResult(value=popt, error=pcov)


async def do_fits(arr: npt.NDArray) -> tuple[PeResult, PeResult, PeResult]:
    """Perform three least squares fits for estimating :math:`P_e`.

    The fits are performed for three different cases: (1) both :math:`B_g` and
    :math:`B_{th}` are valid, (2) only :math:`B_g` is valid, (3) only :math:`B_{th}`
    is valid.

    Args:
        arr (np.ndarray): ...

    Returns:
        tuple[PeResult, PeResult, PeResult]: The estimated values and standard errors
          for the three cases describted above.
    """
    log = get_logger()
    log.debug(f"do_fits({arr = })")

    combo_task = asyncio.create_task(fit_pe(arr[5], arr[6]))
    bg_only_task = asyncio.create_task(fit_pe(arr[7], arr[8]))
    bth_only_task = asyncio.create_task(fit_pe(arr[9], arr[10]))

    return await asyncio.gather(combo_task, bg_only_task, bth_only_task)
