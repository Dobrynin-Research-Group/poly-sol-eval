from typing import overload

import numpy.typing as npt

from polysoleval.globals import *
from polysoleval.models import RepeatUnit


@overload
def phi_to_conc(phi: float, rep_unit: RepeatUnit) -> float:
    r"""Transform the reduced concentration to solution concentration.

    Args:
        phi (float): Value of reduced concentration :math:`\varphi=cl^3`.
        rep_unit (RepeatUnit): Repeat unit projection length and mass.

    Returns:
        float: The solution concentration :math:`c` in units of g/L.
    """
    ...


@overload
def phi_to_conc(phi: npt.NDArray, rep_unit: RepeatUnit) -> npt.NDArray:
    r"""Transform the reduced concentration to solution concentration.

    Args:
        phi (npt.NDArray): Reduced concentration data :math:`\varphi=cl^3`.
        rep_unit (RepeatUnit): Repeat unit projection length and mass.

    Returns:
        npt.NDArray: The solution concentration data :math:`c` in units of g/L.
    """
    ...


def phi_to_conc(phi, rep_unit: RepeatUnit):
    return phi * rep_unit.mass / rep_unit.length**3 / REDUCED_AVOGADRO


@overload
def conc_to_phi(conc: float, rep_unit: RepeatUnit) -> float:
    r"""Transform the solution concentration to reduced concentration.

    Args:
        conc (float): Value of solution concentration :math:`c` in units of g/L.
        rep_unit (RepeatUnit): Repeat unit projection length and mass.

    Returns:
        float: The reduced concentration :math:`\varphi=cl^3`.
    """
    ...


@overload
def conc_to_phi(conc: npt.NDArray, rep_unit: RepeatUnit) -> npt.NDArray:
    r"""Transform the solution concentration to reduced concentration.

    Args:
        conc (npt.NDArray): Solution concentration data :math:`c` in units of g/L.
        rep_unit (RepeatUnit): Repeat unit projection length and mass.

    Returns:
        npt.NDArray: The reduced concentration data :math:`\varphi=cl^3`.
    """
    ...


def conc_to_phi(conc, rep_unit: RepeatUnit):
    return conc * rep_unit.length**3 * REDUCED_AVOGADRO / rep_unit.mass
