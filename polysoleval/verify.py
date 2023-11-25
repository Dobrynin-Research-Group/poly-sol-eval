from typing import BinaryIO

import numpy as np
import numpy.typing as npt


def verify_datafile(
    filestream: BinaryIO,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Confirm that the given filestream is a comma-separated value (CSV) file with
    three columns (for concentration, molecular weight, and specific viscosity) and
    no missing values. Returns the three columns if valid, raises a ValueError
    otherwise.

    Args:
        filestream (BinaryIO): The file-like stream containing the contents of the
          datafile.

    Raises:
        ValueError: Either the datafile does not have three columns or there is invalid
          or missing data.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The values of the three columns.
    """
    data = np.genfromtxt(
        filestream,
        delimiter=",",
        missing_values="",
        filling_values=np.nan,
        ndmin=2,
        unpack=True,
    )

    if data.shape[0] != 3:
        raise ValueError(
            f"invalid number of columns in datafile: expected 3, found {len(data)}",
        )

    bad_data = np.logical_or(np.isnan(data), data <= 0)
    if np.any(bad_data):
        first_row = np.where(bad_data)[0][0]
        raise ValueError(f"invalid data in datafile, row {first_row}")

    conc, mw, visc = data
    return conc, mw, visc
