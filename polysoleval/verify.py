from typing import BinaryIO

import numpy as np
import numpy.typing as npt


def verify_datafile(
    filestream: BinaryIO,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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
