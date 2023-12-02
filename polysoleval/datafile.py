from collections.abc import Generator
from io import BytesIO
from time import sleep
from typing import BinaryIO, Callable
from uuid import UUID, uuid1

import numpy as np
import numpy.typing as npt

GeneratorFunc = Callable[[], Generator[bytes, None, None]]


def validate(
    filestream: BinaryIO,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Confirm that the given filestream is a comma-separated value (CSV) file with
    three columns (for concentration, molecular weight, and specific viscosity) and
    has no missing values. Returns the three columns if valid, raises a ValueError
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


class DatafileHandler:
    def __init__(self):
        self._cache: dict[UUID, BytesIO] = dict()

    def write_file(self, arr: npt.NDArray) -> str:
        b = BytesIO()
        np.savetxt(b, arr.T, fmt="%.5e", delimiter=",")
        uuid = uuid1()

        for _ in range(10):
            if uuid not in self._cache:
                break

        self._cache[uuid] = b
        return str(uuid)

    def wait_delete(self, token: str) -> None:
        sleep(4 * 60 * 60)  # 4 hours
        self._cache.pop(UUID(hex=token), None)

    def get_generator(self, token: str) -> GeneratorFunc:
        b = self._cache[UUID(hex=token)]

        def iter_csv():
            yield from b

        return iter_csv
