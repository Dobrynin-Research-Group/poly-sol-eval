from collections.abc import Generator
from datetime import timedelta
from io import BytesIO
from time import time
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
        self._expiration: dict[UUID, float] = dict()

    def write_file(self, arr: npt.NDArray) -> str:
        b = BytesIO()
        np.savetxt(b, arr.T, fmt="%.5e", delimiter=",", encoding="utf-8")
        uuid = uuid1()

        for _ in range(10):
            if uuid not in self._cache:
                break

        self._cache[uuid] = b
        self._expiration[uuid] = time() + timedelta(hours=4).total_seconds()
        return str(uuid)

    def check_delete(self) -> None:
        now = time()
        for uuid in self._cache:
            if uuid not in self._expiration or now >= self._expiration[uuid]:
                self._expiration.pop(uuid, None)
                self._cache.pop(uuid)

    def get_generator(self, token: str) -> GeneratorFunc:
        b = self._cache[UUID(hex=token)]

        def iter_csv():
            yield from b

        return iter_csv

    def clear(self) -> None:
        self._cache.clear()
        self._expiration.clear()
