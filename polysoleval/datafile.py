from collections.abc import Generator
from datetime import timedelta
from io import BytesIO
from time import time
from typing import BinaryIO, Callable
from uuid import UUID, uuid1

import numpy as np
import numpy.typing as npt

from polysoleval.exceptions import PSSTException
from polysoleval.logging import get_logger

GeneratorFunc = Callable[[], Generator[str, None, None]]


def validate(
    filestream: BinaryIO,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Validate and parse filestream.

    Confirms that the given filestream is a comma-separated value (CSV) file with
    three columns (for concentration, molecular weight, and specific viscosity) and
    has no missing values. Returns the three columns if valid.

    Args:
        filestream (BinaryIO): The file-like stream containing the contents of the
          datafile.

    Raises:
        HTTPException: The datafile is improperly formatted in some way (code 400).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The values of the three columns.
    """
    log = get_logger()
    log.debug(f"validate({filestream = })")

    try:
        data = np.genfromtxt(
            filestream,
            delimiter=",",
            missing_values="",
            filling_values=np.nan,
            ndmin=2,
            unpack=True,
        )
    except Exception as e:
        raise PSSTException.InvalidDatafile from e

    if data.shape[0] != 3:
        raise PSSTException.InvalidDatafileColumns
    if data.shape[1] <= 2:
        raise PSSTException.InvalidDatafileRows

    bad_data = np.logical_or(np.isnan(data), data <= 0)
    if np.any(bad_data):
        # first_row = np.where(bad_data)[0][0]
        raise PSSTException.MissingDatafileData

    conc, mw, visc = data
    return conc, mw, visc


class DatafileHandler:
    def __init__(self):
        self._cache: dict[UUID, str] = dict()
        self._expiration: dict[UUID, float] = dict()
        self._log = get_logger()

    def write_file(self, arr: npt.NDArray) -> str:
        self._log.debug(f"DatafileHanler.write_file({arr = })")

        b = BytesIO()
        np.savetxt(
            b,
            arr,
            fmt="%.5e",
            delimiter=",",
            encoding="utf-8",
            header="x,labels,1y,2y,3y,4xBg,4yBg,4xBth,4yBth,4xCombo,4yCombo",
            comments="",
        )
        uuid = uuid1()
        file_string = b.getvalue().decode()

        # with open("sample_file.csv", "w") as csv:
        #     csv.write(file_string)

        for _ in range(10):
            if uuid not in self._cache:
                break
            uuid = uuid1()

        self._cache[uuid] = file_string
        self._expiration[uuid] = time() + timedelta(hours=4).total_seconds()
        return str(uuid)

    def check_delete(self) -> None:
        self._log.debug("DatafileHanler.check_delete()")

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
