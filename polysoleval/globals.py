from os import environ
from pathlib import Path
from typing import NamedTuple

from polysoleval.models import NeuralNetPair, NeuralNetType
from polysoleval.datafile import DatafileHandler


__all__ = [
    "NetRangePairNames",
    "NU",
    "NU3_1",
    "NU2_1",
    "REDUCED_AVOGADRO",
    "NEURALNETPATH",
    "RANGEPATH",
    "TMPPATH",
    "HANDLER",
    "NEURALNET_TYPES",
    "NEURALNET_PAIRS",
]


class NetRangePairNames(NamedTuple):
    net_name: str
    range_name: str


NU = 0.588  # good solvent exponent
NU3_1 = 3 * NU - 1
NU2_1 = 2 * NU - 1

REDUCED_AVOGADRO = 0.60221408  # Avogadro number divided by 10^24

NEURALNETPATH = Path(environ["modelpath"])
RANGEPATH = Path(environ["rangepath"])
TMPPATH = Path(environ["tmppath"])

HANDLER = DatafileHandler()

NEURALNET_TYPES: dict[str, NeuralNetType] = dict()
NEURALNET_PAIRS: dict[NetRangePairNames, NeuralNetPair] = dict()
