from os import environ
from pathlib import Path
from typing import NamedTuple

from polysoleval.models import NeuralNetPair, NeuralNetType
from polysoleval.datafile import DatafileHandler


class NetRangePairNames(NamedTuple):
    net_name: str
    range_name: str


NEURALNETPATH = Path(environ["modelpath"])
RANGEPATH = Path(environ["rangepath"])
TMPPATH = Path(environ["tmppath"])

HANDLER = DatafileHandler()

NEURALNET_TYPES: dict[str, NeuralNetType] = dict()
NEURALNET_PAIRS: dict[NetRangePairNames, NeuralNetPair] = dict()
