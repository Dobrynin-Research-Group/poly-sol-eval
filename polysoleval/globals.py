from os import environ
from pathlib import Path

from polysoleval.datafile import DatafileHandler
from polysoleval.response_models import ModelInstance, ModelType


MODELPATH = Path(environ["modelpath"])
RANGEPATH = Path(environ["rangepath"])
TMPPATH = Path(environ["tmppath"])

HANDLER = DatafileHandler()

MODEL_TYPES: dict[str, ModelType] = dict()
MODEL_INSTANCES: dict[tuple[str, str], ModelInstance] = dict()
