from pydantic import BaseModel

from polysoleval.models import NeuralNetType, RangeSet, EvaluationCase


class NeuralNetTypes(BaseModel):
    neuralnet_types: list[NeuralNetType]


class ValidRangeSets(BaseModel):
    range_sets: list[RangeSet]


class Evaluation(BaseModel):
    bg_only: EvaluationCase
    bth_only: EvaluationCase
    bg_and_bth: EvaluationCase
    token: str = ""
