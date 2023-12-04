from pydantic import BaseModel

from polysoleval.conversions import phi_to_conc


class NeuralNetTypes(BaseModel):
    neuralnet_types: list[NeuralNetType]


class ValidRangeSets(BaseModel):
    range_sets: list[RangeSet]


class Evaluation(BaseModel):
    bg_only: EvaluationCase
    bth_only: EvaluationCase
    bg_and_bth: EvaluationCase
    token: str = ""
