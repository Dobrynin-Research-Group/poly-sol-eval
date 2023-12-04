from enum import Enum

from fastapi import HTTPException, status


class PSSTException(HTTPException, Enum):
    InvalidNeuralNet = (
        status.HTTP_400_BAD_REQUEST,
        "invalid neural network type",
    )
    NeuralNetNotFound = (
        status.HTTP_404_NOT_FOUND,
        "no pre-trained neural networks of this type",
    )
    InvalidNeuralNetPair = (
        status.HTTP_400_BAD_REQUEST,
        "invalid combination of neural network and range set",
    )
    InvalidDatafile = (
        status.HTTP_400_BAD_REQUEST,
        "uploaded datafile is improperly formatted",
    )
    EvaluationError = (
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "there was an issue with the evaluation",
    )
    DatafileResultNotFound = (
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "result data not found",
    )
