from enum import Enum

from fastapi import HTTPException, status


class PSSTException(HTTPException, Enum):
    InvalidNeuralNet = HTTPException(
        status.HTTP_400_BAD_REQUEST,
        detail="invalid neural network type",
    )
    NeuralNetNotFound = HTTPException(
        status.HTTP_404_NOT_FOUND,
        detail="no pre-trained neural networks of this type",
    )
    InvalidNeuralNetPair = HTTPException(
        status.HTTP_400_BAD_REQUEST,
        detail="invalid combination of neural network and range set",
    )
    InvalidDatafile = HTTPException(
        status.HTTP_400_BAD_REQUEST,
        detail="uploaded datafile is improperly formatted",
    )
    EvaluationError = HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="there was an issue with the evaluation",
    )
    DatafileResultNotFound = HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="result data not found",
    )
