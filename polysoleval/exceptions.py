from enum import Enum

from fastapi import HTTPException, status


class PSSTException(HTTPException, Enum):
    InvalidNeuralNet = (
        status.HTTP_400_BAD_REQUEST,
        "invalid neural network type",
    )
    NeuralNetNotFound = (
        status.HTTP_404_NOT_FOUND,
        "found no pre-trained neural networks of this type",
    )
    InvalidNeuralNetPair = (
        status.HTTP_400_BAD_REQUEST,
        "invalid combination of neural network and range set",
    )
    InvalidDatafile = (
        status.HTTP_400_BAD_REQUEST,
        "uploaded datafile is improperly formatted",
    )
    InvalidDatafileColumns = (
        status.HTTP_400_BAD_REQUEST,
        "uploaded datafile should have three comma-delimited columns",
    )
    InvalidDatafileRows = (
        status.HTTP_400_BAD_REQUEST,
        "uploaded datafile should have at least three rows",
    )
    MissingDatafileData = (
        status.HTTP_400_BAD_REQUEST,
        "uploaded datafile is missing values",
    )
    EvaluationError = (
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "there was an issue with the evaluation",
    )
    DatafileResultNotFound = (
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "result data not found",
    )
    InvalidInternalCallLambda = (
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "invalid call signature in function lamda",
    )
    InvalidInternalCallGLambdaG = (
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "invalid call signature in function g_lamdag",
    )
    InferenceRuntimeError = (
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "error encountered during inference, likely the grid data is the wrong shape",
    )
    FittingValueError = (
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "error encountered during Pe fitting, likely divide-by-zero",
    )
