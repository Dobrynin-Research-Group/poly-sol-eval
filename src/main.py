from enum import Enum
from pathlib import Path
from uuid import uuid1

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
import numpy as np
from pydantic import BaseModel

from polysoleval.evaluate import evaluate_data, RepeatUnit


# TODO: correct error codes
# TODO: simplify/pydantic-ify enums into model urls
MODELS_BASEPATH = Path("./models")
TMP_BASEPATH = Path("./tmp")

MODEL_FILE_PATTERN = r"(?<type>[\d\w]+)/(?<name>[\d\w]+)/(?<date>\d{8})\.pt"
MODEL_LIST = list(MODELS_BASEPATH.glob("**/.pt"))

app = FastAPI()


class EvaluationState(str, Enum):
    RECEIVED_SETTINGS = "received settings"
    UPLOADING_FILE = "uploading file"
    FILE_UPLOADED = "file uploaded"
    PROCESSING = "processing"


class EvaluationResults(BaseModel):
    bg: float
    bth: float
    pe_combo: tuple[float, float]
    pe_bg_only: tuple[float, float]
    pe_bth_only: tuple[float, float]


def get_tmpfile():
    return TMP_BASEPATH / (str(uuid1()) + ".csv")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models/")
async def get_models():
    return {"model_types": [f.name for f in MODELS_BASEPATH.iterdir() if f.is_dir()]}


@app.get("/models/{model_type:str}/")
async def get_model_names(model_type: str):
    type_dir = MODELS_BASEPATH / model_type
    if not type_dir.is_dir():
        return {"error": 404}
    return {"model_names": [f.name for f in type_dir.iterdir() if f.is_dir()]}


@app.get("/models/{model_type:str}/{model_name:str}/")
async def get_model_revisions(model_type: str, model_name: str):
    name_dir = MODELS_BASEPATH / model_type / model_name
    if not name_dir.is_dir():
        return {"error": 404}
    return {"model_revisions": [f.name for f in name_dir.iterdir() if f.is_file()]}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        settings_dict: dict = await websocket.receive_json()
        model_path = MODELS_BASEPATH / settings_dict["model_path"]
        repeat_unit = RepeatUnit(settings_dict["mass"], settings_dict["length"])
        await websocket.send_text(EvaluationState.RECEIVED_SETTINGS)

        tmp_filepath = get_tmpfile()
        await websocket.send_text(EvaluationState.UPLOADING_FILE)
        with open(tmp_filepath, "wb") as f:
            async for b in websocket.iter_bytes():
                f.write(b)
        await websocket.send_text(EvaluationState.FILE_UPLOADED)

        with open(tmp_filepath, "rb") as f:
            conc, mw, spec_visc = np.genfromtxt(
                f,
                delimiter=",",
                missing_values="",
                filling_values=np.nan,
            )

        await websocket.send_text(EvaluationState.PROCESSING)
        result = evaluate_data(model_path, conc, mw, spec_visc, repeat_unit)

        await websocket.send_json(EvaluationResults(**result._asdict()))
        await websocket.close()
    except WebSocketDisconnect:
        return {"error": "disconnected"}
