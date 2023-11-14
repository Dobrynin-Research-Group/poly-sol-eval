from contextlib import asynccontextmanager
from dataclasses import asdict
import datetime
from enum import Enum
from os import environ
from pathlib import Path
import re
from typing import Annotated, Optional

from fastapi import (
    BackgroundTasks,
    Cookie,
    FastAPI,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
import numpy as np
from pydantic import BaseModel, field_validator
import torch

import psst
from psst import models, evaluation
from . import modeldb


# TODO: correct error codes
# TODO: simplify/pydantic-ify enums into model urls
# TODO: output OpenAPI documentation to yaml
MODELS_BASEPATH = Path(environ["modelpath"])
TMP_BASEPATH = Path(environ["tmppath"])

# MODEL_FILE_PATTERN = r"(?<type>[\d\w]+)/(?<name>[\d\w]+)/(?<date>\d{8})\.pt"
# MODEL_LIST = list(MODELS_BASEPATH.glob("**/.pt"))

DB = modeldb.start_session()

ML_MODELS: dict[tuple[str, str], tuple[torch.nn.Module, torch.nn.Module]] = dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load ML models
    for range_path in MODELS_BASEPATH.iterdir():
        for model_file in range_path.iterdir():
            model_name = model_file.stem
            Model: type[torch.nn.Module] | None = getattr(models, model_name, None)
            if Model is None:
                continue
            bg_model = Model()
            bth_model = Model()

            d = torch.load(model_file)
            bg_model.load_state_dict(d["bg_model"])
            bth_model.load_state_dict(d["bth_model"])

            ML_MODELS[(range_path.name, model_name)] = (bg_model, bth_model)

    yield
    # clear ML models


app = FastAPI(lifespan=lifespan)


class InvalidSettings(WebSocketDisconnect):
    pass


class EvaluationResponse(BaseModel):
    model_name: str
    range_name: str
    repeat_unit_length: float
    repeat_unit_mass: float
    filename: str
    num_rows: int

    @field_validator("model_name", "range_name")
    @classmethod
    def is_alphanumeric(cls, v: str):
        if not v.isalnum():
            raise ValueError("name should be alphanumeric")
        return v

    @field_validator("repeat_unit_length", "repeat_unit_mass")
    @classmethod
    def is_positive(cls, v: float):
        if v <= 0:
            raise ValueError("value must be positive")
        return v


class BasicResponse(BaseModel):
    code: int
    message: str | None = None


class EvaluationState(str, Enum):
    AWAITING_SETTINGS = "awaiting settings"
    UPLOADING_FILE = "uploading file"
    PREPROCESSING = "preprocessing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"


def phi_to_conc(phi: float, rep_unit_len: float):
    return phi / rep_unit_len**3 / evaluation.AVOGADRO_CONSTANT * 1e24


def conc_to_phi(conc: float, rep_unit_len: float):
    return conc * rep_unit_len**3 * evaluation.AVOGADRO_CONSTANT / 1e24


class Parameters(BaseModel):
    bg: Optional[float]
    bth: Optional[float]
    pe: float
    pe_variance: Optional[float]
    rep_unit_length: Optional[float]
    rep_unit_mass: Optional[float]


class EvaluationResult(BaseModel):
    bg: Optional[float]
    bth: Optional[float]
    pe: float
    pe_variance: Optional[float]
    kuhn_length: float
    thermal_blob_size: Optional[float]
    dp_of_thermal_blob: Optional[float]
    excluded_volume: Optional[float]
    thermal_blob_conc: Optional[float]
    concentrated_conc: float

    @classmethod
    def create(
        cls,
        *,
        bg: Optional[float],
        bth: Optional[float],
        pe: float,
        pe_variance: Optional[float],
        rep_unit_length: float,
    ):
        thermal_blob_size = None
        dp_of_thermal_blob = None
        excluded_volume = None
        thermal_blob_conc = None
        concentrated_conc = None

        if bg and bth:
            kuhn_length = rep_unit_length / bth**2
            phi_th = bth**3 * (bth / bg) ** (1 / (2 * 0.588 - 1))
            thermal_blob_size = rep_unit_length * bth**2 / phi_th
            dp_of_thermal_blob = (bth**3 / phi_th) ** 2
            thermal_blob_conc = phi_to_conc(phi_th, rep_unit_length)
            excluded_volume = phi_th * kuhn_length**3
        elif bg:
            kuhn_length = rep_unit_length / bg ** (1 / 0.412)
        elif bth:
            kuhn_length = rep_unit_length / bth**2
        else:
            raise ValueError("must supply at least one of bg or bth")
        concentrated_conc = phi_to_conc(
            1 / kuhn_length**2 / rep_unit_length, rep_unit_length
        )

        return cls(
            bg=bg,
            bth=bth,
            pe=pe,
            pe_variance=pe_variance,
            kuhn_length=kuhn_length,
            thermal_blob_size=thermal_blob_size,
            excluded_volume=excluded_volume,
            dp_of_thermal_blob=dp_of_thermal_blob,
            thermal_blob_conc=thermal_blob_conc,
            concentrated_conc=concentrated_conc,
        )


class EvaluateResponse(BaseModel):
    bg_only: EvaluationResult
    bth_only: EvaluationResult
    both_bg_and_bth: EvaluationResult


class ModelState(BaseModel):
    model_name: str
    model_info: str
    range_name: str
    range_str: str
    phi_range: psst.Range
    nw_range: psst.Range
    visc_range: psst.Range
    bg_range: psst.Range
    bth_range: psst.Range
    pe_range: psst.Range
    # num_epochs: int
    # creation_date: datetime.date


# TODO: validate rep_unit_* > 0
class EvalSettings(BaseModel):
    model_name: str
    range_name: str
    model_state_idx: int
    device: str
    rep_unit_length: float
    rep_unit_mass: float


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models")
async def get_models():
    models = list(modeldb.get_model_names(DB))
    return models


@app.get("/ranges")
async def get_ranges():
    ranges = list(modeldb.get_range_names(DB))
    return ranges


@app.get("/new-parameters")
async def get_new_parameters(params: Parameters):
    if params.rep_unit_length is None:
        raise HTTPException(
            status.HTTP_411_LENGTH_REQUIRED,
            detail="rep_unit_length (repeat unit length) required",
        )
    return EvaluationResult.create(
        bg=params.bg,
        bth=params.bth,
        pe=params.pe,
        pe_variance=params.pe_variance,
        rep_unit_length=params.rep_unit_length,
    )


# @app.get("/states")
# async def get_states(model: str | None = None, range_: str | None = None):
#     states = modeldb.get_state_list(DB, model_name=model, range_name=range_)
#     state_reps: list[ModelState] = list()
#     for state in states:
#         r: modeldb.RangeData = state.range
#         m: modeldb.ModelData = state.model
#         state_rep = ModelState(
#             model_name=m.name,
#             model_info="",
#             range_name=r.name,
#             range_str="",
#             phi_range=psst.Range(**asdict(r.phi_range)),
#             nw_range=psst.Range(**asdict(r.nw_range)),
#             visc_range=psst.Range(**asdict(r.visc_range)),
#             bg_range=psst.Range(**asdict(r.bg_range)),
#             bth_range=psst.Range(**asdict(r.bth_range)),
#             pe_range=psst.Range(**asdict(r.pe_range)),
#             num_epochs=state.num_epochs,
#             creation_date=state.creation_date,
#         )
#         state_reps.append(state_rep)
#     return state_reps


@app.post("/evaluate")
async def post_evaluate(
    model_name: Annotated[str, Form(pattern="[\\w\\d]+")],
    range_name: Annotated[str, Form(pattern="[\\w\\d]+")],
    length: Annotated[float, Form(gt=0.0)],
    mass: Annotated[float, Form(gt=0.0)],
    datafile: UploadFile,
):
    # Validate first
    model = modeldb.get_model(DB, name=model_name)
    if model is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"model {model_name} not found"
        )

    range_ = modeldb.get_range(DB, name=range_name)
    if range_ is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"range {range_name} not found"
        )

    try:
        data = np.genfromtxt(
            datafile.file,
            delimiter=",",
            missing_values="",
            filling_values=np.nan,
            ndmin=2,
            unpack=True,
        )
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, e.args[0]) from e

    if data.shape[0] != 3:
        raise HTTPException(
            status.HTTP_406_NOT_ACCEPTABLE,
            detail="invalid number of columns in datafile:"
            f" expected 3, found {data.shape[1]}",
        )

    bad_data = np.logical_or(np.isnan(data), data <= 0)
    if np.any(bad_data):
        first_row = np.where(bad_data)[0][0]
        raise HTTPException(
            status.HTTP_406_NOT_ACCEPTABLE,
            detail=f"invalid data in datafile, row {first_row}",
        )

    conc, mw, visc = data

    # Set up models and config
    bg_model, bth_model = ML_MODELS[(range_name, model_name)]
    range_config = psst.RangeConfig(
        phi_range=asdict(range_.phi_range),
        nw_range=asdict(range_.nw_range),
        visc_range=asdict(range_.visc_range),
        bg_range=asdict(range_.bg_range),
        bth_range=asdict(range_.bth_range),
        pe_range=asdict(range_.pe_range),
    )

    repeat_unit = evaluation.RepeatUnit(length, mass)

    # Do evaluation (two inferences and one curve fit)
    result = evaluation.evaluate_dataset(
        conc,
        mw,
        visc,
        repeat_unit,
        bg_model,
        bth_model,
        range_config,
    )

    bg_only_result = EvaluationResult.create(
        bg=result.bg,
        bth=None,
        pe=result.pe_bg_only.opt,
        pe_variance=result.pe_bg_only.var,
        rep_unit_length=length,
    )
    bth_only_result = EvaluationResult.create(
        bth=result.bth,
        bg=None,
        pe=result.pe_bth_only.opt,
        pe_variance=result.pe_bth_only.var,
        rep_unit_length=length,
    )
    combo_result = EvaluationResult.create(
        bg=result.bg,
        bth=result.bth,
        pe=result.pe_combo.opt,
        pe_variance=result.pe_combo.var,
        rep_unit_length=length,
    )

    return EvaluateResponse(
        bg_only=bg_only_result,
        bth_only=bth_only_result,
        both_bg_and_bth=combo_result,
    )


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         await websocket.send_text(EvaluationState.AWAITING_SETTINGS)
#         settings_dict: dict = await websocket.receive_json()
#         settings = EvalSettings.model_validate(settings_dict)

#         Model = getattr(models, settings.model_name, None)
#         if Model is None:
#             raise InvalidSettings(
#                 code=status.HTTP_400_BAD_REQUEST,
#                 reason=f"Invalid model name: {settings.model_name}",
#             )

#         r = modeldb.get_range(DB, name=settings.range_name)
#         if r is None:
#             raise InvalidSettings(
#                 code=status.HTTP_400_BAD_REQUEST,
#                 reason=f"Invalid range name: {settings.range_name}",
#             )

#         # state_row = modeldb.get_state(DB, settings.model_state_idx)
#         # if state_row is None:
#         #     raise InvalidSettings(
#         #         code=status.HTTP_400_BAD_REQUEST,
#         #         reason=f"Invalid model state index: {settings.model_state_idx}",
#         #     )

#         await websocket.send_json({"status": status.HTTP_202_ACCEPTED})

#         await websocket.send_text(EvaluationState.UPLOADING_FILE)
#         tmp_filepath = "./tmp/hosh.bt"
#         with open(tmp_filepath, "wb") as f:
#             async for b in websocket.iter_bytes():
#                 f.write(b)

#         await websocket.send_text(EvaluationState.PREPROCESSING)
#         bg_model = Model()
#         bth_model = Model()
#         range_config = psst.RangeConfig(
#             phi_range=settings,
#         )

#         repeat_unit = evaluation.RepeatUnit(
#             settings_dict["mass"], settings_dict["length"]
#         )
#         with open(tmp_filepath, "rb") as f:
#             conc, mw, spec_visc = np.genfromtxt(
#                 f,
#                 delimiter=",",
#                 missing_values="",
#                 filling_values=np.nan,
#             )

#         await websocket.send_text(EvaluationState.EVALUATING)

#         evaluation.evaluate_dataset(
#             conc,
#             mw,
#             spec_visc,
#             repeat_unit,
#         )
#         result = evaluation.run(model_path, conc, mw, spec_visc, repeat_unit)

#         await websocket.send_json(EvaluationResults(**result._asdict()))
#     except InvalidSettings as ise:
#         await websocket.send_json({"code": ise.code, "message": ise.reason})
#     except WebSocketDisconnect:
#         pass
#     finally:
#         await websocket.close()
