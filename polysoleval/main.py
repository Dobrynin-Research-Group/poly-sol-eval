from contextlib import asynccontextmanager
from dataclasses import asdict
from os import environ
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile, status
import numpy as np
from pydantic import BaseModel
import torch

import psst
from . import modeldb, evaluate


# TODO: correct error codes
# TODO: simplify/pydantic-ify enums into model urls
# TODO: output OpenAPI documentation to yaml
MODELPATH = Path(environ["modelpath"])
RANGEPATH = Path(environ["rangepath"])
TMPPATH = Path(environ["tmppath"])
DB_PASSWORD_FILE = Path(environ["modeldb_root_password_file"])
DB_PASSWORD = DB_PASSWORD_FILE.read_text().strip()

DB = modeldb.start_session(DB_PASSWORD, MODELPATH, RANGEPATH, update=False)

model_range_spec = tuple[str, str]
bg_bth_models = tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]
ML_MODELS: dict[model_range_spec, bg_bth_models] = dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load ML models
    for model_ in modeldb.get_models(DB):
        model_name = model_.name
        bg_model = None
        bth_model = None

        for range_ in modeldb.get_ranges(DB):
            bg_file = MODELPATH / f"{model_name}-{range_.name}-Bg.pt"
            if bg_file.is_file():
                bg_model = torch.jit.load(bg_file)
                bg_model.eval()

            bth_file = MODELPATH / f"{model_name}-{range_.name}-Bth.pt"
            if bth_file.is_file():
                bth_model = torch.jit.load(bth_file)
                bth_model.eval()

            ML_MODELS[(model_name, range_.name)] = (bg_model, bth_model)

    yield
    # clear ML models
    ML_MODELS.clear()


app = FastAPI(lifespan=lifespan)


def phi_to_conc(phi: float, rep_unit_len: float):
    return phi / rep_unit_len**3 / evaluate.AVOGADRO_CONSTANT * 1e24


def conc_to_phi(conc: float, rep_unit_len: float):
    return conc * rep_unit_len**3 * evaluate.AVOGADRO_CONSTANT / 1e24


class Parameters(BaseModel):
    bg: Optional[float]
    bth: Optional[float]
    pe: float
    pe_variance: Optional[float]
    rep_unit_length: Optional[float]
    rep_unit_mass: Optional[float]


class ModelResponse(BaseModel):
    name: str
    description: Optional[str]


class BasicRange(BaseModel):
    min_value: float
    max_value: float


class RangeResponse(BaseModel):
    name: str

    phi_res: int
    nw_res: int

    phi_range: BasicRange
    nw_range: BasicRange
    visc_range: BasicRange
    bg_range: BasicRange
    bth_range: BasicRange
    pe_range: BasicRange


class EvaluationResult(BaseModel):
    bg: Optional[float] = None
    bth: Optional[float] = None
    pe: float = 0.0
    pe_variance: Optional[float] = None
    kuhn_length: float = 0.0
    thermal_blob_size: Optional[float] = None
    dp_of_thermal_blob: Optional[float] = None
    excluded_volume: Optional[float] = None
    thermal_blob_conc: Optional[float] = None
    concentrated_conc: float = 0.0

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


class EvaluationResponse(BaseModel):
    bg_only: EvaluationResult
    bth_only: EvaluationResult
    both_bg_and_bth: EvaluationResult


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models")
async def get_models():
    return MODELS


@app.get("/ranges")
async def get_ranges():
    return RANGES


@app.post("/new-parameters")
async def post_new_parameters(params: Parameters):
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


@app.post("/evaluate")
async def post_evaluate(
    ml_model_name: Annotated[str, Form(pattern="[\\w\\d]+")],
    range_name: Annotated[str, Form(pattern="[\\w\\d]+")],
    length: Annotated[float, Form(gt=0.0)],
    mass: Annotated[float, Form(gt=0.0)],
    datafile: UploadFile,
):
    # Validate first
    bg_model, bth_model = ML_MODELS.get((ml_model_name, range_name), (None, None))
    if bg_model is None or bth_model is None:
        model_exists = False
        for m in MODELS:
            if m.name == ml_model_name:
                model_exists = True
                break
        if not model_exists:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, detail="Unrecognized model name"
            )

        range_exists = False
        for r in RANGES:
            if r.name == range_name:
                range_exists = True
                break
        if not range_exists:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, detail="Unrecognized range name"
            )

        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Unsupported pair of model and range names",
        )

    range_response = None
    for r in RANGES:
        if r.name == range_name:
            range_response = r
            break
    if range_response is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="Unrecognized range name"
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

    # Set up config
    phi_range = range_response.phi_range.model_dump()
    phi_range["shape"] = range_response.phi_res
    nw_range = range_response.nw_range.model_dump()
    nw_range["shape"] = range_response.nw_res
    range_config = psst.RangeConfig(
        phi_range=phi_range,
        nw_range=nw_range,
        visc_range=range_response.visc_range.model_dump(),
        bg_range=range_response.bg_range.model_dump(),
        bth_range=range_response.bth_range.model_dump(),
        pe_range=range_response.pe_range.model_dump(),
    )

    repeat_unit = evaluate.RepeatUnit(length, mass)

    # Do evaluation (two inferences and three curve fits)
    try:
        result = await evaluate.evaluate_dataset(
            conc,
            mw,
            visc,
            repeat_unit,
            bg_model,
            bth_model,
            range_config,
        )
    except Exception as re:
        detail = "unexpected failure in evaluation\n" + re.args[0]
        raise HTTPException(status.HTTP_417_EXPECTATION_FAILED, detail=detail)

    bg_only_result = EvaluationResult.create(
        bg=result.bg,
        bth=None,
        pe=result.pe_bg_only.opt,
        pe_variance=result.pe_bg_only.var,
        rep_unit_length=length,
    )
    bth_only_result = EvaluationResult.create(
        bg=None,
        bth=result.bth,
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

    return EvaluationResponse(
        bg_only=bg_only_result,
        bth_only=bth_only_result,
        both_bg_and_bth=combo_result,
    )
