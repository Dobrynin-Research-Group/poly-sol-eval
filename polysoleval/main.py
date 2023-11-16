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
from psst import models, evaluation
from . import modeldb


# TODO: correct error codes
# TODO: simplify/pydantic-ify enums into model urls
# TODO: output OpenAPI documentation to yaml
MODELS_BASEPATH = Path(environ["modelpath"])
RANGES_BASEPATH = Path(environ["rangepath"])
TMP_BASEPATH = Path(environ["tmppath"])
DB_PASSWORD_FILE = Path(environ["modeldb_root_password_file"])
DB_PASSWORD = DB_PASSWORD_FILE.read_text().strip()

DB = modeldb.start_session(DB_PASSWORD, MODELS_BASEPATH, RANGES_BASEPATH, update=True)

ML_MODELS: dict[tuple[str, str], tuple[torch.nn.Module, torch.nn.Module]] = dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load ML models
    for model_ in modeldb.get_models(DB):
        model_name = model_.name
        Model: type[torch.nn.Module] | None = getattr(models, model_name, None)
        if Model is None:
            continue
        bg_model = Model()
        bth_model = Model()

        for range_ in modeldb.get_ranges(DB):
            model_file = MODELS_BASEPATH / model_name / (range_.name + ".pt")
            d = torch.load(model_file)
            bg_model.load_state_dict(d["bg_model"])
            bth_model.load_state_dict(d["bth_model"])

            ML_MODELS[(model_name, range_.name)] = (bg_model, bth_model)

    yield
    # clear ML models
    ML_MODELS.clear()


app = FastAPI(lifespan=lifespan)


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


class EvaluationResponse(BaseModel):
    bg_only: EvaluationResult
    bth_only: EvaluationResult
    both_bg_and_bth: EvaluationResult


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models")
async def get_models():
    return [
        ModelResponse(name=model.name, description=model.description)
        for model in modeldb.get_models(DB)
    ]


@app.get("/ranges")
async def get_ranges():
    return [
        RangeResponse(
            name=db_range.name,
            phi_res=db_range.phi_range.shape,
            nw_res=db_range.nw_range.shape,
            phi_range=BasicRange(
                min_value=db_range.phi_range.min_value,
                max_value=db_range.phi_range.max_value,
            ),
            nw_range=BasicRange(
                min_value=db_range.nw_range.min_value,
                max_value=db_range.nw_range.max_value,
            ),
            visc_range=BasicRange(
                min_value=db_range.visc_range.min_value,
                max_value=db_range.visc_range.max_value,
            ),
            bg_range=BasicRange(
                min_value=db_range.bg_range.min_value,
                max_value=db_range.bg_range.max_value,
            ),
            bth_range=BasicRange(
                min_value=db_range.bth_range.min_value,
                max_value=db_range.bth_range.max_value,
            ),
            pe_range=BasicRange(
                min_value=db_range.pe_range.min_value,
                max_value=db_range.pe_range.max_value,
            ),
        )
        for db_range in modeldb.get_ranges(DB)
    ]


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
    model = modeldb.get_model(DB, name=ml_model_name)
    if model is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"model {ml_model_name} not found"
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
    try:
        bg_model, bth_model = ML_MODELS[(ml_model_name, range_name)]
    except KeyError as ke:
        print(ML_MODELS)
        raise ke
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

    return EvaluationResponse(
        bg_only=bg_only_result,
        bth_only=bth_only_result,
        both_bg_and_bth=combo_result,
    )