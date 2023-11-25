from contextlib import asynccontextmanager
from os import environ
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, UploadFile, status

from polysoleval.evaluate import create_result, evaluate_dataset, RepeatUnit
from polysoleval.load import *
from polysoleval.response_models import *
from polysoleval.verify import verify_datafile


MODELPATH = Path(environ["modelpath"])
RANGEPATH = Path(environ["rangepath"])
TMPPATH = Path(environ["tmppath"])

RANGES: list[RangeResponse] = list()
MODELS: list[ModelResponse] = list()
ModelRangeSpec = tuple[str, str]
BgBthModels = tuple[MaybeModel, MaybeModel]
ML_MODELS: dict[ModelRangeSpec, BgBthModels] = dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load range configs
    for range_file in RANGEPATH.glob("*.yaml"):
        RANGES.append(load_range_from_yaml(range_file))

    # load model descriptions
    MODELS = load_models_from_yaml(MODELPATH / "models.yaml")

    # load ML models
    for model_ in MODELS:
        for range_ in RANGES:
            bg_fp = MODELPATH / f"{model_.name}-{range_.name}-Bg.pt"
            bth_fp = MODELPATH / f"{model_.name}-{range_.name}-Bth.pt"
            bg_model = load_model_file(bg_fp)
            bth_model = load_model_file(bth_fp)
            key = (model_.name, range_.name)
            ML_MODELS[key] = (bg_model, bth_model)

    yield

    # clear ML models
    RANGES.clear()
    MODELS.clear()
    ML_MODELS.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"models": MODELS}


@app.get("/models")
async def get_models() -> list[ModelResponse]:
    return MODELS


@app.get("/ranges")
async def get_ranges() -> list[RangeResponse]:
    return RANGES


@app.post("/new-parameters")
async def post_new_parameters(params: InputParameters):
    if params.rep_unit_length is None:
        raise HTTPException(
            status.HTTP_411_LENGTH_REQUIRED,
            detail="rep_unit_length (repeat unit length) required",
        )
    return create_result(
        bg=params.bg,
        bth=params.bth,
        pe=params.pe,
        pe_variance=params.pe_variance,
        rep_unit=RepeatUnit(params.rep_unit_length, params.rep_unit_mass),
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
        conc, mw, visc = verify_datafile(datafile.file)
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, e.args[0]) from e

    # Do evaluation (two inferences and three curve fits)
    try:
        result = await evaluate_dataset(
            conc,
            mw,
            visc,
            RepeatUnit(length, mass),
            bg_model,
            bth_model,
            range_response,
        )
    except Exception as re:
        detail = "unexpected failure in evaluation\n" + re.args[0]
        raise HTTPException(status.HTTP_417_EXPECTATION_FAILED, detail=detail)

    return result
