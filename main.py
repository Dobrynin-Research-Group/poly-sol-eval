from contextlib import asynccontextmanager
from os import environ
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile, status

from polysoleval.globals import *
from polysoleval.response_models import *


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load range configs
    for range_file in RANGEPATH.glob("*.yaml"):
        range_ = load_range_from_yaml(range_file)
        RANGES[range_.name] = range_

    # load model descriptions
    global MODELS
    MODELS = {m.name: m for m in load_models_from_yaml(MODELPATH / "models.yaml")}

    # load ML models
    for model_name in MODELS:
        for range_name in RANGES:
            bg_fp = MODELPATH / f"{model_name}-{range_name}-Bg.pt"
            bth_fp = MODELPATH / f"{model_name}-{range_name}-Bth.pt"
            bg_model = load_model_file(bg_fp)
            bth_model = load_model_file(bth_fp)
            key = (model_name, range_name)
            ML_MODELS[key] = (bg_model, bth_model)

    yield

    # clear ML models
    RANGES.clear()
    MODELS.clear()
    ML_MODELS.clear()


app = FastAPI(lifespan=lifespan)


def get_model_range_pairs(
    model_name: Optional[str] = None, range_name: Optional[str] = None
):
    if model_name is None and range_name is None:
        raise ValueError("either model_name or range_name must be given")

    return_names: list[tuple[ModelResponse, RangeResponse]] = list()

    for m, r in ML_MODELS.keys():
        if r == range_name or m == model_name:
            model_ = MODELS[m]
            range_ = RANGES[r]
            return_names.append((model_, range_))

    return return_names


@app.get("/")
async def root():
    return {"model_names": [m for m in MODELS], "ranges": [r for r in RANGES.values()]}


@app.get("/models")
async def get_models():
    return {"model_names": [m for m in MODELS]}


@app.get("/models/{model_name}")
async def get_model_by_name(model_name: str):
    pairs = get_model_range_pairs(model_name=model_name)
    if pairs:
        return {"pairs": [{"model": m, "range": r} for m, r in pairs]}
    raise HTTPException(status.HTTP_404_NOT_FOUND, detail="model name not found")


@app.get("/ranges")
async def get_ranges():
    return {"ranges": [r for r in RANGES.values()]}


@app.get("/ranges/{range_name}")
async def get_range_by_name(range_name: str):
    pairs = get_model_range_pairs(range_name=range_name)
    if pairs:
        return {"pairs": [{"model": m, "range": r} for m, r in pairs]}
    raise HTTPException(status.HTTP_404_NOT_FOUND, detail="range name not found")


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
    if ml_model_name not in MODELS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="Unrecognized model name"
        )
    if range_name not in RANGES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="Unrecognized range name"
        )

    bg_model, bth_model = ML_MODELS.get((ml_model_name, range_name), (None, None))
    if bg_model is None or bth_model is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Unsupported pair of model and range names",
        )

    range_response = RANGES[range_name]

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
