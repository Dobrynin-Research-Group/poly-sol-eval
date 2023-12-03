from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, status, UploadFile
from fastapi.responses import StreamingResponse

from polysoleval.datafile import validate
from polysoleval.evaluate import evaluate_dataset
from polysoleval.globals import *
from polysoleval.response_models import *


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model descriptions
    types = ModelType.all_from_yaml(MODELPATH / "model_types.yaml")

    global MODEL_TYPES
    MODEL_TYPES = {m.name: m for m in types}

    # load ML models
    for bg_model_path in MODELPATH.glob("*-Bg.pt"):
        model_name, range_name, _ = bg_model_path.stem.split("-")

        model_type = MODEL_TYPES.get(model_name, None)
        if model_type is None:
            continue

        bth_model_path = bg_model_path.with_stem(f"{model_name}-{range_name}-Bth")
        if not bth_model_path.is_file():
            continue

        range_path = RANGEPATH / f"{range_name}.yaml"
        if not range_path.is_file():
            continue
        MODEL_INSTANCES[(model_name, range_name)] = ModelInstance(
            model_type=model_type,
            range_path=range_path,
            bg_model_path=bg_model_path,
            bth_model_path=bth_model_path,
        )

    yield

    # clear ML models
    MODEL_TYPES.clear()
    MODEL_INSTANCES.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return HTTPException(status.HTTP_200_OK)


@app.get("/models")
def get_models() -> ModelTypesResponse:
    """Return every model type.

    Returns:
        _type_: _description_
    """
    return ModelTypesResponse(model_types=list(MODEL_TYPES.values()))


@app.get("/models/{model_type}")
def get_model_instances(model_type: str) -> ModelInstancesResponse:
    """Return all model instances of the given model type.

    Args:
        model_type (str): _description_

    Returns:
        _type_: _description_
    """
    instances = [v.range_set for k, v in MODEL_INSTANCES.items() if k[0] == model_type]
    if instances:
        return ModelInstancesResponse(model_instances=instances)

    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        detail="No valid instances found for the chosen model type",
    )


@app.post("/evaluate")
async def post_evaluate(
    ml_model_name: Annotated[str, Form()],
    range_name: Annotated[str, Form()],
    length: Annotated[float, Form(gt=0.0)],
    mass: Annotated[float, Form(gt=0.0)],
    datafile: UploadFile,
    background_tasks: BackgroundTasks,
) -> EvaluationResponse:
    # Validate first
    instance = MODEL_INSTANCES.get((ml_model_name, range_name), None)
    if instance is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Unsupported pair of model and range names",
        )

    rep_unit = RepeatUnit(length=length, mass=mass)

    try:
        conc, mw, visc = validate(datafile.file)
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, e.args[0]) from e

    # Do evaluation (two inferences and three curve fits)
    try:
        result, arr = await evaluate_dataset(
            conc,
            mw,
            visc,
            rep_unit,
            instance.bg_model,
            instance.bth_model,
            instance.range_set,
        )
    except Exception as re:
        detail = "unexpected failure in evaluation\n" + re.args[0]
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
        ) from re

    result.token = HANDLER.write_file(arr)
    background_tasks.add_task(HANDLER.wait_delete, result.token)
    return result


@app.get(
    "/datafile/{token}", responses={200: {"content": {"text/plain; charset=utf-8": {}}}}
)
def get_datafile(token: str) -> StreamingResponse:
    try:
        file_generator = HANDLER.get_generator(token)
    except KeyError as ke:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="data not found") from ke

    return StreamingResponse(file_generator(), media_type="text/plain")
