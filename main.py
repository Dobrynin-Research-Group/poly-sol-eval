from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, status, UploadFile
from fastapi.responses import StreamingResponse

from polysoleval.models import *
from polysoleval.datafile import validate
from polysoleval.evaluate import evaluate_dataset
from polysoleval.globals import *
from polysoleval import responses


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model descriptions
    global NEURALNET_TYPES
    NEURALNET_TYPES = {
        m.name: m
        for m in NeuralNetType.all_from_yaml(NEURALNETPATH / "model_types.yaml")
    }

    # load ML models
    for bg_net_path in NEURALNETPATH.glob("*-Bg.pt"):
        net_name, range_name, _ = bg_net_path.stem.split("-")

        net_type = NEURALNET_TYPES.get(net_name, None)
        if net_type is None:
            continue

        bth_net_path = bg_net_path.with_stem(f"{net_name}-{range_name}-Bth")
        if not bth_net_path.is_file():
            continue

        range_path = RANGEPATH / f"{range_name}.yaml"
        if not range_path.is_file():
            continue
        NEURALNET_PAIRS[NetRangePairNames(net_name, range_name)] = NeuralNetPair(
            neuralnet_type=net_type,
            range_path=range_path,
            bg_net_path=bg_net_path,
            bth_net_path=bth_net_path,
        )

    yield

    # clear ML models
    NEURALNET_TYPES.clear()
    NEURALNET_PAIRS.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return HTTPException(status.HTTP_200_OK)


@app.get("/models")
def get_models() -> responses.NeuralNetTypes:
    """Return every model type.

    Returns:
        _type_: _description_
    """
    return responses.NeuralNetTypes(neuralnet_types=list(NEURALNET_TYPES.values()))


@app.get("/models/{neuralnet_name}")
def get_model_instances(neuralnet_name: str) -> responses.ValidRangeSets:
    """Return all model instances of the given model type.

    Args:
        model_type (str): _description_

    Returns:
        _type_: _description_
    """
    valid_sets = [
        v.range_set for k, v in NEURALNET_PAIRS.items() if k.net_name == neuralnet_name
    ]
    if valid_sets:
        return responses.ValidRangeSets(range_sets=valid_sets)

    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        detail="No pre-trained networks found for the chosen neural net type",
    )


@app.post("/evaluate")
async def post_evaluate(
    ml_model_name: Annotated[str, Form()],
    range_name: Annotated[str, Form()],
    length: Annotated[float, Form(gt=0.0)],
    mass: Annotated[float, Form(gt=0.0)],
    datafile: UploadFile,
    background_tasks: BackgroundTasks,
) -> responses.Evaluation:
    # Validate first
    instance = NEURALNET_PAIRS.get(NetRangePairNames(ml_model_name, range_name), None)
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
            instance.bg_net,
            instance.bth_net,
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
    "/datafile/{token}",
    responses={200: {"content": {"text/plain; charset=utf-8": {}}}},
)
def get_datafile(token: str) -> StreamingResponse:
    try:
        file_generator = HANDLER.get_generator(token)
    except KeyError as ke:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail="datafile not found"
        ) from ke

    return StreamingResponse(file_generator(), media_type="text/plain")
