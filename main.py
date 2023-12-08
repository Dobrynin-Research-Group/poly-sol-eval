from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import StreamingResponse
from polysoleval.logging import get_logger, setup_logger

from polysoleval.models import *
from polysoleval.datafile import validate
from polysoleval.evaluate import evaluate_dataset
from polysoleval.exceptions import PSSTException
from polysoleval.globals import *
from polysoleval import responses


@asynccontextmanager
async def lifespan(app: FastAPI):
    log = get_logger()
    log = setup_logger(log)
    log.info("set up the logger")

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
    HANDLER.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/", status_code=200)
def root():
    return


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
    if neuralnet_name not in NEURALNET_TYPES.keys():
        raise PSSTException.InvalidNeuralNet

    valid_sets = [
        v.range_set for k, v in NEURALNET_PAIRS.items() if k.net_name == neuralnet_name
    ]
    if valid_sets:
        return responses.ValidRangeSets(range_sets=valid_sets)

    raise PSSTException.NeuralNetNotFound


@app.post("/evaluate")
async def post_evaluate(
    ml_model_name: Annotated[str, Form()],
    range_name: Annotated[str, Form()],
    length: Annotated[float, Form(gt=0.0)],
    mass: Annotated[float, Form(gt=0.0)],
    datafile: UploadFile,
) -> responses.Evaluation:
    HANDLER.check_delete()

    log = get_logger()
    log.debug("starting /evaluate")

    # Validate first
    instance = NEURALNET_PAIRS.get(NetRangePairNames(ml_model_name, range_name), None)
    if instance is None:
        raise PSSTException.InvalidNeuralNetPair

    rep_unit = RepeatUnit(length=length, mass=mass)

    conc, mw, visc = validate(datafile.file)

    # Do evaluation (two inferences and three curve fits)
    try:
        results = await evaluate_dataset(
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
        raise PSSTException.EvaluationError from re

    eval_response = responses.Evaluation.from_params(
        bg=results.bg,
        bth=results.bth,
        pe_combo=results.pe_combo,
        pe_bg=results.pe_bg_only,
        pe_bth=results.pe_bth_only,
        rep_unit=rep_unit,
    )

    eval_response.token = HANDLER.write_file(results.array)
    return eval_response


@app.get("/datafile/{token}")
def get_datafile(token: str) -> StreamingResponse:
    try:
        file_generator = HANDLER.get_generator(token)
    except KeyError as ke:
        raise PSSTException.DatafileResultNotFound from ke
    finally:
        HANDLER.check_delete()

    return StreamingResponse(file_generator(), media_type="text/plain")
