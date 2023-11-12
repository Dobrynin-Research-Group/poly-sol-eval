from datetime import date
import pytest
import sqlalchemy as sqla
import sqlalchemy.orm as orm

from polysoleval import modeldb
import psst

DATABASE_URL = "sqlite:///:memory:"


# Setup the in-memory SQLite database for testing
@pytest.fixture
def session():
    engine = sqla.create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=sqla.StaticPool,
    )
    TestingSessionLocal = orm.sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    modeldb.BaseData.metadata.create_all(engine)
    return TestingSessionLocal()


def test_add_model(session: orm.Session):
    model = modeldb.add_model(session, name="Inception3", filepath="Inception3.py")
    assert model.name == "Inception3"
    assert model.filepath == "Inception3.py"


def test_add_config(session: orm.Session):
    rc = psst.RangeConfig(
        phi_range=psst.Range(3e-5, 0.02, 224, True),
        nw_range=psst.Range(3e-5, 0.02, 224, True),
        visc_range=psst.Range(3e-5, 0.02, 224, True),
        bg_range=psst.Range(3e-5, 0.02, 224, True),
        bth_range=psst.Range(3e-5, 0.02, 224, True),
        pe_range=psst.Range(3e-5, 0.02, 224, True),
    )
    range_ = modeldb.add_range(session, "AridAgar", rc)
    assert range_.name == "AridAgar"
    assert range_.phi_range == modeldb.RangeCols(3e-5, 0.02, 224, True)
    assert range_.nw_range == modeldb.RangeCols(3e-5, 0.02, 224, True)
    assert range_.visc_range == modeldb.RangeCols(3e-5, 0.02, 224, True)
    assert range_.bg_range == modeldb.RangeCols(3e-5, 0.02, 224, True)
    assert range_.bth_range == modeldb.RangeCols(3e-5, 0.02, 224, True)
    assert range_.pe_range == modeldb.RangeCols(3e-5, 0.02, 224, True)


def test_add_state(session: orm.Session):
    _ = modeldb.add_model(session, name="Inception3", filepath="Inception3.py")
    rc = psst.RangeConfig(
        phi_range=psst.Range(3e-5, 0.02, 224, True),
        nw_range=psst.Range(3e-5, 0.02, 224, True),
        visc_range=psst.Range(3e-5, 0.02, 224, True),
        bg_range=psst.Range(3e-5, 0.02, 224, True),
        bth_range=psst.Range(3e-5, 0.02, 224, True),
        pe_range=psst.Range(3e-5, 0.02, 224, True),
    )
    _ = modeldb.add_range(session, "AridAgar", rc)
    state = modeldb.add_state(
        session, date.today(), "./bg_bth_model.pt", 300, "AridAgar", "Inception3"
    )

    model = state.model
    assert model.name == "Inception3"
    assert model.filepath == "Inception3.py"

    range_ = state.range
    assert range_.name == "AridAgar"
    assert range_.phi_range == modeldb.RangeCols(3e-5, 0.02, 224, True)
    assert range_.nw_range == modeldb.RangeCols(3e-5, 0.02, 224, True)
    assert range_.visc_range == modeldb.RangeCols(3e-5, 0.02, 224, True)
    assert range_.bg_range == modeldb.RangeCols(3e-5, 0.02, 224, True)
    assert range_.bth_range == modeldb.RangeCols(3e-5, 0.02, 224, True)
    assert range_.pe_range == modeldb.RangeCols(3e-5, 0.02, 224, True)


def setup_db_items(session: orm.Session):
    _ = modeldb.add_model(session, name="Inception3", filepath="Inception3.py")
    _ = modeldb.add_model(session, name="Vgg13", filepath="Vgg13.py")
    rc = psst.RangeConfig(
        phi_range=psst.Range(3e-5, 0.02, 224, True),
        nw_range=psst.Range(3e-5, 0.02, 224, True),
        visc_range=psst.Range(3e-5, 0.02, 224, True),
        bg_range=psst.Range(3e-5, 0.02, 224, True),
        bth_range=psst.Range(3e-5, 0.02, 224, True),
        pe_range=psst.Range(3e-5, 0.02, 224, True),
    )
    _ = modeldb.add_range(session, "AridAgar", rc)
    _ = modeldb.add_state(
        session,
        creation_date=date.today(),
        filepath="./bg_bth_model.pt",
        num_epochs=300,
        range_name="AridAgar",
        model_name="Inception3",
    )
    session.flush()


def test_get_models(session: orm.Session):
    setup_db_items(session)
    model_list = modeldb.get_model_names(session)
    assert len(model_list) == 2
    assert model_list[0] == "Inception3"
    assert model_list[1] == "Vgg13"


def test_get_ranges(session: orm.Session):
    setup_db_items(session)
    range_list = modeldb.get_range_names(session)
    assert len(range_list) == 1
    assert range_list[0] == "AridAgar"


@pytest.mark.parametrize("model", ["Inception3", None])
@pytest.mark.parametrize("range", ["AridAgar", None])
def test_get_states(session: orm.Session, model: str | None, range: str | None):
    setup_db_items(session)
    state_list = modeldb.get_state_list(session, model, range)
    assert len(state_list) == 1
    first: modeldb.StateData = state_list[0]
    assert first.model.name == "Inception3"
    assert first.range.name == "AridAgar"
    assert first.num_epochs == 300
    assert first.filepath == "./bg_bth_model.pt"
