from dataclasses import dataclass
from datetime import date

import sqlalchemy as sqla
import sqlalchemy.orm as orm

import psst

# from psst import Range

# models: id | name | filepath
# configs: id | name | phi_range | nw_range | visc_range | bg_range | bth_range | pe_range
# states: id | date | filepath | config_id | model_id | num_epochs | optuna_params(?)


def start_session():
    datebase_url = sqla.URL.create(
        "mysql+mysqldb",
        username="jacobs",
        password="nth9ga4u",
        host="localhost",
        database="modeldb",
    )
    engine = sqla.create_engine(
        datebase_url,
        connect_args={"check_same_thread": False},
        poolclass=sqla.StaticPool,
    )
    SessionLocal = orm.sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    return session


@dataclass
class RangeCols:
    min_value: float
    max_value: float
    shape: int = 0
    log_scale: bool = False


class BaseData(orm.DeclarativeBase):
    pass


class ModelData(BaseData):
    __tablename__ = "models"

    id: orm.Mapped[int] = orm.mapped_column(primary_key=True, index=True)
    name: orm.Mapped[str] = orm.mapped_column(unique=True)
    filepath: orm.Mapped[str]
    # states: orm.Mapped[list["StateData"]] = orm.relationship(back_populates="model")


class RangeData(BaseData):
    __tablename__ = "ranges"

    id: orm.Mapped[int] = orm.mapped_column(primary_key=True, index=True)
    name: orm.Mapped[str] = orm.mapped_column(unique=True)
    # states: orm.Mapped[list["StateData"]] = orm.relationship(back_populates="range")

    # Mapped[Range] is incorrect
    phi_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("phi_min"),
        orm.mapped_column("phi_max"),
        orm.mapped_column("phi_num"),
        orm.mapped_column("phi_log"),
    )
    nw_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("nw_min"),
        orm.mapped_column("nw_max"),
        orm.mapped_column("nw_num"),
        orm.mapped_column("nw_log"),
    )
    visc_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("visc_min"),
        orm.mapped_column("visc_max"),
        orm.mapped_column("visc_num"),
        orm.mapped_column("visc_log"),
    )
    bg_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("bg_min"),
        orm.mapped_column("bg_max"),
        orm.mapped_column("bg_num"),
        orm.mapped_column("bg_log"),
    )
    bth_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("bth_min"),
        orm.mapped_column("bth_max"),
        orm.mapped_column("bth_num"),
        orm.mapped_column("bth_log"),
    )
    pe_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("pe_min"),
        orm.mapped_column("pe_max"),
        orm.mapped_column("pe_num"),
        orm.mapped_column("pe_log"),
    )


# class StateData(BaseData):
#     __tablename__ = "states"

#     id: orm.Mapped[int] = orm.mapped_column(primary_key=True, index=True)
#     filepath: orm.Mapped[str]
#     num_epochs: orm.Mapped[int]
#     creation_date: orm.Mapped[date]
#     range_id: orm.Mapped[int] = orm.mapped_column(sqla.ForeignKey("ranges.id"))
#     range: orm.Mapped["RangeData"] = orm.relationship(back_populates="states")
#     model_id: orm.Mapped[int] = orm.mapped_column(sqla.ForeignKey("models.id"))
#     model: orm.Mapped["ModelData"] = orm.relationship(back_populates="states")


def add_model(db: orm.Session, name: str, filepath: str):
    model = ModelData(name=name, filepath=filepath)
    db.add(model)
    db.commit()
    return model


def add_range(
    db: orm.Session,
    name: str,
    range_config: psst.RangeConfig,
):
    range_ = RangeData(
        name=name,
        phi_range=range_config.phi_range,
        nw_range=range_config.nw_range,
        visc_range=range_config.visc_range,
        bg_range=range_config.bg_range,
        bth_range=range_config.bth_range,
        pe_range=range_config.pe_range,
    )
    db.add(range_)
    db.commit()
    return range_


# def add_state(
#     db: orm.Session,
#     creation_date: date,
#     filepath: str,
#     num_epochs: int,
#     range_name: str,
#     model_name: str,
# ):
#     range_id = db.scalar(sqla.select(RangeData.id).filter_by(name=range_name))
#     model_id = db.scalar(sqla.select(ModelData.id).filter_by(name=model_name))
#     state = StateData(
#         creation_date=creation_date,
#         filepath=filepath,
#         range_id=range_id,
#         model_id=model_id,
#         num_epochs=num_epochs,
#     )
#     db.add(state)
#     db.commit()
#     return state


def get_model_names(db: orm.Session):
    sel = sqla.select(ModelData.name)
    return db.scalars(sel).all()


def get_model(db: orm.Session, idx: int | None = None, name: str | None = None):
    if idx:
        sel = sqla.select(ModelData).where(ModelData.id == idx)
    elif name:
        sel = sqla.select(ModelData).where(ModelData.name == name)
    else:
        return None
    return db.scalars(sel).one_or_none()


def get_range_names(db: orm.Session):
    sel = sqla.select(RangeData.name)
    return db.scalars(sel).all()


def get_range(db: orm.Session, idx: int | None = None, name: str | None = None):
    if idx:
        sel = sqla.select(RangeData).where(RangeData.id == idx)
    elif name:
        sel = sqla.select(RangeData).where(RangeData.name == name)
    else:
        return None
    return db.scalars(sel).one_or_none()


# def get_state_list(
#     db: orm.Session, model_name: str | None = None, range_name: str | None = None
# ):
#     statement = sqla.select(StateData)
#     if model_name:
#         model = get_model(db, name=model_name)
#         if model is None:
#             return list()
#         statement = statement.where(StateData.model_id == model.id)
#     if range_name:
#         range_ = get_range(db, name=range_name)
#         if range_ is None:
#             return list()
#         statement = statement.where(StateData.range_id == range_.id)
#     return db.scalars(statement).all()


# def get_state(db: orm.Session, idx: int):
#     statement = sqla.select(StateData).where(StateData.id == idx)
#     return db.scalar(statement)
