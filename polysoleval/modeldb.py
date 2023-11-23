from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
import sqlalchemy as sqla
import sqlalchemy.orm as orm

import psst

# models: id | name | description | model (pickle object)
# configs: id | name | phi_range | nw_range | visc_range | bg_range | bth_range | pe_range


class BaseData(orm.DeclarativeBase):
    pass


def start_session(
    password: str, model_dir: Path, range_dir: Path, update: bool = False
):
    database_url = sqla.URL.create(
        "mariadb",
        username="root",
        password=password,
        host="modeldb",
        database="modeldb",
        port=3306,
    )
    engine = sqla.create_engine(database_url, poolclass=sqla.StaticPool)
    SessionLocal = orm.sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    if not update:
        BaseData.metadata.drop_all(engine)
    BaseData.metadata.create_all(engine)

    yaml = YAML(typ="safe", pure=True)
    model_dict: dict[str, Any] = yaml.load(model_dir / "models.yaml")

    range_files = [f for f in range_dir.iterdir() if f.is_file()]
    range_configs = list()
    for f in range_files:
        try:
            config = psst.RangeConfig.from_file(f)
        except ValueError:
            config = None
        range_configs.append(config)

    if update:
        for model_name, values in model_dict.items():
            update_model(
                session, model_name, values["description"].strip(), values["link"]
            )
        for filepath, config in zip(range_files, range_configs):
            if config:
                update_range(session, filepath.stem, config)
    else:
        for model_name, values in model_dict.items():
            add_model(
                session, model_name, values["description"].strip(), values["link"]
            )
        for filepath, config in zip(range_files, range_configs):
            if config:
                add_range(session, filepath.stem, config)

    return session


class ModelData(BaseData):
    __tablename__ = "models"

    id: orm.Mapped[int] = orm.mapped_column(primary_key=True, index=True)
    name: orm.Mapped[str] = orm.mapped_column(unique=True, type_=sqla.String(32))
    description: orm.Mapped[str] = orm.mapped_column(type_=sqla.String(1024))
    link: orm.Mapped[str] = orm.mapped_column(type_=sqla.String(1024))


def add_model(
    db: orm.Session,
    name: str,
    description: str,
    link: str,
):
    model = ModelData(name=name, description=description, link=link)
    db.add(model)
    db.commit()
    return model


def update_model(
    db: orm.Session,
    name: str,
    description: str | None,
    link: str | None,
):
    sel = sqla.select(ModelData.id, ModelData.name).where(ModelData.name == name)
    row = db.execute(sel).one_or_none()
    if row is None:
        if description is None or link is None:
            return
        add_model(db, name, description, link)
        return

    row_id: int = row.id
    stmt = sqla.update(ModelData).where(ModelData.id == row_id)
    if description:
        stmt = stmt.values(description=description)
    if link:
        stmt = stmt.values(link=link)

    db.execute(stmt)
    db.commit()


def get_models(db: orm.Session):
    sel = sqla.select(ModelData)
    return db.scalars(sel).all()


def get_model(db: orm.Session, idx: int | None = None, name: str | None = None):
    if idx:
        sel = sqla.select(ModelData).where(ModelData.id == idx)
    elif name:
        sel = sqla.select(ModelData).where(ModelData.name == name)
    else:
        return None
    return db.scalars(sel).one_or_none()


@dataclass
class RangeCols:
    min_value: float
    max_value: float
    log_scale: bool = False

    @classmethod
    def from_range(cls, range_: psst.Range):
        return cls(range_.min_value, range_.max_value, range_.log_scale)


class RangeData(BaseData):
    __tablename__ = "ranges"

    id: orm.Mapped[int] = orm.mapped_column(primary_key=True, index=True)
    name: orm.Mapped[str] = orm.mapped_column(unique=True, type_=sqla.String(32))
    num_phi: orm.Mapped[int]
    num_nw: orm.Mapped[int]

    phi_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("phi_min"),
        orm.mapped_column("phi_max"),
        orm.mapped_column("phi_log"),
    )
    nw_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("nw_min"),
        orm.mapped_column("nw_max"),
        orm.mapped_column("nw_log"),
    )
    visc_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("visc_min"),
        orm.mapped_column("visc_max"),
        orm.mapped_column("visc_log"),
    )
    bg_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("bg_min"),
        orm.mapped_column("bg_max"),
        orm.mapped_column("bg_log"),
    )
    bth_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("bth_min"),
        orm.mapped_column("bth_max"),
        orm.mapped_column("bth_log"),
    )
    pe_range: orm.Mapped[RangeCols] = orm.composite(
        orm.mapped_column("pe_min"),
        orm.mapped_column("pe_max"),
        orm.mapped_column("pe_log"),
    )


def add_range(
    db: orm.Session,
    name: str,
    range_config: psst.RangeConfig,
):
    range_ = RangeData(
        name=name,
        num_phi=range_config.phi_range.shape,
        num_nw=range_config.nw_range.shape,
        phi_range=RangeCols.from_range(range_config.phi_range),
        nw_range=RangeCols.from_range(range_config.nw_range),
        visc_range=RangeCols.from_range(range_config.visc_range),
        bg_range=RangeCols.from_range(range_config.bg_range),
        bth_range=RangeCols.from_range(range_config.bth_range),
        pe_range=RangeCols.from_range(range_config.pe_range),
    )
    db.add(range_)
    db.commit()
    return range_


def update_range(
    db: orm.Session,
    name: str,
    range_config: psst.RangeConfig,
):
    sel = sqla.select(RangeData.id, RangeData.name).where(RangeData.name == name)
    row = db.execute(sel).one_or_none()

    if row is None:
        add_range(db, name, range_config)
        return

    update = (
        sqla.update(RangeData)
        .where(RangeData.id == row.id)
        .values(
            num_phi=range_config.phi_range.shape,
            num_nw=range_config.nw_range.shape,
            phi_range=RangeCols.from_range(range_config.phi_range),
            nw_range=RangeCols.from_range(range_config.nw_range),
            visc_range=RangeCols.from_range(range_config.visc_range),
            bg_range=RangeCols.from_range(range_config.bg_range),
            bth_range=RangeCols.from_range(range_config.bth_range),
            pe_range=RangeCols.from_range(range_config.pe_range),
        )
    )
    db.execute(update)
    db.commit()


def get_ranges(db: orm.Session):
    sel = sqla.select(RangeData)
    return db.scalars(sel).all()


def get_range(db: orm.Session, idx: int | None = None, name: str | None = None):
    if idx:
        sel = sqla.select(RangeData).where(RangeData.id == idx)
    elif name:
        sel = sqla.select(RangeData).where(RangeData.name == name)
    else:
        return None
    return db.scalars(sel).one_or_none()
