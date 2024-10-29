# Standard
import datetime
import enum
import fcntl
import logging
import os
import pathlib
import typing
import uuid

# Third Party
import pydantic
import rich
import yaml

logger = logging.getLogger(__name__)


class TrainingPhases(enum.Enum):
    TRAIN1 = "train1"
    TRAIN2 = "train2"
    EVAL1 = "eval1"
    EVAL2 = "eval2"
    DONE = "done"


AutoDatetimeField = pydantic.Field(
    default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
)


class EvalResult(pydantic.BaseModel):
    """Single checkpoint's final score"""

    ended_at_utc: datetime.datetime = AutoDatetimeField
    checkpoint: pydantic.DirectoryPath
    score: float

    @pydantic.field_serializer("checkpoint", "ended_at_utc")
    def call_str_constructor(self, val: typing.Any) -> str:
        return str(val)


class EvalPhaseModel(pydantic.BaseModel):
    """Stores info about evaluation phase"""

    started_at_utc: datetime.datetime = AutoDatetimeField
    ended_at_utc: datetime.datetime | None = None
    checkpoints: list[pydantic.DirectoryPath]
    finished_checkpoints: list[pydantic.DirectoryPath] = []
    results: list[EvalResult] = []
    best_checkpoint: EvalResult | None = None

    @pydantic.field_serializer(
        "checkpoints",
        "finished_checkpoints",
    )
    def pathlibPath_list_to_str(self, paths: list[pathlib.Path]) -> list[str]:
        return [str(path) for path in paths]

    @pydantic.field_serializer("ended_at_utc")
    def serialize_optional_datetime(self, val: datetime.datetime | None):
        if val:
            return str(val)
        return val


class TrainPhaseModel(pydantic.BaseModel):
    """Stores info about training phase. Doesn't store training args because user can get those elsewhere, e.g. config."""

    started_at_utc: datetime.datetime = AutoDatetimeField
    ended_at_utc: datetime.datetime | None = None
    checkpoints: pydantic.DirectoryPath
    # TODO: might want training args here, but can't currently do it
    # without side-effects because model instance is changed inside of _training_phase

    @pydantic.field_serializer("checkpoints")
    def call_str_constructor(self, val: typing.Any) -> str:
        return str(val)

    @pydantic.field_serializer("ended_at_utc")
    def serialize_optional_datetime(self, val: datetime.datetime | None):
        if val:
            return str(val)
        return val


class JournalModel(pydantic.BaseModel):
    run_id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
    started_at_utc: datetime.datetime = AutoDatetimeField
    ended_at_utc: datetime.datetime | None = None
    current_phase: TrainingPhases = TrainingPhases.TRAIN1
    train_1: TrainPhaseModel | None = None
    eval_1: EvalPhaseModel | None = None
    train_2: TrainPhaseModel | None = None
    eval_2: EvalPhaseModel | None = None
    final_output: EvalResult | None = None

    @pydantic.field_serializer("current_phase")
    def enum_to_value(self, enum_val: TrainingPhases) -> str:
        return str(enum_val.value)

    @pydantic.field_serializer("run_id")
    def object_to_str(self, value: typing.Any) -> str:
        return str(value)

    @pydantic.field_serializer("ended_at_utc")
    def serialize_optional_datetime(self, val: datetime.datetime | None):
        if val:
            return str(val)
        return val


class TrainingJournal:
    def __init__(self, journalfile: pathlib.Path):
        if journalfile.is_dir():
            raise ValueError(
                f"Given journal path is to an existing directory: '{str(journalfile)}'"
            )

        # not a dir. Could not exist, or be a file.

        self.was_loaded: bool = False
        self.journalfile = journalfile
        if journalfile.is_file():
            logger.debug(f"Received journal that is a file: {journalfile}")
            with open(journalfile, "r", encoding="utf-8") as f:
                model_dict = yaml.safe_load(stream=f)

            try:
                self.journal = JournalModel.parse_obj(model_dict)
                self.was_loaded = True
                logger.debug("Parsed journal from journalfile.")
            except pydantic.ValidationError as e:
                logging.error(
                    f"Tried to load Journal but encountered an error. If provided, journal must be parseable. {e}"
                )
                self.journal = JournalModel()
        else:
            logger.debug(
                f"Reference to journal wasn't a file. Initializing empty object. {journalfile}"
            )
            self.journal = JournalModel()

    def create_empty_journal(self) -> None:
        # makes empty file available at `journalfile` path
        self.journalfile.parent.mkdir(exist_ok=True, parents=True)
        self.journalfile.touch(exist_ok=True)

    def commit(self, create_new: bool = False) -> None:
        if create_new:
            self.create_empty_journal()

        # try dumping before we open file for writing so yaml parsing can fail before we
        # destroy file content.
        _ = yaml.safe_dump(self.journal.model_dump())

        # implementation by @leseb
        with open(self.journalfile, "w", encoding="utf-8") as f:
            # Acquire an exclusive lock to prevent other processes from writing
            fcntl.flock(f, fcntl.LOCK_EX)

            # Write journal's content
            yaml.safe_dump(data=self.journal.model_dump(), stream=f)

            # Flush the buffer to ensure data is moved to OS buffer
            f.flush()

            # Call fsync to ensure the data is physically written to disk
            os.fsync(f.fileno())
        logger.debug("Model written to disk")

    @property
    def current_phase(self) -> TrainingPhases:
        return self.journal.current_phase

    @current_phase.setter
    def current_phase(self, new_phase: TrainingPhases) -> None:
        self.journal.current_phase = new_phase

    @staticmethod
    def now_utc() -> datetime.datetime:
        """Static helper method for current time in UTC"""
        return datetime.datetime.now(datetime.timezone.utc)

    @staticmethod
    def best_checkpoint(phase_model: EvalPhaseModel) -> EvalResult:
        """Returns the EvalResult object with the highest score."""
        return sorted(phase_model.results, reverse=True, key=lambda c: c.score)[0]

    def print_model_rich(self) -> str:
        """
        Prints model in color, with indentation. Meant to be executed at runtime with
        a logging statement if desired. 'rich' doesn't have a 'format' option so this
        is a not-terrible way to get good output.
        """
        rich.print(self.journal, flush=True)
        return ""
