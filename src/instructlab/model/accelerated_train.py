# Standard
import enum
import functools
import logging
import os
import pathlib
import pprint
import typing

# Third Party
# pylint: disable=ungrouped-imports
from instructlab.training import DistributedBackend, TorchrunArgs, TrainingArgs
import click

# First Party
from instructlab import utils
from instructlab.configuration import _serve
from instructlab.model.backends import backends

# Local
from .phased_training import (
    EvalPhaseModel,
    EvalResult,
    TrainingJournal,
    TrainingPhases,
    TrainPhaseModel,
)

logger = logging.getLogger(__name__)


class SupportedTrainingStrategies(enum.Enum):
    """Available advanced training stratefies"""

    LAB_MULTIPHASE: str = "lab-multiphase"


def accelerated_train(
    train_args: TrainingArgs,
    torch_args: TorchrunArgs,
    strategy: str | None,
    distributed_backend: str,
    phased_phase1_data: pathlib.Path | None,
    phased_phase2_data: pathlib.Path | None,
    phased_base_dir: pathlib.Path,
    phased_phase1_num_epochs: int | None,
    phased_phase1_samples_per_save: int | None,
    phased_phase1_effective_batch_size: int | None,
    phased_phase2_num_epochs: int | None,
    phased_phase2_samples_per_save: int | None,
    phased_phase2_effective_batch_size: int | None,
    enable_serving_output: bool,
    phased_mt_bench_judge: pathlib.Path | None,
    skip_user_confirm: bool,
    force_clear_phased_cache: bool,
    eval_serve: _serve,
    eval_gpus: int,
    training_journal: pathlib.Path | None,
):
    # run_training is a dynamic attribute, pylint is not clever enough
    # to detect it.
    # Third Party
    if strategy == SupportedTrainingStrategies.LAB_MULTIPHASE.value:
        if (
            distributed_backend
            and distributed_backend not in DistributedBackend._value2member_map_
        ):
            # pylint: disable=broad-exception-raised
            raise Exception(
                f"Invalid training backend option '{distributed_backend}' specified. Please specify either `fsdp` or `deepspeed`"
            )

        # pull the trainrandom.randinting and torch args from the flags
        # the flags are populated from the config as a base.
        logger.debug(
            "Rendered training arguments:\n%s", pprint.pformat(train_args.model_dump())
        )

        if not (phased_phase1_data and phased_phase2_data):
            # pylint: disable=broad-exception-raised
            raise Exception(
                "End-to-end training minimally requires: `--phased-phase1-data`, and `--phased-phase2-data`. One or more wasn't correctly specified."
            )

        # if they both exist, must be Path objects
        if not (phased_phase1_data.exists() and phased_phase2_data.exists()):
            raise FileNotFoundError(
                "Data for both phase1 and phase2 must be specified for phased training."
            )

        mt_bench_judge: pathlib.Path
        if phased_mt_bench_judge is None:
            raise FileNotFoundError(
                "No MT-Bench model was provided with '--phased-mt-bench-judge'"
            )
        if not phased_mt_bench_judge.resolve().exists():
            raise FileNotFoundError(
                f"MT-Bench model directory could not be found: {phased_mt_bench_judge}\nMust be an absolute path to a model directory."
            )

        # makes MyPy happy because 'mt_bench_judge' isn't Path | None
        mt_bench_judge = phased_mt_bench_judge

        if training_journal is None:
            training_journal = phased_base_dir / "journalfile.yaml"
        else:
            # might come in as a str so needs to become a path.
            training_journal = pathlib.Path(training_journal)

        # try to load journal. May be empty.
        journal = TrainingJournal(journalfile=training_journal)
        click.secho("\n\n~~~~~~~~~~~~~STARTING MULTI-PHASE TRAINING~~~~~~~~~~~~~")

        # experimental preference.
        if phased_phase1_num_epochs != 7:
            click.secho(
                f"Running phased training with '{phased_phase1_num_epochs}' epochs.\nNote: 7 epochs is the recommended amount for optimal performance.",
                fg="yellow",
            )

        if journal.was_loaded:
            click.secho(
                f"There was an existing training journal at: '{str(training_journal)}'"
            )
            journal.print_model_rich()
            click.secho(
                f"WARNING: Existing training journal state must correspond to state in '{str(phased_base_dir)}'",
                bg="yellow",
                fg="black",
            )
            click.secho("Alternative behavior is undefined.", bg="yellow", fg="black")
        else:
            click.secho(
                f"No training journal found. Will initialize at: '{str(journal.journalfile)}'",
                bg="yellow",
                fg="black",
            )
            journal.commit(create_new=True)

        user_clear_cache: bool = False
        if not skip_user_confirm:
            click.secho(
                "Metadata (checkpoints, the training journal) may have been saved from a previous training run."
            )
            click.secho(
                "By default, training will resume from this metadata if it exists."
            )
            click.secho(
                "Alternatively, the metadata can be cleared, and training can start from scratch."
            )
            click.secho("\nWould you like to START TRAINING FROM THE BEGINNING?")

            user_clear_cache = click.confirm(
                "'y' clears metadata to start new training, 'N' tries to resume: "
            )

        user_clear_cache = user_clear_cache or force_clear_phased_cache

        if user_clear_cache:
            training_journal.unlink(
                missing_ok=True
            )  # delete whatever the old journal was
            journal = TrainingJournal(
                journalfile=training_journal
            )  # create an empty journal
            journal.commit(create_new=True)  # save it.

        _prepare_phased_base_dir(phased_base_dir, delete_subdirs=user_clear_cache)
        _run_phased_training(
            train_args=train_args,
            torch_args=torch_args,
            base_dir=phased_base_dir,
            phase1_data=phased_phase1_data,
            phase1_num_epochs=phased_phase1_num_epochs,
            phase1_samples_per_save=phased_phase1_samples_per_save,
            phase1_checkpoints_dir=phased_base_dir / "phase1" / "checkpoints",
            phased_phase1_effective_batch_size=phased_phase1_effective_batch_size,
            phase2_data=phased_phase2_data,
            phase2_num_epochs=phased_phase2_num_epochs,
            phase2_samples_per_save=phased_phase2_samples_per_save,
            phase2_checkpoints_dir=phased_base_dir / "phase2" / "checkpoints",
            phased_phase2_effective_batch_size=phased_phase2_effective_batch_size,
            phase2_eval_cache=phased_base_dir / "phase2" / "eval_cache",
            mtbench_judge=mt_bench_judge,
            enable_serving_output=enable_serving_output,
            journal=journal,
            eval_serve=eval_serve,
            eval_gpus=eval_gpus,
        )
    else:
        # Third Party
        from instructlab.training import (
            run_training,  # pylint: disable=no-name-in-module
        )

        run_training(train_args=train_args, torch_args=torch_args)


def _run_phased_training(
    train_args: TrainingArgs,
    torch_args: TorchrunArgs,
    base_dir: pathlib.Path,
    phase1_data: pathlib.Path,
    phase1_num_epochs: int | None,
    phase1_samples_per_save: int | None,
    phase1_checkpoints_dir: pathlib.Path,
    phased_phase1_effective_batch_size: int | None,
    phase2_data: pathlib.Path,
    phase2_num_epochs: int | None,
    phase2_samples_per_save: int | None,
    phase2_checkpoints_dir: pathlib.Path,
    phased_phase2_effective_batch_size: int | None,
    phase2_eval_cache: pathlib.Path,
    mtbench_judge: pathlib.Path,
    enable_serving_output: bool,
    journal: TrainingJournal,
    eval_serve: _serve,
    eval_gpus: int,
) -> None:
    if journal.current_phase == TrainingPhases.DONE:
        click.secho(
            "The selected Training Journal suggests that training has finished, with 'current_phase=done' in the journalfile.",
            fg="cyan",
        )
        return

    # make mypy happy
    phase_model: TrainPhaseModel | EvalPhaseModel | None = None

    if journal.current_phase == TrainingPhases.TRAIN1:
        click.secho("Training Phase 1/2...", fg="cyan")

        phase_model = journal.journal.train_1
        if phase_model is None:
            phase_model = TrainPhaseModel(checkpoints=phase1_checkpoints_dir)
            journal.journal.train_1 = phase_model
        journal.commit()

        _training_phase(
            train_args=train_args.model_copy(deep=True),
            torch_args=torch_args,
            data_path=phase1_data,
            checkpoint_dir=phase1_checkpoints_dir,
            num_epochs=phase1_num_epochs,
            samples_per_save=phase1_samples_per_save,
            effective_batch_size=phased_phase1_effective_batch_size,
            # model override not necessary because we expect model to come from ctx.params.model_path.
        )

        phase_model.ended_at_utc = TrainingJournal.now_utc()

        journal.current_phase = TrainingPhases.TRAIN2
        journal.commit()

        logger.debug("Finished training #1\n%s", journal.print_model_rich())
    else:
        click.secho("SKIPPING: Training Phase 1/2; already in Journal", fg="cyan")

    # if journal.current_phase == TrainingPhases.EVAL1:
    #     click.secho("MMLU evaluation for Phase 1...", fg="cyan")

    # NOTE: requires hf_format sub-directory. Training sets this up.
    # phase1_checkpoints_dir = phase1_checkpoints_dir / "hf_format"

    # phase_model = journal.journal.eval_1
    # if phase_model is None:
    #     # if it's not None, it already exists and may have 'results', so we shouldn't overwrite it.
    #     phase_model = EvalPhaseModel(
    #         checkpoints=list(phase1_checkpoints_dir.iterdir())
    #     )
    #     journal.journal.eval_1 = phase_model
    # journal.commit()

    # best_checkpoint = _evaluate_dir_of_checkpoints(
    #     eval_func=_mmlu, phase_model=phase_model, journal=journal
    # )

    # phase_model.best_checkpoint = best_checkpoint
    # phase_model.ended_at_utc = TrainingJournal.now_utc()

    #     journal.current_phase = TrainingPhases.TRAIN2
    #     journal.commit()
    #     logger.debug("Finished eval #1\n%s", journal.print_model_rich())
    # else:
    #     click.secho(
    #         "SKIPPING: MMLU evaluation for Phase 1; already in Journal", fg="cyan"
    #     )

    if journal.current_phase == TrainingPhases.TRAIN2:
        click.secho("Training Phase 2/2...", fg="cyan")

        phase_model = journal.journal.train_2
        if phase_model is None:
            phase_model = TrainPhaseModel(checkpoints=phase2_checkpoints_dir)
            journal.journal.train_2 = phase_model
        journal.commit()

        # if journal.journal.eval_1 is None:
        #     raise RuntimeError(
        #         "Training journal field 'eval_1' cannot be None for phase 'train_2'"
        #     )

        # NOTE:
        # this is a recent change, implemented to ignore MMLU. We just look at the checkpoints
        # from the phase 1 training and take the most recent one.
        phase1_checkpoints_dir_hf = phase1_checkpoints_dir / "hf_format"
        if not phase1_checkpoints_dir_hf.exists():
            raise FileNotFoundError(
                f"{phase1_checkpoints_dir_hf} doesn't exist. This likely means that no checkpoints were saved from phase 1."
            )

        phase1_checkpoints = sorted(
            list(phase1_checkpoints_dir_hf.iterdir()),
            reverse=True,
            # XXX(osilkin): This line takes the checkpoint name "samples_NNN" and tells sorted
            #               to use the last NNN string as an integer
            key=lambda x: int(str(x).rsplit("_", maxsplit=1)[-1]),
        )

        if len(phase1_checkpoints) == 0:
            raise FileNotFoundError(
                f"{phase1_checkpoints_dir_hf} Has no checkpoints. This likely means that no checkpoints were saved from phase 1."
            )

        next_checkpoint = phase1_checkpoints[0]

        _training_phase(
            train_args=train_args.model_copy(deep=True),
            torch_args=torch_args,
            data_path=phase2_data,
            checkpoint_dir=phase2_checkpoints_dir,
            model_override=next_checkpoint,  # type: ignore
            num_epochs=phase2_num_epochs,
            samples_per_save=phase2_samples_per_save,
            effective_batch_size=phased_phase2_effective_batch_size,
        )

        phase_model.ended_at_utc = TrainingJournal.now_utc()

        journal.current_phase = TrainingPhases.EVAL2
        journal.commit()
        logger.debug("Finished training #2\n%s", journal.print_model_rich())
    else:
        click.secho("SKIPPING: Training Phase 2/2; already in Journal", fg="cyan")

    if journal.current_phase == TrainingPhases.EVAL2:
        click.secho("MT-Bench evaluation for Phase 2...", fg="cyan")

        phase2_checkpoints_dir = phase2_checkpoints_dir / "hf_format"
        phase2_eval_cache = base_dir / "phase2" / "eval_cache"
        phase_model = journal.journal.eval_2
        if phase_model is None:
            # if it's not None, it already exists and may have 'results', so we shouldn't overwrite it.
            phase_model = EvalPhaseModel(
                checkpoints=list(phase2_checkpoints_dir.iterdir())
            )
            journal.journal.eval_2 = phase_model
        journal.commit()

        best_checkpoint = _evaluate_dir_of_checkpoints(
            phase_model=phase_model,
            journal=journal,
            eval_func=functools.partial(
                _mtbench,
                eval_serve=eval_serve,
                eval_gpus=eval_gpus,
                eval_cache=phase2_eval_cache,
                mtbench_judge=mtbench_judge,
                enable_serving_output=enable_serving_output,
            ),
        )

        phase_model.best_checkpoint = best_checkpoint
        phase_model.ended_at_utc = TrainingJournal.now_utc()

        journal.current_phase = TrainingPhases.DONE
        journal.journal.final_output = best_checkpoint
        journal.journal.ended_at_utc = TrainingJournal.now_utc()
        journal.commit()
        logger.debug("Finished eval #2\n%s", journal.print_model_rich())

    else:
        click.secho(
            "SKIPPING: MT-Bench evaluation for Phase 2; already in Journal", fg="cyan"
        )

    output_checkpoint: EvalResult | None = journal.journal.final_output
    if not output_checkpoint:
        raise RuntimeError(
            "At the end of training, but no 'final_output' checkpoint in TrainingJournal"
        )

    click.secho(
        f"Training finished! Best final checkpoint: {output_checkpoint.checkpoint} with score: {output_checkpoint.score}\nJournal: {str(journal.journalfile)}",
        fg="green",
    )


def _training_phase(
    train_args: TrainingArgs,
    torch_args: TorchrunArgs,
    data_path: pathlib.Path,
    model_override: pathlib.Path | None = None,
    num_epochs: int | None = None,
    samples_per_save: int | None = None,
    checkpoint_dir: pathlib.Path | None = None,
    effective_batch_size: int | None = None,
) -> None:
    """A single step of phased training that supports key param overriding."""

    # Third Party
    from instructlab.training import run_training  # pylint: disable=no-name-in-module

    logger.debug(
        f"Phased Training -- training phase -- Overriding data_path: {train_args.data_path} with {data_path}"
    )

    # NOTE: have to cast pathlib.Path to str because Pydantic models require this. Here and below.
    train_args.data_path = str(data_path)

    if checkpoint_dir:
        train_args.ckpt_output_dir = str(checkpoint_dir)

    if model_override:
        logger.debug(
            f"Phased Training -- training phase -- Overriding model_path: {train_args.model_path} with {model_override}"
        )
        train_args.model_path = str(model_override)

    if num_epochs:
        logger.debug(
            f"Phased Training -- training phase -- Overriding num epochs: {train_args.num_epochs} with {num_epochs}"
        )
        train_args.num_epochs = num_epochs

    if samples_per_save is not None:
        logger.debug(
            f"Phased Training -- training phase -- Overriding samples per save: {train_args.save_samples} with {samples_per_save}"
        )
        train_args.save_samples = samples_per_save

    if effective_batch_size:
        logger.debug(
            f"Phased Training -- training phase -- Overriding effective batch size: {train_args.effective_batch_size} with {effective_batch_size}"
        )
        train_args.effective_batch_size = effective_batch_size

    click.secho(
        f"TrainingArgs for current phase: {pprint.pformat(train_args)}", fg="cyan"
    )

    run_training(train_args=train_args, torch_args=torch_args)


def _prepare_phased_base_dir(
    phased_base_dir: pathlib.Path, delete_subdirs: bool = True
) -> None:
    """Adds phase1 and phase2 directories in phased_base_dir.
    In each, adds `checkpoints` and `eval_cache` subdirectories.

    Also adds training `journalfile.yaml`

    Args:
        phased_base_dir: directory wrapping phase1 and phase2 cached data.
    """

    logger.debug(f"Phased Training -- Preparing phased base dir: {phased_base_dir}")

    phase1_dir_path = phased_base_dir / "phase1"
    phase2_dir_path = phased_base_dir / "phase2"

    for p in [phase1_dir_path, phase2_dir_path]:
        if delete_subdirs:
            utils.clear_directory(p)
        _setup_phase_dirs(p)


def _setup_phase_dirs(path: pathlib.Path) -> None:
    """Creates {path}/checkpoints and {path}/eval_cache directories."""

    # TODO: these sub-directory names are hard-coded here but they
    # could be parameterized in config.
    logger.debug(f"Phased Training -- Created phase directories for {path}")
    ckpt_path = path / "checkpoints"
    eval_cache_path = path / "eval_cache"

    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(eval_cache_path, exist_ok=True)


def _mmlu(model: pathlib.Path) -> float:
    # Third Party
    from instructlab.eval.mmlu import MMLU_TASKS, MMLUEvaluator
    import torch

    tasks = MMLU_TASKS
    if os.environ.get("INSTRUCTLAB_EVAL_MMLU_MIN_TASKS") is not None:
        tasks = tasks[:4]
    evaluator = MMLUEvaluator(model, tasks=tasks)

    # type the variable because MyPy doesn't seem to honor the types of the spread tuple
    ckpt_score: float
    ckpt_score, _ = evaluator.run()

    logging.debug("Phased Training -- MMLU eval phase -- Clearing PyTorch cache")
    torch.cuda.empty_cache()

    return ckpt_score


def _mtbench(
    model: pathlib.Path,
    eval_serve: _serve,
    eval_gpus: int,
    eval_cache: pathlib.Path,
    mtbench_judge: pathlib.Path,
    enable_serving_output: bool,
) -> float:
    # TODO: optimization: run all generations in serial and then do all judgments at once to save time loading/unloading prometheus.
    # Third Party
    from instructlab.eval.mt_bench import MTBenchEvaluator
    import torch

    # First Party
    from instructlab.model.evaluate import get_gpus, get_model_name, launch_server

    explicit_gpus = None
    gpus, effective_gpus = get_gpus(eval_serve, eval_gpus)
    if gpus and gpus > 0:
        # gpus are specified in config for evaluate
        logger.debug("Using gpus from config")
        explicit_gpus = gpus
    elif effective_gpus > 0:
        # tensor-parallel size specified in serving config
        logger.debug("Using gpus from --tensor-parallel-size in config")
    else:
        # TODO: Should be parameterized by the user specific for training
        explicit_gpus = torch.cuda.device_count()
        effective_gpus = explicit_gpus

    model_name = get_model_name(str(model))
    judge_model_name = get_model_name(str(mtbench_judge))

    evaluator = MTBenchEvaluator(
        model_name=model_name,
        judge_model_name=judge_model_name,
        output_dir=str(eval_cache),
        merge_system_user_message=True,  # TODO: expose this to the user
    )

    server = None
    model_serve_url = None
    try:
        logger.debug("Starting model server for mt-bench answer generation")
        server, model_serve_url, effective_gpus = launch_server(
            eval_serve=eval_serve,
            tls_client_cert=None,
            tls_client_key=None,
            tls_client_passwd=None,
            tls_insecure=False,
            model=str(model),
            model_name=model_name,
            gpus=explicit_gpus,
            max_workers="auto",
            enable_serving_output=enable_serving_output,
            backend=backends.VLLM,
        )
        logger.debug("Generating mt-bench answers")
        evaluator.gen_answers(
            model_serve_url, max_workers="auto", serving_gpus=effective_gpus
        )
    finally:
        if server is not None:
            server.shutdown()

    try:
        logger.debug("Starting model server for mt-bench answer judgment")
        server, model_serve_url, effective_gpus = launch_server(
            eval_serve=eval_serve,
            tls_client_cert=None,
            tls_client_key=None,
            tls_client_passwd=None,
            tls_insecure=False,
            model=str(mtbench_judge),
            model_name=judge_model_name,
            gpus=explicit_gpus,
            max_workers="auto",
            backend=backends.VLLM,
            enable_serving_output=enable_serving_output,
        )
        logger.debug("Judging mt-bench answers")
        mt_bench_results: tuple = evaluator.judge_answers(
            model_serve_url, max_workers="auto", serving_gpus=effective_gpus
        )
        ckpt_score: float = mt_bench_results[0]
    finally:
        if server is not None:
            server.shutdown()

    return ckpt_score


def _evaluate_dir_of_checkpoints(
    eval_func: typing.Callable[..., float],
    phase_model: EvalPhaseModel,
    journal: TrainingJournal,
) -> EvalResult:
    """Run eval_func on all model checkpoints in a directory."""
    # TODO: parallelize MMLU over available GPUs

    # doing this to avoid removing checkpoints from same list that we're iterating over.
    checkpoints_todo = list(
        set(phase_model.checkpoints) - set(phase_model.finished_checkpoints)
    )

    if len(checkpoints_todo) == 0:
        raise RuntimeError(
            "No checkpoints were evaluated, 'checkpoints_todo' was empty in journal."
        )

    for checkpoint in checkpoints_todo:
        logger.debug(str(checkpoint))
        checkpoint_score = eval_func(model=checkpoint)

        phase_model.results.append(
            EvalResult(
                score=checkpoint_score,
                checkpoint=checkpoint,
                ended_at_utc=TrainingJournal.now_utc(),
            )
        )

        phase_model.finished_checkpoints.append(checkpoint)
        journal.commit()

        click.secho(
            f"CHECKPOINT EVALUATION: {str(checkpoint)} SCORED {checkpoint_score}",
            fg="red",
            bg="cyan",
        )

    return TrainingJournal.best_checkpoint(phase_model=phase_model)
