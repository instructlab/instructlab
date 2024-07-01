from typing import Any, Tuple, Callable
from pathlib import Path
import os
from functools import partial

import torch

from instructlab.training import (
    DeepSpeedOptions,
    LoraOptions,
    TorchrunArgs,
    TrainingArgs,
    run_training,
)
from instructlab.eval.mmlu import (
    MMLUEvaluator,
    MMLU_TASKS,
)  # waiting on this to be available in the upstream
from instructlab.eval.mt_bench import MTBenchEvaluator


def _e2e_training_step(
    train_args: TrainingArgs,
    torch_args: TorchrunArgs,
    model_override: str | None = None,
    ckpt_dir_override: str | None = None,
) -> None:
    """A single step of end-to-end training.

    Args:
        train_args: training arguments
        torch_args: torchrun initialization arguments
        checkpoint_dir_override: re-routes checkpoint path to known, static training directory. Defaults to None.
    """

    if ckpt_dir_override:
        train_args.ckpt_output_dir = ckpt_dir_override

    if model_override:
        train_args.model_path = model_override

    run_training(train_args=train_args, torch_args=torch_args)


def _mmlu(ckpt_path: str) -> float:
    evaluator = MMLUEvaluator(
        ckpt_path,
        tasks=MMLU_TASKS,
        few_shots=2,  # TODO need to know this param
        batch_size=torch.cuda.device_count(),  # TODO:  should we parameterize this now or wait until later?
    )

    ckpt_score, _ = evaluator.run()
    return ckpt_score


def _mtbench(
    ckpt_path: str, eval_cache_path: str, judge_name: str, judge_url: str
) -> float:
    evaluator = MTBenchEvaluator(
        model_name=ckpt_path,
        judge_model_name=judge_name,
        output_dir=eval_cache_path,
        max_workers=torch.cuda.device_count(),
    )

    evaluator.gen_answers(ckpt_path)  # writes results to eval_cache_path

    ckpt_score, _, _ = evaluator.judge_answers(judge_url)
    return ckpt_score


def _evaluate_dir_of_checkpoints(
    checkpoints_dir_path: str, eval_func: Callable[..., float]
) -> Tuple[float, str]:
    """Run eval_func on all model checkpoints in a directory, returning the best performing.

    Args:
        checkpoints_dir_path: directory of model checkpoints
        eval_func: arbitrary evaluation function. careful! could be a partial.

    Returns:
        (checkpoint_score, checkpoint_path)
    """

    results: list[Tuple[float, str]] = []
    directory = Path(checkpoints_dir_path)
    for ckpt_path in directory.iterdir():
        if ckpt_path.is_dir():
            ckpt_score = eval_func(ckpt_path=ckpt_path)
            results.append((ckpt_score, ckpt_path))

    # shape of: [tuple(score, path)]
    ckpt_performance = sorted(results, reverse=True)
    best_ckpt = ckpt_performance[0]
    return best_ckpt


def run_e2e_training(
    params: dict[str, Any], train_args: TrainingArgs, torch_args: TorchrunArgs
) -> str:
    """Runs phased training end-to-end with inter-training phase checkpoint picking.

    Args:
        params (dict[str, Any]): _description_
        train_args (TrainingArgs): _description_
        torch_args (TorchrunArgs): _description_

    Returns:
        best checkpoint from the final evaluation, phase10.
    """

    checkpoints_dir = os.path.abspath(params.e2e_checkpoints_dir)
    p5_checkpoints_dir = os.path.join(checkpoints_dir, "phase5")
    p10_checkpoints_dir = os.path.join(checkpoints_dir, "phase10")

    assert os.path.exists(
        p5_checkpoints_dir
    ), f"{p5_checkpoints_dir} needs to exist for end-to-end training!"
    assert os.path.exists(
        p10_checkpoints_dir
    ), f"{p10_checkpoints_dir} needs to exist for end-to-end training!"

    # TODO: probably want some more asserts to make sure that this will actually run to the end.

    # Phase05:
    _e2e_training_step(
        train_args=train_args,
        torch_args=torch_args,
        ckpt_dir_override=p5_checkpoints_dir,
    )
    _, p5_best_ckpt_path = _evaluate_dir_of_checkpoints(
        checkpoints_dir_path=p5_checkpoints_dir, eval_func=_mmlu
    )
    # ---------------------

    # Phase10:
    _e2e_training_step(
        train_args=train_args,
        torch_args=torch_args,
        model_override=p5_best_ckpt_path,
        ckpt_dir_override=p10_checkpoints_dir,
    )
    _, p10_best_ckpt_path = _evaluate_dir_of_checkpoints(
        checkpoints_dir_path=p10_checkpoints_dir,
        eval_func=partial(
            _mtbench,
            eval_cache_path=params.e2e_eval_cache_path,
            judge_name=params.e2e_mtbench_judge_name,
            judge_url=params.e2e_mtbench_judge_url,
        ),
    )
    # ---------------------

    return p10_best_ckpt_path
