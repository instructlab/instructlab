#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import os
import shutil

# Third Party
from click.testing import CliRunner
import instructlab.training

# First Party
from instructlab import lab
import instructlab.cli.model.train
import instructlab.model.accelerated_train

# This test is designed to be single threaded
ORIG_RUN_TRAINING = instructlab.training.run_training
ORIG_EVAL_CHECKPOINTS = instructlab.model.accelerated_train._evaluate_dir_of_checkpoints

INTENTIONAL_TRAINING_FAILURE_MESSAGE = "INTENTIONAL TRAINING FAILURE"
INTENTIONAL_MT_BENCH_FAILURE_MESSAGE = "INTENTIONAL MT_BENCH FAILURE"


def fail_before_first_run_training(*args, **kwargs):
    print("Training called, failing intentionally")
    raise Exception(INTENTIONAL_TRAINING_FAILURE_MESSAGE)


def fail_before_second_run_training(*args, **kwargs):
    print("Training called, running")
    ORIG_RUN_TRAINING(*args, **kwargs)
    print("Setting up next training phase to fail")
    instructlab.training.run_training = fail_before_first_run_training


def fail_mt_bench(*args, **kwargs):
    print("MT_Bench called, failing intentionally")
    raise Exception(INTENTIONAL_MT_BENCH_FAILURE_MESSAGE)


def run_training_phase(
    knowledge_data_path,
    skills_data_path,
    phased_base_dir,
    config,
    expected_exit_code=0,
    fail_on_phase1=False,
    fail_on_phase2=False,
    fail_on_mt_bench=False,
):
    instructlab.training.run_training = ORIG_RUN_TRAINING
    instructlab.model.accelerated_train._evaluate_dir_of_checkpoints = (
        ORIG_EVAL_CHECKPOINTS
    )

    if fail_on_phase1:
        instructlab.training.run_training = fail_before_first_run_training
    elif fail_on_phase2:
        instructlab.training.run_training = fail_before_second_run_training
    elif fail_on_mt_bench:
        instructlab.model.accelerated_train._evaluate_dir_of_checkpoints = fail_mt_bench

    os.makedirs(phased_base_dir, exist_ok=True)

    cli_runner = CliRunner()
    with cli_runner.isolated_filesystem():
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config",
                config,
                "model",
                "train",
                "--pipeline",
                "accelerated",
                "--strategy",
                "lab-multiphase",
                "--phased-phase1-data",
                knowledge_data_path,
                "--phased-phase2-data",
                skills_data_path,
                "--phased-phase1-num-epochs",
                "1",
                "--phased-phase2-num-epochs",
                "1",
                "--phased-base-dir",
                phased_base_dir,
                "--skip-user-confirm",
            ],
        )

    print(
        f"TRAIN OUTPUT: fail_on_phase1={fail_on_phase1}, fail_on_phase2={fail_on_phase2}, fail_on_mt_bench={fail_on_mt_bench}"
    )
    print(result.output)
    if fail_on_phase1 or fail_on_phase2:
        assert INTENTIONAL_TRAINING_FAILURE_MESSAGE in result.output
        if fail_on_phase1:
            assert_phase1_started(result)
        elif fail_on_phase2:
            assert_phase1_and_phase2_started(result)
    elif fail_on_mt_bench:
        assert INTENTIONAL_MT_BENCH_FAILURE_MESSAGE in result.output
        assert_phase2_resumed_and_eval_started(result)
    else:
        assert_completion_resumed(result)
    if fail_on_phase1 or fail_on_phase2 or fail_on_mt_bench:
        assert result.exception is not None
    assert result.exit_code == expected_exit_code


def assert_phase1_started(result):
    assert "Training Phase 1/2..." in result.output


def assert_phase2_started(result):
    assert "Training Phase 2/2..." in result.output


def assert_phase2_eval_started(result):
    assert "MT-Bench evaluation for Phase 2..." in result.output


def assert_phase1_in_journal(result):
    assert "SKIPPING: Training Phase 1/2; already in Journal" in result.output


def assert_phase1_and_phase2_started(result):
    assert_phase1_started(result)
    assert_phase2_started(result)


def assert_phase2_resumed_and_eval_started(result):
    assert_phase2_started(result)
    assert_phase2_eval_started(result)
    assert_phase1_in_journal(result)


def assert_completion_resumed(result):
    assert_phase2_eval_started(result)
    assert_phase1_in_journal(result)
    assert "SKIPPING: Training Phase 2/2; already in Journal" in result.output
    assert "Training finished! Best final checkpoint:" in result.output


def test_phased_training_resume(
    knowledge_data_path, skills_data_path, data_home_path, config
):
    phased_base_dir = os.path.join(data_home_path, "instructlab", "phased-resume")

    print("Running training to fail on phase 1")
    run_training_phase(
        knowledge_data_path,
        skills_data_path,
        phased_base_dir,
        config,
        expected_exit_code=1,
        fail_on_phase1=True,
    )
    print("Running training to fail on phase 2")
    run_training_phase(
        knowledge_data_path,
        skills_data_path,
        phased_base_dir,
        config,
        expected_exit_code=1,
        fail_on_phase2=True,
    )
    print("Running training to fail on mt_bench")
    run_training_phase(
        knowledge_data_path,
        skills_data_path,
        phased_base_dir,
        config,
        expected_exit_code=1,
        fail_on_mt_bench=True,
    )
    print("Running training to completion")
    run_training_phase(knowledge_data_path, skills_data_path, phased_base_dir, config)

    shutil.rmtree(phased_base_dir)

    print("Finished phased training resume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phased Training")
    parser.add_argument("--knowledge-data-path", help="Path to knowledge data path")
    parser.add_argument("--skills-data-path", help="Path to skills data path")
    parser.add_argument("--data-home-path", help="Data home path")
    parser.add_argument("--config", help="Path to instructlab config")
    args = parser.parse_args()
    knowledge_data_path = args.knowledge_data_path
    skills_data_path = args.skills_data_path
    data_home_path = args.data_home_path
    config = args.config

    test_phased_training_resume(
        knowledge_data_path, skills_data_path, data_home_path, config
    )
