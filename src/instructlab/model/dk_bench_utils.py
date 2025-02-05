# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
from datetime import datetime
from typing import List
import enum
import logging
import os
import pathlib
import statistics as stats

# Third Party
from instructlab.eval.mt_bench_common import (
    get_openai_client as get_local_openai_client,
)
from instructlab.eval.ragas import ModelConfig, RagasEvaluator
from openai import OpenAI, OpenAIError
from openpyxl import load_workbook  # type: ignore
from openpyxl.styles import Border, Font, Side  # type: ignore
from ragas.evaluation import EvaluationResult  # type: ignore
import pandas as pd

# First Party
from instructlab import client_utils

# Local
from ..configuration import _serve
from .evaluate import get_model_name as get_local_model_name
from .evaluate import launch_server, validate_model

logger = logging.getLogger(__name__)


class IOFileType(enum.Enum):
    CSV = "csv"
    JSONL = "jsonl"
    XLSX = "xlsx"


def validate_input_questions(input_questions: pathlib.Path) -> None:
    """
    Ensure input questions file exists, is not a directory
    and is a '.jsonl' file.
    Args:
        input_questions (Path): The path to the input questions .jsonl file
    Returns:
        None
    """
    if not input_questions.exists():
        raise ValueError(
            f"Input questions file {input_questions} does not exist",
        )

    if input_questions.is_dir():
        raise ValueError(
            f"Input questions file {input_questions} is a directory",
        )

    # need to make sure input questions is a jsonl file
    if input_questions.suffix.lstrip(".") != IOFileType.JSONL.value:
        raise ValueError(
            f"Invalid file type: {input_questions}. Expected a '.jsonl' file."
        )


def validate_output_file_formats(file_formats: List[str]) -> None:
    """
    Validates that file formats passed in for the file containing scores and
    responses is a valid output format. Valid file formats are one of:
    jsonl, xlsx, or csv.
    Args:
        file_formats (List[str]): A list of all of the file format strings
                                  passed in by the user.
    Returns:
        None
    """
    for file_format in file_formats:
        if not any(file_format == item.value for item in IOFileType):
            raise ValueError(
                f"File format {file_format} is not a valid output format. Format must be one of csv, xlsx, jsonl"
            )


def create_results_file_name(
    file_format: str, output_dir: str, timestamp: str, model_name: str
) -> str:
    """
    Validates that file formats passed in for the file containing scores and
    responses is a valid output format. Valid file formats are one of:
    jsonl, xlsx, or csv.

    This function assumes output_dir is already created and is a directory.
    Args:
        file_formats (str): The file format for the output file. One of "csv",
                            "xlsx", "jsonl".
        output_dir (str):   Output directory for results and  scores file
        timestamp (str):    timestamp in .iso format that is part of the
                            name of the output file.
        model_name (str):   Name of model to be used in the name of the output
                            file.
    Returns:
        str: The file name containing the scores and model responses
    """
    if file_format not in [
        IOFileType.CSV.value,
        IOFileType.JSONL.value,
        IOFileType.XLSX.value,
    ]:
        raise ValueError("File format is not one of: csv, xlsx, jsonl")

    # remove any trailing slashes from user provided output_dir
    output_dir = os.path.normpath(output_dir)

    # make directory for models results
    model_results_dir = f"{output_dir}/{model_name}"
    os.makedirs(model_results_dir, exist_ok=True)

    return f"{model_results_dir}/results_{timestamp}.{file_format}"


def print_results(
    result: EvaluationResult, results_files: List[str], model_name: str
) -> None:
    """
    Prints a scoring report for DK-Bench
    Args:
        result (Evaluation):       ragas EvaluationResult to parse for scores
                                   to each question.
        results_files (List[str]): List of files with scores and responses
        model_name (str):          Name of model that generated responses for
                                   DK-Bench to evaluate againist reference answer.
    Returns:
        None
    """
    print("\n")
    print("# DK-BENCH REPORT")
    print(f"\n## MODEL: {model_name}\n")
    total_score = 0
    for i, score in enumerate(result.scores):
        print(f"Question #{i+1}:     {score['domain_specific_rubrics']}/5")
        total_score += score["domain_specific_rubrics"]

    average = total_score / len(result.scores)
    print("----------------------------")
    print(f"Average Score:   {average:.2f}/5")
    print(f"Total Score:     {total_score}/{len(result.scores)*5}\n")

    print("Responses and scores are written to:")
    for file in results_files:
        print(f"{file}")
    print("\n")


def create_excel_results_file(excel_file: str, result: EvaluationResult) -> None:
    """
    Writes an excel file based on the result of an evaluation.
    The excel files has two sheets. The first is a summary sheet
    with a table of individual question scores, average, total score and median.

    The second sheet has the score, the question (user_input), reference,
    response, model name, and evaluation run, similar to the contents of the
    .jsonl and .csv files.

    Args:
        result (Evaluation):    ragas EvaluationResult to parse for scores
                                for summary sheet and questions, references,
                                and responses sheet.
        excel_file (str):       Name of excel file to be created the summary
                                and dataset sheets.
    Returns:
        None
    """
    scores = [score["domain_specific_rubrics"] for score in result.scores]
    question_indices = [f"Q{i + 1}" for i in range(len(scores))]

    col1 = ["Average", "Total Score", "Median", "Question"] + question_indices
    col2 = [stats.mean(scores), sum(scores), stats.median(scores), "Score"] + scores

    summary_data = {
        "Metric": col1,
        "Value": col2,
    }

    summary_df = pd.DataFrame(summary_data)

    response_df = result.dataset.to_pandas()
    response_df["scores"] = scores

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        # df with contents similar to those in the .jsonl and .csv output files
        response_df.to_excel(writer, sheet_name="dataset", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    # Add a visual separation for row 5 on the summary sheet before question number and scores are output.
    wb = load_workbook(excel_file)
    summary_sheet = wb["Summary"]
    for cell in summary_sheet[5]:  # Row 5 (Question, Score)
        cell.font = Font(bold=True)
        cell.border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

    wb.save(excel_file)


def write_results(
    result: EvaluationResult, file_formats: List[str], output_dir: str, model_name: str
) -> List[str]:
    """
    Writes results files for DK-Bench for each file format provided.
    Files have the name {output_dir}/responses-{model_name}-{timestamp}.{file_type}"

    Each of the entries in a file has the following fields:
    {model_name, scores, evaluation_run, user_input, response, reference}

    The list of file formats can be assumed to be valid.

    Args:
        result (Evaluation):      ragas EvaluationResult to parse for scores
                                  for summary sheet and questions, references,
                                  and responses sheet.
        file_formats (List[str]): List of file formats of files to write results to.
        output_dir (str):         Directory for results to be written out to
        model_name (str):         Model name to be used in the results file name.
    Returns:
        List[str]:                A list of strings that are the file names of the files
                                  with responses and scores for DK-Bench
    """
    response_df = result.dataset.to_pandas()
    response_df["model_name"] = model_name

    scores = [score["domain_specific_rubrics"] for score in result.scores]
    response_df["scores"] = scores

    timestamp = datetime.now().isoformat()
    response_df["timestamp"] = f"{timestamp}"

    results_files = []
    for fmt in file_formats:
        results_file = create_results_file_name(fmt, output_dir, timestamp, model_name)

        if IOFileType.JSONL.value == fmt:
            response_df.to_json(f"{results_file}", orient="records", lines=True)
        elif IOFileType.CSV.value == fmt:
            response_df.to_csv(f"{results_file}", index=False)
        elif IOFileType.XLSX.value == fmt:
            create_excel_results_file(results_file, result)

        results_files.append(results_file)
        logger.debug("DK-Bench responses and results written to %s", results_file)

    return results_files


def is_judge_model_name_valid(judge_model_name: str, api_key: str) -> bool:
    """
    Verifies whether or not the judge model provided is the name of a valid
    OpenAI model to use for the judge in DK-Bench evaluation.

    Args:
        judge_model_name (str):   Name of the judge model to check validity of.
        api_key (str):            OpenAI API key.
    Returns:
        bool:                     Whether or not the judge_model_name is a valid
                                  OpenAI model name.
    """
    try:
        client = OpenAI(
            base_url="https://api.openai.com/v1/",
            api_key=api_key,
        )
        models = client.models.list()
    except OpenAIError as exc:
        raise client_utils.ClientException(f"Connection Error {exc}") from exc

    return any(judge_model_name == model.id for model in models.data)


def run_dk_bench(
    serve_config: _serve,
    tls_insecure: bool,
    tls_client_cert: str,
    tls_client_key: str,
    tls_client_passwd: str,
    model: str,
    max_workers: str | int | None,
    gpus: int | None,
    backend: str | None,
    enable_serving_output: bool,
    input_questions: str,
    system_prompt: str,
    temperature: float,
    judge_model_name: str,
) -> tuple[EvaluationResult, str]:
    """
    Wrapper for running one iteration of DK-Bench evaluation.

    Args:
        serve_config (_serve):           Name of the judge model to check validity of.
        tls_insecure (bool):             TLS is secure bool for launch_server
        tls_client_cert (str):           TLS client cert for launch_server
        tls_client_key (str):            TLS client key for launch_server
        tls_client_passwd (str):         TLS client password for launch_server
        model (str):                     Model to generate responses for
                                         evaluation.
        max_workers (str | int | None):  Max workers
        gpus (int | None):               Number of gpus to use when serving
                                         with vLLM.
        backend (str | None):            Serving backend for local model
        enable_serving_output (bool):    Whether to dump full vLLM output
                                         into foreground.
        input_questions (str):           Path to file with input questions
                                         and references.
        system_prompt (str):             System prompt for model generating
                                         responses.
        temperature (float):             Chat temperature for generating
                                         responses.
        judge_model_name (str):          OpenAI Judge model name.
    Returns:
        result (Evaluation):             ragas EvaluationResult to parse for
                                         scores for summary sheet and questions,
                                         references, and responses sheet.
        model_name (str):                Model name of model responses were collected
                                         from.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "Environment variable 'OPENAI_API_KEY' must be set to run the Judge model in DK-Bench."
        )
    judge_openai_api_key = os.environ["OPENAI_API_KEY"]

    if not is_judge_model_name_valid(judge_model_name, judge_openai_api_key):
        raise ValueError("Judge model name must be a valid OpenAI GPT model")

    # RagasEvaluator.run() expects a Path of a .jsonl file
    input_questions_path = pathlib.Path(input_questions).resolve()
    validate_input_questions(input_questions_path)

    try:
        test_df = pd.read_json(input_questions_path, orient="records", lines=True)
        if "response" in test_df.columns:
            logger.info(
                "Input file %s already contains responses for evaluation. Responses from %s will not be collected for this file.",
                input_questions_path,
                model,
            )
            get_responses_from_model = False
        else:
            get_responses_from_model = True

    except Exception as exc:
        raise ValueError(
            f"Contents of {input_questions_path} cannot be loaded as JSON. Please ensure it is a valid '.jsonl' file."
        ) from exc

    evaluator = RagasEvaluator()
    if get_responses_from_model:
        logger.debug(
            "Input file needs responses for evaluation. Getting responses from user configured model %s.",
            model,
        )
        server = None
        validate_model(model)
        try:
            logger.debug("Model being evaluated in DK-Bench is local")
            model_name = get_local_model_name(model)
            server, api_base, _ = launch_server(
                eval_serve=serve_config,
                tls_insecure=tls_insecure,
                tls_client_cert=tls_client_cert,
                tls_client_key=tls_client_key,
                tls_client_passwd=tls_client_passwd,
                model=model,
                model_name=model_name,
                max_workers=max_workers,
                gpus=gpus,
                backend=backend,
                enable_serving_output=enable_serving_output,
            )
            openai_client = get_local_openai_client(
                model_api_base=api_base, api_key=None
            )
            model_config = ModelConfig(
                model_name=model_name,
                temperature=temperature,
                system_prompt=system_prompt,
            )
            result = evaluator.run(
                dataset=input_questions_path,
                student_model=model_config,
                student_openai_client=openai_client,
                judge_model_name=judge_model_name,
                judge_openai_api_key=judge_openai_api_key,
            )

        finally:
            if server is not None:
                server.shutdown()
    # evaluation on just a dataset with responses already provided
    else:
        result = evaluator.run(
            dataset=input_questions_path,
            judge_model_name=judge_model_name,
            judge_openai_api_key=judge_openai_api_key,
        )
        model_name = "no-model-provided"

    return result, model_name
