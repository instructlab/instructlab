# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
from datetime import datetime
import enum
import logging
import pathlib
import os
import yaml

# Third Party
import httpx
from openai import OpenAI, OpenAIError
import pandas as pd
from ragas.callbacks import ChainRun
from ragas.dataset_schema import EvaluationDataset, EvaluationResult

# First Party
from instructlab.eval.ragas import ModelConfig, RagasEvaluator, RunConfig, Sample
from instructlab.eval.mt_bench_common import get_openai_client
from instructlab import client_utils

# Local
from ..utils import is_model_gguf, is_model_safetensors, validate_taxonomy_file
from ..client_utils import http_client, ClientException
from .evaluate import launch_server, get_model_name
from ..configuration import DEFAULTS

logger = logging.getLogger(__name__)

class IOFileType(enum.Enum):
    CSV: str = "csv"
    JSONL: str = "jsonl"
    XLSX: str = "xlsx"
    QNA_YAML: str = "qna.yaml"

def model_is_local(model: str) -> bool:
    if model is not None and os.path.exists(model):
        model_path = pathlib.Path(model)
        valid_model = False
        if model_path.is_dir():
            valid_model = is_model_safetensors(model_path)
        elif model_path.is_file():
            valid_model = is_model_gguf(model_path)
        return valid_model
    else:
        return False

def is_judge_model_name_valid(judge_model_name, api_key):
    try:
        client = OpenAI(
            base_url="https://api.openai.com/v1/",
            api_key=api_key,
        )
        models = client.models.list()
    except OpenAIError as exc:
        raise client_utils.ClientException(f"Connection Error {exc}") from exc

    model_ids = [model.id for model in models.data if model.id.startswith("gpt-")]
    return judge_model_name in model_ids

def get_endpoint_model_name(endpoint: str, api_key: str, http_client: httpx.Client) -> str:
    try:
        client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
            timeout=DEFAULTS.CONNECTION_TIMEOUT,
            http_client=http_client,
        )
        models = client.models.list()
    except OpenAIError as exc:
        raise client_utils.ClientException(f"Connection Error {exc}") from exc

    if len(models.data) != 1:
        raise client_utils.ClientException(f"More than one model at endpoint, pass in model name manually")

    return models.data[0].id

def create_results_file_name(file_format: IOFileType, output_dir: str) -> str:
    file_type = "jsonl"
    if IOFileType.CSV.value == file_format:
        file_type = "csv"
    if IOFileType.XLSX.value == file_format:
        file_type = "xlsx"

    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")

    output_path = pathlib.Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        output_dir = DEFAULTS.EVAL_DATA_DIR
        logger.info(f"Output dir provided is not a directory or does not exist, writing results to output directory {output_dir}")

    return f"{output_dir}/responses-{timestamp}.{file_type}"

def read_qna_yaml(yaml_file) -> pd.DataFrame:
    qna_list = []

    _, errors = validate_taxonomy_file(yaml_file)
    if errors:
        logger.info("Skipping {yaml_file} due to errors. Run `ilab taxonomy diff` on a taxonomy with this qna.yaml file to get errors")
        return pd.DataFrame()

    with yaml_file.open("r") as file:
        qna = yaml.safe_load(file)
        for seed_example in qna["seed_examples"]:
            for questions_and_answers in seed_example["questions_and_answers"]:
                qna_list.append({
                    "user_input": questions_and_answers["question"].strip(),
                    "reference": questions_and_answers["answer"].strip()
                })
    return pd.DataFrame(qna_list)

def get_input_df_from_file(file, evaluator) -> pd.DataFrame:
    input_df = pd.DataFrame()

    # skips file's that are just the directory when traversing subdirectories
    if file.is_dir():
        return input_df

    try:
        if IOFileType.CSV.value == file.suffix.strip('.'):
            input_df = pd.read_csv(file)
        elif IOFileType.JSONL.value == file.suffix.strip('.'):
            input_df = pd.read_json(file, orient="records", lines=True)
        elif IOFileType.XLSX.value == file.suffix.strip('.'):
            input_df = pd.read_excel(file)
        elif IOFileType.QNA_YAML.value == file.name:
            input_df = read_qna_yaml(file)
        else:
            logger.info(f"Ignoring reading {file}. Invalid file format. File extension must be .csv, .jsonl, or .xlsx")
            return input_df

        evaluator.validate_dataset(input_df)
    except ValueError as exc:
        print(f"Error in {file}. {exc}")
        return pd.DataFrame()

    logger.info(f"Added {file} to the evaluation dataset")
    return input_df

def get_input_df(input_questions, evaluator) -> pd.DataFrame:
    input_questions_path = pathlib.Path(input_questions)
    input_df = pd.DataFrame()

    if input_questions_path.exists():
        if input_questions_path.is_dir():
            for file in input_questions_path.rglob("*"):
                file_df = get_input_df_from_file(file,evaluator)
                input_df = pd.concat([input_df, file_df], ignore_index=True)
            # final validation of entire file after all concats
            evaluator.validate_dataset(input_df)
        if input_questions_path.is_file():
            input_df = get_input_df_from_file(input_questions_path,evaluator)

    input_df = input_df.drop_duplicates(subset=["user_input"])

    return input_df

def get_responses_from_model(ctx, evaluator, input_df, model, model_name, model_prompt, temperature, max_workers, gpus, backend, enable_serving_output) -> pd.DataFrame:
    student_api_key = None
    server = None
    if model_is_local(model):
        logger.info(f"Model is local")
        if model_name == None:
            model_name = get_model_name(model)
        server, api_base, effective_gpus = launch_server(
            eval_serve=ctx.obj.config.serve,
            tls_client_cert=ctx.params["tls_client_cert"],
            tls_client_key=ctx.params["tls_client_key"],
            tls_client_passwd=ctx.params["tls_client_passwd"],
            tls_insecure=ctx.params["tls_insecure"],
            model=model,
            model_name=model_name,
            max_workers=max_workers,
            gpus=gpus,
            backend=backend,
            enable_serving_output=enable_serving_output,
        )
    else:
        logger.info(f"Model is an endpoint")
        http_params = client_utils.http_client(
            {
                "tls_client_cert": ctx.params["tls_client_cert"],
                "tls_client_key": ctx.params["tls_client_key"],
                "tls_client_passwd": ctx.params["tls_client_passwd"],
                "tls_insecure": ctx.params["tls_insecure"],
            }
        )

        student_api_key = os.environ.get("STUDENT_ENDPOINT_API_KEY", None)
        if student_api_key is None:
            logger.info("API_KEY for model at endpoint {model} is not set. To set it set the environment variable $STUDENT_ENDPOINT_API_KEY")

        if model_name == None:
            model_name = get_endpoint_model_name(model, student_api_key, http_params)
        api_base = model

    openai_client = get_openai_client(model_api_base=api_base, api_key=student_api_key)
    logger.info(f"Model name is {model_name}")
    logger.info(f"Student Model Prompt is {model_prompt}")
    logger.info(f"Temperature is {temperature}")
    model_config = ModelConfig(model_name=model_name, temperature=temperature, system_prompt=model_prompt)
    input_df = evaluator.generate_answers_from_model(input_df, model_config, openai_client)

    return input_df, server

def print_results(result):
    print("\n")

    # TODO: Make this output much prettier
    print(f"############################")
    print("Scores")
    print(f"############################")
    total_score = 0
    question_num = 0
    for score in result.scores:
        question_num += 1
        print(f"Question #{question_num}:     {score['domain_specific_rubrics']}/5")
        total_score += score['domain_specific_rubrics']

    average = total_score/len(result.scores)
    print(f"----------------------------")
    print(f"Average Score:   {average}/5")
    print(f"Total Score:     {total_score}/{question_num*5}")

def write_results(result, file_formats, output_dir):
        response_df = result.dataset.to_pandas()
        scores = [score["domain_specific_rubrics"] for score in result.scores]
        response_df['scores'] = scores
        file_formats = file_formats.split(",")

        print(f"\nResponses and scores written to:")
        for fmt in file_formats:
            if IOFileType.JSONL.value == fmt:
                results_file = create_results_file_name(IOFileType.JSONL.value, output_dir)
                response_df.to_json(f"{results_file}", orient="records", lines=True)
            elif IOFileType.CSV.value == fmt:
                results_file = create_results_file_name(IOFileType.CSV.value, output_dir)
                response_df.to_csv(f"{results_file}", index=False)
            elif IOFileType.XLSX.value == fmt:
                results_file = create_results_file_name(IOFileType.XLSX.value, output_dir)
                response_df.to_excel(f"{results_file}", sheet_name="dataset", index=False)
            else:
                logger.info("Output format {fmt} is not valid")
                continue

            print(f"{results_file}")
        
def run_llmaaj(ctx, model, max_workers, gpus, backend, enable_serving_output, input_questions, output_file_formats, output_dir, model_prompt, temperature, model_name, judge_model_name):

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Environment variable 'OPENAI_API_KEY' must be set to run the Judge model in LLMaaJ.")

    judge_api_key = os.environ.get("OPENAI_API_KEY", None)
    if not is_judge_model_name_valid(judge_model_name, judge_api_key):
        raise ValueError("Judge model name must be a valid OpenAI GPT model")

    logger.info(f"Judge model name is: {judge_model_name}")

    logger.info(f"Input questions path is {input_questions}")
    evaluator = RagasEvaluator()
    input_df = get_input_df(input_questions, evaluator)

    server = None
    try:
        if model is not None:
            input_df, server = get_responses_from_model(ctx, evaluator, input_df, model, model_name, model_prompt, temperature, max_workers, gpus, backend, enable_serving_output)

        result = evaluator.run(
            dataset=input_df, judge_model_name=judge_model_name, judge_openai_api_key=judge_api_key
        )

        print_results(result)
        write_results(result, output_file_formats, output_dir)

    finally:
        if model_is_local(model):
            if server is not None:
                server.shutdown()
