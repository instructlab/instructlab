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
import click
from openai import OpenAI, OpenAIError
import pandas as pd
from ragas.callbacks import ChainRun
from ragas.dataset_schema import EvaluationDataset, EvaluationResult
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

# First Party
from instructlab.eval.ragas import ModelConfig, RagasEvaluator, RunConfig, Sample, _DEFAULT_SYSTEM_PROMPT
from instructlab.eval.mt_bench_common import get_openai_client
from instructlab import client_utils

# Local
from ..utils import is_model_gguf, is_model_safetensors, validate_taxonomy_file
from ..client_utils import http_client, ClientException
from .evaluate import launch_server, get_model_name
from ..configuration import DEFAULTS

logger = logging.getLogger(__name__)

class llmaaj_judge(BaseModel):
    name: str = Field(
        default='gpt-4o',
        description="Judge model name for judge doing evaluation in the LLMaaJ evaluation. Model name must be an Open AI GPT model.",
    )

class llmaaj_model(BaseModel):
    location: str = Field(
        description="Endpoint or path for model being evaluated in LLMaaJ evaluation",
    )
    prompt: str | None = Field(
        default=_DEFAULT_SYSTEM_PROMPT,
        description="Prompt by model to get responses to be evaluated in LLMaaJ evaluation.",
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature setting for model to get responses to be evaluated in LLMaaJ evaluation.",
    )
    api_key: str | None = Field(
        default=None,
        description="Name of environment variable that contains the models api key. The '$' is not needed in front of the variable name. For example 'MODEL_1_API_KEY' is a correct instance for inptu not '$MODEL_1_API_KEY'.",
    )

class llmaaj_config(BaseModel):
    """Class describing configuration of LLMaaJ evaluation benchmark."""

    input_questions: str = Field(
        description="File or directory with questions, reference answers, and responses for the LLMaaJ evaluation. Permitted files can either be appropriately formatted .csv, .jsonl, or .xlsx files or a qna.yaml from the taxonomy. If a directory is specified, the directory must contain one or more appropriately formatted files or the files must exist in one of it's subdirectories.",
    )
    output_dir: str = Field(
        default_factory=lambda: DEFAULTS.EVAL_DATA_DIR,
        description="Directory where evaluation results are stored.",
    )

    output_file_formats: list[str] | None = Field(
        default=['jsonl'],
        description="A list of all of the file types the LLMaaJ evaluation should output it's results in. Options include 'csv', 'xlsx', and 'jsonl'. If this variable is not specified, results are output in .jsonl format.",
        examples=[
            ["csv", "xlsx"],
        ],
    )

    models: list[llmaaj_model] | None = Field(
        default_factory=list,
        description="A list of models to collect responses and be evaluated in the LLMaaJ evaluation.",
        examples=[
            "model: http://model.endpoint.com/v1\n    temperature: 0.0\n    api_key: MODEL1_ENDPOINT_API_KEY\n    prompt: 'With the best of your knowledge what is the answer to the following question'\n    model_name: foo",
            "model: http://model2.endpoint.com/v1\n    api_key: MODEL2_ENDPOINT_API_KEY",
            "model: /path/to/a/model\n    model_name: trained_model",
        ],
    )
    judge: llmaaj_judge = Field(
        default_factory=llmaaj_judge,
        description="LLMaaJ judge settings",
    )


class IOFileType(enum.Enum):
    CSV: str = "csv"
    JSONL: str = "jsonl"
    XLSX: str = "xlsx"
    QNA_YAML: str = "qna.yaml"

def print_example_job_file():
    print("Example of a valid job file is:\n")
    model1 = llmaaj_model(location="https://endpoint.of.model1.com/v1", prompt="sample prompt", api_key="MODEL_API_KEY_ENV_VAR", name="model-1")
    model2 = llmaaj_model(location="/path/to/local/model", name="trained-model", prompt="another prompt", )
    models = [model1, model2]
    example_config = llmaaj_config(models=models, input_questions="/path/to/questions")

    example_config_dict = example_config.dict()
    yaml_output = yaml.dump(example_config_dict, sort_keys=False)
    print(yaml_output)

def load_job_file(llmaaj_job_file) -> llmaaj_config:
    llmaaj_cfg = {}
    try:
        with open(llmaaj_job_file, 'r') as file:
            llmaaj_cfg = yaml.safe_load(file)
    except OSError:
        click.secho(
            f"Error {llmaaj_job_file} could not be loaded properly.",
            fg="red",
        )
        raise click.exceptions.Exit(1)

    try:
        cfg = llmaaj_config(**llmaaj_cfg)
    except ValidationError as e:
        click.secho(
            f"Job file {llmaaj_job_file} is not valid",
            fg="red",
        )
        print_example_job_file()
        raise click.exceptions.Exit(1)
    return cfg

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

def create_results_file_name(file_format: IOFileType, output_dir: str, timestamp: str, model_name: str) -> str:
    file_type = "jsonl"
    if IOFileType.CSV.value == file_format:
        file_type = "csv"
    if IOFileType.XLSX.value == file_format:
        file_type = "xlsx"

    output_path = pathlib.Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        output_dir = DEFAULTS.EVAL_DATA_DIR
        logger.info(f"Output dir provided is not a directory or does not exist, writing results to output directory {output_dir}")

    return f"{output_dir}/responses-{model_name}-{timestamp}.{file_type}"

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

def get_responses_from_model(ctx, evaluator, input_df, model, model_prompt, temperature, api_key, max_workers, gpus, backend, enable_serving_output) -> pd.DataFrame:
    server = None
    if model_is_local(model):
        logger.info(f"Model is local")
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

        model_name = get_endpoint_model_name(model, student_api_key, http_params)
        api_base = model

    openai_client = get_openai_client(model_api_base=api_base, api_key=student_api_key)
    logger.info(f"Model name is {model_name}")
    logger.info(f"Student Model Prompt is {model_prompt}")
    logger.info(f"Temperature is {temperature}")
    model_config = ModelConfig(model_name=model_name, temperature=temperature, system_prompt=model_prompt)
    input_df = evaluator.generate_answers_from_model(input_df, model_config, openai_client)

    return input_df, server, model_name

def make_run_dir(output_dir):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S-%f")
    run_dir = f"{output_dir}/job-{timestamp}"

    path_does_not_exist = not os.path.exists(output_dir) 
    path_is_not_dir =  not os.path.isdir(output_dir)
    if path_does_not_exist or path_is_not_dir:
        run_dir = f"{DEFAULTS.EVAL_DATA_DIR}/job-{timestamp}",

    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def print_header():
    # TODO: Make this output much prettier
    print("\n")
    print(f"############################")
    print("SCORES")
    print(f"############################")

def print_results(result, model_name):
    print(f"\nMODEL: {model_name}\n")
    total_score = 0
    question_num = 0
    for score in result.scores:
        question_num += 1
        print(f"Question #{question_num}:     {score['domain_specific_rubrics']}/5")
        total_score += score['domain_specific_rubrics']

    average = total_score/len(result.scores)
    print(f"----------------------------")
    print(f"Average Score:   {average}/5")
    print(f"Total Score:     {total_score}/{question_num*5}\n")

def write_results(result, file_formats, output_dir, model_name):
        response_df = result.dataset.to_pandas()
        scores = [score["domain_specific_rubrics"] for score in result.scores]
        response_df['scores'] = scores

        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S-%f")

        print(f"Responses and scores written to:")
        for fmt in file_formats:
            if IOFileType.JSONL.value == fmt:
                results_file = create_results_file_name(IOFileType.JSONL.value, output_dir, timestamp, model_name)
                response_df.to_json(f"{results_file}", orient="records", lines=True)
            elif IOFileType.CSV.value == fmt:
                results_file = create_results_file_name(IOFileType.CSV.value, output_dir, timestamp, model_name)
                response_df.to_csv(f"{results_file}", index=False)
            elif IOFileType.XLSX.value == fmt:
                results_file = create_results_file_name(IOFileType.XLSX.value, output_dir, timestamp, model_name)
                response_df.to_excel(f"{results_file}", sheet_name="dataset", index=False)
            else:
                logger.info("Output format {fmt} is not valid")
                continue

            print(f"{results_file}")
        
def run_llmaaj(ctx, model, max_workers, gpus, backend, enable_serving_output, input_questions, output_dir, model_prompt, temperature, judge_model_name, api_key=None) -> tuple[EvaluationResult, str | None]:

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
            input_df, server, model_name = get_responses_from_model(ctx, evaluator, input_df, model, model_prompt, temperature, api_key, max_workers, gpus, backend, enable_serving_output)

        result = evaluator.run(
            dataset=input_df, judge_model_name=judge_model_name, judge_openai_api_key=judge_api_key
        )

    finally:
        if model_is_local(model):
            if server is not None:
                server.shutdown()

    return result, model_name
