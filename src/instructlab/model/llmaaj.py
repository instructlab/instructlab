# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
from datetime import datetime
import enum
import logging
import pathlib
import os

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
from ..utils import is_model_gguf, is_model_safetensors
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
    if os.path.exists(model):
        model_path = pathlib.Path(model)
        valid_model = False
        if model_path.is_dir():
            valid_model = is_model_safetensors(model_path)
        elif model_path.is_file():
            valid_model = is_model_gguf(model_path)
        return valid_model
    else:
        return False

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

def run_llmaaj(ctx, model, max_workers, gpus, backend, enable_serving_output, input_questions, output_file_formats, output_dir, model_prompt, temperature, model_name, judge_model_name):
    logger.info(f"Input questions path is {input_questions}")

    try:
        student_api_key = None
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
            student_api_key = os.environ.get("STUDENT_ENDPOINT_API_KEY", None)
            http_params = client_utils.http_client(
                {
                    "tls_client_cert": ctx.params["tls_client_cert"],
                    "tls_client_key": ctx.params["tls_client_key"],
                    "tls_client_passwd": ctx.params["tls_client_passwd"],
                    "tls_insecure": ctx.params["tls_insecure"],
                }
            )

            if model_name == None:
                model_name = get_endpoint_model_name(model, student_api_key, http_params)
            api_base = model

        openai_client = get_openai_client(model_api_base=api_base, api_key=student_api_key)
        logger.info(f"Model name is {model_name}")
        logger.info(f"Student Model Prompt is {model_prompt}")
        logger.info(f"Temperature is {temperature}")
        student_model_config = ModelConfig(model_name=model_name, temperature=temperature, system_prompt=model_prompt)
        judge_api_key = os.environ.get("OPENAI_API_KEY", None)
        logger.info(f"Judge model name is: {judge_model_name}")

        evaluator = RagasEvaluator()
        result = evaluator.run(
            dataset=input_questions, student_model=student_model_config, student_openai_client=openai_client, judge_model_name=judge_model_name, judge_openai_api_key=judge_api_key
        )
        print("\n")

        # TODO: Make this output much prettier
        print("###########")
        print("Scores are:")
        for score in result.scores:
            print(f"Score is: {score['domain_specific_rubrics']}")
            
        response_df = result.dataset.to_pandas()
        scores = [score["domain_specific_rubrics"] for score in result.scores]
        response_df['scores'] = scores
        output_formats = output_file_formats.split(",")

        if IOFileType.JSONL.value in output_formats:
            results_file = create_results_file_name(IOFileType.JSONL.value, output_dir)
            response_df.to_json(f"{results_file}", orient="records", lines=True)
            print(f"Responses and scores written to {results_file}")

        if IOFileType.CSV.value in output_formats:
            results_file = create_results_file_name(IOFileType.CSV.value, output_dir)
            response_df.to_csv(f"{results_file}", index=False)
            print(f"Responses and scores written to {results_file}")

        if IOFileType.XLSX.value in output_formats:
            results_file = create_results_file_name(IOFileType.XLSX.value, output_dir)
            response_df.to_excel(f"{results_file}", sheet_name="Summary", index=False)
            print(f"Responses and scores written to {results_file}")

    finally:
        if model_is_local(model):
            if server is not None:
                server.shutdown()
