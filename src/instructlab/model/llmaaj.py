# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
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

def create_results_file_name() -> str:
    return f"{DEFAULTS.EVAL_DATA_DIR}/responses.jsonl"

def run_llmaaj(ctx, model, max_workers, gpus, backend, enable_serving_output, input_questions):
    try:
        student_api_key = None
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
            student_api_key = os.environ.get("STUDENT_ENDPOINT_API_KEY", None)
            http_params = client_utils.http_client(
                {
                    "tls_client_cert": ctx.params["tls_client_cert"],
                    "tls_client_key": ctx.params["tls_client_key"],
                    "tls_client_passwd": ctx.params["tls_client_passwd"],
                    "tls_insecure": ctx.params["tls_insecure"],
                }
            )

            model_name = get_endpoint_model_name(model, student_api_key, http_params)
            api_base = model

        logger.info(f"Model name is {model_name}")
        logger.info(f"Input questions path is {input_questions}")
        #base_ds_df = pd.read_json(f"{input_questions}", orient="records", lines=True)
        openai_client = get_openai_client(model_api_base=api_base, api_key=student_api_key)
        student_model_config = ModelConfig(model_name=model_name)
        run_config = RunConfig(max_retries=3, max_wait=60, seed=42, timeout=30)
        judge_api_key = os.environ.get("OPENAI_API_KEY", None)

        evaluator = RagasEvaluator()
        result = evaluator.run(
            #dataset=base_ds, run_config=run_config,
            dataset=input_questions, student_model=student_model_config, run_config=run_config, student_openai_client=openai_client, judge_model_name="gpt-4o-mini", judge_openai_api_key=judge_api_key
        )


        response_df = result.dataset.to_pandas()
        results_file = create_results_file_name()
        response_df.to_json(f"{results_file}", orient="records", lines=True)
        logger.info(f"Responses written to {results_file}")
        print("\n###########")
        print("Scores are:")
        for score in result.scores:
            print(score['domain_specific_rubrics'])
        """
        # code in case I don't want to call OpenAI
        interim_df = pd.DataFrame(
                {"user_input": user_question1, "response": student_model_response1, "reference": golden_answer1},
                {"user_input": user_question2, "response": student_model_response2, "reference": golden_answer2}
        )

        evaluation_dataset = EvaluationDataset.from_pandas(interim_df)
        _unimportant_ragas_traces = {
            "default": ChainRun(
                run_id="42",
                parent_run_id=None,
                name="root",
                inputs={"system": "null", "user": "null"},
                outputs={"assistant": "null"},
                metadata={"user_id": 1337},
            )
        }
        temp_res = EvaluationResult(
            scores=[{'domain_specific_metric': 10}, {'domain_specific_metric': 8}],
            dataset=evaluation_dataset,
            ragas_traces=_unimportant_ragas_traces,
        )
        """

    finally:
        if model_is_local(model):
            if server is not None:
                server.shutdown()
    return

