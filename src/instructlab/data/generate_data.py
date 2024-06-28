# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import logging
import os
import time

# Third Party
from datasets import Dataset
import httpx
from instructlab.sdg import SDG, utils
from instructlab.sdg.default_flows import (
    MODEL_FAMILY_MERLINITE,
    MODEL_FAMILY_MIXTRAL,
    MMLUBenchFlow,
    SimpleKnowledgeFlow,
    SynthKnowledgeFlow,
)
from instructlab.sdg.pipeline import Pipeline
from instructlab.sdg.utils import chunking, models
from instructlab.sdg.utils.taxonomy import (
    leaf_node_to_samples,
    read_taxonomy_leaf_nodes,
)
import openai

# First Party
# pylint: disable=ungrouped-imports
from instructlab.utils import get_sysprompt

logger = logging.getLogger(__name__)


def _unescape(s):
    return bytes(s, "utf-8").decode("utf-8")


# This is a hack because the simple workflow returns a q/a pair as a single output.
# We could possibly try to ask for them separately, but it would cost twice the inference
# API calls. All of this is because the smallest models we use on small environments
# for testing and demos weren't good enough to follow the strict formatting instructions used
# in the full pipeline.
def _get_question(logger, synth_example):
    if "question" in synth_example:
        return synth_example["question"]

    if "output" not in synth_example:
        raise utils.GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[0].strip() + "?" if len(parts) == 2 else ""


# This is also a hack. See the comment above _get_question.
def _get_response(logger, synth_example):
    if "response" in synth_example:
        return synth_example["response"]

    if "output" not in synth_example:
        raise GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


def _gen_train_data(logger, machine_instruction_data, output_file_train):
    train_data = []
    for synth_example in machine_instruction_data:
        logger.debug(synth_example)
        user = _get_question(logger, synth_example)
        if len(synth_example.get("context", "")) > 0:
            user += "\n" + synth_example["context"]
        train_data.append(
            {
                "system": get_sysprompt(),
                "user": _unescape(user),
                "assistant": _unescape(_get_response(logger, synth_example)),
            }
        )

    with open(output_file_train, "w", encoding="utf-8") as outfile:
        for entry in train_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")


def _gen_test_data(
    leaf_nodes,
    output_file_test,
):
    test_data = []
    for _, leaf_node in leaf_nodes.items():
        for seed_example in leaf_node:
            user = seed_example["instruction"]  # question

            if len(seed_example["input"]) > 0:
                user += "\n" + seed_example["input"]  # context

            test_data.append(
                {
                    "system": get_sysprompt(),
                    "user": _unescape(user),
                    "assistant": _unescape(seed_example["output"]),  # answer
                }
            )

    with open(output_file_test, "w", encoding="utf-8") as outfile:
        for entry in test_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")


def generate_data(
    api_base,
    tls_insecure,
    model_family: str,
    yaml_rules: Optional[str] = None,
    output_dir: Optional[str] = None,
    taxonomy: Optional[str] = None,
    taxonomy_base: Optional[str] = None,
    model_name: Optional[str] = None,
    # TODO - not yet used, but should be presumably
    num_instructions_to_generate: Optional[int] = None,
    console_output=True,
    api_key: Optional[str] = None,
    chunk_word_count=None,
    server_ctx_size=None,
    tls_client_cert: Optional[str] = None,
    tls_client_key: Optional[str] = None,
    tls_client_passwd: Optional[str] = None,
    # TODO need to update the CLI to specify which profile to use (simple or full at the moment)
    profile: Optional[str] = "simple",
):
    generate_start = time.time()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not (taxonomy and os.path.exists(taxonomy)):
        raise GenerateException(f"Error: taxonomy ({taxonomy}) does not exist.")

    leaf_nodes = read_taxonomy_leaf_nodes(taxonomy, taxonomy_base, yaml_rules)
    if not leaf_nodes:
        raise GenerateException("Error: No new leaf nodes found in the taxonomy.")

    name = Path(model_name).stem  # Just in case it is a file path
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    output_file_generated = f"generated_{name}_{date_suffix}.json"
    output_file_test = f"test_{name}_{date_suffix}.jsonl"
    output_file_train = f"train_{name}_{date_suffix}.jsonl"

    _gen_test_data(
        leaf_nodes,
        os.path.join(output_dir, output_file_test),
    )

    logger.debug(f"Generating to: {os.path.join(output_dir, output_file_generated)}")

    orig_cert = (tls_client_cert, tls_client_key, tls_client_passwd)
    cert = tuple(item for item in orig_cert if item)
    verify = not tls_insecure
    client = openai.OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(cert=cert, verify=verify),
    )

    if models.get_model_family(model_family, model_name) == "mixtral":
        model_family = MODEL_FAMILY_MIXTRAL
    else:
        model_family = MODEL_FAMILY_MERLINITE

    # TODO -- llama-cpp doesn't support batching, we need to get a hint from the CLI
    # about whether we can turn this on (whether vllm is used or not)
    batched = False

    flow_types = []
    if profile == "full":
        flow_types.append(MMLUBenchFlow)
        flow_types.append(SynthKnowledgeFlow)
    elif profile == "simple":
        flow_types.append(SimpleKnowledgeFlow)
    else:
        raise SystemExit(f"Error: profile ({profile}) is not supported.")

    sdg = SDG(
        [
            Pipeline(flow_type(client, model_family, model_name, batched).get_flow())
            for flow_type in flow_types
        ]
    )

    if console_output:
        logger.info(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )

    generated_data = None
    for leaf_node in leaf_nodes.values():
        samples = leaf_node_to_samples(leaf_node)

        if not samples:
            # TODO - expected in e2e tests since we haven't integrated skills yet
            logger.error("No samples found in leaf node")
            continue

        # TODO this is broken, just trying to get initial integration to run
        # pylint: disable=consider-using-enumerate
        for i in range(len(samples)):
            samples[i]["document"] = chunking.chunk_document(
                documents=samples[i]["document"],
                server_ctx_size=server_ctx_size,
                chunk_word_count=chunk_word_count,
            )[0]

        # TODO -- there is a parameter for how many samples to generate, but we ignore it so far

        ds = Dataset.from_list(samples)
        new_generated_data = sdg.generate(ds)
        generated_data = (
            new_generated_data
            if generated_data is None
            else generated_data + new_generated_data
        )
        logger.info("Generated %d samples" % len(generated_data))
        logger.debug("Generated data: %s" % generated_data)

    if generated_data is None:
        generated_data = []

    _gen_train_data(logger, generated_data, os.path.join(output_dir, output_file_train))

    # TODO
    # This is for backwards compatibility. The file existing previously, so we'll keep it for now.
    # I believe the github bot assumes it is present for presenting generated data to a taxonomy
    # reviewer or contributor. Otherwise, I don't see a consumer of it in this repo or the
    # `ilab` CLI.
    _gen_train_data(
        logger, generated_data, os.path.join(output_dir, output_file_generated)
    )

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")
