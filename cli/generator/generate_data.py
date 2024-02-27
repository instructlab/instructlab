from datetime import datetime
import time
from typing import Optional
import json
import os
from os.path import splitext
import random
import re
import string
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import yaml
from git import Repo

import numpy as np
import tqdm
from rouge_score import rouge_scorer

from . import utils
from ..config.config import Config


def encode_prompt(prompt_instructions, prompt_file):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(prompt_file).read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response.message.content
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        # if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
        #     continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(fr'{idx}\.\s+(Instruction|Input|Output):', inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_data(
    logger,
    config: Config,
    output_dir: Optional[str] = None,
    taxonomy: Optional[str] = None,
    seed_tasks_path: Optional[str] = None,
    prompt_file_path: Optional[str] = None,
    model_name: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_instructions_to_generate: Optional[int] = None,
    num_prompt_instructions=2,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
):
    # Load generate configuration from config file unless already overwritten by the CLI
    if not taxonomy:
        taxonomy = config.get_generate_taxonomy()
    if not seed_tasks_path:
        seed_tasks_path = config.get_generate_seed_task_path()
    if not prompt_file_path:
        prompt_file_path = config.get_generate_prompt_file_path()
    if not model_name:
        model_name = config.get_generate_model()
    if not num_cpus:
        num_cpus = config.get_generate_num_cpus()
    if not num_instructions_to_generate:
        num_instructions_to_generate = config.get_generate_num_instructions()

    seed_instruction_data = []
    generate_start = time.time()

    # check taxonomy first then seed_tasks_path
    # throw an error if both not found
    if taxonomy and os.path.exists(taxonomy):
        # Default output_dir to taxonomy file's dir
        output_dir = output_dir or os.path.dirname(os.path.abspath(taxonomy))
        is_file = os.path.isfile(taxonomy)
        if is_file:
            if splitext(taxonomy)[1] != ".yaml":  # File name standard
                raise SystemExit(f"Error: taxonomy ({taxonomy}) is not a directory or file with '.yaml' extension.")
            try:
                with open(taxonomy, 'r') as file:
                    contents = yaml.safe_load(file)
                    for t in contents:
                        seed_instruction_data.append(
                            {"instruction": t["question"], "input": "", "output": t["answer"]})
            except Exception as e:
                print(e.__repr__, " in ", file)
                print(e)
                raise SystemExit(yaml.YAMLError(f"taxonomy file () has YAML errors!  Exiting."))

        else:  # taxonomy is_dir
            # Gather the new or changed YAMLs using git diff
            repo = Repo("taxonomy")
            updated_taxonomy_files = [u for u in repo.untracked_files if splitext(u)[1].lower() in [".yaml", ".yml"]] + \
                [d.a_path for d in repo.index.diff(None) if splitext(d.a_path)[1].lower() in [".yaml", ".yml"]]
            errors = 0
            warnings = 0
            for f in updated_taxonomy_files:
                if splitext(f)[1] != ".yaml":
                    logger.warn(f"WARNING: Skipping {f}! Use lowercase '.yaml' extension instead.")
                    errors += 1
                    continue
                file_path = os.path.join("taxonomy", f)
                try:
                    with open(file_path, 'r') as file:
                        contents = yaml.safe_load(file)
                        for t in contents:
                            q = t["question"]
                            a = t["answer"]
                            if not q or not a:
                                logger.warn(f"Skipping {file_path} because question and/or answer is empty!")
                                warnings += 1
                                continue
                            seed_instruction_data.append(
                                {"instruction": q, "input": "", "output": a})
                except Exception as e:
                    errors += 1
                    print(e.__repr__, " in ", file_path)
                    logger.error(e)

            if warnings:
                logger.warn(
                    f"{warnings} warnings (see above) due to taxonomy files that were not usable.")
            if errors:
                raise SystemExit(yaml.YAMLError(f"{errors} taxonomy files with YAML errors!  Exiting."))
    
    elif seed_tasks_path and os.path.exists(seed_tasks_path):
        output_dir = output_dir or os.path.dirname(os.path.abspath(seed_tasks_path))
        seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
        seed_instruction_data = [
            {"instruction": t["instruction"], "input": t["instances"][0]["input"],
             "output": t["instances"][0]["output"]}
            for t in seed_tasks
        ]
    else:
        raise SystemExit(f"Error: both taxonomy ({taxonomy}) and ({seed_tasks_path}) do not exist.")

    seeds = len(seed_instruction_data)
    logger.debug(f"Loaded {seeds} human-written seed instructions from {taxonomy or seed_tasks_path}")
    if not seeds:
        raise SystemExit("Nothing to generate. Exiting.")

    name = Path(model_name).stem  # Just in case it is a file path
    output_file = f"generated_{name}_{datetime.now().replace(microsecond=0).isoformat()}.json"
    logger.debug(f"Generating to: {os.path.join(output_dir, output_file)}")

    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        logger.debug(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions, prompt_file_path)
            batch_inputs.append(prompt)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            # Comment out extra info not currently being used:
            # most_similar_instructions = {
            #    all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            # }
            KEEP_ROUGE_SCORES_LT = 0.7   # TODO: PARAM
            if max(rouge_scores) > KEEP_ROUGE_SCORES_LT:
                continue
            else:
                keep += 1
            # Comment out extra info not currently being used:
            # instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            # instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        logger.debug(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        logger.debug(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, output_file))

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")
