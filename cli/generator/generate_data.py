# Standard
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from os.path import splitext
from pathlib import Path
from typing import Optional
import json
import os
import random
import re
import string
import time

try:
    # Third Party
    import git
except ImportError:
    pass
# Third Party
from rouge_score import rouge_scorer

# import numpy as np
import tqdm
import yaml

# Local
from . import utils

DEFAULT_PROMPT_TEMPLATE = """\
You are asked to come up with a set of 20 diverse task instructions under {taxonomy}. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.
3. The type of instructions should not have topic diversity. The list should include follow the same topic and category.
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The instructions should be in English.
6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
7. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
8. Not all instructions require input. For example, when a instruction asks about some general information, "what is the highest peak in the world", it is not necessary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
9. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.

List of 20 tasks:
"""


class GenerateException(Exception):
    """An exception raised during generate step."""


def check_prompt_file(prompt_file_path):
    """Check for prompt file."""
    try:
        with open(prompt_file_path, encoding="utf=8") as file:
            prompt_template = file.read()
    except FileNotFoundError:
        print(f"Cannot find {prompt_file_path}. Using default prompt.")
        prompt_template = DEFAULT_PROMPT_TEMPLATE
    prompt_template = prompt_template + "\n"
    return prompt_template


def encode_prompt(prompt_instructions, prompt):
    """Encode multiple prompt instructions into a single string."""
    idx = 0
    prompt = prompt.format(taxonomy=prompt_instructions[0]["taxonomy_path"])
    # pylint: disable=unused-variable
    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, prompt_input, prompt_output, taxonomy_path,) = (
            task_dict["instruction"],
            task_dict["input"],
            task_dict["output"],
            task_dict["taxonomy_path"],
        )
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt_input = "<noinput>" if prompt_input.lower() == "" else prompt_input
        prompt += "###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{prompt_input}\n"
        prompt += f"{idx + 1}. Output:\n{prompt_output}\n"
    prompt += "###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = (
        f"{num_prompt_instructions + 1}. Instruction:" + response.message.content
    )
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        # if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
        #     continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(rf"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        inst = splitted_data[2].strip()
        prompt_input = splitted_data[4].strip()
        prompt_input = "" if prompt_input.lower() == "<noinput>" else prompt_input
        prompt_output = splitted_data[6].strip()
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
        # We found that the model tends to add "write a program" to some existing instructions
        # which lead to a lot of such instructions and it's confusing whether the model needs
        # to write a program or directly output the result, so here we filter them out.
        # NOTE: this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append(
            {"instruction": inst, "input": prompt_input, "output": prompt_output}
        )
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def get_seed_examples(contents):
    if "seed_examples" in contents:
        return contents["seed_examples"]
    return contents


def generate_data(
    logger,
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
    rouge_threshold: Optional[float] = None,
):
    seed_instruction_data = []
    generate_start = time.time()

    # check taxonomy first then seed_tasks_path
    # throw an error if both not found
    # pylint: disable=broad-exception-caught,raise-missing-from
    if taxonomy and os.path.exists(taxonomy):

        is_file = os.path.isfile(taxonomy)
        if is_file:
            # Default output_dir to taxonomy file's dir
            output_dir = output_dir or os.path.dirname(os.path.abspath(taxonomy))
            seed_instruction_data, warnings, errors = read_taxonomy_file(
                logger, taxonomy
            )
            if warnings:
                logger.warn(
                    f"{warnings} warnings (see above) due to taxonomy file not (fully) usable."
                )
            if errors:
                raise SystemExit(yaml.YAMLError("Taxonomy file with errors! Exiting."))
        else:  # taxonomy is_dir
            # Default output_dir to taxonomy dir, using the dir not the parent
            output_dir = output_dir or os.path.abspath(taxonomy)
            # Gather the new or changed YAMLs using git diff
            try:
                updated_taxonomy_files = get_taxonomy_diff(taxonomy)
            except NameError as exc:
                raise GenerateException("`git` binary not found") from exc
            total_errors = 0
            total_warnings = 0
            for f in updated_taxonomy_files:
                file_path = os.path.join(taxonomy, f)
                data, warnings, errors = read_taxonomy_file(logger, file_path)
                total_warnings += warnings
                total_errors += errors
                if data:
                    seed_instruction_data.extend(data)
            if total_warnings:
                logger.warn(
                    f"{total_warnings} warnings (see above) due to taxonomy files that were not (fully) usable."
                )
            if total_errors:
                raise SystemExit(
                    yaml.YAMLError(
                        f"{total_errors} taxonomy files with errors! Exiting."
                    )
                )
    elif seed_tasks_path and os.path.exists(seed_tasks_path):
        output_dir = output_dir or os.path.dirname(os.path.abspath(seed_tasks_path))
        with open(seed_tasks_path, "r", encoding="utf-8") as seed_tasks_file:
            seed_tasks = [json.loads(l) for l in seed_tasks_file]
        seed_instruction_data = [
            {
                "instruction": t["instruction"],
                "input": t["instances"][0]["input"],
                "output": t["instances"][0]["output"],
            }
            for t in seed_tasks
        ]
    else:
        raise SystemExit(
            f"Error: both taxonomy ({taxonomy}) and ({seed_tasks_path}) do not exist."
        )

    seeds = len(seed_instruction_data)
    logger.debug(
        f"Loaded {seeds} human-written seed instructions from "
        f"{taxonomy or seed_tasks_path}"
    )
    if not seeds:
        raise SystemExit("Nothing to generate. Exiting.")

    test_data = []
    for seed_example in seed_instruction_data:
        user = seed_example["instruction"]
        if len(seed_example["input"]) > 0:
            user += "\n" + seed_example["input"]
        test_data.append(
            {
                "system": utils.SYSTEM_PROMPT,
                "user": user,
                "assistant": seed_example["output"],
            }
        )

    name = Path(model_name).stem  # Just in case it is a file path
    output_file = (
        f"generated_{name}_{datetime.now().replace(microsecond=0).isoformat()}.json"
    )
    output_file_train = (
        f"train_{name}_{datetime.now().replace(microsecond=0).isoformat()}.jsonl"
    )
    output_file_test = (
        f"test_{name}_{datetime.now().replace(microsecond=0).isoformat()}.jsonl"
    )
    logger.debug(f"Generating to: {os.path.join(output_dir, output_file)}")

    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    train_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        logger.debug(
            f"Loaded {len(machine_instruction_data)} machine-generated instructions"
        )

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
    all_instruction_tokens = [
        scorer._tokenizer.tokenize(inst) for inst in all_instructions
    ]

    prompt_template = check_prompt_file(prompt_file_path)
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            try:
                prompt_instructions = random.sample(
                    seed_instruction_data, num_prompt_instructions
                )
            except ValueError as exc:
                raise GenerateException(
                    f"There was a problem with the new data, please make sure the yaml is formatted correctly, and there is enough new data({num_prompt_instructions}+ Q&A) or decrease `num_prompt_instructions`, (currently {num_prompt_instructions})"
                ) from exc
            prompt = encode_prompt(prompt_instructions, prompt_template)
            batch_inputs.append(prompt)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            # Hard-coded to maximize length. Requests will be automatically adjusted.
            max_tokens=3072,
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
            new_instructions = post_process_gpt3_response(
                num_prompt_instructions, result
            )
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(
                instruction_data_entry["instruction"]
            )
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
            if max(rouge_scores) > rouge_threshold:
                continue
            keep += 1
            # Comment out extra info not currently being used:
            # instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            # instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        logger.debug(
            f"Request {request_idx} took {request_duration:.2f}s, "
            f"processing took {process_duration:.2f}s"
        )
        logger.debug(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, output_file))
        for synth_example in machine_instruction_data:
            user = synth_example["instruction"]
            if len(synth_example["input"]) > 0:
                user += "\n" + synth_example["input"]
            train_data.append(
                {
                    "system": utils.SYSTEM_PROMPT,
                    "user": user,
                    "assistant": synth_example["output"],
                }
            )
        # utils.jdump(train_data, os.path.join(output_dir, output_file_train))
        with open(
            os.path.join(output_dir, output_file_train), "w", encoding="utf-8"
        ) as outfile:
            for entry in train_data:
                json.dump(entry, outfile)
                outfile.write("\n")
        # utils.jdump(test_data, os.path.join(output_dir, output_file_test))
        with open(
            os.path.join(output_dir, output_file_test), "w", encoding="utf-8"
        ) as outfile:
            for entry in test_data:
                json.dump(entry, outfile)
                outfile.write("\n")

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")


def get_taxonomy_diff(repo="taxonomy"):
    repo = git.Repo(repo)
    untracked_files = [
        u for u in repo.untracked_files if splitext(u)[1].lower() in [".yaml", ".yml"]
    ]
    modified_files = [
        d.a_path
        for d in repo.index.diff(None)
        if splitext(d.a_path)[1].lower() in [".yaml", ".yml"]
    ]
    updated_taxonomy_files = untracked_files + modified_files

    return updated_taxonomy_files


def read_taxonomy_file(logger, file_path):
    seed_instruction_data = []
    warnings = 0
    errors = 0
    if splitext(file_path)[1] != ".yaml":
        logger.warn(
            f"Skipping {file_path}! Use lowercase '.yaml' extension instead."
        )
        warnings += 1
        return None, warnings, errors
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            contents = yaml.safe_load(file)
            if not contents:
                logger.warn(f"Skipping {file_path} because it is empty!")
                warnings += 1
                return None, warnings, errors
            tax_path = "->".join(file_path.split(os.sep)[1:-1])
            for t in get_seed_examples(contents):
                q = t["question"]
                a = t["answer"]
                if not q:
                    logger.warn(
                        f"Skipping entry in {file_path} "
                        + "because question is empty!"
                    )
                    warnings += 1
                if not a:
                    logger.warn(
                        f"Skipping entry in {file_path} "
                        + "because answer is empty!"
                    )
                    warnings += 1
                if not q or not a:
                    continue
                tax_path = "->".join(file_path.split(os.sep)[1:-1])
                seed_instruction_data.append(
                    {
                        "instruction": q,
                        "input": "",
                        "output": a,
                        "taxonomy_path": tax_path,
                    }
                )
    except Exception as e:
        errors += 1
        print(e.__repr__, " in ", file_path)
        logger.error(e)

    return seed_instruction_data, warnings, errors
