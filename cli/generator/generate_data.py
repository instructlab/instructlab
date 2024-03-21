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

# Third Party
from jinja2 import Template

try:
    # Third Party
    import git
except ImportError:
    pass

# Third Party
from rouge_score import rouge_scorer
import gitdb

# import numpy as np
import tqdm
import yaml

# Local
from . import utils

DEFAULT_PROMPT_TEMPLATE = """\
You are asked to come up with a set of 5 diverse task instructions under {{taxonomy}}{{" for the task \\"%s\\""|format(task_description)  if task_description}}. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.
{% if not document -%}
3. The type of instructions should not have topic diversity. The list should follow the same topic and category.
{% else -%}
3. The type of instructions should be similar to provided examples. The generated instruction and the output should be grounded in the provided document.
{% endif -%}
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The instructions should be in English.
6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
{% if not document -%}
7. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
8. Not all instructions require input. For example, when an instruction asks about some general information, "what is the highest peak in the world", it is not necessary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
9. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.
{% else -%}
7. The output should be an appropriate response to the input and the instruction. Long outputs are preferable.
{% endif %}

{% if not document -%}
List of 5 tasks:
{% else -%}
Based on below document provide a list of 5 tasks:

Document:
{{document}}

Here are some examples to help you understand the type of questions that are asked for this document:
{% endif -%}
"""


_WORD_DENYLIST = [
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
    prompt_template = prompt_template.strip() + "\n"
    return prompt_template


def encode_prompt(prompt_instructions, prompt):
    """Encode multiple prompt instructions into a single string."""
    idx = 0
    document = prompt_instructions[0].get("document")
    prompt = Template(prompt).render(
        taxonomy=prompt_instructions[0]["taxonomy_path"],
        task_description=prompt_instructions[0]["task_description"],
        document=document,
    )

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
        if any(find_word_in_string(word, inst) for word in _WORD_DENYLIST):
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


def get_version(contents):
    if "version" in contents:
        version = contents["version"]
        try:
            version = int(version)
        except ValueError:
            pass
        return version
    return 1


def get_instructions_from_model(
    logger,
    request_idx,
    instruction_data_pool,
    prompt_template,
    api_base,
    api_key,
    model_name,
    num_prompt_instructions,
    request_batch_size,
    temperature,
    top_p,
):
    batch_inputs = []
    for _ in range(request_batch_size):
        # only sampling from the seed tasks
        try:
            prompt_instructions = random.sample(
                instruction_data_pool, num_prompt_instructions
            )
        except ValueError as exc:
            raise GenerateException(
                f"There was a problem with the new data, please make sure the "
                f"yaml is formatted correctly, and there is enough "
                f"new data({num_prompt_instructions}+ Q&A))"
            ) from exc
        prompt = encode_prompt(prompt_instructions, prompt_template)
        batch_inputs.append(prompt)
    decoding_args = utils.OpenAIDecodingArguments(
        temperature=temperature,
        n=1,
        # Hard-coded to maximize length.
        # Requests will be automatically adjusted.
        max_tokens=3072,
        top_p=top_p,
        stop=["\n5", "5.", "5."],
    )
    request_start = time.time()
    results = utils.openai_completion(
        api_base=api_base,
        api_key=api_key,
        prompts=batch_inputs,
        model_name=model_name,
        batch_size=request_batch_size,
        decoding_args=decoding_args,
    )
    request_duration = time.time() - request_start

    post_process_start = time.time()
    instruction_data = []
    for result in results:
        new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
        # make sure the generated instruction carried over extra fields
        prompt_ins_0 = prompt_instructions[0]
        for new_ins in new_instructions:
            new_ins["taxonomy_path"] = prompt_ins_0["taxonomy_path"]
            new_ins["task_description"] = prompt_ins_0["task_description"]
            new_ins["document"] = prompt_ins_0["document"]
        instruction_data += new_instructions

    post_process_duration = time.time() - post_process_start
    logger.debug(
        f"Request {request_idx} took {request_duration:.2f}s, "
        f"post-processing took {post_process_duration:.2f}s"
    )

    return instruction_data


def generate_data(
    logger,
    api_base,
    output_dir: Optional[str] = None,
    taxonomy: Optional[str] = None,
    taxonomy_base: Optional[str] = None,
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
    console_output=True,
    api_key: Optional[str] = None,
):
    seed_instruction_data = []
    generate_start = time.time()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # check taxonomy first then seed_tasks_path
    # throw an error if both not found
    # pylint: disable=broad-exception-caught,raise-missing-from
    if taxonomy and os.path.exists(taxonomy):
        seed_instruction_data = read_taxonomy(logger, taxonomy, taxonomy_base)
    else:
        raise SystemExit(f"Error: taxonomy ({taxonomy}) does not exist.")

    seeds = len(seed_instruction_data)
    logger.debug(
        f"Loaded {seeds} human-written seed instructions from "
        f"{taxonomy or seed_tasks_path}"
    )
    if not seeds:
        raise SystemExit("Nothing to generate. Exiting.")

    def unescape(s):
        return bytes(s, "utf-8").decode("utf-8")

    test_data = []
    for seed_example in seed_instruction_data:
        user = seed_example["instruction"]
        if len(seed_example["input"]) > 0:
            user += "\n" + seed_example["input"]
        test_data.append(
            {
                "system": utils.SYSTEM_PROMPT,
                "user": unescape(user),
                "assistant": unescape(seed_example["output"]),
            }
        )

    name = Path(model_name).stem  # Just in case it is a file path
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    output_file = f"generated_{name}_{date_suffix}.json"
    output_file_train = f"train_{name}_{date_suffix}.jsonl"
    output_file_test = f"test_{name}_{date_suffix}.jsonl"
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
    if console_output:
        print(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )
    all_taxonomy_paths = list(set(e["taxonomy_path"] for e in seed_instruction_data))
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        # Pick taxonomy path
        selected_taxonomy = all_taxonomy_paths[request_idx % len(all_taxonomy_paths)]
        logger.info(f"Selected taxonomy path {selected_taxonomy}")
        # Filter the pool
        instruction_data_pool = [
            e
            for e in seed_instruction_data + machine_instruction_data
            if e["taxonomy_path"] == selected_taxonomy
        ]
        instruction_data = get_instructions_from_model(
            logger,
            request_idx,
            instruction_data_pool,
            prompt_template,
            api_base,
            api_key,
            model_name,
            num_prompt_instructions,
            request_batch_size,
            temperature,
            top_p,
        )

        total = len(instruction_data)
        keep = 0
        assess_start = time.time()
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenized instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(
                instruction_data_entry["instruction"]
            )
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            instruction_data_entry["taxonomy_path"] = selected_taxonomy
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
            if console_output:
                print(
                    f"Q> {instruction_data_entry['instruction']}\nI> {instruction_data_entry['input']}\nA> {instruction_data_entry['output']}\n"
                )
        progress_bar.update(keep)
        assess_duration = time.time() - assess_start
        logger.debug(f"Assessing generated samples took {assess_duration:.2f}s")
        logger.debug(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, output_file))
        for synth_example in machine_instruction_data:
            user = synth_example["instruction"]
            if len(synth_example["input"]) > 0:
                user += "\n" + synth_example["input"]
            train_data.append(
                {
                    "system": utils.SYSTEM_PROMPT,
                    "user": unescape(user),
                    "assistant": unescape(synth_example["output"]),
                }
            )
        # utils.jdump(train_data, os.path.join(output_dir, output_file_train))
        with open(
            os.path.join(output_dir, output_file_train), "w", encoding="utf-8"
        ) as outfile:
            for entry in train_data:
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write("\n")
        # utils.jdump(test_data, os.path.join(output_dir, output_file_test))
        with open(
            os.path.join(output_dir, output_file_test), "w", encoding="utf-8"
        ) as outfile:
            for entry in test_data:
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write("\n")

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")


def istaxonomyfile(fn):
    topleveldir = fn.split("/")[0]
    if fn.endswith(".yaml") and topleveldir in ["compositional_skills", "knowledge"]:
        return True
    return False


def get_taxonomy_diff(repo="taxonomy", base="origin/main"):
    repo = git.Repo(repo)
    untracked_files = [u for u in repo.untracked_files if istaxonomyfile(u)]

    branches = [b.name for b in repo.branches]

    head_commit = None
    if "/" in base:
        re_git_branch = re.compile(f"remotes/{base}$", re.MULTILINE)
    elif base in branches:
        re_git_branch = re.compile(f"{base}$", re.MULTILINE)
    else:
        try:
            head_commit = repo.commit(base)
        except gitdb.exc.BadName as exc:
            raise SystemExit(
                yaml.YAMLError(
                    f'Couldn\'t find the taxonomy git ref "{base}" from the current HEAD'
                )
            ) from exc

    # Move backwards from HEAD until we find the first commit that is part of base
    # then we can take our diff from there
    current_commit = repo.commit("HEAD")
    while not head_commit:
        branches = repo.git.branch("-a", "--contains", current_commit.hexsha)
        if re_git_branch.findall(branches):
            head_commit = current_commit
            break
        try:
            current_commit = current_commit.parents[0]
        except IndexError as exc:
            raise SystemExit(
                yaml.YAMLError(
                    f'Couldn\'t find the taxonomy base branch "{base}" from the current HEAD'
                )
            ) from exc

    modified_files = [
        d.b_path
        for d in head_commit.diff(None)
        if not d.deleted_file and istaxonomyfile(d.b_path)
    ]

    updated_taxonomy_files = list(set(untracked_files + modified_files))
    return updated_taxonomy_files


# pylint: disable=broad-exception-caught
def read_taxonomy_file(logger, file_path):
    seed_instruction_data = []
    warnings = 0
    errors = 0
    if splitext(file_path)[1] != ".yaml":
        logger.warn(f"Skipping {file_path}! Use lowercase '.yaml' extension instead.")
        warnings += 1
        return None, warnings, errors
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            contents = yaml.safe_load(file)
            if not contents:
                logger.warn(f"Skipping {file_path} because it is empty!")
                warnings += 1
                return None, warnings, errors
            version = get_version(contents)
            if version != 1:
                logger.warn(
                    f"Skipping {file_path} because its version, {version}, is not understood. You may need a newer version of this command."
                )
                warnings += 1
                return None, warnings, errors
            tax_path = "->".join(file_path.split(os.sep)[1:-1])
            task_description = contents.get("task_description")
            document = contents.get("document")
            for t in get_seed_examples(contents):
                q = str(t["question"])
                a = str(t["answer"])
                c = str(t.get("context", ""))
                if not q:
                    logger.warn(
                        f"Skipping entry in {file_path} " + "because question is empty!"
                    )
                    warnings += 1
                if not a:
                    logger.warn(
                        f"Skipping entry in {file_path} " + "because answer is empty!"
                    )
                    warnings += 1
                if not q or not a:
                    continue
                seed_instruction_data.append(
                    {
                        "instruction": q,
                        "input": c,
                        "output": a,
                        "taxonomy_path": tax_path,
                        "task_description": task_description,
                        "document": document,
                    }
                )
    except Exception as e:
        errors += 1
        print(e.__repr__, " in ", file_path)
        logger.error(e)

    return seed_instruction_data, warnings, errors


def read_taxonomy(logger, taxonomy, taxonomy_base):
    seed_instruction_data = []
    is_file = os.path.isfile(taxonomy)
    if is_file:
        seed_instruction_data, warnings, errors = read_taxonomy_file(logger, taxonomy)
        if warnings:
            logger.warn(
                f"{warnings} warnings (see above) due to taxonomy file not (fully) usable."
            )
        if errors:
            raise SystemExit(yaml.YAMLError("Taxonomy file with errors! Exiting."))
    else:  # taxonomy is_dir
        # Gather the new or changed YAMLs using git diff
        try:
            updated_taxonomy_files = get_taxonomy_diff(taxonomy, taxonomy_base)
        except NameError as exc:
            raise utils.GenerateException("`git` binary not found") from exc
        total_errors = 0
        total_warnings = 0
        if updated_taxonomy_files:
            logger.info("Found new taxonomy files :")
            for e in updated_taxonomy_files:
                logger.info(f"* {e}")
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
                yaml.YAMLError(f"{total_errors} taxonomy files with errors! Exiting.")
            )
    return seed_instruction_data
