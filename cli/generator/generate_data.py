# Standard
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from os.path import splitext
from pathlib import Path
from typing import Optional
import json
import logging
import os
import random
import re
import string
import time

# Third Party
from jinja2 import Template
from rouge_score import rouge_scorer
import click
import tqdm
import yaml

# Local
from ..utils import get_documents, get_taxonomy_diff
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

DEFAULT_YAML_RULES = """\
extends: relaxed

rules:
  line-length:
    max: 120
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


class _WordPolicy:
    def __init__(self, data, allow_name, deny_name):
        self.allow = self.__orderedTuple(data.get(allow_name, ()))
        self.deny = self.__orderedTuple(data.get(deny_name, ()))

    def __orderedTuple(self, value):
        if not isinstance(value, list):
            return ()
        return tuple(sorted(value))

    def apply(self, data):
        for word in self.allow:
            data.discard(word)
        for word in self.deny:
            data.add(word)

    def __repr__(self):
        return "allow = " + repr(self.allow) + ", deny = " + repr(self.deny)


class _SeedGroups:
    def __init__(self, seed_instruction_data):
        grouping = self.__grouping = {}
        for inst in seed_instruction_data:
            policy = inst["word_policy"]
            entry_key = (policy.allow, policy.deny)
            if entry_key not in grouping:
                grouping[entry_key] = [inst]
            else:
                grouping[entry_key].append(inst)

    def get_group(self, seed):
        policy = seed["word_policy"]
        entry_key = (policy.allow, policy.deny)
        return self.__grouping[entry_key]

    def debug_info(self, logger):
        if not logger.isEnabledFor(logging.DEBUG):
            return
        groups = self.__grouping.items()
        size = len(groups)
        detail = []
        for group in groups:
            policy = group[0]
            count = len(group[1])
            if len(policy[0]) == 0 and len(policy[1]) == 0:
                policy = "default"
            detail.append(f"{policy}: {count} entries")
        logger.debug(f"Defined {size} seed groups: {detail}")


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
    """Encode multiple prompt instructions into a single string.
    If documents exist, randomly select one."""
    idx = 0
    document = None
    document_list = prompt_instructions[0].get("document")

    if document_list:
        document = random.choice(document_list)

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


def writeline2file(logfile, line):
    t = datetime.now().replace(microsecond=0).isoformat()
    with open(logfile, "a", encoding="utf-8") as fp:
        fp.write(f"{t} - {line}\n")


def post_process_gpt3_response(
    num_prompt_instructions, response, discarded_file, policy: _WordPolicy
):
    if response is None:
        return [], 0
    raw_instructions = (
        f"{num_prompt_instructions + 1}. Instruction:" + response.message.content
    )
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    discarded = 0
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        # if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
        #     continue
        idx += num_prompt_instructions + 1

        if not inst.strip():
            continue

        splitted_data = re.split(rf"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            writeline2file(
                discarded_file,
                f"Discarded instruction(didn't match expected format, idx={idx}): "
                + repr(inst),
            )
            discarded += 1
            continue
        inst = splitted_data[2].strip()
        prompt_input = splitted_data[4].strip()
        prompt_input = "" if prompt_input.lower() == "<noinput>" else prompt_input
        prompt_output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            writeline2file(
                discarded_file,
                "Discarded instruction(wrong number of words): " + repr(splitted_data),
            )
            discarded += 1
            continue
        # filter based on keywords that are not suitable for language models.
        exclude = set(_WORD_DENYLIST)

        # Removes allowed words, and adds denied words
        policy.apply(exclude)

        if any(find_word_in_string(word, inst) for word in exclude):
            writeline2file(
                discarded_file,
                "Discarded instruction(contained a word from the denylist): "
                + repr(splitted_data),
            )
            discarded += 1
            continue
        # We found that the model tends to add "write a program" to some existing instructions
        # which lead to a lot of such instructions and it's confusing whether the model needs
        # to write a program or directly output the result, so here we filter them out.
        # NOTE: this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            writeline2file(
                discarded_file,
                "Discarded instruction(began with 'Write a program'): "
                + repr(splitted_data),
            )
            discarded += 1
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            writeline2file(
                discarded_file,
                "Discarded instruction(began with punctuation): " + repr(splitted_data),
            )
            discarded += 1
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            writeline2file(
                discarded_file,
                "Discarded instruction(began with non-ascii): " + repr(splitted_data),
            )
            discarded += 1
            continue
        instructions.append(
            {"instruction": inst, "input": prompt_input, "output": prompt_output}
        )
    return instructions, discarded


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
    output_file_discarded,
):
    batch_inputs = []
    seed_groups = _SeedGroups(instruction_data_pool)
    seed_groups.debug_info(logger)
    selected_data = instruction_data_pool
    for _ in range(request_batch_size):
        # only sampling from the seed tasks
        try:
            if selected_data is instruction_data_pool:
                # pick from the whole set, at random to get the group, then pick again only within group
                pick = random.choice(selected_data)
                selected_data = seed_groups.get_group(pick)

            prompt_instructions = random.sample(selected_data, num_prompt_instructions)
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
    total_discarded = 0
    for result in results:
        # make sure the generated instruction carried over extra fields
        prompt_ins_0 = prompt_instructions[0]
        policy = prompt_ins_0["word_policy"]
        new_instructions, discarded = post_process_gpt3_response(
            num_prompt_instructions, result, output_file_discarded, policy
        )
        total_discarded += discarded
        # make sure the generated instruction carried over extra fields
        for new_ins in new_instructions:
            new_ins["taxonomy_path"] = prompt_ins_0["taxonomy_path"]
            new_ins["task_description"] = prompt_ins_0["task_description"]
            new_ins["document"] = prompt_ins_0["document"]
            new_ins["word_policy"] = policy
        instruction_data += new_instructions

    post_process_duration = time.time() - post_process_start
    logger.debug(
        f"Request {request_idx} took {request_duration:.2f}s, "
        f"post-processing took {post_process_duration:.2f}s"
    )

    return instruction_data, total_discarded


def generate_data(
    logger,
    api_base,
    yaml_rules: Optional[str] = None,
    output_dir: Optional[str] = None,
    taxonomy: Optional[str] = None,
    taxonomy_base: Optional[str] = None,
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
        seed_instruction_data = read_taxonomy(
            logger, taxonomy, taxonomy_base, yaml_rules
        )
    else:
        raise SystemExit(f"Error: taxonomy ({taxonomy}) does not exist.")

    seeds = len(seed_instruction_data)
    logger.debug(f"Loaded {seeds} human-written seed instructions from {taxonomy}")
    if not seeds:
        raise SystemExit("Nothing to generate. Exiting.")

    def unescape(s):
        return bytes(s, "utf-8").decode("utf-8")

    test_data = []
    for seed_example in seed_instruction_data:
        user = seed_example["instruction"]
        if len(seed_example["input"]) > 0:
            user += "\n" + seed_example["input"]
        try:
            test_data.append(
                {
                    "system": utils.SYSTEM_PROMPT,
                    "user": unescape(user),
                    "assistant": unescape(seed_example["output"]),
                }
            )
        except TypeError as exc:
            click.secho(
                f"Error reading seed examples: {exc}. Please make sure your answers are verbose enough.",
                fg="red",
            )
            raise click.exceptions.Exit(1)

    name = Path(model_name).stem  # Just in case it is a file path
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    output_file = f"generated_{name}_{date_suffix}.json"
    output_file_train = f"train_{name}_{date_suffix}.jsonl"
    output_file_test = f"test_{name}_{date_suffix}.jsonl"
    output_file_discarded = os.path.join(
        output_dir, f"discarded_{name}_{date_suffix}.log"
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
    if console_output:
        print(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )

    all_taxonomy_paths = list(set(e["taxonomy_path"] for e in seed_instruction_data))
    total_discarded = 0
    total_rouged = 0
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
        instruction_data, discarded = get_instructions_from_model(
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
            output_file_discarded,
        )
        total_discarded += discarded
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
                total_rouged += 1
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
        logger.debug(
            f"Generated {total} instructions(discarded {discarded}), rouged {total - keep}, kept {keep} instructions"
        )
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

    if total_discarded or total_rouged:
        logger.info(
            f"{len(machine_instruction_data)} instructions generated, {total_discarded} discarded due to format (see {output_file_discarded}), {total_rouged} discarded due to rouge score"
        )
    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")


# pylint: disable=broad-exception-caught
def read_taxonomy_file(logger, file_path, yaml_rules: Optional[str] = None):
    # pylint: disable=C0415
    # Third Party
    from yamllint import linter
    from yamllint.config import YamlLintConfig

    seed_instruction_data = []
    warnings = 0
    errors = 0
    # file should end with ".yaml" explicitly
    if splitext(file_path)[1] != ".yaml":
        logger.warn(f"Skipping {file_path}! Use lowercase '.yaml' extension instead.")
        warnings += 1
        return None, warnings, errors
    # read file if extension is correct
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            # do general YAML linting if specified
            if yaml_rules is not None:
                try:
                    yaml_config = YamlLintConfig(file=yaml_rules)
                except FileNotFoundError:
                    logger.debug(f"Cannot find {yaml_rules}. Using default rules.")
                    yaml_config = YamlLintConfig(DEFAULT_YAML_RULES)
            else:
                yaml_config = YamlLintConfig(DEFAULT_YAML_RULES)
            for p in linter.run(file, yaml_config):
                errors += 1
                logger.error(
                    f"error found in file {file.name}: {p.desc} {p.line} {p.rule}"
                )
                return None, warnings, errors
        # do more explict checking of file contents
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
            # get seed instruction data
            tax_path = "->".join(file_path.split(os.sep)[1:-1])
            task_description = contents.get("task_description")
            documents = contents.get("document")
            if documents:
                documents = get_documents(documents)
                logger.info("Content from git repo fetched")
            for t in get_seed_examples(contents):
                q = t["question"]
                a = t["answer"]
                c = t.get("context")
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
                        "input": "" if not c else c,
                        "output": a,
                        "taxonomy_path": tax_path,
                        "task_description": task_description,
                        "document": documents,
                        "word_policy": _WordPolicy(t, "allow_words", "deny_words"),
                    }
                )
    except Exception as e:
        errors += 1
        raise (f"Exception {e} raised in {file_path}") from e

    return seed_instruction_data, warnings, errors


def read_taxonomy(logger, taxonomy, taxonomy_base, yaml_rules):
    seed_instruction_data = []
    is_file = os.path.isfile(taxonomy)
    if is_file:  # taxonomy is file
        seed_instruction_data, warnings, errors = read_taxonomy_file(
            logger, taxonomy, yaml_rules
        )
        if warnings:
            logger.warn(
                f"{warnings} warnings (see above) due to taxonomy file not (fully) usable."
            )
        if errors:
            raise SystemExit(yaml.YAMLError("Taxonomy file with errors! Exiting."))
    else:  # taxonomy is dir
        # Gather the new or changed YAMLs using git diff
        try:
            updated_taxonomy_files = get_taxonomy_diff(taxonomy, taxonomy_base)
        except NameError as exc:
            raise utils.GenerateException("`git` binary not found") from exc
        total_errors = 0
        total_warnings = 0
        if updated_taxonomy_files:
            logger.debug("Found new taxonomy files:")
            for e in updated_taxonomy_files:
                logger.debug(f"* {e}")
        for f in updated_taxonomy_files:
            file_path = os.path.join(taxonomy, f)
            data, warnings, errors = read_taxonomy_file(logger, file_path, yaml_rules)
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
