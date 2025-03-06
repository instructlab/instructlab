# DK-Bench (Domain-Knowledge Bench)

## Background

Current knowledge evaluation in InstructLab, MMLU-Branch, evaluates models on their answers to multiple choice questions.
In MMLU-Branch, the model either gets the answer right or wrong, but there's no in between to give the model credit on moderately correct or incorrect answers.

Additionally, there was no way to bring a custom evaluation data set to run on InstructLab trained models.

DK-Bench (Domain-Knowledge Bench) provides the ability to bring custom evaluation questions and score the models answers on a sliding scale.

## Description

DK-Bench provides a flexible way for users to bring custom evaluation questions, reference answers, and collect a model's answers to the questions.
The response to each of question is given a score from 1-5 by a judge model.
The current default judge model is GPT-4o by OpenAI.

### User Input

To run DK-Bench, users must provide a `.jsonl` file containing every question they wish to ask a model to answer and evaluate.

Each line must be valid JSON, and contain the entries `user_input` for the question, and `reference` for the ground truth answer to the question.

Here is an example of a `.jsonl` file that can be used in DK-Bench with 2 questions:

```json
{"user_input":"What is the capital of Canada?","reference":"The capital of Canada is Ottawa."}
{"user_input":"What is the capital of Mexico?","reference":"The capital of Mexico is Mexico City."}
```

### Rubric

Each response given by a model is compared to the reference answer and graded on the following 1 - 5 scale by the judge model:

| Score | Criteria |
| ---------------- | -------- |
| 1 | The response is entirely incorrect, irrelevant, or does not align with the reference in any meaningful way |
| 2 | The response partially matches the reference but contains major errors, significant omissions, or irrelevant information |
| 3 | The response aligns with the reference overall but lacks sufficient detail, clarity, or contains minor inaccuracies |
| 4 | The response is mostly accurate, aligns closely with the reference, and contains only minor issues or omissions |
| 5 | The response is fully accurate, completely aligns with the reference, and is clear, thorough, and detailed |

### Output

After running DK-Bench, users will see a report with the question by question score, the average, and the total for their run.

```shell
# DK-BENCH REPORT

## MODEL: granite-7b-lab

Question #1:     5/5
Question #2:     5/5
Question #3:     5/5
Question #4:     5/5
Question #5:     2/5
Question #6:     3/5
Question #7:     2/5
Question #8:     3/5
Question #9:     5/5
Question #10:     5/5
----------------------------
Average Score:   4.00/5
Total Score:     40/50

Responses and scores are written to:
~/.local/share/instructlab/internal/eval_data/dk_bench/granite-7b-lab/results_2025-02-04T16:18:02.337010.jsonl
```

The original question, the responses a model gave for that specific question, the reference answer, and the score of the judge model are all output to an output file.

The output file can be `.jsonl` (the default and seen below), `.csv`, or `.xlsx`. This can be set using the `--output-file-formats` flag or in the `dk_bench` section of the configuration file.

```json
{"user_input":"What is the capital of Canada?","response":"The capital of Canada is Ottawa.","reference":"The capital of Canada is Ottawa.","scores":5}
{"user_input":"What is the capital of Mexico?","response":"The capital of Mexico is Mexico City.","reference":"The capital of Mexico is Mexico City.","scores":5}
```

## Examples

> [!NOTE]
> The DK-Bench evaluation requires the environment variables `OPENAI_API_KEY` to be set. It will not run if this variable is not set.
> To set your OpenAI API Key in the shell run:
> `export OPENAI_API_KEY=<API KEY HERE>`

DK-Bench has many flags that are applied when the `ilab model evaluate` command is used to run DK-Bench.

Here is the most basic example of using DK-Bench, by passing the path to the `.jsonl` dataset into `--input-questions`, and the path to a local model via `--model`:

```shell
ilab model evaluate --benchmark dk_bench --input-questions /home/use/path/to/questions.jsonl --model ~/.cache/instructlab/models/instructlab/granite-7b-lab
```

```shell
(venv) $ ilab model evaluate --benchmark dk_bench --input-questions /home/use/path/to/questions.jsonl --model ~/.cache/instructlab/models/instructlab/granite-7b-lab
INFO 2025-02-04 16:15:55,707 numexpr.utils:162: NumExpr defaulting to 16 threads.
INFO 2025-02-04 16:15:55,955 datasets:59: PyTorch version 2.4.0 available.
INFO 2025-02-04 16:15:58,231 instructlab.model.evaluate:585: Using local model found at '~/.cache/instructlab/models/instructlab/granite-7b-lab' for '--model'
...
INFO 2025-02-04 16:16:35,583 instructlab.model.backends.vllm:145: vLLM engine successfully started at http://127.0.0.1:52591/v1
~/instructlab/venv/lib64/python3.11/site-packages/instructlab/eval/ragas.py:220: LangChainDeprecationWarning: The class `ChatOpenAI` was deprecated in LangChain 0.0.10 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import ChatOpenAI``.
  critic_lm = ChatOpenAI(model=judge_model_name, api_key=judge_openai_api_key)
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:06<00:00,  1.44it/s]
INFO 2025-02-04 16:17:56,038 instructlab.model.backends.vllm:494: Waiting for GPU VRAM reclamation...█████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:06<00:00,  1.21it/s]

# DK-BENCH REPORT

## MODEL: granite-7b-lab

Question #1:     5/5
Question #2:     5/5
Question #3:     5/5
Question #4:     5/5
Question #5:     2/5
Question #6:     3/5
Question #7:     2/5
Question #8:     3/5
Question #9:     5/5
Question #10:     5/5
----------------------------
Average Score:   4.00/5
Total Score:     40/50

Responses and scores are written to:
~/.local/share/instructlab/internal/eval_data/dk_bench/granite-7b-lab/results_2025-02-04T16:18:02.337010.jsonl

INFO 2025-02-04 16:18:02,337 instructlab.model.evaluate:176: ᕦ(òᴗóˇ)ᕤ Model Evaluation with DK-Bench completed! ᕦ(òᴗóˇ)ᕤ
```

You can specify multiple output formats with the `--output-file-formats` flag:

```shell
(venv) $ ilab model evaluate --benchmark dk_bench --input-questions questions.jsonl --model ~/.cache/instructlab/models/instructlab/granite-7b-lab --output-file-formats csv,xlsx --output-dir ~/dk-bench-results-dir
INFO 2025-02-04 16:15:55,707 numexpr.utils:162: NumExpr defaulting to 16 threads.
...
Responses and scores are written to:
~/dk-bench-results-dir/granite-7b-lab/results_2025-02-04T16:18:02.337010.csv
~/dk-bench-results-dir/granite-7b-lab/results_2025-02-04T16:18:02.337010.xlsx

INFO 2025-02-04 16:18:02,337 instructlab.model.evaluate:176: ᕦ(òᴗóˇ)ᕤ Model Evaluation with DK-Bench completed! ᕦ(òᴗóˇ)ᕤ
```

## Configurations & Flags for Advanced Users

You can adjust how the student model generates data with the following flags:

The `--temperature` flag determines how creative the model's generations are. With `0.0` (the default) the model will greedily pick the likeliest token, and higher values will result in the model's outputs being more random.
The `--system-prompt` flag sets the system prompt for the student model when answering the questions. If none is provided, we try to guess the best default based on the model being used.
The `--judge-model` flag sets the judge model giving out the score for each comparison of response and reference answer. The default is `gpt-4o`. The judge model currently needs to be an OpenAI model.

All of the flags for DK-Bench can be set in the config file as well.

## Running DK-Bench from pre-generated responses

If you've already generated the answers ahead of time, or would like to avoid spinning up a student model for this benchmark, you can also pass a `.jsonl` file containing the model's responses in the `response` field.

```shell
(venv) $ ilab model evaluate --benchmark dk_bench --input-questions /home/user/path/to/questions-with-responses.jsonl
INFO 2025-02-04 16:15:55,707 numexpr.utils:162: NumExpr defaulting to 16 threads.
...
# DK-BENCH REPORT

## MODEL: no-model-provided

Question #1:     1/5
Question #2:     5/5
----------------------------
Average Score:   3.00/5
Total Score:     6/10

Responses and scores are written to:
~/.local/share/instructlab/internal/eval_data/dk_bench/no-model-provided/results_2025-02-04T16:18:02.337010.jsonl

INFO 2025-02-04 16:18:02,337 instructlab.model.evaluate:176: ᕦ(òᴗóˇ)ᕤ Model Evaluation with DK-Bench completed! ᕦ(òᴗóˇ)ᕤ
```

Here `questions-with-responses.jsonl` include `response` fields:

```json
{"user_input":"What is the capital of Canada?","response":"The capital of Canada is Toronto.","reference":"The capital of Canada is Ottawa."}
{"user_input":"What is the capital of Mexico?","response":"The capital of Mexico is Mexico City.","reference":"The capital of Mexico is Mexico City."}
```
