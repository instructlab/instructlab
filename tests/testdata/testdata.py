# SPDX-License-Identifier: Apache-2.0

documents = [
    """Knowledge is an awareness of facts, a familiarity with individuals and situations,
      or a practical skill. Knowledge of facts, also called propositional knowledge, is often characterized
      as true belief that is distinct from opinion or guesswork by virtue of justification.
      While there is wide agreement among philosophers that propositional knowledge is a form of true belief,
      many controversies focus on justification. This includes questions like how to understand justification,
      whether it is needed at all, and whether something else besides it is needed.
      These controversies intensified in the latter half of the 20th century due to a series of thought experiments
      called Gettier cases that provoked alternative definitions."""
]

knowledge_seed_instruction = [
    {
        "instruction": "What is this knowledge about?",
        "input": "",
        "output": "It's a knowledge that makes the tests more knowledgeable",
        "taxonomy_path": "knowledge->textbook->puns-copy->general",
        "task_description": "For knowledge tests",
        "document": documents,
    },
    {
        "instruction": "question2",
        "input": "",
        "output": "answer2",
        "taxonomy_path": "knowledge->textbook->puns-copy->general",
        "task_description": "For knowledge tests",
        "document": documents,
    },
    {
        "instruction": "question3",
        "input": "",
        "output": "answer3",
        "taxonomy_path": "knowledge->textbook->puns-copy->general",
        "task_description": "For knowledge tests",
        "document": documents,
    },
    {
        "instruction": "question4",
        "input": "",
        "output": "answer4",
        "taxonomy_path": "knowledge->textbook->puns-copy->general",
        "task_description": "For knowledge tests",
        "document": documents,
    },
    {
        "instruction": "Question5",
        "input": "",
        "output": "Answer5",
        "taxonomy_path": "knowledge->textbook->puns-copy->general",
        "task_description": "For knowledge tests",
        "document": documents,
    },
]

generate_data_return_value = [
    {
        "instruction": "3. Tell me a pun about water.",
        "input": "",
        "output": "Why did the scarecrow win an award?\nBecause he was outstanding in his field!",
        "taxonomy_path": "compositional_skills->writing->freeform->jokes->puns-copy->general",
        "task_description": "to teach a large language model to come up with puns",
        "document": None,
    },
    {
        "instruction": "4. Give me a pun about books.",
        "input": "",
        "output": "Why don't books ever get lost on the shelf?\nBecause they are always on the cover!",
        "taxonomy_path": "compositional_skills->writing->freeform->jokes->puns-copy->general",
        "task_description": "to teach a large language model to come up with puns",
        "document": None,
    },
]
