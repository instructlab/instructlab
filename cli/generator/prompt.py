_DEFAULT_PROMPT_TEMPLATE = """\
You are asked to come up with a set of 5 diverse task instructions under {taxonomy}{task_description_str}. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.
3. The type of instructions should not have topic diversity. The list should follow the same topic and category.
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The instructions should be in English.
6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
7. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
8. Not all instructions require input. For example, when a instruction asks about some general information, "what is the highest peak in the world", it is not necessary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
9. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.

List of 5 tasks:
"""

_DEFAULT_PROMPT_TEMPLATE_DOCUMENT = """\
You are asked to come up with a set of 5 diverse task instructions under {taxonomy}{task_description_str}. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.
3. The type of instructions should be similar to provided examples. The generated instruction and the output should be grounded in the provided document.
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The instructions should be in English.
6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
7. The output should be an appropriate response to the input and the instruction. Long outputs are preferable.

Based on below document provide a list of 5 tasks:

Document:
{document}

Here are some examples to help you understand the type of questions that are asked for this document:
"""


def get_default_template():
    return _DEFAULT_PROMPT_TEMPLATE


def get_document_template():
    return _DEFAULT_PROMPT_TEMPLATE_DOCUMENT
