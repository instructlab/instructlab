from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator, ValidationError
from ruamel.yaml import YAML

class YorNValidator(Validator):
    def validate(self, document):
        text = document.text

        if text and not (text == 'y' or text == 'n'):
            raise ValidationError(message='Answer must be y or n',cursor_position=0)

class KorSValidator(Validator):
    def validate(self, document):
        text = document.text

        if text and not (text == 'k' or text == 's'):
            raise ValidationError(message='Answer must be k or s',cursor_position=0)

def run_feed_cli(taxonomy_path, num_questions):
    print("Welcome to Lab Feed")
    task_type = prompt('Are you adding new knowledge (k) or a new skill (s)? Input k or s: ', validator=KorSValidator())

    if task_type == 'k':
        taxonomy_path += '/knowledge'
    else:
        taxonomy_path += '/compositional_skills'

    file_path = taxonomy_path + '/qna.yaml'

    data = {}

    data["task_description"] = prompt('description of task: ')
    data["created_by"] = prompt('github user name: ')
    data["seed_examples"] = []
    i = 0
    for i in range(num_questions):
        num_questions = str(num_questions)
        print(f"======= Addition {i+1} of {num_questions} =======")
        context = prompt('context: ')
        question = prompt('question: ')
        answer = prompt('answer: ')
        attribution_exists = prompt('attribution (y/n): ', validator=YorNValidator())
        attribution = {}
        example = {}
        example["question"] = question
        example["answer"] = answer
        if len(context) != 0:
            example["context"] = context
        if attribution_exists == 'y':
            attribution["source"] = prompt('source: ')
            attribution["license"] = prompt('license: ')
            example["attribution"] = []
            example["attribution"].append(attribution)
        data["seed_examples"].append(example)

    file = open(file_path, 'w')
    yaml = YAML()
    yaml.explicit_start = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.dump(data, file)
    file.close()
    if task_type == 'k':
        print(f"wrote new knowledge to {file_path}")
    else:
        print(f"wrote new skill to {file_path}")
