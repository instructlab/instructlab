import json

SYS_PROMPT = "You are Labrador, an AI language model developed by IBM DMF (Data Model Factory) Alignment Team. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."

def format_text(obj):
    return f"""\
<|system|>
{obj['system']}
<|user|>
{obj['user']}
<|assistant|>
{obj['assistant']}<|endoftext|>\
"""

def make_data():
    is_lab_gen = True
    if is_lab_gen:
        # This branch uses data from `lab generate`
        # train_gen.jsonl and test_gen.jsonl are the two files produced by `lab generate`
        for fn in ['data_puns/train_gen.jsonl', 'data_puns/test_gen.jsonl']:

            # Load the JSON Lines file
            with open(fn, 'r') as f:
                data = [json.loads(line) for line in f]

            # Add the "text" field with value "x" to each object
            data_new = []
            for obj in data:
                data_new.append({'text': format_text(obj)})

            # Save the modified objects back to the JSON Lines file
            # TODO make the naming conversion more robust
            if "train" in fn:
                n = len(data_new) // 10 * 8
                with open(fn.replace("_gen", ""), 'w') as f:
                    for obj in data_new[:n]:
                        f.write(json.dumps(obj) + '\n')
                with open(fn.replace("_gen", "").replace("train", "valid"), 'w') as f:
                    for obj in data_new[n:]:
                        f.write(json.dumps(obj) + '\n')
            else:
                with open(fn.replace("_gen", ""), 'w') as f:
                    for obj in data_new:
                        f.write(json.dumps(obj) + '\n')
    else:
        # This branch is to use Shiv generated data
        # You can ignore for now
        fn = "data_puns_shiv/raw.jsonl"

        # Load the JSON Lines file
        with open(fn, 'r') as f:
            data = [json.loads(line) for line in f]

        # Add the "text" field with value "x" to each object
        data_new = []
        for obj in data:
            obj_new = {}
            obj_new["system"] = SYS_PROMPT
            obj_new["user"] = obj["inputs"]
            obj_new["assistant"] = obj["targets"]
            data_new.append(obj_new | {'text': format_text(obj_new)})

        # Save the modified objects back to the JSON Lines file
        n = len(data_new) // 10 * 7
        m = len(data_new) // 10 * 2 + n
        with open(fn.replace("raw.jsonl", "train.jsonl"), 'w') as f:
            for obj in data_new[:n]:
                f.write(json.dumps(obj) + '\n')
        with open(fn.replace("raw.jsonl", "valid.jsonl"), 'w') as f:
            for obj in data_new[n:m]:
                f.write(json.dumps(obj) + '\n')
        with open(fn.replace("raw.jsonl", "test.jsonl"), 'w') as f:
            for obj in data_new[:m]:
                f.write(json.dumps(obj) + '\n')

if __name__ == "__main__":
    make_data()