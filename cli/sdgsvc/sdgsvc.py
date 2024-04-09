# Standard
import json
import os
import time

# Third Party
import requests
import yaml


def datagen_svc(url, taxonomy_files, output_dir, cert, verify, num_samples):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_files = []
    for tf in taxonomy_files:
        if not os.path.exists(tf):
            raise FileNotFoundError(f"File {tf} not found")

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        # Open file, parse as yaml
        with open(tf, "r", encoding="utf-8") as f:
            tf_data = yaml.safe_load(f)
            # TODO, this is the only value the service expects for now ...
            tf_data["mm_model_id"] = "mistralai/mixtral-8x7b-instruct-v0-1"
            tf_data["num_samples"] = num_samples

        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(tf_data),
            cert=cert,
            verify=verify,
            timeout=60 * 10,
        )

        fn = os.path.join(
            output_dir, f"sdg_{int(time.time())}_{tf.split('/')[-2]}.json"
        )
        with open(fn, "w", encoding="utf-8") as f:
            try:
                f.write(json.dumps(response.json(), indent=4))
            except requests.exceptions.JSONDecodeError:
                f.write("Error: Response from SDG service was not valid JSON.\n\n")
                f.write("---\n\n")
                f.write(response.text)

        output_files.append(fn)

    return output_files
