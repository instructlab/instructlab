# Standard
import contextlib
import sys

# Third Party
import yaml

# WARNING:
# This script is a utility meant to be used in scripts/e2e.sh to edit the entries
# in the ilab config from the CLI. It should not be used edit the ilab config by users


def update_yaml(file_path, key, new_value):
    try:
        with open(file_path, "r") as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML: {exc}")
                sys.exit(1)

        keys = key.split(".")
        values = []

        with contextlib.suppress(ValueError):
            new_value = int(new_value)

        for key in keys:
            if key in data:
                values.append(data)
                data = data[key]
            else:
                print(f"Error: Key '{key}' not found in the YAML file.")
                sys.exit(1)

        if len(keys) != len(values):
            print("Error: Keys and Values list sizes are not equal")
            sys.exit(1)

        keys.reverse()
        values.reverse()
        values[0][keys[0]] = new_value
        for i in range(1, len(keys)):
            values[i][keys[i]] = values[i - 1]
        data = values[len(keys) - 1]

        try:
            with open(file_path, "w") as file:
                try:
                    yaml.safe_dump(data, file)
                except yaml.YAMLError as exc:
                    print(f"Error dumping YAML: {exc}")
                    sys.exit(1)
        except OSError:
            print(f"Error: The file '{file_path}' was not found.")
            sys.exit(1)

        print(f"Updated '{key}' to '{new_value}' in {file_path}")
    except OSError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python config_editor.py <path to yaml config file> <key(s)> <new value>"
        )
        print("Ex: ")
        print("   python config_editor.py config.yaml evaluate.gpus 4")
        print(
            "   python config_editor.py config.yaml generate.teacher.vllm.model_family mixtral"
        )
        sys.exit(1)

    print(
        "\033[31mWARNING: This script is a utility only meant to be used in scripts/e2e.sh. It should not be used by users to edit the ilab config.yaml\033[0m"
    )

    yaml_file = sys.argv[1]
    yaml_keys = sys.argv[2]
    new_value = sys.argv[3]

    update_yaml(yaml_file, yaml_keys, new_value)
