#!/bin/bash -ex

# If arg $1, use that branch from GitHub, else assume run in repo root.
if [ -z "$1" ]
  then
    echo "No argument for branch, assuming from clone."
    INSTALL_FROM=".."
  else
    INSTALL_FROM="git+https://github.com/instruct-lab/cli@$1"
fi


# Create a directory right here to use
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Use mktemp for unique name, but we'll keep the dir
TEMP_DIR=`mktemp -d -p "$HERE" -t test_run_XXXXXXXX` || exit 1
echo "test run in: $TEMP_DIR"
cd $TEMP_DIR

python3 -m venv venv
source venv/bin/activate
pip install $INSTALL_FROM
lab init      # Initializes environment for labrador

lab list      # List taxonomy YAML files -- nothing modified

echo "^^^ 'lab list' should have found no modified/new files"

# Add a key error / modify an existing file
echo "- answer: 'An answer without an question?'" >> taxonomy/compositional_skills/writing/freeform/jokes/puns/qna.yaml
# Add a YAML ParserError (The apostrophe inside the single quotes!) in a new file (copy)
cp taxonomy/compositional_skills/writing/freeform/jokes/puns/qna.yaml taxonomy/compositional_skills/writing/freeform/jokes/puns/testrun_qna.yaml
echo "- answer: 'okay'" >> taxonomy/compositional_skills/writing/freeform/jokes/puns/qna.yaml
echo "  question: 'don't quote me'" >> taxonomy/compositional_skills/writing/freeform/jokes/puns/qna.yaml

lab list      # List taxonomy YAML files
echo "^^^ Up there 'lab list' should have found 2 modified or new jokes/puns/ yaml files"

lab generate  # Generates synthetic data to enhance your example data
echo "--- Down there 'lab generate' should have found 2 modified or new jokes/puns/ yaml files"
echo "|   ... and reported a ParserError and a KeyError"
echo "V   ... and then exits!"

# lab download  # Download the model(s) to train
# lab serve     # Start a local server
# lab chat      # Run a chat using the modified model
# lab test      # Perform rudimentary tests of the model
# lab train     # Trains labrador model
