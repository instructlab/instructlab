#!/usr/bin/env bash

pre-commit run --all-files
RETURN_CODE=$?

function echoWarning() {
  LIGHT_YELLOW='\033[1;33m'
  NC='\033[0m' # No Color
  echo -e "${LIGHT_YELLOW}${1}${NC}"
}

if [ "$RETURN_CODE" -ne 0 ]; then
  if [ "${CI}" != "true" ]; then
    echoWarning "☝️ This appears to have failed, but actually your files have been formatted."
    echoWarning "Make a new commit with these changes before making a pull request."
  else
    echoWarning "This test failed because your code isn't formatted correctly."
    echoWarning 'Locally, run `make run fmt`, it will appear to fail, but change files.'
    echoWarning "Add the changed files to your commit and this stage will pass."
  fi

  exit $RETURN_CODE
fi
