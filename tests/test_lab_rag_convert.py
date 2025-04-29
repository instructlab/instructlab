# SPDX-License-Identifier: Apache-2.0

# Standard
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union
from unittest.mock import patch

# Third Party
from click.testing import CliRunner
from docling.backend.docling_parse_v2_backend import (  # type: ignore  # noqa: F401
    DoclingParseV2DocumentBackend,
)
from docling.datamodel.base_models import (  # type: ignore  # noqa: F401
    ConversionStatus,
    InputFormat,
)
from docling.datamodel.document import ConversionResult  # type: ignore  # noqa: F401
from docling.datamodel.document import InputDocument  # type: ignore  # noqa: F401
from docling.document_converter import FormatOption  # type: ignore  # noqa: F401
import pytest

# First Party
from instructlab import lab
from instructlab.feature_gates import FeatureGating, FeatureScopes, GatedFeatures
from instructlab.rag.convert import _load_converter_and_format_options
from tests.test_feature_gates import dev_preview


class MockDocumentConverter:
    def __init__(
        self,
        allowed_formats: Optional[list[InputFormat]] = None,  # pylint: disable=unused-argument; noqa: ARG002
        format_options: Optional[Dict[InputFormat, FormatOption]] = None,  # pylint: disable=unused-argument; noqa: ARG002
    ):
        pass

    def convert_all(
        self,
        source: Iterable[Union[Path, BytesIO]],  # pylint: disable=unused-argument; noqa: ARG002
        raises_on_error: bool = True,  # pylint: disable=unused-argument; noqa: ARG002
    ) -> Iterator[ConversionResult]:
        # Third Party
        for doc in source:
            print(type(doc))
            in_doc = InputDocument(
                path_or_stream=doc,
                format=InputFormat.PDF,
                backend=DoclingParseV2DocumentBackend,
            )
            conv_res = ConversionResult(input=in_doc, status=ConversionStatus.SUCCESS)
            yield conv_res


def test_rag_convert_errors_with_useful_message_when_not_enabled():
    runner = CliRunner()
    env = runner.make_env({"ILAB_FEATURE_SCOPE": "Default"})
    result = runner.invoke(lab.ilab, ["--config=DEFAULT", "rag", "convert"], env=env)

    assert not FeatureGating.feature_available(GatedFeatures.RAG)

    # check that the error message contains the environment variable name and the feature
    # scope level; a (heuristic) check on the message being both up-to-date and useful
    assert FeatureGating.env_var_name in result.output
    assert FeatureScopes.DevPreviewNoUpgrade.value in result.output


def run_rag_convert_test(
    params: List[str],
    expected_strings: List[str],  # pylint: disable=unused-argument; noqa: ARG002
    expected_output_file: Union[Path, None],  # pylint: disable=unused-argument; noqa: ARG002
    should_succeed: bool,
):
    """
    Core logic for testing conversion using the CLI runner and checking for expected output string and expected output file.
    params = A list of parameters that would follow 'ilab rag config' on the command line.
    expected_strings = A list of strings that are expected to appear in the output when running with these parameters.
    expected_output_file = A fully qualified path to a file we would expect to be generated from running with these parameters.
    should_succeed = True iff the we would expect the command to succeed (complete with a 0 exit code).
    """
    _load_converter_and_format_options()
    runner = CliRunner()
    with patch("instructlab.rag.convert.DocumentConverter", MockDocumentConverter):
        result = runner.invoke(
            lab.ilab, ["--config=DEFAULT", "rag", "convert"] + params
        )
        if should_succeed:
            assert (
                result.exit_code == 0
            ), f"Unexpected failure for parameters {params}: {result.output}"
        for expected_string in expected_strings:
            assert expected_string in result.output
        if expected_output_file is not None:
            assert expected_output_file.exists()
        else:
            assert (
                result.exit_code != 0
            ), f"Unexpected success for parameters {params}: {result.output}"


@dev_preview
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_convert_pdf_from_directory(tmp_path: Path):
    """
    Tests converting from the sample PDF in tests/testdata/documents/pdf.
    Verifies that it says that it is processing and finished that sample PDF.
    Also verifies that the expected JSON file exists in the temp directory at the end.
    """
    test_input_dir = "tests/testdata/documents/pdf"
    test_output_dir = tmp_path / "convert-outputs"
    params = ["--input-dir", str(test_input_dir), "--output-dir", str(test_output_dir)]
    expected_strings = ["How to use YAML with InstructLab Page 1.pdf"]
    expected_output_file = (
        test_output_dir / "How to use YAML with InstructLab Page 1.json"
    )
    run_rag_convert_test(params, expected_strings, expected_output_file, True)


@dev_preview
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_convert_md_from_directory(tmp_path: Path):
    """
    Tests converting from the sample Markdown file in tests/testdata/documents/md.
    Verifies that the expected JSON file exists in the temp directory at the end.
    """
    test_input_dir = "tests/testdata/documents/md"
    test_output_dir = tmp_path / "convert-outputs"
    params = ["--input-dir", str(test_input_dir), "--output-dir", str(test_output_dir)]
    expected_strings = ["Transforming source files ['hello.md']"]
    expected_output_file = test_output_dir / "hello.json"
    run_rag_convert_test(params, expected_strings, expected_output_file, True)


# Note: This test uses a local taxonomy that references a file in a remote git repository.  For that reason, this
# test won't pass when run on a machine with no connection to the internet.  It wil also fail if the repository
# is not working or if it ever gets deleted.  That's not ideal, but we do need to test these capabilities.
# TODO: Consider re-working this with a mock for the github server.
@dev_preview
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_convert_md_from_taxonomy(tmp_path: Path):
    """
    Tests converting from the sample Markdown file in a github repo referenced in tests/testdata/sample_taxonomy.
    The specific file referenced is phoenix.md in https://github.com/RedHatOfficial/rhelai-taxonomy-data,
    because that is the one in
    https://github.com/instructlab/sdg/blob/main/tests/testdata/test_valid_knowledge_skill.yaml.
    Verifies that the expected JSON file exists in the temp directory at the end.
    """
    test_input_taxonomy = "tests/testdata/sample_taxonomy"
    test_output_dir = tmp_path / "convert-outputs"
    params = [
        "--taxonomy-path",
        str(test_input_taxonomy),
        "--taxonomy-base",
        "empty",
        "--output-dir",
        str(test_output_dir),
    ]
    expected_strings = ["Transforming source files ['phoenix.md']"]
    expected_output_file = test_output_dir / "phoenix.json"
    run_rag_convert_test(params, expected_strings, expected_output_file, True)


# Note that there is no test for converting pdf from taxonomy.  The tests above verify that you can convert
# both PDF and md and that you can convert from both a directory and a taxonomy.  Testing a PDF from a
# taxonomy too seems redundant, and these tests are already taking a lot of time so I don't want to add
# another expensive one for that purpose.


@dev_preview
def test_convert_from_missing_directory_fails(tmp_path: Path):
    """
    Verifies that converting from a non-existent directory fails.
    """
    test_input_dir = "tests/testdata/documents/no-such-directory"
    test_output_dir = tmp_path / "convert-outputs"
    params = ["--input-dir", str(test_input_dir), "--output-dir", str(test_output_dir)]
    run_rag_convert_test(params, [], None, False)


@dev_preview
def test_convert_from_non_directory_fails(tmp_path: Path):
    """
    Verifies that converting fails when the input directory is a file and not a directory.
    """
    test_input_dir = "tests/testdata/documents/md/hello.md"
    test_output_dir = tmp_path / "convert-outputs"
    params = ["--input-dir", str(test_input_dir), "--output-dir", str(test_output_dir)]
    run_rag_convert_test(params, [], None, False)
