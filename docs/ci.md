# CI for InstructLab

## Unit tests

Unit tests are designed to test specific InstructLab components or features in isolation. In CI, these tests are run on Python 3.11 and Python 3.10 on CPU-only Ubuntu
runners as well as Python 3.11 on CPU-only MacOS runners. Generally, new code should be adding or modifying unit tests.

All unit tests currently live in the `tests/` directory and are run with [pytest](https://docs.pytest.org/) via [tox](https://tox.wiki/).

## Functional tests

Functional tests are designed to test InstructLab components or features in tandem, but not necessarily as part of a complex workflow. In CI, these tests are run on
Python 3.11 and Python 3.10 on CPU-only Ubuntu runners as well as Python 3.11 on CPU-only MacOS runners. New code may or may not need a functional test but should strive
to implement one if possible.

The functional test script is Shell-based and can be found at `scripts/functional-tests.sh`.

## End-to-end (E2E) tests

End-to-end (E2E) tests are designed to test the InstructLab workflow from end-to-end, mimicking real user behavior.

The E2E test script is Shell-based and can be found at `scripts/basic-workflow-tests.sh`.
This script takes arguments that control which features are used to allow
varying test coverage based on the resources available on a given test runner.

There is currently a ["small" t-shirt size E2E job](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-t4-x1.yml)
that is run when the [E2E starter job](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-starter.yml) successfully completes.
The starter job runs automatically on all PRs and after commits merge to `main` or release
branches. Its successful completion depends upon the successful completion of any linting type jobs.

There is also a ["medium" t-shirt size E2E job](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-a10g-x1.yml) and ["large" t-shirt size E2E job](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-a10g-x4.yml) that can be triggered manually on the [actions
page](https://github.com/instructlab/instructlab/actions) for the repository.
These run on a variety of instance types and can be run at the discretion of
repo maintainers.

The "large" t-shirt size E2E job also runs automatically against the `main` branch at 11AM every day.

### E2E Test Coverage Options

You can specify the following flags to test various features of `ilab` with the
`basic-workflow-tests.sh` script - you can see examples of these being used within
the E2E job configuration files found
[here](https://github.com/instructlab/instructlab/blob/main/.github/workflows/).

| Flag | Feature |
| --- | --- |
| `e` | Run model evaluation |
| `F` | Run "fullsize" SDG |
| `L` | Run legacy training |
| `m` | Run minimal configuration (lower number of instructions and training epochs) |
| `M` | Use Mixtral model (4-bit quantized) instead of Merlinite (4-bit quantized) |
| `P` | Use the phased training within the 'full' training library |
| `T` | Use the 'full' training library rather than legacy training |
| `v` | Run with vLLM for serving |

### Current E2E Jobs

| Name | T-Shirt Size | Runner Host | Instance Type | OS | GPU Type | Script flags | Runs when? | Slack reporting? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`e2e-nvidia-t4-x1.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-t4-x1.yml) | Small | AWS | [`g4dn.2xlarge`](https://aws.amazon.com/ec2/instance-types/g4/) | CentOS Stream 9 | 1 x NVIDIA Tesla T4 w/ 16 GB VRAM | `m` | Pull Requests, Push to `main` or `release-*` branch | No |
| [`e2e-nvidia-a10g-x1.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-a10g-x1.yml) | Medium | AWS |[`g5.2xlarge`](https://aws.amazon.com/ec2/instance-types/g5/) | CentOS Stream 9 | 1 x NVIDIA A10G w/ 24 GB VRAM | `mf` | Manually by Maintainers | No |
| [`e2e-nvidia-a10g-x4.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-a10g-x4.yml) | Large | AWS |[`g5.12xlarge`](https://aws.amazon.com/ec2/instance-types/g5/) | CentOS Stream 9 | 4 x NVIDIA A10G w/ 24 GB VRAM (98 GB) | `mevFMT` | Manually by Maintainers, Automatically against `main` branch at 8AM/8PM UTC | Yes |

### E2E Test Coverage Matrix

| Area | Feature | [`e2e-nvidia-t4-x1.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-t4-x1.yml) | [`e2e-nvidia-a10g-x1.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-a10g-x1.yml) | [`e2e-nvidia-a10g-x4.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-a10g-x4.yml) |
| --- | --- | --- | --- | --- |
| **Serving**  | llama-cpp                |✅|✅|✅ (temporary)|
|              | vllm                     |⎯|⎯|✅|
| **Generate** | simple                   |✅|✅|⎯|
|              | full                     |⎯|⎯|✅|
| **Training** | legacy+Linux             |⎯|❌|⎯|
|              | legacy+Linux+4-bit-quant |✅|⎯|⎯|
|              | training-lib             |⎯|⎯|✅|
| **Eval**     | eval                     |⎯|❌(*2)|✅|

Points of clarification (*):

1. The `training-lib` testing is not testing using the output of the Generate step. <https://github.com/instructlab/instructlab/issues/1655>
2. The `eval` testing is not evaluating the output of the Training step. <https://github.com/instructlab/instructlab/issues/1642>

### Current E2E Status against `main`

![`e2e-nvidia-t4-x1.yaml` on `main`](https://github.com/instructlab/instructlab/actions/workflows/e2e-nvidia-t4-x1.yml/badge.svg?branch=main)
![`e2e-nvidia-a10g-x1.yaml` on `main`](https://github.com/instructlab/instructlab/actions/workflows/e2e-nvidia-a10g-x1.yml/badge.svg?branch=main)
![`e2e-nvidia-a10g-x4.yaml` on `main`](https://github.com/instructlab/instructlab/actions/workflows/e2e-nvidia-a10g-x4.yml/badge.svg?branch=main)

### Slack reporting

Some E2E jobs send their results to the InstructLab Slack workspace channel `#cli-e2e-results` via the `Son of Jeeves` bot. This has been implemented
via [the official Slack GitHub Action](https://github.com/slackapi/slack-github-action?tab=readme-ov-file#technique-2-slack-app). You can see which jobs currently have reporting via the "Current E2E Jobs" table above.

### Triggering an E2E job via GitHub Web UI

For the E2E jobs that can be launched manually, they take an input field that
specifies the PR number or git branch to run them against. If you run them
against a PR, they will automatically post a comment to the PR when the tests
begin and end so it's easier for those involved in the PR to follow the results.

1. Visit the [Actions tab](https://github.com/instructlab/instructlab/actions).
2. Click on one of the E2E workflows on the left side of the page.
3. Click on the `Run workflow` button on the right side of the page.
4. Enter a branch name or a PR number in the input field.
5. Click the green `Run workflow` button.

Here is an example of using the GitHub Web UI to launch an E2E workflow:

![GitHub Actions Run Workflow Example](images/github-actions-run-workflow-example.png)
