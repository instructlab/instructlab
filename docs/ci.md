# CI for InstructLab

## Unit tests

All unit tests currently live in the `tests/` directory.

## Functional tests

The functional test script can be found at `scripts/functional-tests.sh`

## End-to-end (E2E) tests

The end-to-end test script can be found at `scripts/basic-workflow-tests.sh`.
This script takes arguments that control which features are used to allow
varying test coverage based on the resources available on a given test runner.

There is currently a [default `E2E`
job](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e.yml)
that runs automatically on all PRs and after commits merge to `main` or release
branches.

We are currently doing a trial of running the smallest AWS-based e2e workflow
automatically against PRs, as well.

There are other E2E jobs that can be triggered manually on the [actions
page](https://github.com/instructlab/instructlab/actions) for the repository.
These run on a variety of instance types and can be run at the discretion of
repo maintainers.

### E2E Test Coverage

You can specify the following flags to test various features of `ilab` with the
`basic-workflow-tests.sh` script - you can see examples of these being used within
the E2E job configuration files found
[here](https://github.com/instructlab/instructlab/blob/main/.github/workflows/).

| Flag | Feature |
| --- | --- |
| `e` | Run model evaluation |
| `m` | Run minimal configuration |
| `M` | Use Mixtral model (4-bit quantized) |
| `f` | Run "fullsize" training |
| `F` | Run "fullsize" SDG |
| `v` | Run with vLLM for serving |
| `T` | Use the 'full' training library rather than legacy training |

### Trigger via GitHub Web UI

For the E2E jobs that are launched manually, they take an input field that
specifies the PR number or git branch to run them against. If you run them
against a PR, they will automatically post a comment to the PR when the tests
begin and end so it's easier for those involved in the PR to follow the results.

1. Visit the [Actions tab](https://github.com/instructlab/instructlab/actions).
2. Click on the [E2E test](https://github.com/instructlab/instructlab/actions/workflows/e2e.yml)
   workflow on the left side of the page.
3. Click on the `Run workflow` button on the right side of the page.
4. Enter a branch name or a PR number in the input field.
5. Click the green `Run workflow` button.

Here is an example of using the GitHub Web UI to launch an E2E workflow:

![GitHub Actions Run Workflow Example](images/github-actions-run-workflow-example.png)

### Current GPU-enabled Runners

The project currently supports the usage of the following runners for the E2E jobs:

* [GitHub built-in GPU runner](https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners/about-larger-runners#specifications-for-gpu-larger-runners) -- referred to as `ubuntu-gpu` in our workflow files Only `e2e.yml` uses this runner.
* Ephemeral GitHub runners launched on demand on AWS. Most workflows work this way, granting us access to a wider variety of infrastructure at lower cost.

### E2E Workflows

| File | T-Shirt Size | Runner Host |Instance Type | GPU Type | OS |
| --- | --- | --- | --- | --- | --- |
| [`e2e.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e.yml) | Small | GitHub | N/A | 1 x NVIDIA Tesla T4 w/ 16 GB VRAM | Ubuntu |
| [`e2e-nvidia-t4-x1.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-t4-x1.yml) | Small | AWS | [`g4dn.2xlarge`](https://aws.amazon.com/ec2/instance-types/g4/) | 1 x NVIDIA Tesla T4 w/ 16 GB VRAM | CentOS Stream 9 |
| [`e2e-nvidia-a10g-x1.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-a10g-x1.yml) | Medium | AWS |[`g5.2xlarge`](https://aws.amazon.com/ec2/instance-types/g5/) | 1 x NVIDIA A10G w/ 24 GB VRAM | CentOS Stream 9 |
| [`e2e-nvidia-a10g-x4.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-a10g-x4.yml) | Large | AWS |[`g5.12xlarge`](https://aws.amazon.com/ec2/instance-types/g5/) | 4 x NVIDIA A10G w/ 24 GB VRAM (98 GB) | CentOS Stream 9 |

### E2E Test Matrix

| Area | Feature | [`e2e.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e.yml) (small) ![`e2e.yaml` on `main`](https://github.com/instructlab/instructlab/actions/workflows/e2e.yml/badge.svg?branch=main)| [`e2e-nvidia-t4-x1.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-t4-x1.yml) (small) | [`e2e-nvidia-a10g-x1.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-a10g-x1.yml) (medium) | [`e2e-nvidia-a10g-x4.yml`](https://github.com/instructlab/instructlab/blob/main/.github/workflows/e2e-nvidia-a10g-x4.yml) (large) ![`e2e-nvidia-a10g-x4.yaml` on `main`](https://github.com/instructlab/instructlab/actions/workflows/e2e-nvidia-a10g-x4.yml/badge.svg?branch=main)|
| --- | --- | --- | --- | --- | --- |
| **Serving**  | llama-cpp                |✅|✅|✅|✅ (temporary)|
|              | vllm                     |⎯|⎯|⎯|✅|
| **Generate** | simple                   |✅|✅|✅|⎯|
|              | full                     |⎯|⎯|⎯|✅|
| **Training** | legacy+Linux             |⎯|⎯|❌|⎯|
|              | legacy+Linux+4-bit-quant |✅|✅|⎯|⎯|
|              | training-lib             |⎯|⎯|⎯|✅|
| **Eval**     | eval                     |⎯|⎯|❌(*2)|✅|

Points of clarification (*):

1. The `training-lib` testing is not testing using the output of the Generate step. <https://github.com/instructlab/instructlab/issues/1655>
2. The `eval` testing is not evaluating the output of the Training step. <https://github.com/instructlab/instructlab/issues/1540>
