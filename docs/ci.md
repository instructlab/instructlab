# CI for InstructLab

## End-to-end CI Job

This CI job is triggered right before merging a PR to the `main` branch when a PR received two
approvals. Alternatively, it can be triggered manually by adding the `e2e-trigger` label to a PR.
Additionally, the workflow will be triggered manually by a maintainer before branching a new release.

It runs as much of the `ilab` workflow as it can on the GPU-enabled worker we
have available through GitHub Actions.

Any PR that makes functional changes that may affect the `ilab` workflow is
a good candidate for running this workflow. It does not run automatically since
the cost of this workflow is substantially higher than all other CI jobs that
run on normal runners.

### Trigger via GitHub Actions

1. Visit the [Actions tab](https://github.com/instructlab/instructlab/actions).
2. Click on the [E2E test](https://github.com/instructlab/instructlab/actions/workflows/e2e.yml)
   workflow on the left side of the page.
3. Click on the `Run workflow` button on the right side of the page.
4. Enter a branch name or a PR number in the input field.
5. Click the green `Run workflow` button.

### Trigger via PR Label

The CI job can also be triggered on a PR by adding the `e2e-trigger` label.
