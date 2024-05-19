# CI for InstructLab

## End-to-end CI Job

This CI job is manually triggered by `instructlab` repo triagers or maintainers.
It runs as much of the `ilab` workflow as it can on the GPU-enabled worker we
have available through GitHub Actions.

### Trigger via GitHub Actions

1. Visit the [Actions tab](https://github.com/instructlab/instructlab/actions).
2. Click on the [E2E test](https://github.com/instructlab/instructlab/actions/workflows/e2e.yml)
   workflow on the left side of the page.
3. Click on the `Run workflow` button on the right side of the page.
4. Enter a branch name or a PR number in the input field.
5. Click the green `Run workflow` button.

### Trigger via PR Label

The CI job can also be triggered on a PR by adding the `e2e-trigger` label.
