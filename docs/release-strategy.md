# InstructLab CLI Release Strategy

This document discusses the release strategy and processes for the
`instructlab` Python package built from the
<https://github.com/instructlab/instructlab> git repository.

## Versioning Scheme

Releases use a `X.Y.Z` numbering scheme.

X-stream release are for major releases. At this stage in the project a major release has not been cut and we expect each release to be a new Y-stream.

Z-stream releases are meant for critical bug and documentation fixes. Z-stream releases are cut as maintainers see fit.

## Schedule

The project currently operates on a time-based release schedule.
The goal is to release a new Y-stream every 2-3 weeks though the time span remains flexible based on the discretion of the maintainers team.

The cadence for major releases starting from 1.0 onward will be determined as the project matures.

A schedule will be updated in a markdown file on the <https://github.com/instructlab/instructlab> GitHub repository.

## Release Tracking

Release planning is tracked via [milestones](https://github.com/instructlab/instructlab/milestones) in GitHub. Milestones on GitHub will exist for the next two releases.

PRs and Issues associated with the next two milestones will be prioritized for review and merging into the `main` branch of instructlab.

## Git Branches and Tags

Every `X.Y` release stream gets a new branch.

Each release, `X.Y.Z`, exists as a tag named `vX.Y.Z`.

## Release Branch Maintenance

Maintenance efforts are only on the most recent Y-stream.
Critical bug fixes are backported to the most recent `X.Y` branch that contains the `stable` tag for a Z-stream release.

## Release Mechanics

Release mechanics are done by a Release Manager identified for that release.
The Release Manager is a member of the CLI Maintainers team that has agreed to take on these responsibilities.
The Release Manager can change on a per-release basis.
The Release Manager for each release is identified in the Description of the Milestone used to plan and track that release on GitHub.

The following are the steps for how Y-stream and Z-stream releases gets cut.

1. Maintainers determine a commit on the main branch that will serve as the basis for the next release.
    - For a Z-stream release skip this step.
2. For a Y-Stream release: create a new branch in the format `release-vX.Y`.
    - For a Z-Stream release skip this step.
3. For a Z-Stream release: backport all relevant commits from `main` to the `release-X.Y` branch.
    - For a Y-Stream release skip this step.
4. Create a new release on GitHub. The following is automated:
    - Tagging the branch on GitHub
    - Creating a change log on GitHub
    - Version number of the Python package is derived from the tag name
5. Validate the release (manual testing).
6. Move the `stable` tag to the newest release.
7. Announce release on the following channels:
    - InstructLab Slack
8. Create a milestone on GitHub for the next release without a milestone.
    - For a Z-Stream release skip this step.
