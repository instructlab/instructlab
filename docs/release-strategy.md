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
Critical bug fixes are backported to the most recent release branch.

## Release Mechanics

Release mechanics are done by a Release Manager identified for that release.
The Release Manager is a member of the CLI Maintainers team that has agreed to take on these responsibilities.
The Release Manager can change on a per-release basis.
The Release Manager for each release is identified in the Description of the Milestone used to plan and track that release on GitHub.

The following are the steps for how Y-stream and Z-stream releases gets cut.

### Y-Stream

1. Determine a commit on the main branch that will serve as the basis for the next release - most of the time this should be the latest commit.
1. Create a new release branch in the format `release-vX.Y` off of the determined commit (will match `main` if the latest commit is chosen).
1. Validate the release branch with an [E2E test](ci.md).
1. Create a new release on GitHub targeting the release branch and using the latest Y-Stream tag as the previous release (e.g. `0.15.1` preceeds `0.16.0`).
1. Move the `stable` tag to the new release (note this tag is set to be deprecated on September 1st, 2024 - users should use PyPi to install the latest "stable" release)
1. Announce release via the following:
    - The `#announce` channel on Slack
    - The `announce` mailing list
1. Create a milestone on GitHub for the next release without a milestone.

### Z-Stream

1. Backport all relevant commits from `main` to the `release-vX.Y` branch - this can be done automatically with Mergify or manually if preferred. A backport using Mergify is done by adding a comment to the PR with the change merged to `main` with the contents `@Mergifyio backport <release-vX.Y>`.
1. Validate the release branch with an [E2E test](ci.md).
1. Create a new release on GitHub targeting the release branch and using the previous Z-Stream tag as the previous release (e.g. `0.15.0` preceeds `0.15.1`).
1. Move the `stable` tag to the new release (note this tag is set to be deprecated on September 1st, 2024 - users should use PyPi to install the latest "stable" release)
1. Announce release via the following:
    - The `#announce` channel on Slack
    - The `announce` mailing list
