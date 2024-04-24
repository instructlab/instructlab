# InstructLab CLI Release Strategy

This document discusses the release strategy and processes for the
`instructlab` Python package built from the
<https://github.com/instructlab/instructlab> git repository.

## Versioning Scheme

Releases use a `X.Y.Z` numbering scheme.

X-stream release are for major releases. At this stage in the project a major release has not been cut and we expect each release (2-3 weeks) to be a new Y-stream.

Backwards compatibility will exist within a Y-stream for example within 0.1.8 and 0.1.9 and between Y-stream releases for update purposes. For example between 0.1.9 and 0.2.0.

Z-stream releases are meant for critical bug and documentation fixes. Backward compatibility is guaranteed between Z-stream releases. Z-stream releases are cut as maintainers see fit.

## Schedule

The project currently operates on a time-based release schedule.
Before releasing version 1.0, the goal is to release a new Y-stream every 2-3 weeks though the time span remains flexible based on the discretion of the maintainers team.

The cadence for major releases starting from 1.0 and onward will be determined as the project matures.

A schedule will be updated in a markdown file on the <https://github.com/instructlab/instructlab> Github repository.

## Release Tracking

Release planning is tracked via milestones in Github. Milestones on Github will exist for the next two releases.

Issues and PRs for milestones are set, monitored, and changed at weekly grooming sessions.
Only PRs and issues associated with the next two milestones will be prioritized and merged into the main branch of instructlab

## Git Branches and Tags

Every `X.Y` release stream gets a new branch.

Each release, `X.Y.Z`, exists as a tag on the `X.Y` branch.

## Release Branch Maintenance

Maintenance efforts are primarily focused on the `main` branch.
Occasionally, critical bug fixes may be backported to the most recent
`X.Y` branch for a Z-stream bug-fix release. Bug fixes in general are not
actively backported to release branches.

We will not have a X-stream release unless a total re-architecture of instructlab is occurring.

## Release Mechanics

The following are the steps for how a release gets cut.

1. Maintainers determine a commit on the main branch that will serve as the basis for the next release.
    - For a Z-stream release skip this step.
2. For a y-Stream release: create a new branch in the format release-X.YY.0.
    - For a Z-Stream release skip this step
3. Create a commit for the release
4. Open up a PR
5. Maintainers Merge PR
    - The PR will require 2 maintainer approvals.
7. Create a new release on Github. The following is automated
    - Publishing to PyPI
    - Tagging the branch on Github
    - Creating a change log on Github
8. Announce release on the following channels
    - Community Slack
    - The Instructlab community mailing list
    - Update the releases page on the Instructlab website
9. Create a milestone on Github for the next release without a milestone
    - The date for this milestone is set to 2-3 weeks after the next milestone. This date can be tweaked at the next refinement session
