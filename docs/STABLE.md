Moving the stable tag
=====================

Because we do not have a PyPI package where folks can just install latest, our `README.md` instructions describe installing from the GitHub using a `stable` tag.

To support this, we move the stable tag to the latest release, as needed.

This can be done via the following steps:

Assume that the upstream repository (`instructlab/instructlab`) is using the `upstream` Git remote

Get current tags from the upstream:

```ShellSession
$ git switch main
$ git fetch upstream
$ git pull upstream
# Just making sure you are up-to-date locally
$ git tag --list
$ git show-ref --tags
```

Using v0.x.y as the example desired stable version:

```ShellSession
$ git show v0.x.y
```

1. Verify that this is the tag/commit you want for stable.
2. Double check with the short SHA hashes on tags [here](https://github.com/instructlab/instructlab/tags)
3. Copy the SHA hash

I usually test first w/o --force and expect an error if I have everything right.

```ShellSession
$ git tag stable v0.x.y
fatal: tag 'stable' already exists
```

Next, add the `-f` (force) flag to force the change locally.

```ShellSession
$ git tag -f stable v0.x.y
```

Verify the tag SHA hashes look correct:

```ShellSession
$ git show-ref --tags
```

You can then push the new tag upstream with `-f` (force) flag - I usually test first without `--force` and expect an error if I have everything right.

```ShellSession
$ git push upstream stable
! [rejected]        stable -> stable (already exists)
error: failed to push some refs to 'https://github.com/instructlab/instructlab.git'
hint: Updates were rejected because the tag already exists in the remote.
```

If you are sure, push with force to upstream and prepare to live with the consequences of your actions.

```ShellSession
git push -f upstream stable
```

Finally, check the tags on the GitHub web UI to ensure everything looks correct.
