Moving the stable tag
=====================

Because we do not have a PyPI package where folks can just install latest, our `README.md` instructions describe installing from the GitHub using a "stable" tag.

To support this, we move the stable tag to the latest release, as needed.

This can be done by:

Get current:

```ShellSession
$ git switch main
$ git fetch
$ git pull
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

Push the new tag to remote (origin) with `-f` (force) flag.

I usually test first w/o --force and expect an error if I have everything right.

```ShellSession
$ git push origin stable
! [rejected]        stable -> stable (already exists)
error: failed to push some refs to 'https://github.com/instructlab/instructlab.git'
hint: Updates were rejected because the tag already exists in the remote.
```

If you are sure. Push with force.

```ShellSession
git push -f origin stable
```

Check the Tags on GitHub web.
