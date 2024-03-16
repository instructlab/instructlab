"""Do not copy SELinux context

Workaround for https://github.com/python/cpython/issues/83074
"""

# sitecustomize is imported by site.py on interpreter startup
# Standard
import errno
import os
import shutil

if hasattr(os, "listxattr"):

    def _patched_copyxattr(src, dst, *, follow_symlinks=True):
        """Copy extended filesystem attributes from `src` to `dst`.

        Overwrite existing attributes.

        If `follow_symlinks` is false, symlinks won't be followed.
        """
        try:
            names = os.listxattr(src, follow_symlinks=follow_symlinks)
        except OSError as e:
            if e.errno not in (errno.ENOTSUP, errno.ENODATA, errno.EINVAL):
                raise
            return

        # WORKAROUND: do not copy system, security, or trusted attributes.
        # Only copy user attributes.
        names = [name for name in names if name.startswith("user.")]

        for name in names:
            try:
                value = os.getxattr(src, name, follow_symlinks=follow_symlinks)
                os.setxattr(dst, name, value, follow_symlinks=follow_symlinks)
            except OSError as e:
                if e.errno not in (
                    errno.EPERM,
                    errno.ENOTSUP,
                    errno.ENODATA,
                    errno.EINVAL,
                    errno.EACCES,
                ):
                    raise

    shutil._copyxattr = _patched_copyxattr
