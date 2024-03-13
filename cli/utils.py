# Standard
import copy
import functools
import os
import platform
import sys

# Third Party
import click


def macos_requirement(echo_func, exit_exception):
    """Adds a check for MacOS before running a method.

    :param echo_func: Echo function accepting message and fg parameters to print the error.
    :param exit_exception: Exit exception to raise in case the MacOS requirement is not fulfilled.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_macos_with_m_chip():
                echo_func(
                    message=f"`lab {func.__name__}` is only implemented for macOS with M-series chips for now",
                    fg="red",
                )
                raise exit_exception(1)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_macos_with_m_chip():
    """Checks if the OS is MacOS with an ARM processor"""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def expand_path(path):
    """Expands ~ and environment variables from the given path"""
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    return path


def make_lab_diff_aliases(cli, diff):
    lab_list = copy.deepcopy(diff)
    lab_list.name = "list"
    lab_list.help = """
    Lists taxonomy files that have changed since <taxonomy-base>.
    Similar to 'git diff <ref>'
    """
    lab_list.deprecated = True

    def lab_list_callback(*args, **kwargs):
        click.secho(
            "DeprecationWarning: Use `lab diff` instead.",
            fg="red",
        )
        retval = diff.callback(*args, **kwargs)
        return retval

    lab_list.callback = lab_list_callback
    cli.add_command(lab_list)

    lab_check = copy.deepcopy(diff)
    lab_check.name = "check"
    lab_check.help = "Check that taxonomy is valid"
    lab_check.deprecated = True
    # use `--quiet` for current `lab check` behavior
    lab_check.params = lab_check.params[:3]

    def lab_check_callback(*args, **kwargs):
        click.secho(
            "DeprecationWarning: Use `lab diff --quiet` instead.",
            fg="red",
        )
        retval = diff.callback(*args, **kwargs, quiet=True)
        return retval

    lab_check.callback = lab_check_callback
    cli.add_command(lab_check)
