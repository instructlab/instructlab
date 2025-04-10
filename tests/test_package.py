# SPDX-License-Identifier: Apache-2.0
"""Verify package metadata"""

# Standard
from importlib.metadata import metadata
import typing

# Third Party
from packaging.requirements import Requirement
from packaging.version import Version
import pytest

PKG_NAME = "instructlab"
# expected hardware extras
HW_EXTRAS = frozenset({"cpu", "cuda", "hpu", "mps", "rocm"})
# special cases
EXTRA_CHECKS = {
    "hpu": {
        "torch": Version("2.6.0"),
        "transformers": Version("4.43.0"),
    }
}


def iter_requirements(pkg: str = PKG_NAME) -> typing.Iterable[Requirement]:
    m = metadata(pkg)
    requires = m.get_all("Requires-Dist")
    assert requires is not None
    for r in requires:
        yield Requirement(r)


def test_provides_extra():
    m = metadata(PKG_NAME)
    assert set(m.get_all("Provides-Extra")).issuperset(HW_EXTRAS)


def test_require_no_url_req():
    # PyPI does not accept packages with URL requirements
    for req in iter_requirements():
        assert req.url is None, req


@pytest.mark.parametrize("hw_extra", sorted(HW_EXTRAS))
@pytest.mark.parametrize("py_version", ["3.10", "3.11"])
def test_package_conflict(py_version: str, hw_extra: str) -> None:
    if py_version != "3.11" and hw_extra == "hpu":
        pytest.skip("Intel Gaudi only supports 3.11")

    base: dict[str, Requirement] = {}
    hw: dict[str, Requirement] = {}
    for req in iter_requirements():
        # override version for environment
        base_env = {
            "implementation_version": f"{py_version}.0",
            "python_full_version": f"{py_version}.0",
            "python_version": py_version,
        }
        extra_env = base_env.copy()
        extra_env["extra"] = hw_extra

        if req.marker is None or req.marker.evaluate(base_env):
            # no marker or no optional requirement
            base[req.name] = req
        elif req.marker.evaluate(extra_env):
            # matching optional requirement
            hw[req.name] = req

    for name, hwreq in hw.items():
        basereq = base.get(name)
        if basereq is None:
            continue
        for specifier in hwreq.specifier:
            # naive check for common version conflicts
            # allow pre-releases for Gaudi
            if specifier.operator in {"~=", "==", "<=", ">="}:
                version: Version = Version(specifier.version)
                assert basereq.specifier.contains(
                    version, prereleases=version.is_prerelease
                ), (basereq, hwreq)

    # verify special cases against base requirements
    if hw_extra in EXTRA_CHECKS:
        for name, basereq in base.items():
            extra_check = EXTRA_CHECKS[hw_extra].get(name)
            if extra_check is not None:
                assert basereq.specifier.contains(
                    extra_check, prereleases=extra_check.is_prerelease
                ), (basereq, extra_check)
