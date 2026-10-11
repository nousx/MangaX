# -*- coding: utf-8 -*-
"""
Package checking utilities
"""

import functools
import itertools
import pathlib
import subprocess
import sys
from typing import List, Optional

try:
    # packaging < 22.0
    from packaging.requirements import Requirement
except ImportError:
    try:
        # packaging >= 22.0
        from packaging.requirements import Requirement
    except (ImportError, ModuleNotFoundError):
        # Fallback: parse requirements manually
        import re
        class Requirement:
            def __init__(self, requirement_string):
                self.requirement_string = requirement_string
                # Simple regex to extract package name
                match = re.match(r'^([a-zA-Z0-9\-_\.]+)', requirement_string.strip())
                self.name = match.group(1) if match else requirement_string

from packaging.utils import canonicalize_name

try:
    import importlib.metadata as importlib_metadata
except (ModuleNotFoundError, ImportError):
    import importlib_metadata
from packaging.version import Version


def package_version(name: str) -> Optional[Version]:
    """Get the version of an installed package"""
    try:
        return Version(importlib_metadata.distribution(canonicalize_name(name)).version)
    except importlib_metadata.PackageNotFoundError:
        return None


def _nonblank(text):
    """Leave out empty lines and comment lines"""
    return text and not text.startswith('#')


def _is_requirement(line):
    """Whether a line is a dependency line (pip options are left out)"""
    line = line.strip()
    # Leave out empty lines and comments
    if not line or line.startswith('#'):
        return False
    # Leave out pip options (--xxx or -x)
    if line.startswith('-'):
        return False
    # Dependencies given as a URL (starting with http:// or https://) have to be kept;
    # these are direct links to wheel files
    return True


@functools.singledispatch
def yield_lines(iterable):
    """Extract the meaningful lines"""
    return itertools.chain.from_iterable(map(yield_lines, iterable))


@yield_lines.register(str)
def _(text):
    return filter(_nonblank, map(str.strip, text.splitlines()))


def drop_comment(line):
    """Remove comments"""
    return line.partition(' #')[0]


def join_continuation(lines):
    """Join continuation lines"""
    lines = iter(lines)
    for item in lines:
        while item.endswith('\\'):
            try:
                item = item[:-2].strip() + next(lines)
            except StopIteration:
                return
        yield item


def load_req_file(requirements_file: str) -> List[str]:
    """Load a requirements file"""
    with pathlib.Path(requirements_file).open(encoding='utf-8') as reqfile:
        lines = join_continuation(map(drop_comment, yield_lines(reqfile)))
        # Leave out pip options (such as --extra-index-url)
        valid_reqs = [line for line in lines if _is_requirement(line)]
        return list(map(lambda x: str(Requirement(x)), valid_reqs))


def _yield_reqs_to_install(req: Requirement, current_extra: str = ''):
    """Check which dependencies need installing"""
    if req.marker and not req.marker.evaluate({'extra': current_extra}):
        return

    try:
        version_str = importlib_metadata.distribution(req.name).version
    except importlib_metadata.PackageNotFoundError:
        yield req
    else:
        if not version_str:
            # A damaged or incomplete dist-info exists but there is no version to compare; reinstall this dependency.
            yield req
            return

        # For PyTorch and similar packages, remove the local version identifier (such as +cu128) before comparing
        # For example: 2.9.1+cu128 -> 2.9.1
        version_base = version_str.split('+')[0]

        # Check with the base version number first
        if req.specifier.contains(version_base, prereleases=True):
            # The version matches: check the sub-dependencies
            for child_req in (importlib_metadata.metadata(req.name).get_all('Requires-Dist') or []):
                child_req_obj = Requirement(child_req)
                need_check, ext = False, None
                for extra in req.extras:
                    if child_req_obj.marker and child_req_obj.marker.evaluate({'extra': extra}):
                        need_check = True
                        ext = extra
                        break
                if need_check:
                    yield from _yield_reqs_to_install(child_req_obj, ext)
        else:
            # The version does not match, but a newer installed version also counts as satisfied
            # For example: ==2.8.0 is required and 2.9.1 is installed, which counts as satisfied (backward compatible)
            try:
                installed_version = Version(version_base)
                # Check whether the specifier has an exact version requirement (==)
                has_exact_match = any(spec.operator == '==' for spec in req.specifier)

                if has_exact_match:
                    # There is an exact version requirement: check whether the installed version is newer
                    # Extract the required version number
                    for spec in req.specifier:
                        if spec.operator == '==':
                            required_version = Version(spec.version)
                            if installed_version >= required_version:
                                # The installed version is newer or equal: satisfied
                                return
                            break

                # Otherwise the version does not match and it has to be installed
                yield req
            except Exception:
                # The version could not be parsed: handle it with the original logic
                yield req


def _check_req(req: Requirement):
    """Check whether a single dependency is satisfied"""
    return not bool(list(itertools.islice(_yield_reqs_to_install(req), 1)))


def get_missing_packages(reqs: List[str]) -> List[str]:
    """Get the list of packages that are missing or need updating"""
    missing = []
    for req_str in reqs:
        req = Requirement(req_str)
        if not _check_req(req):
            missing.append(req_str)
    return missing


def check_reqs(reqs: List[str]) -> bool:
    """Check whether all dependencies are satisfied"""
    return all(map(lambda x: _check_req(Requirement(x)), reqs))


def check_req_file(requirements_file: str) -> bool:
    """Check whether the dependencies in a requirements file are satisfied"""
    try:
        return check_reqs(load_req_file(requirements_file))
    except Exception as e:
        print(f'Checking the dependency file failed: {e}')
        return False


def get_missing_packages_from_file(requirements_file: str) -> List[str]:
    """Get the list of packages in a requirements file that are missing or need updating"""
    try:
        reqs = load_req_file(requirements_file)
        return get_missing_packages(reqs)
    except Exception as e:
        print(f'Checking the dependency file failed: {e}')
        return []

