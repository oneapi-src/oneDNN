################################################################################
# Copyright 2021-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import functools
import sys


@functools.total_ordering
class Version:
    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch

    @property
    def _as_tuple(self):
        return self.major, self.minor, self.patch

    def __lt__(self, other):
        return self._as_tuple < other._as_tuple

    def __eq__(self, other):
        return self._as_tuple == other._as_tuple


def get_version():
    return Version(*map(int, sys.version.split(" ")[0].split(".")))


def check_version():
    return get_version() >= Version(3, 7, 0)


def dedent(multiline):
    lines = multiline.split("\n")
    if len(lines) == 1:
        return lines[0].strip()
    indent = min(len(line) - len(line.lstrip()) for line in lines[1:])
    return (lines[0] + "\n".join(line[indent:] for line in lines[1:])).strip()
