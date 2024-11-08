#!/usr/bin/python3

# *******************************************************************************
# Copyright 2025 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
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
# *******************************************************************************

import sys


def compare_two_benchdnn(file1, file2, tolerance=0.05):
    """
    Compare two benchdnn output files
    """
    with open(file1) as f:
        r1 = f.readlines()

    with open(file2) as f:
        r2 = f.readlines()

    # Trim non-formatted lines and split the prolem from time
    r1 = [x.split(",") for x in r1[1:-3]]
    r2 = [x.split(",") for x in r2[1:-3]]

    # Convert to dict and trim \n
    r1 = {x[0]: float(x[1][:-1]) for x in r1}
    r2 = {x[0]: float(x[1][:-1]) for x in r2}

    if len(r1) != len(r2):
        raise Exception("The number of benchdnn runs do not match")

    for prb, time1 in r1.items():
        if prb not in r2:
            raise Exception(f"{prb} exists in {file1} but not {file2}")

        if r2[prb] / time1 > 1 + tolerance:
            raise Exception(f"{prb} has regressed by {round(r2[prb] / time1, 2)}x")


if __name__ == "__main__":
    compare_two_benchdnn(sys.argv[1], sys.argv[2])
