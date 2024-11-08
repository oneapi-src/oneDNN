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
import os


def compare_two_benchdnn(file1, file2, tolerance=0.05):
    """
    Compare two benchdnn output files
    """
    with open(file1) as f:
        r1 = f.readlines()

    with open(file2) as f:
        r2 = f.readlines()

    # Trim non-formatted lines and split the prolem from time
    r1 = [x.split(",") for x in r1 if x[0:8] == "--mode=P"]
    r2 = [x.split(",") for x in r2 if x[0:8] == "--mode=P"]

    # Convert to dict and trim \n
    r1 = [(x[0], float(x[1][:-1])) for x in r1]
    r2 = [(x[0], float(x[1][:-1])) for x in r2]

    if len(r1) != len(r2):
        raise Exception("The number of benchdnn runs do not match")

    print("%prb%,%-time(old)%,%-time(new)%,%passed%")

    passed = True
    failed_tests = []
    for idx, item in enumerate(r1):
        prb, time1 = item
        if prb != r2[idx][0]:
            raise Exception(f"{prb} exists in {file1} but not {file2}")

        res_str = f"{prb}, {time1}, {r2[idx][1]}"
        print(res_str)

        if time1 != 0: # Incompatible tests would return 0 so avoid division by 0
            test_pass = (r2[idx][1] - time1) / time1 < tolerance
            if not test_pass:
                failed_tests.append(res_str)
                passed = False

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"pass={passed}")

    if passed:
        print("Regression tests passed")
    else:
        print("\n----The following tests did not pass:----")
        print("\n".join(failed_tests) + "\n")
        raise Exception("Some regression tests did not pass")

if __name__ == "__main__":
    compare_two_benchdnn(sys.argv[1], sys.argv[2])
