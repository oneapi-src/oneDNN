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
from collections import defaultdict
from scipy.stats import ttest_ind
import warnings
import statistics


def compare_two_benchdnn(file1, file2, tolerance=0.05):
    """
    Compare two benchdnn output files
    """
    with open(file1) as f:
        r1 = f.readlines()

    with open(file2) as f:
        r2 = f.readlines()

    # Trim non-formatted lines and split the problem from time
    r1 = [x.split(",") for x in r1 if x[0:8] == "--mode=P"]
    r2 = [x.split(",") for x in r2 if x[0:8] == "--mode=P"]

    if (len(r1) == 0) or (len(r2) == 0):
        warnings.warn("One or both of the test results have zero lines")
    if len(r1) != len(r2):
        warnings.warn("The number of benchdnn runs do not match")

    r1_samples = defaultdict(list)
    r2_samples = defaultdict(list)

    for k, v in r1:
        r1_samples[k].append(float(v[:-1]))
    for k, v in r2:
        r2_samples[k].append(float(v[:-1]))

    failed_tests = []
    times = {}
    for prb, r1_times in r1_samples.items():
        if prb not in r2_samples:
            warnings.warn(f"{prb} exists in {file1} but not {file2}")
            continue

        r2_times = r2_samples[prb]

        res = ttest_ind(r2_times, r1_times, alternative='greater')
        r1_med = statistics.median(r1_times)
        r2_med = statistics.median(r2_times)
        times[prb] = (r1_med, r2_med)
        times_str = f" {times[prb][0]} vs {times[prb][1]}"

        # pass the test if:
        # the t-test passes (i.e. pvalue > 0.05) OR
        # both the median time and min time has not 
        # slowed down by more than 10%
        passed = res.pvalue > 0.05 or \
                ((r2_med - r1_med) / r1_med < 0.1 and \
                (min(r2_times) - min(r1_times)) / min(r1_times) < 0.1)
        if not passed:
            failed_tests.append(prb + times_str)
            passed = False

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(f"pass={not failed_tests}", file=f)

    if not failed_tests:
        print("Regression tests passed")
    else:
        message = "\n----The following regression tests failed:----\n" + \
                    "\n".join(failed_tests) + "\n"
        if "GITHUB_OUTPUT" in os.environ:
            out_message = message.replace("\n", "%0A")
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f'message={out_message}', file=f)
        print(message)
        raise Exception("Some regression tests failed")

if __name__ == "__main__":
    compare_two_benchdnn(sys.argv[1], sys.argv[2])
