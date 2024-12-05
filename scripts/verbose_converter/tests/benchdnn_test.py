#!/usr/bin/env python
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

import argparse
import os
import subprocess
import sys
from argparse import RawTextHelpFormatter
from collections import defaultdict
from typing import Dict, List


class TestingException(RuntimeError):
    def __init__(self, msg):
        from src.utils import dedent  # type: ignore[import-not-found]

        super().__init__(dedent(msg))


class FailedCase(TestingException):
    def __init__(self, status: str, repro: str):
        super().__init__(f"Failed case: {status}: {repro}")


def convert_dir_benchdnn2verbose(dir):
    mapping = {
        "FWD_D": "forward_training",
        "FWD_B": "forward_training",
        "FWD_I": "forward_inference",
        "BWD_D": "backward_data",
        "BWD_W": "backward_weights",
        "BWD_DW": "backward",
    }
    return mapping.get(dir, "undef")


def filter_verbose(verbose: str, driver: str, filter_event: str):
    found_cases: List[str] = []
    tentative_cases: Dict[str, List[str]] = defaultdict(list)
    for line in verbose.split("\n"):
        if "__REPRO" in line:
            # n: STATUS (Status message) __REPRO: repro
            _, status_info, repro = map(str.strip, line.split(":", 2))
            status_and_message = status_info.rsplit(None, 1)[0]
            status = status_and_message.split("(", 1)[0].strip()
            # workaround for nvim-treesitter indent bug: )
            argname = "prop" if driver == "rnn" else "dir"
            known_prop_kind: str = "undef"
            for part in repro.split():
                if part.startswith(f"--{argname}="):
                    value = part[len(argname) + 3 :]
                    known_prop_kind = convert_dir_benchdnn2verbose(value)
                    break

            cases = tentative_cases[known_prop_kind]
            tentative_cases.clear()
            if status == "SKIPPED":
                continue
            elif "FAILED" in status:
                raise FailedCase(status, repro)
            elif not cases:
                continue
            found_cases.append(cases[-1])
        elif line.startswith("onednn_verbose,"):
            # Detect driver
            parts = line.split(",")
            try:
                float(parts[2])  # check for timestamp
            except ValueError:
                pass
            else:
                parts.pop(2)
            try:
                component = parts[2]
                event, *_ = parts[3].split(":", 1)
                primitive = parts[5]
                impl_name = parts[6]
                prop_kind = parts[7]
            except IndexError:
                continue
            if component != "primitive" or event not in filter_event:
                continue
            if get_driver(primitive) != driver:
                continue
            # Filter out transform routine till it's properly supported. Use
            # impl name for that due to it's the only difference between two
            # ukernel calls.
            if driver == "brgemm" and impl_name == "pack_B":
                continue
            # Remove primitive creation/run time
            try:
                float(parts[-1])
            except ValueError:
                continue
            without_time = ",".join(parts[:-1])
            # Filter out fill reorders. Only the last one is real.
            tentative_cases[prop_kind].append(without_time)
            if prop_kind != "undef":
                # In case the reproducer uses the default prop kind
                tentative_cases["undef"].append(without_time)
    return "\n".join(found_cases)


def generate_verbose(path_to_benchdnn, engine, driver, batch):
    benchdnn_exe = path_to_benchdnn + "/benchdnn"
    sub_env = os.environ.copy()
    sub_env["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "0"

    # Runtime dimension require execution verbose output.
    # BRGEMM driver through ukernel API supports verbose only at execution.
    profile_mode = "create"
    benchdnn_mode = "I"
    if driver in ("matmul", "reorder", "brgemm"):
        profile_mode = "exec"
        benchdnn_mode = "R"
    # Add extra noise (dispatch, etc.) to ensure it gets filtered out
    sub_env["ONEDNN_VERBOSE"] = f"dispatch,error,check,profile_{profile_mode}"

    sub_args = [
        benchdnn_exe,
        f"--engine={engine}",
        f"--{driver}",
        f"--mode={benchdnn_mode}",
        f"--batch={batch}",
    ]
    try:
        sub = subprocess.run(
            sub_args,
            capture_output=True,
            text=True,
            env=sub_env,
        )
    except Exception as e:
        raise TestingException(
            f"subprocess.run() raised exception: {e!s}"
        ) from None

    if sub.returncode != 0:
        # most likely converter generated incorrect batch file
        raise TestingException(
            f"""
             subprocess.run() returned {sub.returncode},
             args: {sub_args}
             stderr: {sub.stderr}
             """
        )

    filter_event = "exec" if benchdnn_mode == "R" else "create"
    return filter_verbose(sub.stdout, driver, filter_event)


def generate_batch(verbose, driver):
    import verbose_converter  # type: ignore[import-not-found]

    verbose = verbose.splitlines()
    aggregate_opts = [
        "engine",
        "prim_kind",
        "impl",
        "prop_kind",
        "mds",
        "exts",
        "alg_kind",
        "shapes",
    ]
    data = verbose_converter.convert(
        parser="oneDNN",
        input=verbose,
        action="generate",
        generator="benchdnn",
        split_output=True,
        agg_keys=aggregate_opts,
    )

    filename = f"{driver}.test.generated"
    output = data.get(driver, "")
    with open(filename, "w") as fd:
        fd.write(f"{output}\n")
    return filename


def compare(driver, ref_v, comp_v):
    def filter_lines(lines):
        for line in lines.splitlines():
            if driver in line:
                yield line

    def without_impl(verbose_line):
        parts = verbose_line.split(",")
        return ",".join(parts[:6] + parts[7:])

    def find_named_entry(name, entries):
        for entry in entries:
            entry_name, *entry_args = entry.split(":")
            if entry_name == name:
                return entry_args
        return None

    def accept_results(r, c):
        if r == c:
            return True

        # TODO: Handle cases with non-unique md tags
        #  * multiple size-1 dimensions with the same stride
        #  * multiple dimensions with 0 stride
        if driver == "matmul":
            # In matmul cases with runtime dims that resolve to ones, the bias
            # memory descriptor will potentially have the wrong mask printed in
            # the verbose line. We do not maintain enough information to always
            # print the correct mask, but the reference and computed verbose
            # lines will match, up to implementation name.
            parts = r.split(",")
            mds = parts[8].split()
            aux = parts[10].split()
            shapes = parts[11].split(":", 1)
            wei, act = list(map(lambda x: list(map(int, x.split("x"))), shapes))
            if find_named_entry("bia", mds) is None:
                return False
            rt_dim_mask = find_named_entry("runtime_dims_masks", aux)
            if rt_dim_mask is None:
                return False
            wei_mask, act_mask = list(map(int, rt_dim_mask))
            if wei[-2] == 1 and wei_mask & (1 << (len(wei) - 2)):
                return without_impl(r) == without_impl(c)
            if act[-1] == 1 and act_mask & (1 << (len(act) - 1)):
                return without_impl(r) == without_impl(c)
        elif driver == "sum":
            # There is no information in a sum verbose line about scales, so if
            # dispatch depends on particular scale values, the implementation
            # may change with default scales. In this case, we check that the
            # rest of the verbose line is the same.
            return without_impl(r) == without_impl(c)
        return False

    file_map = {"reference": ref_v, "computed": comp_v}
    for r, c in zip(filter_lines(ref_v), filter_lines(comp_v)):
        if accept_results(r, c):
            continue
        for log_type, content in file_map.items():
            with open(f"{driver}.{log_type}.log", "w") as fd:
                fd.write(content)
        raise TestingException(
            f"""
             verboses do not match
             ref: {r}
             com: {c}
             """
        )


def test(path_to_benchdnn, engine, driver, batch):
    ref_verbose = generate_verbose(path_to_benchdnn, engine, driver, batch)
    # XXX: Maybe generate batch and run benchdnn for each verbose line
    # separately to detect error on case level and not on batch level?
    # The reason behind testing on batch level is that ref_verbose generator
    # might introduce multiple verbose lines for single line in batch file
    com_batch = generate_batch(ref_verbose, driver)
    com_verbose = generate_verbose(path_to_benchdnn, engine, driver, com_batch)
    compare(driver, ref_verbose, com_verbose)
    # XXX: Maybe run an additional loop
    #    ref -> ref verbose -> com 1 -> com 1 verbose -> com 2 -> com 2 verbose
    # Comparing com 1 and com 2 verbose instead would address the special cases
    # in accept_results. We can even compare just the cases where ref and com 1
    # don't match.


def main():
    relpath = "../../../build/tests/benchdnn"
    realpath = os.path.dirname(os.path.realpath(__file__))
    realpath_benchdnn = os.path.realpath(f"{realpath}/{relpath}")
    args_parser = argparse.ArgumentParser(
        description="benchdnn test", formatter_class=RawTextHelpFormatter
    )
    args_parser.add_argument(
        "-e",
        "--engine",
        default="cpu",
        choices=("cpu", "gpu"),
        help="Engine to use to run tests",
    )
    args_parser.add_argument(
        "-d",
        "--dataset",
        default=f"{realpath}/dataset_simple",
        help="input with benchdnn batch files",
    )
    args_parser.add_argument(
        "-b",
        "--benchdnn_path",
        default=realpath_benchdnn,
        help="Path to benchdnn executable",
    )
    args_parser.add_argument(
        "-i",
        "--inputs_path",
        default=f"{realpath_benchdnn}/inputs",
        help="Path to benchdnn batch files",
    )
    args = args_parser.parse_args()

    failed = False
    with open(args.dataset, "r") as dataset:
        for case in dataset.readlines():
            case = case.split("#", 1)[0].strip()
            if not case:
                continue
            driver, batch = case.split(",")
            batch = batch.split("\n", 1)[0]
            batch_file_path = f"{args.inputs_path}/{driver}/{batch}"
            test_info = f"BENCHDNN TEST: {args.engine}, {driver}, {batch}"
            try:
                test(args.benchdnn_path, args.engine, driver, batch_file_path)
            except Exception as e:
                print(f"{test_info}: FAILED {e!s}")
                failed = True
            else:
                print(f"{test_info}: PASSED")
    return failed


def get_driver(primitive: str):
    import src.benchdnn_generator as bg  # type: ignore[import-not-found]

    try:
        converter = bg.get_converter(primitive)
    except KeyError:
        return None
    else:
        return converter.driver


# Add parent dir to sys.path to make verbose_converter visible for test
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
