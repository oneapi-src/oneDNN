#!/usr/bin/python3

# *******************************************************************************
# Copyright 2024 Arm Limited and affiliates.
# Copyright 2024 Intel Corporation
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

import argparse
import subprocess
import re

# Ensure the scope ends in a colon and that same level scopes are
# comma delimited.
# Current implementation only checks the first level scope as ':' can be used
# in the commit description (ex: TBB::tbb or bf16:bf16).
# TODO: Limit scopes to an acceptable list of tags.
def __scopeCheck(msg: str):
    status = "Message scope: "

    if not re.match('^[a-z0-9]+(, [a-z0-9]+)*: ', msg):
        print(f"{status} FAILED: Commit message must follow the format "
               "<scope>:[ <scope>:] <short description>")
        return False

    print(f"{status} OK")
    return True

# Ensure  a character limit for the first line.
def __numCharacterCheck(msg: str):
    status = "Message length:"
    if len(msg) <= 72:
        print(f"{status} OK")
        return True
    else:
        print(f"{status} FAILED: Commit message summary must not "
               "exceed 72 characters.")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("head", help="Head commit of PR branch")
    parser.add_argument("base", help="Base commit of PR branch")
    args = parser.parse_args()
    base: str = args.base
    head: str = args.head

    commit_range = base + ".." + head
    messages = subprocess.run(["git", "rev-list", "--format=oneline",
        commit_range], capture_output=True, text=True).stdout

    is_ok = True
    for i in messages.splitlines():
      print(i)
      commit_msg=i.split(' ', 1)[1]
      result = __numCharacterCheck(commit_msg)
      is_ok = is_ok and result
      result = __scopeCheck(commit_msg)
      is_ok = is_ok and result

    if is_ok:
        print("All commmit messages are formatted correctly. ")
    else:
        print("Some commit message checks failed. Please align commit messages "
              "with Contributing Guidelines and update the PR.")
        exit(1)


if __name__ == "__main__":
    main()
