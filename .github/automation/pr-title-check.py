#!/usr/bin/python3

# *******************************************************************************
# Copyright 2024 Arm Limited and affiliates.
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


# * Ensuring the scopes end in colon and same level scopes are comma delimited.
# TODO: Limit scopes to an acceptable list of tags.
def __scopeCheck(msg: str):
    firstLine = (msg.partition("\n")[0]).strip()

    if not ":" in firstLine:
        raise ValueError(
            f"Please see contribution guidelines. First line must contain a scope ending in a colon. Got: {firstLine}"
        )

    # The last element of the split is the title, which we don't care about. Remove it.
    scopesArray = firstLine.split(":")[:-1]
    print("---")
    print(f"Scopes: {scopesArray}")

    for scopes in scopesArray:
        print("---")
        print(f"Same-level scope: {scopes}")
        numWords = len(scopes.split())
        numCommas = scopes.count(",")
        print(f"Number of words in scope: {numWords}")
        print(f"Number of commas in scope: {numCommas}")

        if numWords != numCommas + 1:
            raise ValueError(
                f"Please see contribution guidelines. Same-level scopes must be seperated by a comma. If this is true then words == commas + 1."
            )


# * Ensuring a character limit for the first line.
def __numCharacterCheck(msg: str):
    summary = msg.partition("\n")[0]
    msgSummaryLen = len(summary)
    if msgSummaryLen >= 72:
        raise ValueError(
            f"Please see contribution guidelines. Message summary must be less than 72. Got: {msgSummaryLen}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("msg", help="Commit message to check.")
    args = parser.parse_args()
    msg: str = args.msg
    print(f"msg: {msg}")
    __numCharacterCheck(msg)
    __scopeCheck(msg)


if __name__ == "__main__":
    main()
