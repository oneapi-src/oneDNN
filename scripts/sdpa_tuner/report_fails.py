#!/usr/bin/env python3
# *******************************************************************************
# Copyright 2026 Intel Corporation
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
"""Compare the tiler's validity rules with what a sweep observed.

    python report_fails.py sweep1.csv sweep2.csv ...

Lists the configs that failed to build (status fail) but that the library's
fwd_describe() still accepts, grouped by config with the problem features
that usually explain a generator limit, and the configs that ran but the
library rejects (a rule that is too broad). Reproduce one failed config
with ONEDNN_VERBOSE=all --mode=I to read the generator's reason before
adding a rule to select.cpp, then rerun this script: both lists should be
empty.
"""

import csv
import os
import sys
from collections import OrderedDict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_fwd_model as fm  # noqa: E402
import sdpa_model as sm  # noqa: E402


def features(r):
    c = [int(x) for x in r["cfg"].split(",")]
    return "sg=%d kq=%d vs=%d qt=%d dmax=%s al=%s/%s tk=%s" % (
        c[4] * c[5],
        c[0] * c[1],
        c[2] * c[3],
        c[1] * c[5],
        r.get("p_d_max_kq", "?"),
        r.get("p_q_align", "?"),
        r.get("p_k_align", "?"),
        "1" if r.get("p_transpose_k") == "True" else "0",
    )


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    lib_path = ""
    for a in sys.argv[1:]:
        if a.startswith("--lib="):
            lib_path = a.split("=", 1)[1]
    if not paths:
        sys.exit(__doc__)
    lib = sm.Library(lib_path or None)
    fm.LIB = lib
    accepted: Dict[str, List[Dict[str, str]]] = OrderedDict()
    rejected: Dict[str, List[Dict[str, str]]] = OrderedDict()
    n = fails = 0
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("cfg"):
                    continue
                n += 1
                rr = dict(r)
                rr["ms_min"] = rr.get("ms_min") or "0"
                row = fm.Row(rr)
                valid = lib.estimate(row.hw, row.prb, r["cfg"]) is not None
                if r["status"] == "fail":
                    fails += 1
                    if valid:
                        g = accepted.setdefault(r["cfg"], [])
                        g.append(r)
                elif not valid:
                    rejected.setdefault(r["cfg"], []).append(r)
    print(
        "%d rows, %d failed; failed configs the library still accepts: %d, "
        "running configs it rejects: %d"
        % (
            n,
            fails,
            sum(len(v) for v in accepted.values()),
            sum(len(v) for v in rejected.values()),
        )
    )
    for title, groups in (
        ("failed but accepted", accepted),
        ("ran but rejected", rejected),
    ):
        if not groups:
            continue
        print("\n%s:" % title)
        for cfg, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print("  %-26s x%-3d %s" % (cfg, len(rs), features(rs[0])))
            seen = set()
            for r in rs:
                p = r["problem"]
                if p in seen:
                    continue
                seen.add(p)
                print("      %s" % p[:100])
                if len(seen) >= 3:
                    break


if __name__ == "__main__":
    main()
