/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "self/self.hpp"

namespace self {

int bench(int argc, char **argv) {
    SAFE(argc <= 1 ? OK : FAIL, CRIT);

    (void)argv;
    auto &bs = benchdnn_stat;
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        bs.skipped = bs.tests;
        return OK;
    }

    common();
    res();
    conv();
    bnorm();
    memory();
    graph();
    norm();

    return bs.tests == bs.passed ? OK : FAIL;
}

} // namespace self
