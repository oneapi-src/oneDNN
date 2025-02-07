/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef UTILS_RES_HPP
#define UTILS_RES_HPP

#include "oneapi/dnnl/dnnl_types.h"

#include "utils/timer.hpp"

#include <string>
#include <vector>

/* result structure */
enum res_state_t {
    UNTESTED = 0,
    PASSED,
    SKIPPED,
    MISTRUSTED,
    UNIMPLEMENTED,
    INVALID_ARGUMENTS,
    FAILED,
    LISTED,
    INITIALIZED,
    EXECUTED,
};

enum dir_t {
    DIR_UNDEF = 0,
    FLAG_DAT = 1,
    FLAG_WEI = 2,
    FLAG_BIA = 4,
    FLAG_FWD = 32,
    FLAG_BWD = 64,
    FLAG_INF = 128,
    FWD_D = FLAG_FWD + FLAG_DAT,
    FWD_I = FLAG_FWD + FLAG_DAT + FLAG_INF,
    FWD_B = FLAG_FWD + FLAG_DAT + FLAG_BIA,
    BWD_D = FLAG_BWD + FLAG_DAT,
    BWD_DW = FLAG_BWD + FLAG_DAT + FLAG_WEI,
    BWD_W = FLAG_BWD + FLAG_WEI,
    BWD_WB = FLAG_BWD + FLAG_WEI + FLAG_BIA,
};

struct check_mem_size_args_t {
    check_mem_size_args_t() = default;
    check_mem_size_args_t(
            const_dnnl_primitive_desc_t pd, bool want_input, dir_t dir)
        : pd(pd), want_input(want_input), dir(dir) {}

    // Input args.
    const_dnnl_primitive_desc_t pd = nullptr;
    bool want_input = false;
    dir_t dir = DIR_UNDEF; // See ANCHOR: MEM_CHECK_ARGS_DIR;

    // Output args:
    // `sizes` used to validate OpenCL memory requirements.
    std::vector<size_t> sizes;
    // `total_size_device` specifies memory allocated on device for a test obj.
    // It's an accumulated result of `sizes` values.
    size_t total_size_device = 0;
    // `total_size_ref` specifies Memory allocated for reference computations
    // (`C` mode only). This value can represent either memory sizes needed for
    // a naive reference implementation on plain formats, or memory sizes needed
    // for a prim_ref (--fast-ref) test object which can utilize blocked
    // formats.
    size_t total_size_ref = 0;
    // `total_size_compare` specifies memory allocated for comparison results
    // tensor (`C` mode only).
    size_t total_size_compare = 0;
    // `total_size_mapped` specifies memory allocated for mapped buffers on the
    // host (GPU backend only).
    size_t total_size_mapped = 0;
    // `total_ref_md_size` specifies the additional tag::abx f32 memory
    // required for correctness check.
    // * The first element refers to the total memory for input reference
    // * The second element refers to the total memory for output reference
    // The args are used in memory estimation for graph driver only.
    size_t total_ref_md_size[2] = {0, 0};
    // `scratchpad_size` specifies a scratchpad size for specific checks.
    size_t scratchpad_size = 0;
};

struct res_t {
    res_state_t state;
    size_t errors, total;
    timer::timer_map_t timer_map;
    std::string impl_name;
    std::string prim_ref_repro;
    std::string reason;
    // TODO: fuse `ibytes` and `obytes` into `mem_size_args`.
    size_t ibytes, obytes;
    check_mem_size_args_t mem_size_args;
};

#endif
