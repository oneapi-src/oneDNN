/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <cstring>
#include <vector>

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_types.h"

#include "utils/timer.hpp"

struct check_mem_size_args_t {

    check_mem_size_args_t() = default;
    check_mem_size_args_t(const_dnnl_primitive_desc_t pd, bool want_input)
        : pd(pd)
        , want_input(want_input)
        , is_scratchpad(false)
        , total_size_device(0)
        , total_size_cpu(0)
        , scratchpad_size(0) {
        // initialize the memory size for reference path
        memset(total_ref_md_size, 0, sizeof(total_ref_md_size));
    }

    // Input args.
    const_dnnl_primitive_desc_t pd;
    bool want_input;
    bool is_scratchpad;

    // Output args:
    // `sizes` used to validate OpenCL memory requirements.
    std::vector<size_t> sizes;
    // `total_size_device` specifies memory allocated on device for a test obj.
    size_t total_size_device;
    // `total_size_cpu` specifies:
    // * Memory allocated for reference ocmputations (`C` mode only).
    // * Memory allocated for comparison results (`C` mode only).
    // * Memory allocated for mapping device memory (GPU backend only).
    // * Memory allocated on CPU for a test obj (CPU backend only).
    size_t total_size_cpu;
    // `total_ref_md_size` specifies the additional tag::abx f32 memory
    // required for correctness check.
    // * The first element refers to the total memory for input reference
    // * The second element refers to the total memory for output reference
    // The args are used in memory estimation for graph driver only.
    size_t total_ref_md_size[2];
    // `scratchpad_size` specifies a scratchpad size for specific checks.
    size_t scratchpad_size;
};

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

struct res_t {
    res_state_t state;
    size_t errors, total;
    timer::timer_map_t timer_map;
    std::string impl_name;
    std::string prim_ref_repro;
    std::string reason;
    size_t ibytes, obytes;

    // TODO: merge mem_check_dir into check_mem_size_args_t
    dir_t mem_check_dir = DIR_UNDEF;
    check_mem_size_args_t mem_size_args;
};

#endif
