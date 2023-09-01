/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <cassert>
#include <vector>

#include "oneapi/dnnl/dnnl_types.h"

#include "common.hpp"
#include "utils/data_kind.hpp"

struct data_kind_entry_t {
    data_kind_t dk;
    std::vector<int> exec_args;
};

static data_kind_entry_t data_kind_table[] = {
        {SRC, {DNNL_ARG_SRC, DNNL_ARG_DIFF_SRC}},
        {WEI, {DNNL_ARG_WEIGHTS, DNNL_ARG_DIFF_WEIGHTS}},
        {BIA, {DNNL_ARG_BIAS, DNNL_ARG_DIFF_BIAS}},
        {DST, {DNNL_ARG_DST, DNNL_ARG_DIFF_DST}},
        {DAT_TOTAL, {DNNL_ARG_SCRATCHPAD}},
};

data_kind_t exec_arg2data_kind(int arg) {
    for (const auto &e : data_kind_table) {
        for (const auto &a : e.exec_args) {
            if (a == arg) return e.dk;
        }
    }

    int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
            - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
    bool is_post_ops_arg = (arg & post_ops_range);
    bool is_dw_post_op = (arg & DNNL_ARG_ATTR_POST_OP_DW);
    bool is_scales_arg = (arg & DNNL_ARG_ATTR_SCALES);
    bool is_zero_point_arg = (arg & DNNL_ARG_ATTR_ZERO_POINTS);
    if (!is_post_ops_arg && !is_dw_post_op && !is_scales_arg
            && !is_zero_point_arg)
        BENCHDNN_PRINT(0, "Error: arg \'%d\' was not recognized\n", arg);

    const auto table_size = sizeof(data_kind_table) / sizeof(*data_kind_table);
    return data_kind_table[table_size - 1].dk;
}

const char *data_kind2str(data_kind_t kind) {
    switch (kind) {
        case SRC: return "SRC";
        case SRC_1: return "SRC_ADD";
        case WEI: return "WEI";
        case BIA: return "BIA";
        case DST: return "DST";
        case DIFF_DST: return "DIFF_DST";
        case ACC: return "ACC";
        case MEAN: return "MEAN";
        case VAR: return "VAR";
        case SC: return "SC";
        case SH: return "SH";
        case DST_ITER: return "DST_ITER";
        case DST_ITER_C: return "DST_ITER_C";
        case AUGRU_ATTENTION: return "AUGRU_ATTENTION";
        case SRC_ITER: return "SRC_ITER";
        case SRC_ITER_C: return "SRC_ITER_C";
        case WEI_ITER: return "WEI_ITER";
        case WEI_PEEPHOLE: return "WEI_PEEPHOLE";
        case WEI_PROJECTION: return "WEI_PROJECTION";
        default: assert(!"incorrect data kind");
    }
    return "incorrect data kind";
}
