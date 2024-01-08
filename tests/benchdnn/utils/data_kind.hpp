/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef UTILS_DATA_KIND_HPP
#define UTILS_DATA_KIND_HPP

#define BENCHDNN_DNNL_ARG_UNDEF 0

enum data_kind_t {
    SRC = 0,
    WEI,
    BIA,
    DST,
    DIFF_DST,
    ACC,
    // bnorm, lnorm
    SRC_1,
    MEAN,
    VAR,
    SC,
    SH,
    // rnn
    DST_ITER,
    DST_ITER_C,
    AUGRU_ATTENTION,
    SRC_ITER,
    SRC_ITER_C,
    WEI_ITER,
    WEI_PEEPHOLE,
    WEI_PROJECTION,

    DAT_TOTAL,
};
const char *data_kind2str(data_kind_t kind);

// Returns correspondent `data_kind_t` value to a given execution `arg` value.
data_kind_t exec_arg2data_kind(int arg);

// Returns correspondent execution `arg` value to a given `data_kind_t` value.
int data_kind2exec_arg(data_kind_t dk);

#endif
