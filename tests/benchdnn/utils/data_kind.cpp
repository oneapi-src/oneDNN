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

#include "utils/data_kind.hpp"

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
