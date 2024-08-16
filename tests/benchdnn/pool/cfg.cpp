/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"

#include "pool/pool.hpp"

namespace pool {

cfg_t::cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds) {
    output_data_kind_ = (prb->dir & FLAG_FWD) ? DST : SRC;
    for (const auto kind : kinds) {
        auto orig_data_type = prb->get_dt(kind);
        auto data_type = deduce_cfg_data_type(orig_data_type, prb->attr, kind);
        cfg_entry_.emplace(kind,
                cfg_entry_t {
                        kind, orig_data_type, data_type, get_cfg_map(kind)});
    }

    // Keep values for average algorithms positive to prevent cancellation err.
    if (prb->alg != alg_t::max) {
        set_range_min(SRC, 0);
        set_range_min(DST, 0);
    }

    BENCHDNN_PRINT(6, "%s SRC_%s=[%d;%d]\n", "[FILL_CFG]",
            dt2str(this->get_dt(SRC)), get_range_min(SRC), get_range_max(SRC));
}

cfg_t::cfg_entry_t::cfg_map_t cfg_t::get_cfg_map(data_kind_t kind) const {
    static const cfg_t::cfg_entry_t::cfg_map_t cfg_map = {
            {{dnnl_f64}, {-2048, 2048}},
            {{dnnl_f32}, {-2048, 2048}},
            {{dnnl_s32}, {-2048, 2048}},
            {{dnnl_bf16}, {-32, 32}},
            {{dnnl_f16}, {-32, 32}},
            {{dnnl_f8_e5m2}, {-8, 8}},
            {{dnnl_f8_e4m3}, {-8, 8}},
            {{dnnl_s8}, {INT8_MIN, INT8_MAX}},
            {{dnnl_u8}, {0, UINT8_MAX}},
    };

    switch (kind) {
        case SRC: return cfg_map;
        case DST: return cfg_map;
        default: assert(!"unsupported data kind"); break;
    }
    static cfg_t::cfg_entry_t::cfg_map_t dummy;
    return dummy;
}

} // namespace pool
