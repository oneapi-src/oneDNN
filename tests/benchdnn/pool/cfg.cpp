/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

std::string str2cfg(const char *str) {
#define CASE(cfg) \
    if (!strcasecmp(STRINGIFY(cfg), str)) return str
    CASE(f32);
    CASE(f64);
    CASE(s32);
    CASE(f16);
    CASE(bf16);
    CASE(s8);
    CASE(u8);
    CASE(s8u8);
    CASE(u8s8);
    CASE(s8f32);
    CASE(f32s8);
    CASE(u8f32);
    CASE(f32u8);
    CASE(s8f16);
    CASE(f16s8);
    CASE(u8f16);
    CASE(f16u8);
#undef CASE
    BENCHDNN_PRINT(0, "Config name \'%s\' is not supported.\n", str);
    SAFE_V(CRIT);
    return std::string();
}

int handle_legacy_cfg(
        std::vector<dnnl_data_type_t> &dt, const std::string &cfg) {
    BENCHDNN_PRINT(0, "%s\n",
            "Warning: `--cfg=CFG` option is deprecated. Use `--dt=DT[:DT] "
            "instead.");

    if (cfg == "f32")
        dt = {dnnl_f32};
    else if (cfg == "f64")
        dt = {dnnl_f64};
    else if (cfg == "s32")
        dt = {dnnl_s32};
    else if (cfg == "f16")
        dt = {dnnl_f16};
    else if (cfg == "bf16")
        dt = {dnnl_bf16};
    else if (cfg == "s8")
        dt = {dnnl_s8};
    else if (cfg == "u8")
        dt = {dnnl_u8};
    else if (cfg == "u8s8")
        dt = {dnnl_u8, dnnl_s8};
    else if (cfg == "s8u8")
        dt = {dnnl_s8, dnnl_u8};
    else if (cfg == "u8f32")
        dt = {dnnl_u8, dnnl_f32};
    else if (cfg == "f32u8")
        dt = {dnnl_f32, dnnl_u8};
    else if (cfg == "s8f32")
        dt = {dnnl_s8, dnnl_f32};
    else if (cfg == "f32s8")
        dt = {dnnl_f32, dnnl_s8};
    else if (cfg == "u8f16")
        dt = {dnnl_u8, dnnl_f16};
    else if (cfg == "f16u8")
        dt = {dnnl_f16, dnnl_u8};
    else if (cfg == "s8f16")
        dt = {dnnl_s8, dnnl_f16};
    else if (cfg == "f16s8")
        dt = {dnnl_f16, dnnl_s8};
    else {
        BENCHDNN_PRINT(0, "Error: Config name \'%s\' is not supported.\n",
                cfg.c_str());
        return FAIL;
    }
    return OK;
}

} // namespace pool
