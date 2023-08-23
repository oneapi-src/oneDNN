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

#include "matmul.hpp"

namespace matmul {

cfg_t::cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds) {
    for (const auto kind : kinds) {
        auto orig_data_type = prb->get_dt(kind);
        auto data_type = deduce_cfg_data_type(orig_data_type, prb->attr, kind);
        cfg_entry_.emplace(kind,
                cfg_entry_t {
                        kind, orig_data_type, data_type, get_cfg_map(kind)});
    }

    // Use wider dst to test proper u8 loads.
    const bool is_int8_and_wide_dst = this->get_dt(SRC) == dnnl_u8
            && dnnl_data_type_size(this->get_dt(WEI)) == 1
            && dnnl_data_type_size(this->get_dt(DST)) >= 4;
    if (is_int8_and_wide_dst) { set_range_max(SRC, 160); }

    BENCHDNN_PRINT(6,
            "[FILL_CFG] SRC_%s=[%d;%d]; WEI_%s=[%d;%d]; DST_%s=[%d;%d];\n",
            dt2str(this->get_dt(SRC)), get_range_min(SRC), get_range_max(SRC),
            dt2str(this->get_dt(WEI)), get_range_min(WEI), get_range_max(WEI),
            dt2str(this->get_dt(DST)), get_range_min(DST), get_range_max(DST));
}

// Adjust density based on accumulation chain.
float cfg_t::get_density(const cfg_t::density_args_t &density_args) const {
    float density = 1.f;
    if (!has_bench_mode_bit(mode_bit_t::corr) || density_args.data_kind != SRC)
        return density;

    const int64_t safe_n_acc = get_safe_n_acc();
    assert(safe_n_acc > 0);

    // Bump density for some empiric value for int8 validation to hit saturation
    // bound.
    float safe_density = (float)safe_n_acc / density_args.n_acc;
    if (is_int8()) safe_density *= 3.f;
    density = MIN2(density, safe_density);

    BENCHDNN_PRINT(6, "%s safe_n_acc=%d density=%f\n", "[FILL_CFG]",
            (int)safe_n_acc, density);

    return density;
}

cfg_t::cfg_entry_t::cfg_map_t cfg_t::get_cfg_map(data_kind_t kind) const {
    static const cfg_t::cfg_entry_t::cfg_map_t src_cfg_map = {
            {{dnnl_f32}, {-64, 64}},
            {{dnnl_bf16}, {-4, 4}},
            {{dnnl_f16}, {-4, 4}},
            {{dnnl_s8}, {-4, 4}},
            {{dnnl_u8}, {0, 8}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t wei_cfg_map = {
            {{dnnl_f32}, {-128, 128}},
            {{dnnl_bf16}, {-8, 8}},
            {{dnnl_f16}, {-2, 2}},
            {{dnnl_s8}, {-4, 4}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t bia_cfg_map = {
            {{dnnl_f32}, {-8, 8}},
            {{dnnl_bf16}, {-8, 8}},
            {{dnnl_f16}, {-8, 8}},
            {{dnnl_s8}, {-8, 8}},
            {{dnnl_u8}, {0, 8}},
            {{dnnl_s32}, {-8, 8}},
            // Bias can be empty, which is expressed through undefined dt.
            // This entry allows to modify a bias range in general path without
            // branching whether bias is present or not.
            // Applicable for graph driver data displacer.
            {{dnnl_data_type_undef}, {0, 0}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t dst_cfg_map = {
            {{dnnl_f32}, {-8, 8}},
            {{dnnl_bf16}, {-8, 8}},
            {{dnnl_f16}, {-4, 4}},
            {{dnnl_s8}, {-4, 4}},
            {{dnnl_u8}, {0, 8}},
            {{dnnl_s32}, {-128, 128}},
    };

    switch (kind) {
        case SRC: return src_cfg_map;
        case WEI: return wei_cfg_map;
        case BIA: return bia_cfg_map;
        case DST: return dst_cfg_map;
        default: assert(!"unsupported data kind"); break;
    }
    static cfg_t::cfg_entry_t::cfg_map_t dummy;
    return dummy;
}

} // namespace matmul
