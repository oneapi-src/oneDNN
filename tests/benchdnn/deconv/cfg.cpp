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

#include "deconv/deconv.hpp"

namespace deconv {

cfg_t::cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds) {
    output_data_kind_ = (prb->dir & FLAG_FWD) ? DST
            : (prb->dir & FLAG_WEI)           ? WEI
                                              : SRC;
    for (const auto kind : kinds) {
        auto orig_data_type = prb->get_dt(kind);
        auto data_type = deduce_cfg_data_type(orig_data_type, prb->attr, kind);
        cfg_entry_.emplace(kind,
                cfg_entry_t {
                        kind, orig_data_type, data_type, get_cfg_map(kind)});
    }

    adjust_ranges_for_safe_n_acc();

    // Use wider dst to test proper u8 loads.
    const bool is_int8_and_wide_dst
            = dnnl_data_type_size(this->get_dt(WEI)) == 1
            && dnnl_data_type_size(this->get_dt(DST)) >= 4;
    if (is_int8_and_wide_dst) { this->set_range_max(SRC, 160); }

    // For s8s8 weights have to be even to comply with adjust_scale of 0.5f.
    // Divide the range by factor of two here, and multiply values by factor
    // of two when do filling.
    const bool is_s8s8
            = this->get_dt(SRC) == dnnl_s8 && this->get_dt(WEI) == dnnl_s8;
    if (is_s8s8) {
        this->set_range_min(WEI, -2);
        this->set_range_max(WEI, 2);
    }

    BENCHDNN_PRINT(6,
            "[FILL_CFG] SRC_%s=[%d;%d]; WEI_%s=[%d;%d]; DST_%s=[%d;%d];\n",
            dt2str(this->get_dt(SRC)), get_range_min(SRC), get_range_max(SRC),
            dt2str(this->get_dt(WEI)), get_range_min(WEI), get_range_max(WEI),
            dt2str(this->get_dt(DST)), get_range_min(DST), get_range_max(DST));
}

// Adjust density based on accumulation chain.
float cfg_t::get_density(const cfg_t::density_args_t &density_args) const {
    float density = 1.f;
    std::string safe_n_acc_str = "N/A";

    const data_kind_t allowed_non_dense_kind
            = output_data_kind_ == DST ? SRC : DST;

    if (has_bench_mode_bit(mode_bit_t::corr)
            && density_args.data_kind == allowed_non_dense_kind) {
        int64_t safe_n_acc = get_safe_n_acc();
        assert(safe_n_acc > 0);
        safe_n_acc_str = std::to_string(safe_n_acc);

        // Bump density for some empiric value for int8 validation to hit
        // saturation bound.
        float safe_density = (float)safe_n_acc / density_args.n_acc;
        if (is_int8()) safe_density *= 3.f;
        density = MIN2(density, safe_density);
    }

    BENCHDNN_PRINT(6, "[FILL_CFG][%s] n_acc=%lld safe_n_acc=%s; density=%f\n",
            data_kind2str(density_args.data_kind),
            (long long)density_args.n_acc, safe_n_acc_str.c_str(), density);

    return density;
}

cfg_t::cfg_entry_t::cfg_map_t cfg_t::get_cfg_map(data_kind_t kind) const {
    static const cfg_t::cfg_entry_t::cfg_map_t src_cfg_map = {
            {{dnnl_f64}, {-32, 32}},
            {{dnnl_f32}, {-32, 32}},
            {{dnnl_bf16}, {-4, 4}},
            {{dnnl_f16}, {-4, 4}},
            {{dnnl_s8}, {-4, 4}},
            {{dnnl_u8}, {0, 8}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t wei_cfg_map = {
            {{dnnl_f64}, {-32, 32}},
            {{dnnl_f32}, {-32, 32}},
            {{dnnl_bf16}, {-8, 8}},
            {{dnnl_f16}, {-2, 2}},
            {{dnnl_s8}, {-4, 4}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t bia_cfg_map = {
            {{dnnl_f64}, {-8, 8}},
            {{dnnl_f32}, {-8, 8}},
            {{dnnl_bf16}, {-8, 8}},
            {{dnnl_f16}, {-8, 8}},
            {{dnnl_s8}, {-8, 8}},
            {{dnnl_u8}, {0, 8}},
            {{dnnl_s32}, {-8, 8}},
    };

    static const cfg_t::cfg_entry_t::cfg_map_t dst_cfg_map = {
            {{dnnl_f64}, {-8, 8}},
            {{dnnl_f32}, {-8, 8}},
            {{dnnl_bf16}, {-4, 4}},
            {{dnnl_f16}, {-4, 4}},
            {{dnnl_s8}, {-4, 4}},
            {{dnnl_u8}, {0, 160}},
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

std::string str2cfg(const char *str) {
    std::string s;
#define CASE(cfg) \
    if (!strcasecmp(STRINGIFY(cfg), str)) return s = str, s
    CASE(f16);
    CASE(f32);
    CASE(f32_wino);
    CASE(f64);
    CASE(u8s8f32);
    CASE(u8s8f16);
    CASE(u8s8bf16);
    CASE(u8s8s32);
    CASE(u8s8s8);
    CASE(u8s8u8);
    CASE(s8s8f32);
    CASE(s8s8f16);
    CASE(s8s8bf16);
    CASE(s8s8s32);
    CASE(s8s8s8);
    CASE(s8s8u8);
    CASE(bf16bf16f32);
    CASE(bf16bf16f16);
    CASE(bf16bf16s8);
    CASE(bf16bf16u8);
    CASE(bf16bf16bf16);
    CASE(f32bf16bf16);
    CASE(bf16f32bf16);
    CASE(f32f32s8);
    CASE(f32f32u8);
    CASE(f16f16f32);
    CASE(f16f16s8);
    CASE(f16f16u8);
    CASE(f32f16f16);
    CASE(f16f32f16);
#undef CASE
    BENCHDNN_PRINT(0, "Config name \'%s\' is not supported.\n", str);
    SAFE_V(CRIT);
    return std::string();
}

int handle_legacy_cfg(
        std::vector<dnnl_data_type_t> &dt, const std::string &cfg) {
    BENCHDNN_PRINT(0, "%s\n",
            "Warning: `--cfg=CFG` option is deprecated. Use `--dt=DT[:DT:DT] "
            "instead.");

    if (cfg == "f32")
        dt = {dnnl_f32};
    else if (cfg == "f32_wino")
        dt = {dnnl_f32};
    else if (cfg == "bf16bf16bf16")
        dt = {dnnl_bf16};
    else if (cfg == "f16")
        dt = {dnnl_f16};
    else if (cfg == "f64")
        dt = {dnnl_f64};
    else if (cfg == "u8s8f32")
        dt = {dnnl_u8, dnnl_s8, dnnl_f32};
    else if (cfg == "u8s8f16")
        dt = {dnnl_u8, dnnl_s8, dnnl_f16};
    else if (cfg == "u8s8bf16")
        dt = {dnnl_u8, dnnl_s8, dnnl_bf16};
    else if (cfg == "u8s8s32")
        dt = {dnnl_u8, dnnl_s8, dnnl_s32};
    else if (cfg == "u8s8s8")
        dt = {dnnl_u8, dnnl_s8, dnnl_s8};
    else if (cfg == "u8s8u8")
        dt = {dnnl_u8, dnnl_s8, dnnl_u8};
    else if (cfg == "s8s8f32")
        dt = {dnnl_s8, dnnl_s8, dnnl_f32};
    else if (cfg == "s8s8f16")
        dt = {dnnl_s8, dnnl_s8, dnnl_f16};
    else if (cfg == "s8s8bf16")
        dt = {dnnl_s8, dnnl_s8, dnnl_bf16};
    else if (cfg == "s8s8s32")
        dt = {dnnl_s8, dnnl_s8, dnnl_s32};
    else if (cfg == "s8s8s8")
        dt = {dnnl_s8, dnnl_s8, dnnl_s8};
    else if (cfg == "s8s8u8")
        dt = {dnnl_s8, dnnl_s8, dnnl_u8};
    else if (cfg == "f16f16f32")
        dt = {dnnl_f16, dnnl_f16, dnnl_f32};
    else if (cfg == "f16f16s8")
        dt = {dnnl_f16, dnnl_f16, dnnl_s8};
    else if (cfg == "f16f16u8")
        dt = {dnnl_f16, dnnl_f16, dnnl_u8};
    else if (cfg == "bf16bf16f32")
        dt = {dnnl_bf16, dnnl_bf16, dnnl_f32};
    else if (cfg == "bf16bf16f16")
        dt = {dnnl_bf16, dnnl_bf16, dnnl_f16};
    else if (cfg == "bf16bf16s8")
        dt = {dnnl_bf16, dnnl_bf16, dnnl_s8};
    else if (cfg == "bf16bf16u8")
        dt = {dnnl_bf16, dnnl_bf16, dnnl_u8};
    else if (cfg == "f32bf16bf16")
        dt = {dnnl_f32, dnnl_bf16, dnnl_bf16};
    else if (cfg == "f32f32s8")
        dt = {dnnl_f32, dnnl_f32, dnnl_s8};
    else if (cfg == "f32f32u8")
        dt = {dnnl_f32, dnnl_f32, dnnl_u8};
    else if (cfg == "bf16f32bf16")
        dt = {dnnl_bf16, dnnl_f32, dnnl_bf16};
    else if (cfg == "f32f16f16")
        dt = {dnnl_f32, dnnl_f16, dnnl_f16};
    else if (cfg == "f16f32f16")
        dt = {dnnl_f16, dnnl_f32, dnnl_f16};
    else {
        BENCHDNN_PRINT(0, "Error: Config name \'%s\' is not supported.\n",
                cfg.c_str());
        return FAIL;
    }
    return OK;
}

} // namespace deconv
