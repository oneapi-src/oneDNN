/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "brgemm.hpp"

namespace brgemm {

// Adjust density based on accumulation chain.
float cfg_t::get_density(const cfg_t::density_args_t &density_args) const {
    float density = 1.f;
    if (!is_bench_mode(CORR) || density_args.data_kind != SRC) return density;

    // Find the number of accumulators safe to use with the following equations:
    // Integer value can be expressed exactly with floating-point is
    // `PREC = (1 << std::numeric_limit::digits(dst_dt))`.
    // SUM_1_N(VALUES) <= PREC.   This should hold to get precise answer.
    // SUM_1_N(VALUES) <= N_ACC * MAX_VALUE <= PREC.  It's a top estimate, where
    // MAX_VALUE = MAX_VAL_SRC * MAX_VAL_WEI.
    // SAFE_N_ACC <= PREC / MAX_VALUE.

    const auto &cfg_e_src = cfg_entry_[SRC];
    const auto &cfg_e_wei = cfg_entry_[WEI];
    const auto &cfg_e_dst = cfg_entry_[DST];

    const int64_t max_value
            = cfg_e_src.get_range_abs_max() * cfg_e_wei.get_range_abs_max();
    const int64_t safe_n_acc
            = (1LL << digits_dt(cfg_e_dst.get_dt())) / max_value;
    assert(safe_n_acc > 0);
    density /= div_up(density_args.n_acc, safe_n_acc);
    return density;
}

// Using pow2 values allows to avoid catastrophic cancellation.
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

} // namespace brgemm
