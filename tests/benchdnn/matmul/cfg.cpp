/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

// Adjust density based on accumulation chain.
float cfg_t::get_density(data_kind_t dk, int64_t n_acc) const {
    float density = 1.f;
    if (dk != SRC) return density;

    // Find the number of accumulators save to use with the following equations:
    // Integer value can be expressed exactly with floating-point is
    // `PREC = (1 << std::numeric_limit::digits(dst_dt))`.
    // SUM_1_N(VALUES) <= PREC.   This should hold to get precise answer.
    // SUM_1_N(VALUES) <= N_ACC * MAX_VALUE <= PREC.  It's a top estimate, where
    // MAX_VALUE = MAX_VAL_SRC * MAX_VAL_WEI.
    // SAFE_N_ACC <= PREC / MAX_VALUE.

    const auto &cfg_e_src = cfg_entry[SRC];
    const auto &cfg_e_wei = cfg_entry[WEI];
    const auto &cfg_e_dst = cfg_entry[DST];

    const int64_t max_value
            = cfg_e_src.get_range_abs_max() * cfg_e_wei.get_range_abs_max();
    const int64_t safe_n_acc = (1 << digits_dt(cfg_e_dst.get_dt())) / max_value;
    assert(safe_n_acc > 0);
    density /= div_up(n_acc, safe_n_acc);
    return density;
}

// Using pow2 values allows to avoid catastrophic cancellation.
const cfg_entry_t::cfg_map_t &cfg_entry_t::get_cfg_map() const {
    static const cfg_entry_t::cfg_map_t cfg_map = {
            // f32
            {{SRC, dnnl_f32}, {-64, 64}},
            {{WEI, dnnl_f32}, {-128, 128}},
            {{BIA, dnnl_f32}, {-8, 8}},
            {{DST, dnnl_f32}, {-8, 8}},
            // bf16
            {{SRC, dnnl_bf16}, {-4, 4}},
            {{WEI, dnnl_bf16}, {-8, 8}},
            {{BIA, dnnl_bf16}, {-8, 8}},
            {{DST, dnnl_bf16}, {-8, 8}},
            // f16
            {{SRC, dnnl_f16}, {-4, 4}},
            {{WEI, dnnl_f16}, {-2, 2}},
            {{BIA, dnnl_f16}, {-8, 8}},
            {{DST, dnnl_f16}, {-4, 4}},
            // s8
            {{SRC, dnnl_s8}, {-4, 4}},
            {{WEI, dnnl_s8}, {-4, 4}},
            {{BIA, dnnl_s8}, {-8, 8}},
            {{DST, dnnl_s8}, {-4, 4}},
            // u8
            {{SRC, dnnl_u8}, {0, 8}},
            {{BIA, dnnl_u8}, {0, 8}},
            {{DST, dnnl_u8}, {0, 8}},
            // s32
            {{BIA, dnnl_s32}, {-8, 8}},
            {{DST, dnnl_s32}, {-128, 128}},
    };
    return cfg_map;
}

const cfg_entry_t::cfg_range_t &cfg_entry_t::get_cfg_range() const {
    const cfg_key_t key(data_kind_, data_type_);
    const auto &cfg_map = get_cfg_map();
    const auto it = cfg_map.find(key);
    if (it != cfg_map.end()) return (*it).second;
    assert(!"unexpected");
    static cfg_entry_t::cfg_range_t dummy;
    return dummy;
}

std::string str2cfg(const char *str) {
    std::string s;
#define CASE(cfg) \
    if (!strcasecmp(STRINGIFY(cfg), str)) return s = str, s;
    CASE(f32);
    CASE(f16);
    CASE(f16f16s8);
    CASE(f16f16u8);
    CASE(u8s8f32);
    CASE(u8s8s32);
    CASE(u8s8s8);
    CASE(u8s8u8);
    CASE(s8s8f32);
    CASE(s8s8s32);
    CASE(s8s8s8);
    CASE(s8s8u8);
    CASE(s8s8bf16);
    CASE(u8s8bf16);
    CASE(s8s8f16);
    CASE(u8s8f16);
    CASE(bf16bf16f32);
    CASE(bf16bf16bf16);
    CASE(f32bf16bf16);
    CASE(bf16f32bf16);
#undef CASE

    BENCHDNN_PRINT(0, "Config name \'%s\' is not supported.\n", str);
    SAFE_V(CRIT);
    return std::string();
}

void handle_legacy_cfg(
        std::vector<dnnl_data_type_t> &dt, const std::string &cfg) {
    if (cfg == "f32")
        dt = {dnnl_f32};
    else if (cfg == "bf16bf16bf16")
        dt = {dnnl_bf16};
    else if (cfg == "f16")
        dt = {dnnl_f16};
    else if (cfg == "f16f16s8")
        dt = {dnnl_f16, dnnl_f16, dnnl_s8};
    else if (cfg == "f16f16u8")
        dt = {dnnl_f16, dnnl_f16, dnnl_u8};
    else if (cfg == "u8s8f32")
        dt = {dnnl_u8, dnnl_s8, dnnl_f32};
    else if (cfg == "u8s8s32")
        dt = {dnnl_u8, dnnl_s8, dnnl_s32};
    else if (cfg == "u8s8s8")
        dt = {dnnl_u8, dnnl_s8, dnnl_s8};
    else if (cfg == "u8s8u8")
        dt = {dnnl_u8, dnnl_s8, dnnl_u8};
    else if (cfg == "s8s8f32")
        dt = {dnnl_s8, dnnl_s8, dnnl_f32};
    else if (cfg == "s8s8s32")
        dt = {dnnl_s8, dnnl_s8, dnnl_s32};
    else if (cfg == "s8s8s8")
        dt = {dnnl_s8, dnnl_s8, dnnl_s8};
    else if (cfg == "s8s8u8")
        dt = {dnnl_s8, dnnl_s8, dnnl_u8};
    else if (cfg == "s8s8bf16")
        dt = {dnnl_s8, dnnl_s8, dnnl_bf16};
    else if (cfg == "u8s8bf16")
        dt = {dnnl_u8, dnnl_s8, dnnl_bf16};
    else if (cfg == "s8s8f16")
        dt = {dnnl_s8, dnnl_s8, dnnl_f16};
    else if (cfg == "u8s8f16")
        dt = {dnnl_u8, dnnl_s8, dnnl_f16};
    else if (cfg == "bf16bf16f32")
        dt = {dnnl_bf16, dnnl_bf16, dnnl_f32};
    else if (cfg == "f32bf16bf16")
        dt = {dnnl_f32, dnnl_bf16, dnnl_bf16};
    else if (cfg == "bf16f32bf16")
        dt = {dnnl_bf16, dnnl_f32, dnnl_bf16};
    else {
        BENCHDNN_PRINT(
                0, "Config name \'%s\' is not supported.\n", cfg.c_str());
        SAFE_V(CRIT);
    }
}

} // namespace matmul
