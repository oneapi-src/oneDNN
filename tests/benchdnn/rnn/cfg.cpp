/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

#include <set>

#include "dnnl_common.hpp"
#include "rnn/rnn.hpp"

namespace rnn {

namespace {

#define CASE(KIND, ENTRY) \
    if (kind == (KIND)) return ENTRY
#define DEFAULT(ENTRY) return ENTRY
#define END_LIST \
    SAFE_V(CRIT); \
    return F32_ENTRY

#define CFG_INTERNAL(name, alias) \
    struct conf_##name##_t : dt_conf_t { \
        using dt_conf_t::dt_conf_t; \
        const entry_t &operator[](rnn_data_kind_t kind) const override; \
    } conf_##name(STRINGIFY(alias)); \
    const dt_conf_t::entry_t &conf_##name##_t::operator[]( \
            rnn_data_kind_t kind) const

std::set<const dt_conf_t *> cfg_list;
#define CFG(name) \
    struct conf_##name##_t : dt_conf_t { \
        using dt_conf_t::dt_conf_t; \
        const entry_t &operator[](rnn_data_kind_t kind) const override; \
    } conf_##name(STRINGIFY(name)); \
    static auto __reg_##name = cfg_list.insert(&conf_##name); \
    const dt_conf_t::entry_t &conf_##name##_t::operator[]( \
            rnn_data_kind_t kind) const

// f32
#define MIN_F32 0.0f
#define MAX_F32 .999999f
#define MEAN_F32 .5f
#define STDDEV_F32 0.01f
#define EPS_F32 epsilon_dt(dnnl_f32)
const int f32_max_exact = 1 << 24;
dt_conf_t::entry_t F32_ENTRY {dnnl_f32, -f32_max_exact, f32_max_exact, MIN_F32,
        MAX_F32, MEAN_F32, STDDEV_F32, EPS_F32};

#define UNUSED_REG_VAR(name) UNUSED(__reg_##name)

CFG(f32) {
    UNUSED_REG_VAR(f32);
    return F32_ENTRY;
}

// bf16
#define MIN_BF16 0.0f
#define MAX_BF16 .999999f
#define MEAN_BF16 .5f
#define STDDEV_BF16 0.01f
#define EPS_BF16 epsilon_dt(dnnl_bf16)
dt_conf_t::entry_t BF16_ENTRY_BF16 {dnnl_bf16, -f32_max_exact, f32_max_exact,
        MIN_BF16, MAX_BF16, MEAN_BF16, STDDEV_BF16, EPS_BF16};
dt_conf_t::entry_t BF16_ENTRY_F32 {dnnl_f32, -f32_max_exact, f32_max_exact,
        MIN_F32, MAX_F32, MEAN_F32, STDDEV_F32, EPS_BF16};

CFG(bf16f32) {
    UNUSED_REG_VAR(bf16f32);
    CASE(SRC_LAYER, BF16_ENTRY_BF16);
    CASE(SRC_ITER, BF16_ENTRY_BF16);
    CASE(WEIGHTS_LAYER, BF16_ENTRY_BF16);
    CASE(WEIGHTS_ITER, BF16_ENTRY_BF16);
    CASE(DST_ITER, BF16_ENTRY_BF16);
    CASE(DST_LAYER, BF16_ENTRY_BF16);
    CASE(AUGRU_ATTENTION, BF16_ENTRY_BF16);
    DEFAULT(BF16_ENTRY_F32);
}

CFG(bf16) {
    UNUSED_REG_VAR(bf16);
    CASE(SRC_LAYER, BF16_ENTRY_BF16);
    CASE(SRC_ITER, BF16_ENTRY_BF16);
    CASE(SRC_ITER_C, BF16_ENTRY_BF16);
    CASE(WEIGHTS_LAYER, BF16_ENTRY_BF16);
    CASE(WEIGHTS_ITER, BF16_ENTRY_BF16);
    CASE(WEIGHTS_PEEPHOLE, BF16_ENTRY_F32);
    CASE(WEIGHTS_PROJECTION, BF16_ENTRY_F32);
    CASE(BIAS, BF16_ENTRY_BF16);
    CASE(DST_ITER, BF16_ENTRY_BF16);
    CASE(DST_ITER_C, BF16_ENTRY_BF16);
    CASE(DST_LAYER, BF16_ENTRY_BF16);
    CASE(AUGRU_ATTENTION, BF16_ENTRY_BF16);
    DEFAULT(BF16_ENTRY_F32);
}

// bf32
dt_conf_t::entry_t BF32_ENTRY {dnnl_f32, -f32_max_exact, f32_max_exact,
        MIN_BF16, MAX_BF16, MEAN_BF16, STDDEV_BF16, EPS_BF16};
CFG_INTERNAL(bf32, f32) {
    CASE(BIAS, F32_ENTRY);
    DEFAULT(BF32_ENTRY);
}

// f16
#define MIN_F16 0.0f
#define MAX_F16 .999999f
#define MEAN_F16 .5f
#define STDDEV_F16 0.01f
#define EPS_F16 epsilon_dt(dnnl_f16)
dt_conf_t::entry_t F16_ENTRY {dnnl_f16, -f32_max_exact, f32_max_exact, MIN_F16,
        MAX_F16, MEAN_F16, STDDEV_F16, EPS_F16};
dt_conf_t::entry_t F16_ENTRY_F32 {dnnl_f32, -f32_max_exact, f32_max_exact,
        MIN_F32, MAX_F32, MEAN_F32, STDDEV_F32, EPS_F16};

CFG(f16) {
    UNUSED_REG_VAR(f16);
    CASE(SRC_LAYER, F16_ENTRY);
    CASE(SRC_ITER, F16_ENTRY);
    CASE(SRC_ITER_C, F16_ENTRY);
    CASE(WEIGHTS_LAYER, F16_ENTRY);
    CASE(WEIGHTS_ITER, F16_ENTRY);
    CASE(WEIGHTS_PEEPHOLE, F16_ENTRY_F32);
    CASE(WEIGHTS_PROJECTION, F16_ENTRY_F32);
    CASE(BIAS, F16_ENTRY);
    CASE(DST_ITER, F16_ENTRY);
    CASE(DST_ITER_C, F16_ENTRY);
    CASE(DST_LAYER, F16_ENTRY);
    CASE(AUGRU_ATTENTION, F16_ENTRY);
    DEFAULT(F16_ENTRY_F32);
}

CFG(f16f32) {
    UNUSED_REG_VAR(f16f32);
    CASE(SRC_LAYER, F16_ENTRY);
    CASE(SRC_ITER, F16_ENTRY);
    CASE(WEIGHTS_LAYER, F16_ENTRY);
    CASE(WEIGHTS_ITER, F16_ENTRY);
    CASE(DST_ITER, F16_ENTRY);
    CASE(DST_LAYER, F16_ENTRY);
    CASE(AUGRU_ATTENTION, F16_ENTRY);
    DEFAULT(F16_ENTRY_F32);
}

// f16_math_mode
dt_conf_t::entry_t F16_MATH_ENTRY {dnnl_f32, -f32_max_exact, f32_max_exact,
        MIN_F16, MAX_F16, MEAN_F16, STDDEV_F16, EPS_F16};
CFG_INTERNAL(f16_math, f32) {
    CASE(BIAS, F32_ENTRY);
    DEFAULT(F16_MATH_ENTRY);
}

// s8
#define EPS_U8 4e-3
#define EPS_S8 8e-3

#define MIN_U8 0.0f
#define MAX_U8 127.f
#define MEAN_U8 28.f
#define STDDEV_U8 16.f

#define MIN_S8 (-64.f)
#define MAX_S8 64.f
#define MEAN_S8 8.f
#define STDDEV_S8 32.f
#define MEAN_WEIGHT_S8 0.f

dt_conf_t::entry_t U8_ENTRY_U8_EXACT {
        dnnl_u8, 0, UINT8_MAX, MIN_U8, MAX_U8, MEAN_U8, STDDEV_U8, 0.f};
dt_conf_t::entry_t U8_ENTRY_U8 {
        dnnl_u8, 0, UINT8_MAX, MIN_U8, MAX_U8, MEAN_U8, STDDEV_U8, EPS_U8};
dt_conf_t::entry_t U8_ENTRY_S8 {dnnl_s8, INT8_MIN, INT8_MAX, MIN_S8, MAX_S8,
        MEAN_WEIGHT_S8, STDDEV_S8, EPS_S8};
dt_conf_t::entry_t U8_ENTRY_F32 {dnnl_f32, -f32_max_exact, f32_max_exact,
        MIN_F32, MAX_F32, MEAN_F32, STDDEV_F32, EPS_F32};

dt_conf_t::entry_t S8_ENTRY_S8_EXACT {
        dnnl_s8, INT8_MIN, INT8_MAX, 0, MAX_S8, MEAN_S8, STDDEV_S8, 0.f};
dt_conf_t::entry_t S8_ENTRY_S8 {
        dnnl_s8, INT8_MIN, INT8_MAX, 0, MAX_S8, MEAN_S8, STDDEV_S8, EPS_S8};
dt_conf_t::entry_t S8_ENTRY_WEIGHT_S8 {dnnl_s8, INT8_MIN, INT8_MAX, MIN_S8,
        MAX_S8, MEAN_WEIGHT_S8, STDDEV_S8, EPS_S8};
dt_conf_t::entry_t S8_ENTRY_F32 {dnnl_f32, -f32_max_exact, f32_max_exact,
        MIN_F32, MAX_F32, MEAN_F32, STDDEV_F32, EPS_F32};

CFG(u8u8u8u8) {
    UNUSED_REG_VAR(u8u8u8u8);
    CASE(SRC_LAYER, U8_ENTRY_U8);
    CASE(SRC_ITER, U8_ENTRY_U8);
    CASE(SRC_ITER_C, U8_ENTRY_F32);
    CASE(WEIGHTS_LAYER, U8_ENTRY_S8);
    CASE(WEIGHTS_ITER, U8_ENTRY_S8);
    CASE(WEIGHTS_PEEPHOLE, U8_ENTRY_F32);
    CASE(WEIGHTS_PROJECTION, U8_ENTRY_S8);
    CASE(BIAS, U8_ENTRY_F32);
    CASE(DST_ITER, U8_ENTRY_U8);
    CASE(DST_ITER_C, U8_ENTRY_F32);
    CASE(DST_LAYER, U8_ENTRY_U8_EXACT);
    CASE(AUGRU_ATTENTION, U8_ENTRY_U8);
    END_LIST;
}

CFG(u8u8u8f32) {
    UNUSED_REG_VAR(u8u8u8f32);
    CASE(SRC_LAYER, U8_ENTRY_U8);
    CASE(SRC_ITER, U8_ENTRY_U8);
    CASE(SRC_ITER_C, U8_ENTRY_F32);
    CASE(WEIGHTS_LAYER, U8_ENTRY_S8);
    CASE(WEIGHTS_ITER, U8_ENTRY_S8);
    CASE(WEIGHTS_PEEPHOLE, U8_ENTRY_F32);
    CASE(WEIGHTS_PROJECTION, U8_ENTRY_S8);
    CASE(BIAS, U8_ENTRY_F32);
    CASE(DST_ITER, U8_ENTRY_U8);
    CASE(DST_ITER_C, U8_ENTRY_F32);
    CASE(DST_LAYER, U8_ENTRY_F32);
    CASE(AUGRU_ATTENTION, U8_ENTRY_U8);
    END_LIST;
}

CFG(f32u8f32u8) {
    UNUSED_REG_VAR(f32u8f32u8);
    CASE(SRC_LAYER, U8_ENTRY_U8);
    CASE(SRC_ITER, U8_ENTRY_F32);
    CASE(SRC_ITER_C, U8_ENTRY_F32);
    CASE(WEIGHTS_LAYER, U8_ENTRY_S8);
    CASE(WEIGHTS_ITER, U8_ENTRY_S8);
    CASE(WEIGHTS_PEEPHOLE, U8_ENTRY_F32);
    CASE(WEIGHTS_PROJECTION, U8_ENTRY_S8);
    CASE(BIAS, U8_ENTRY_F32);
    CASE(DST_ITER, U8_ENTRY_F32);
    CASE(DST_ITER_C, U8_ENTRY_F32);
    CASE(DST_LAYER, U8_ENTRY_U8_EXACT);
    CASE(AUGRU_ATTENTION, U8_ENTRY_U8);
    END_LIST;
}

CFG(f32u8f32f32) {
    UNUSED_REG_VAR(f32u8f32f32);
    CASE(SRC_LAYER, U8_ENTRY_U8);
    CASE(SRC_ITER, U8_ENTRY_F32);
    CASE(SRC_ITER_C, U8_ENTRY_F32);
    CASE(WEIGHTS_LAYER, U8_ENTRY_S8);
    CASE(WEIGHTS_ITER, U8_ENTRY_S8);
    CASE(WEIGHTS_PEEPHOLE, U8_ENTRY_F32);
    CASE(WEIGHTS_PROJECTION, U8_ENTRY_S8);
    CASE(BIAS, U8_ENTRY_F32);
    CASE(DST_ITER, U8_ENTRY_F32);
    CASE(DST_ITER_C, U8_ENTRY_F32);
    CASE(DST_LAYER, U8_ENTRY_F32);
    CASE(AUGRU_ATTENTION, U8_ENTRY_U8);
    END_LIST;
}

CFG(s8s8s8s8) {
    UNUSED_REG_VAR(s8s8s8s8);
    CASE(SRC_LAYER, S8_ENTRY_S8);
    CASE(SRC_ITER, S8_ENTRY_S8);
    CASE(SRC_ITER_C, S8_ENTRY_F32);
    CASE(WEIGHTS_LAYER, S8_ENTRY_WEIGHT_S8);
    CASE(WEIGHTS_ITER, S8_ENTRY_WEIGHT_S8);
    CASE(WEIGHTS_PEEPHOLE, S8_ENTRY_F32);
    CASE(WEIGHTS_PROJECTION, S8_ENTRY_WEIGHT_S8);
    CASE(BIAS, S8_ENTRY_F32);
    CASE(DST_ITER, S8_ENTRY_S8);
    CASE(DST_ITER_C, S8_ENTRY_F32);
    CASE(DST_LAYER, S8_ENTRY_S8_EXACT);
    CASE(AUGRU_ATTENTION, U8_ENTRY_U8);
    END_LIST;
}

CFG(s8s8s8f32) {
    UNUSED_REG_VAR(s8s8s8f32);
    CASE(SRC_LAYER, S8_ENTRY_S8);
    CASE(SRC_ITER, S8_ENTRY_S8);
    CASE(SRC_ITER_C, S8_ENTRY_F32);
    CASE(WEIGHTS_LAYER, S8_ENTRY_WEIGHT_S8);
    CASE(WEIGHTS_ITER, S8_ENTRY_WEIGHT_S8);
    CASE(WEIGHTS_PEEPHOLE, S8_ENTRY_F32);
    CASE(WEIGHTS_PROJECTION, S8_ENTRY_WEIGHT_S8);
    CASE(BIAS, S8_ENTRY_F32);
    CASE(DST_ITER, S8_ENTRY_S8);
    CASE(DST_ITER_C, S8_ENTRY_F32);
    CASE(DST_LAYER, S8_ENTRY_F32);
    CASE(AUGRU_ATTENTION, U8_ENTRY_U8);
    END_LIST;
}

CFG(f32s8f32s8) {
    UNUSED_REG_VAR(f32s8f32s8);
    CASE(SRC_LAYER, S8_ENTRY_S8);
    CASE(SRC_ITER, S8_ENTRY_F32);
    CASE(SRC_ITER_C, S8_ENTRY_F32);
    CASE(WEIGHTS_LAYER, S8_ENTRY_WEIGHT_S8);
    CASE(WEIGHTS_ITER, S8_ENTRY_WEIGHT_S8);
    CASE(WEIGHTS_PEEPHOLE, S8_ENTRY_F32);
    CASE(WEIGHTS_PROJECTION, S8_ENTRY_WEIGHT_S8);
    CASE(BIAS, S8_ENTRY_F32);
    CASE(DST_ITER, S8_ENTRY_F32);
    CASE(DST_ITER_C, S8_ENTRY_F32);
    CASE(DST_LAYER, S8_ENTRY_S8_EXACT);
    CASE(AUGRU_ATTENTION, U8_ENTRY_U8);
    END_LIST;
}

CFG(f32s8f32f32) {
    UNUSED_REG_VAR(f32s8f32f32);
    CASE(SRC_LAYER, S8_ENTRY_S8);
    CASE(SRC_ITER, S8_ENTRY_F32);
    CASE(SRC_ITER_C, S8_ENTRY_F32);
    CASE(WEIGHTS_LAYER, S8_ENTRY_WEIGHT_S8);
    CASE(WEIGHTS_ITER, S8_ENTRY_WEIGHT_S8);
    CASE(WEIGHTS_PEEPHOLE, S8_ENTRY_F32);
    CASE(WEIGHTS_PROJECTION, S8_ENTRY_WEIGHT_S8);
    CASE(BIAS, S8_ENTRY_F32);
    CASE(DST_ITER, S8_ENTRY_F32);
    CASE(DST_ITER_C, S8_ENTRY_F32);
    CASE(DST_LAYER, S8_ENTRY_F32);
    CASE(AUGRU_ATTENTION, U8_ENTRY_U8);
    END_LIST;
}

} // namespace

const dt_conf_t &dt_conf_t::create(const std::string &str, const attr_t &attr) {
    if (str == "f32") {
        if (dnnl::impl::utils::one_of(attr.fpmath_mode.mode,
                    dnnl_fpmath_mode_bf16, dnnl_fpmath_mode_tf32))
            return conf_bf32;
        if (attr.fpmath_mode.mode == dnnl_fpmath_mode_f16) return conf_f16_math;
    }
    for (const auto cfg : cfg_list)
        if (cfg->str() == str) return *cfg;
    SAFE_V(CRIT);
    return conf_f32;
}

} // namespace rnn
