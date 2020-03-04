/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include "dnnl_common.hpp"
#include "rnn/rnn.hpp"

namespace rnn {

/* cfgs definition
arrays:
input,
states,
c_states,
weights_input,
weights_states,
bias,
dst_last_iteration,
dst_c_last_iteration,
dst_last_layer,
dst_diff_input,
dst_diff_states,
dst_c_diff_states,
dst_diff_weights_input,
dst_diff_weights_states,
dst_diff_bias,
diff_last_iteration,
diff_c_last_iteration,
diff_last_layer,
weights_peephole,
dst_diff_weights_peephole,
params: {data_type, min, max, f_min, f_max, f_mean, f_stddev, eps}
*/

#define EPS_F32 epsilon_dt(dnnl_f32)

#define MIN_F32 0.0f
#define MAX_F32 .999999f
#define MEAN_F32 .5f
#define STDDEV_F32 0.01f

#define F32_ENTRY_INEXACT \
    { \
        dnnl_f32, -int_max_exact, int_max_exact, MIN_F32, MAX_F32, MEAN_F32, \
                STDDEV_F32, EPS_F32 \
    }

const int int_max_exact = 1 << 24;
const _dt_conf_t conf_f32 = {
        F32_ENTRY_INEXACT, //input
        F32_ENTRY_INEXACT, //states
        F32_ENTRY_INEXACT, //c_states
        F32_ENTRY_INEXACT, //weights_input
        F32_ENTRY_INEXACT, //weights_states
        F32_ENTRY_INEXACT, //bias
        F32_ENTRY_INEXACT, //dst_last_iteration
        F32_ENTRY_INEXACT, //dst_c_last_iteration
        F32_ENTRY_INEXACT, //dst_last_layer
        F32_ENTRY_INEXACT, //dst_diff_input
        F32_ENTRY_INEXACT, //dst_diff_states
        F32_ENTRY_INEXACT, //dst_diff_c_states
        F32_ENTRY_INEXACT, //dst_diff_weights_input
        F32_ENTRY_INEXACT, //dst_diff_weights_states
        F32_ENTRY_INEXACT, //dst_diff_bias
        F32_ENTRY_INEXACT, //diff_last_iteration
        F32_ENTRY_INEXACT, //diff_c_last_iteration
        F32_ENTRY_INEXACT, //diff_last_layer
        F32_ENTRY_INEXACT, //weights_peephole
        F32_ENTRY_INEXACT, //dst_diff_weights_peephole
};

#define EPS_BF16 epsilon_dt(dnnl_bf16)

#define MIN_BF16 0.0f
#define MAX_BF16 .999999f
#define MEAN_BF16 .5f
#define STDDEV_BF16 0.01f

#define BF16_ENTRY_INEXACT \
    { \
        dnnl_bf16, -int_max_exact, int_max_exact, MIN_BF16, MAX_BF16, \
                MEAN_BF16, STDDEV_BF16, EPS_BF16 \
    }

#define BF16_ENTRY_F32_INEXACT \
    { \
        dnnl_f32, -int_max_exact, int_max_exact, MIN_F32, MAX_F32, MEAN_F32, \
                STDDEV_F32, EPS_BF16 \
    }

const _dt_conf_t conf_bf16 = {
        BF16_ENTRY_INEXACT, //input
        BF16_ENTRY_INEXACT, //states
        BF16_ENTRY_F32_INEXACT, //c_states
        BF16_ENTRY_INEXACT, //weights_input
        BF16_ENTRY_INEXACT, //weights_states
        BF16_ENTRY_F32_INEXACT, //bias
        BF16_ENTRY_INEXACT, //dst_last_iteration
        BF16_ENTRY_F32_INEXACT, //dst_c_last_iteration
        BF16_ENTRY_INEXACT, //dst_last_layer
        BF16_ENTRY_F32_INEXACT, //dst_diff_input
        BF16_ENTRY_F32_INEXACT, //dst_diff_states
        BF16_ENTRY_F32_INEXACT, //dst_diff_c_states
        BF16_ENTRY_F32_INEXACT, //dst_diff_weights_input
        BF16_ENTRY_F32_INEXACT, //dst_diff_weights_states
        BF16_ENTRY_F32_INEXACT, //dst_diff_bias
        BF16_ENTRY_F32_INEXACT, //diff_last_iteration
        BF16_ENTRY_F32_INEXACT, //diff_c_last_iteration
        BF16_ENTRY_F32_INEXACT, //diff_last_layer
        BF16_ENTRY_F32_INEXACT, //weights_peephole
        BF16_ENTRY_F32_INEXACT, //dst_diff_weights_peephole
};

#define EPS_U8 4e-3
#define EPS_S8 8e-3

#define MIN_U8 0.0f
#define MAX_U8 127.f
#define MEAN_U8 28.f
#define STDDEV_U8 16.f

#define MIN_S8 -63.f
#define MAX_S8 63.f
#define MEAN_S8 0.f
#define STDDEV_S8 32.f

#define U8_ENTRY_U8_EXACT \
    { dnnl_u8, 0, UINT8_MAX, MIN_U8, MAX_U8, MEAN_U8, STDDEV_U8, 0.f }
#define U8_ENTRY_U8_INEXACT \
    { dnnl_u8, 0, UINT8_MAX, MIN_U8, MAX_U8, MEAN_U8, STDDEV_U8, EPS_U8 }
#define U8_ENTRY_S8_EXACT \
    { dnnl_s8, INT8_MIN, INT8_MAX, MIN_S8, MAX_S8, MEAN_S8, STDDEV_S8, 0.f }
#define U8_ENTRY_F32_EXACT \
    { \
        dnnl_f32, -int_max_exact, int_max_exact, MIN_F32, MAX_F32, MEAN_F32, \
                STDDEV_F32, 0.0f \
    }
#define U8_ENTRY_F32_INEXACT \
    { \
        dnnl_f32, -int_max_exact, int_max_exact, MIN_F32, MAX_F32, MEAN_F32, \
                STDDEV_F32, EPS_F32 \
    }

const _dt_conf_t conf_u8u8u8u8 = {
        U8_ENTRY_U8_EXACT, //input
        U8_ENTRY_U8_EXACT, //states
        U8_ENTRY_F32_EXACT, //c_states
        U8_ENTRY_S8_EXACT, //weights_input
        U8_ENTRY_S8_EXACT, //weights_states
        U8_ENTRY_F32_EXACT, //bias
        U8_ENTRY_U8_INEXACT, //dst_iter
        U8_ENTRY_F32_INEXACT, //dst_c_last_iteration
        U8_ENTRY_U8_EXACT, //dst_layer
};
const _dt_conf_t conf_u8u8u8f32 = {
        U8_ENTRY_U8_EXACT, //input
        U8_ENTRY_U8_EXACT, //states
        U8_ENTRY_F32_EXACT, //c_states
        U8_ENTRY_S8_EXACT, //weights_input
        U8_ENTRY_S8_EXACT, //weights_states
        U8_ENTRY_F32_EXACT, //bias
        U8_ENTRY_U8_INEXACT, //dst_iter
        U8_ENTRY_F32_INEXACT, //dst_c_last_iteration
        U8_ENTRY_F32_INEXACT, //dst_last_layer
};
const _dt_conf_t conf_f32u8f32u8 = {
        U8_ENTRY_U8_EXACT, //input
        U8_ENTRY_F32_EXACT, //states
        U8_ENTRY_F32_EXACT, //c_states
        U8_ENTRY_S8_EXACT, //weights_input
        U8_ENTRY_S8_EXACT, //weights_states
        U8_ENTRY_F32_EXACT, //bias
        U8_ENTRY_F32_INEXACT, //dst_iter
        U8_ENTRY_F32_INEXACT, //dst_c_last_iteration
        U8_ENTRY_U8_EXACT, //dst_last_layer
};
const _dt_conf_t conf_f32u8f32f32 = {
        U8_ENTRY_U8_EXACT, //input
        U8_ENTRY_F32_EXACT, //states
        U8_ENTRY_F32_EXACT, //c_states
        U8_ENTRY_S8_EXACT, //weights_input
        U8_ENTRY_S8_EXACT, //weights_states
        U8_ENTRY_F32_EXACT, //bias
        U8_ENTRY_F32_INEXACT, //dst_iter
        U8_ENTRY_F32_INEXACT, //dst_c_last_iteration
        U8_ENTRY_F32_INEXACT, //dst_last_layer
};

const int int_max_exact_half = 1 << 11;
#define F16_ENTRY_INEXACT \
    { \
        dnnl_f16, -int_max_exact_half, int_max_exact_half, 0.0f, 0.999999f, \
                0.5f, 0.01f, epsilon_dt(dnnl_f16) \
    }

const _dt_conf_t conf_f16 = {
        F16_ENTRY_INEXACT, //input
        F16_ENTRY_INEXACT, //states
        F16_ENTRY_INEXACT, //c_states
        F16_ENTRY_INEXACT, //weights_input
        F16_ENTRY_INEXACT, //weights_states
        F16_ENTRY_INEXACT, //bias
        F16_ENTRY_INEXACT, //dst_last_iteration
        F16_ENTRY_INEXACT, //dst_c_last_iteration
        F16_ENTRY_INEXACT, //dst_last_layer
};

const dt_conf_t *str2cfg(const char *str) {
#define CASE(cfg) \
    if (!strcasecmp(STRINGIFY(cfg), str)) return CONCAT2(conf_, cfg)
    CASE(f32);
    CASE(bf16);
    CASE(f16);
    CASE(u8u8u8u8);
    CASE(u8u8u8f32);
    CASE(f32u8f32u8);
    CASE(f32u8f32f32);
#undef CASE
    []() {
        SAFE(FAIL, CRIT);
        return 0;
    }();
    return (const dt_conf_t *)1;
}

std::ostream &operator<<(std::ostream &s, const dt_conf_t *cfg) {
#define CASE(_cfg) \
    if (cfg == CONCAT2(conf_, _cfg)) return s << STRINGIFY(_cfg)
    CASE(f32);
    CASE(bf16);
    CASE(f16);
    CASE(u8u8u8u8);
    CASE(u8u8u8f32);
    CASE(f32u8f32u8);
    CASE(f32u8f32f32);
#undef CASE
    SAFE_V(FAIL);
    return s;
}
} // namespace rnn
