/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

#include "deconv/deconv.hpp"

#define HALF_MAX 65504
#define HALF_MIN (-65504)

namespace deconv {

/* cfgs definition
 * arrays: SRC, WEI, BIA, DST, ACC
 * params: {data_type, min, max, f_min, f_max, f_base, f_step, f_sparsity, eps}
 */

const int int_max_exact_half = 1 << 11;
const _dt_conf_t conf_f16 = {
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -4, 4, 0, 1, .25,
                0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -2, 2, -2, 1, 1.0,
                0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -6, 6, 0, 1, 1.0,
                0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -4, 4, 0, 1, .25,
                0.},
        {dnnl_f16},
};

const int int_max_exact = 1 << 24;
const _dt_conf_t conf_f16f16f32 = {
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -4, 4, 0, 1, .25,
                0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -2, 2, -2, 1, 1.0,
                0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f16},
};

const _dt_conf_t conf_f32 = {
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .25, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, 1.0, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -512, 512, 0, 1, 1.0, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .25, 0.},
        {dnnl_f32},
};

const _dt_conf_t conf_f32_wino = {
        {dnnl_f32, -FLT_MAX, FLT_MAX, -16, 128, 3, 1, .25, 1e-5},
        {dnnl_f32, -FLT_MAX, FLT_MAX, 2, 64, 2, 1, .75, 6e-6},
        {dnnl_f32, -FLT_MAX, FLT_MAX, 1, 128, 1, 1, .25, 2e-7},
        {dnnl_f32, -FLT_MAX, FLT_MAX, -16, 128, 3, 1, .25, 2e-5},
        {dnnl_f32},
};

const _dt_conf_t conf_f64 = {
        {dnnl_f64, -int_max_exact, int_max_exact, -32, 32, 0, 1, .25, 0.},
        {dnnl_f64, -int_max_exact, int_max_exact, -32, 32, 0, 1, 1.0, 0.},
        {dnnl_f64, -int_max_exact, int_max_exact, -512, 512, 0, 1, 1.0, 0.},
        {dnnl_f64, -int_max_exact, int_max_exact, -32, 32, 0, 1, .25, 0.},
        {dnnl_f64},
};

const _dt_conf_t conf_f32_with_bf16_fpmath = {
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32},
};

const _dt_conf_t conf_f32_with_tf32_fpmath = {
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32},
};

const _dt_conf_t conf_f16_wino = {
        {dnnl_f16, HALF_MIN, HALF_MAX, -2, 16, 0, 1, .25, 5e-3},
        {dnnl_f16, HALF_MIN, HALF_MAX, 1, 6, -2, 1, .5, 6e-3},
        {dnnl_f16, HALF_MIN, HALF_MAX, 1, 2048, 0, 1, .25, 2e-3},
        {dnnl_f16, HALF_MIN, HALF_MAX, -2, 8, 0, 1, .25, 7e-3},
        {dnnl_f16},
};

const _dt_conf_t conf_bf16bf16f32 = {
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32},
};

const _dt_conf_t conf_bf16bf16f16 = {
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -4, 4, 0, 1, .25,
                0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -4, 4, 0, 1, .25,
                0.},
        {dnnl_f16},
};

const _dt_conf_t conf_bf16bf16s8 = {
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_f32},
};

const _dt_conf_t conf_bf16bf16u8 = {
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_u8, 0, UINT8_MAX, 0, UINT8_MAX, 0, 1, .25, 0.},
        {dnnl_f32},
};

const _dt_conf_t conf_bf16bf16bf16 = {
        /* eps is 1e-2 because of loss in precision of
     * output when converted from fp32 to bf16.
     * oneDNN output is compared against reference computed in fp32.*/
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 1e-2},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 1e-2},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 1e-2},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 1e-2},
        {dnnl_f32},
};

const _dt_conf_t conf_f32bf16bf16 = {
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32},
};

const _dt_conf_t conf_f32f32s8 = {
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_f32},
};

const _dt_conf_t conf_f32f32u8 = {
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_s8, 0, UINT8_MAX, 0, UINT8_MAX, 0, 1, .25, 0.},
        {dnnl_f32},
};

const _dt_conf_t conf_f16f16s8 = {
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -4, 4, 0, 1, .25,
                0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -2, 2, -2, 1, 1.0,
                0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -8, 8, 0, 1, 1.0,
                0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_f16},
};

const _dt_conf_t conf_f16f16u8 = {
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -4, 4, 0, 1, .25,
                0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -2, 2, -2, 1, 1.0,
                0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -8, 8, 0, 1, 1.0,
                0.},
        {dnnl_u8, 0, UINT8_MAX, 0, UINT8_MAX, 0, 1, .25, 0.},
        {dnnl_f16},
};

const _dt_conf_t conf_bf16f32bf16 = {
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 0.},
        {dnnl_f32},
};

const _dt_conf_t conf_u8s8f32 = {
        {dnnl_u8, 0, UINT8_MAX, 0, UINT8_MAX, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -255, 255, 0, 1, .25, 0.},
        {dnnl_s32},
};

const _dt_conf_t conf_u8s8f16 = {
        {dnnl_u8, 0, UINT8_MAX, 0, 8, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -2, 2, -2, 1, 1.0, 0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -8, 8, 0, 1, 1.0,
                0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -4, 4, 0, 1, .25,
                0.},
        {dnnl_s32},
};

const _dt_conf_t conf_u8s8bf16 = {
        {dnnl_u8, 0, UINT8_MAX, 0, UINT8_MAX, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 1e-2},
        {dnnl_s32},
};

const _dt_conf_t conf_u8s8s32 = {
        {dnnl_u8, 0, UINT8_MAX, 0, UINT8_MAX, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_s32, INT32_MIN, INT32_MAX, -255, 255, 0, 1, .25, 0.},
        {dnnl_s32},
};

const _dt_conf_t conf_u8s8s8 = {
        {dnnl_u8, 0, UINT8_MAX, 0, 8, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -127, 127, 0, 1, .25, 0.},
        {dnnl_s32},
};

const _dt_conf_t conf_u8s8u8 = {
        {dnnl_u8, 0, UINT8_MAX, 0, 8, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -3, 5, 0, 1, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_u8, 0, UINT8_MAX, 0, 255, 0, 1, .25, 0.},
        {dnnl_s32},
};

const _dt_conf_t conf_s8s8f32 = {
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -8, 3, 0, 4, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -255, 255, 0, 1, .25, 0.},
        {dnnl_s32},
};

const _dt_conf_t conf_s8s8f16 = {
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -8, 3, 0, 4, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_f16, -int_max_exact_half, int_max_exact_half, -4, 4, 0, 1, .25,
                0.},
        {dnnl_s32},
};

const _dt_conf_t conf_s8s8bf16 = {
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -8, 3, 0, 4, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_bf16, -int_max_exact, int_max_exact, -32, 32, 0, 1, .75, 1e-2},
        {dnnl_s32},
};

const _dt_conf_t conf_s8s8s32 = {
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -8, 3, 0, 4, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_s32, INT32_MIN, INT32_MAX, -255, 255, 0, 1, .25, 0.},
        {dnnl_s32},
};

const _dt_conf_t conf_s8s8s8 = {
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -8, 3, 0, 4, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -127, 127, 0, 1, .25, 0.},
        {dnnl_s32},
};

const _dt_conf_t conf_s8s8u8 = {
        {dnnl_s8, INT8_MIN, INT8_MAX, -5, 5, 0, 1, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -4, 7, 0, 4, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
        {dnnl_u8, 0, UINT8_MAX, 0, 255, 0, 1, .25, 0.},
        {dnnl_s32},
};

const _dt_conf_t conf_u8s8f32_wino = {
        {dnnl_u8, 0, UINT8_MAX, 0, 239, 0, 4, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -72, 71, 0, 9, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -9, 35, 0, 9, .25, 0.},
        {dnnl_f32, -FLT_MAX, FLT_MAX, -16, 128, 0, 1, .25, 2e-5}, // oscale
        {dnnl_s32},
};

const _dt_conf_t conf_u8s8s32_wino = {

        {dnnl_u8, 0, UINT8_MAX, 0, 239, 0, 4, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -72, 71, 0, 9, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -9, 35, 0, 9, .25, 0.},
        {dnnl_s32, INT32_MIN, INT32_MAX, -255, 255, 0, 1, .25, 0.},
        {dnnl_s32},
};

const _dt_conf_t conf_u8s8s8_wino = {
        {dnnl_u8, 0, UINT8_MAX, 0, 239, 0, 4, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -72, 71, 0, 9, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -9, 35, 0, 9, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -127, 127, 0, 1, .25, 0.},
        {dnnl_s32},
};

const _dt_conf_t conf_u8s8u8_wino = {
        {dnnl_u8, 0, UINT8_MAX, 0, 239, 0, 4, .25, 0.},
        {dnnl_s8, INT8_MIN, INT8_MAX, -72, 71, 0, 9, .25, 0.},
        {dnnl_f32, INT32_MIN, INT32_MAX, -9, 35, 0, 9, .25, 0.},
        {dnnl_u8, 0, UINT8_MAX, 0, 255, 0, 1, .25, 0.},
        {dnnl_s32},
};

const dt_conf_t *str2cfg(const char *str) {
#define CASE(cfg) \
    if (!strcasecmp(STRINGIFY(cfg), str)) return CONCAT2(conf_, cfg)
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
    CASE(u8s8f32_wino);
    CASE(u8s8s32_wino);
    CASE(u8s8s8_wino);
    CASE(u8s8u8_wino);
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
    CASE(u8s8f32_wino);
    CASE(u8s8s32_wino);
    CASE(u8s8s8_wino);
    CASE(u8s8u8_wino);
    CASE(f16f16f32);
    CASE(f16f16s8);
    CASE(f16f16u8);
    CASE(bf16bf16f32);
    CASE(bf16bf16f16);
    CASE(bf16bf16s8);
    CASE(bf16bf16u8);
    CASE(bf16bf16bf16);
    CASE(f32bf16bf16);
    CASE(f32f32s8);
    CASE(f32f32u8);
    CASE(bf16f32bf16);
#undef CASE
    SAFE_V(FAIL);
    return s;
}

const dt_conf_t *auto_cfg(const alg_t alg, const dt_conf_t *cfg) {
    if (alg != WINO) return cfg;

    std::stringstream ss;
    ss << cfg << "_wino";
    const std::string cpp_pstr = ss.str();
    const char *cfg_s = cpp_pstr.c_str();
#define CASE(_cfg_) \
    if (!strcmp(cfg_s, STRINGIFY(_cfg_))) return CONCAT2(conf_, _cfg_)
    CASE(f32_wino);
    CASE(f16_wino);
    CASE(u8s8f32_wino);
    CASE(u8s8s32_wino);
    CASE(u8s8s8_wino);
    CASE(u8s8u8_wino);
#undef CASE
    return cfg;
}

} // namespace deconv
