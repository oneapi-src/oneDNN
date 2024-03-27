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

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dnnl_common.hpp"
#include "oneapi/dnnl/dnnl.h"

#include "reorder.hpp"

namespace reorder {

const float int_max_exact = 1 << 24;
const float f16_max_exact = 1 << 11;

#define REG(dt, min, max) \
    const dt_conf_s CONCAT2(_conf_, dt) = {CONCAT2(dnnl_, dt), min, max}; \
    const dt_conf_t CONCAT2(conf_, dt) = &CONCAT2(_conf_, dt);

REG(f32, -int_max_exact, int_max_exact);
REG(f64, -int_max_exact, int_max_exact);
REG(f16, -f16_max_exact, f16_max_exact);
REG(bf16, -int_max_exact, int_max_exact);
REG(f8_e5m2, -f16_max_exact, f16_max_exact);
REG(f8_e4m3, -f16_max_exact, f16_max_exact);
// Do not exceed max float value representable in integer. Otherwise, we get
// a correctness issue caused by different computations in reference and the
// library.
REG(s32, INT_MIN, BENCHDNN_S32_TO_F32_SAT_CONST);
REG(s8, INT8_MIN, INT8_MAX);
REG(u8, 0, UINT8_MAX);
REG(s4, -7, 8);
REG(u4, 0, 15);

#undef REG

dt_conf_t dt2cfg(dnnl_data_type_t dt) {
#define CASE(cfg) \
    if (CONCAT2(dnnl_, cfg) == dt) return CONCAT2(conf_, cfg)
    CASE(f32);
    CASE(f64);
    CASE(f16);
    CASE(bf16);
    CASE(f8_e5m2);
    CASE(f8_e4m3);
    CASE(s32);
    CASE(s8);
    CASE(u8);
    CASE(s4);
    CASE(u4);
#undef CASE
    SAFE_V(FAIL);
    return conf_f32;
}

dnnl_data_type_t cfg2dt(dt_conf_t cfg) {
#define CASE(_cfg) \
    if (cfg == CONCAT2(conf_, _cfg)) return CONCAT2(dnnl_, _cfg)
    CASE(f32);
    CASE(f64);
    CASE(f16);
    CASE(bf16);
    CASE(f8_e5m2);
    CASE(f8_e4m3);
    CASE(s32);
    CASE(s8);
    CASE(u8);
    CASE(s4);
    CASE(u4);
#undef CASE
    SAFE_V(FAIL);
    return dnnl_f32;
}

} // namespace reorder
