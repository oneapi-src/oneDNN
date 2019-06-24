/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "mkldnn.h"

#include "mkldnn_common.hpp"

#include "pool/pool.hpp"

namespace pool {

/* cfgs definition
 * arrays: SRC, UNUSED, UNUSED, DST
 * params: {data_type, min, max, f_min, f_max, eps}
 */

const _dt_conf_t conf_f32 = {
    // though integers are expected, eps is needed to cover division error
    {mkldnn_f32, -FLT_MAX,   FLT_MAX,    -2048,      2048, 3e-7},
    {},
    {},
    {mkldnn_f32, -FLT_MAX,   FLT_MAX,    -2048,      2048, 3e-7},
};

const _dt_conf_t conf_s32 = {
    {mkldnn_s32,  INT_MIN,   INT_MAX,    -2048,      2048, 0.},
    {},
    {},
    {mkldnn_s32,  INT_MIN,   INT_MAX,    -2048,      2048, 0.},
};

const float16_t flt16_max = mkldnn::impl::nstl::numeric_limits<float16_t>::max();
const _dt_conf_t conf_f16 = {
    {mkldnn_f16, -flt16_max, flt16_max, -32, 32, 1e-3},
    {},
    {},
    {mkldnn_f16, -flt16_max, flt16_max, -32, 32, 1e-3},
};

#define BFLT16_MAX 3.38953138925153547590470800371487866880e+38F
const _dt_conf_t conf_bf16 = {
    /* Although integers are expected, eps is needed to cover
     * for the division error */
    {mkldnn_bf16, -BFLT16_MAX, BFLT16_MAX, -32, 32, 1e-2},
    {},
    {},
    {mkldnn_bf16, -BFLT16_MAX, BFLT16_MAX, -32, 32, 5e-2},
};
#undef BFLT16_MAX

const _dt_conf_t conf_s8 = {
    {mkldnn_s8,  INT8_MIN,  INT8_MAX, INT8_MIN,  INT8_MAX, 0.},
    {},
    {},
    {mkldnn_s8,  INT8_MIN,  INT8_MAX, INT8_MIN,  INT8_MAX, 0.},
};

const _dt_conf_t conf_u8 = {
    {mkldnn_u8,         0, UINT8_MAX,        0, UINT8_MAX, 0.},
    {},
    {},
    {mkldnn_u8,         0, UINT8_MAX,        0, UINT8_MAX, 0.},
};

const dt_conf_t *str2cfg(const char *str) {
#define CASE(cfg) \
    if (!strcasecmp(STRINGIFY(cfg), str)) return CONCAT2(conf_,cfg)
    CASE(f32);
    CASE(s32);
    CASE(f16);
    CASE(bf16);
    CASE(s8);
    CASE(u8);
#undef CASE
    []() { SAFE(FAIL, CRIT); return 0; }();
    return (const dt_conf_t *)1;
}

const char *cfg2str(const dt_conf_t *cfg) {
#define CASE(_cfg) if (cfg == CONCAT2(conf_,_cfg)) return STRINGIFY(_cfg)
    CASE(f32);
    CASE(s32);
    CASE(f16);
    CASE(bf16);
    CASE(s8);
    CASE(u8);
#undef CASE
    []() { SAFE(FAIL, CRIT); return 0; }();
    return NULL;
}

}
