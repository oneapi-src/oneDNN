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

#include "mkldnn.h"
#include "mkldnn_common.hpp"

#include "bnorm.hpp"

namespace bnorm {

/* cfgs definition
 * arrays: DATA = 5, MEAN, VAR, SS
 * params: {data_type, min, max, f_min, f_max, f_base, f_step, f_sparsity, eps}
 * now used: {data_type, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, UNUSED,
 *            eps}
 */

const int int_max_exact = 1<<24;
const _dt_conf_t conf_f32 = {
    {static_cast<mkldnn_data_type_t>(0), 0, 0, 0, 0, 0, 0, 0, 0.},
    {static_cast<mkldnn_data_type_t>(0), 0, 0, 0, 0, 0, 0, 0, 0.},
    {static_cast<mkldnn_data_type_t>(0), 0, 0, 0, 0, 0, 0, 0, 0.},
    {static_cast<mkldnn_data_type_t>(0), 0, 0, 0, 0, 0, 0, 0, 0.},
    {static_cast<mkldnn_data_type_t>(0), 0, 0, 0, 0, 0, 0, 0, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact,  -32,  32, 0, 1, .35, 2e-7},
    {mkldnn_f32, -int_max_exact, int_max_exact,  -32,  32, 0, 1, 1.0, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact, -512, 512, 0, 1, 1.0, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact,  -32,  32, 0, 1, .35, 0.},
};

const _dt_conf_t conf_s8 = {
    {static_cast<mkldnn_data_type_t>(0), 0, 0, 0, 0, 0, 0, 0, 0.},
    {static_cast<mkldnn_data_type_t>(0), 0, 0, 0, 0, 0, 0, 0, 0.},
    {static_cast<mkldnn_data_type_t>(0), 0, 0, 0, 0, 0, 0, 0, 0.},
    {static_cast<mkldnn_data_type_t>(0), 0, 0, 0, 0, 0, 0, 0, 0.},
    {static_cast<mkldnn_data_type_t>(0), 0, 0, 0, 0, 0, 0, 0, 0.},
    {mkldnn_s8,        INT8_MIN,      INT8_MAX,   -5,   5, 0, 1, .25, 1.},
    {mkldnn_f32, -int_max_exact, int_max_exact,  -32,  32, 0, 1, 1.0, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact, -512, 512, 0, 1, 1.0, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact,  -32,  32, 0, 1, .25, 0.},
};

const dt_conf_t *str2cfg(const char *str) {
#define CASE(cfg) \
    if (!strcasecmp(STRINGIFY(cfg), str)) return CONCAT2(conf_,cfg)
    CASE(f32);
    CASE(s8);
#undef CASE
    []() { SAFE(FAIL, CRIT); return 0; }();
    return (const dt_conf_t *)1;
}

const char *cfg2str(const dt_conf_t *cfg) {
#define CASE(_cfg) if (cfg == CONCAT2(conf_,_cfg)) return STRINGIFY(_cfg)
    CASE(f32);
    CASE(s8);
#undef CASE
    []() { SAFE(FAIL, CRIT); return 0; }();
    return NULL;
}

}
