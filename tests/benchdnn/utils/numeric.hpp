/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef UTILS_NUMERIC_HPP
#define UTILS_NUMERIC_HPP

#include "oneapi/dnnl/dnnl_types.h"

#define BENCHDNN_S32_TO_F32_SAT_CONST 2147483520.f

// These types coming from the library. Should supply dnnl::impl:: prefix.
// using bfloat16_t = dnnl::impl::bfloat16_t;
// using float16_t = dnnl::impl::float16_t;

template <dnnl_data_type_t>
struct prec_traits;

int digits_dt(dnnl_data_type_t dt);
float epsilon_dt(dnnl_data_type_t dt);
float lowest_dt(dnnl_data_type_t dt);
float max_dt(dnnl_data_type_t dt);
float saturate_and_round(dnnl_data_type_t dt, float value);
bool is_integral_dt(dnnl_data_type_t dt);
float maybe_saturate(dnnl_data_type_t dt, float value);
float round_to_nearest_representable(dnnl_data_type_t dt, float value);

#endif
