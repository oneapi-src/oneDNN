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

#ifndef REF_CONV_HPP
#define REF_CONV_HPP

#include "conv/conv.hpp"

namespace conv {

void compute_ref_bwd_bias(const prb_t *prb, const args_t &args);

void compute_wino_ref_fwd(const prb_t *prb, const args_t &args);
void compute_wino_ref_bwd_d(const prb_t *prb, const args_t &args);
void compute_wino_ref_bwd_w(const prb_t *prb, const args_t &args);

} // namespace conv

#endif
