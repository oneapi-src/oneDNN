/*******************************************************************************
* Copyright 2018-2022 Intel Corporation
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

#ifndef DECONV_HPP
#define DECONV_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"

#include "conv/conv_common.hpp"

namespace deconv {

int transpose_data_wei(
        const conv::prb_t *prb, const dnn_mem_t &wei, const dnn_mem_t &wei_tr);

void skip_unimplemented_prb(const conv::prb_t *prb, res_t *res);
void skip_invalid_prb(const conv::prb_t *prb, res_t *res);
void compute_ref(const conv::prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int doit(const conv::prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace deconv
#endif
