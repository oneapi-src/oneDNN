/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef CONV_DW_FUSION_HPP
#define CONV_DW_FUSION_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"

#include "conv/conv.hpp"

namespace conv_dw_fusion {

using desc_t = conv::desc_t;
using prb_t = conv::prb_t;
using alg_t = conv::alg_t;
using cfg_t = conv::cfg_t;

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
void get_fused_conv_dst_dims(const int ndims,
        const attr_t::post_ops_t::entry_t &po_entry, const dnnl_dims_t dims,
        dnnl_dims_t fused_conv_dst_dims);

} // namespace conv_dw_fusion

#endif
