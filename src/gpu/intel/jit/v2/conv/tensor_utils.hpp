/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_TENSOR_UTILS_HPP
#define GPU_INTEL_JIT_V2_CONV_TENSOR_UTILS_HPP

#include "gpu/intel/jit/ir/problem.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

layout_desc_t make_conv_layout_desc(
        tensor_kind_t tensor_kind, bool src_dst_with_group = false);
layout_desc_t make_conv_algo_layout_desc(
        prop_kind_t prop, tensor_kind_t tensor_kind);
layout_tag_t make_conv_layout_tag(
        tensor_kind_t tensor_kind, const std::string &s);
layout_tag_t make_conv_layout_tag(tensor_kind_t tensor_kind,
        dim_idx_t conv_ndims, const memory_desc_t &md);
layout_t make_conv_layout(tensor_kind_t tensor_kind, const layout_tag_t &_tag,
        bool is_dw, const prb_reqs_t &reqs, uint32_t mask = 0xFFFFFFFF);

class dim_mapper_manager_t {
public:
    dim_mapper_manager_t() = default;
    dim_mapper_manager_t(prop_kind_t prop, const prb_reqs_t &reqs);
    const dim_mapper_t &mapper(tensor_kind_t tensor) const;

private:
    expr_t kw_idx = pvars::kw.index_var();
    expr_t kh_idx = pvars::kh.index_var();
    expr_t kd_idx = pvars::kd.index_var();
    expr_t id_idx = pvars::id.index_var();
    expr_t ih_idx = pvars::ih.index_var();
    expr_t iw_idx = pvars::iw.index_var();
    expr_t od_idx = pvars::od.index_var();
    expr_t oh_idx = pvars::oh.index_var();
    expr_t ow_idx = pvars::ow.index_var();

    dim_mapper_t init_src_mapper() const;
    dim_mapper_t init_wei_mapper() const;
    dim_mapper_t init_dst_mapper() const;
    dim_mapper_t init_bias_mapper() const;

    prop_kind_t prop_ = prop_kind::undef;
    prb_reqs_t reqs_;
    dim_mapper_t src_mapper_;
    dim_mapper_t wei_mapper_;
    dim_mapper_t dst_mapper_;
    dim_mapper_t bias_mapper_;
};

dim_mapper_t extend_mapper(
        const dim_mapper_t &mapper, const pvar_t &extra_dim, char letter);

std::vector<pvar_t> skip_mask(
        const view_t &view, const pvar_tile_t &tile, const prb_reqs_t &reqs);

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
