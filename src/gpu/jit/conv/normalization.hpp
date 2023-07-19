/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_JIT_CONV_NORMALIZATION_HPP
#define GPU_JIT_CONV_NORMALIZATION_HPP

#include "gpu/jit/ir/post_ops.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class conv_problem_t;
class conv_post_op_view_mapper_t : public post_op_view_mapper_t {
public:
    conv_post_op_view_mapper_t(
            const gemm_schedule_t &schedule, const conv_problem_t &prb)
        : post_op_view_mapper_t(schedule.c_view())
        , schedule_(schedule)
        , prb_(prb) {}

    view_t create_view(const type_t &type, uint32_t mask) const override {
        return post_op_view_mapper_t::create_view(type, normalize_mask(mask));
    }

    view_t create_view(const memory_desc_t &md) const override;

    view_t try_create_bias_view(uint32_t mask) const override;

    // Checks if convolution computes output elements that are out of bound in
    // the output tensor. This can happen due to spatial padding.
    //
    // For example for forward convolution OW is padded to OW_PADDED. Then if
    // ow >= OW (out of bounds) and iw = ow * SW - PW + kw * (DW + 1) < IW (in
    // bounds) convolution computes an out-of-bound element which is not
    // generally zero. This requires special handling if there are post-ops
    // followed the convolution.
    bool is_spurious_spatial(int dim_idx) const override;
    bool need_to_restore_zero_padding() const override;
    bool use_dst_in_sum_post_op() const override;
    bool can_use_scales() const override;
    bool can_use_simple_src_zps() const override { return false; }

private:
    uint32_t normalize_mask(uint32_t orig_mask) const;

    const gemm_schedule_t &schedule_;
    const conv_problem_t &prb_;
};

void normalize_conv_layouts(layout_t &src_layout, layout_t &wei_layout,
        layout_t &dst_layout, layout_t &bia_layout, bool with_groups, int g,
        int ic, int oc, bool is_dw, int reduced_dim, bool add_groups);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
