/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "cpu/zero_point_utils.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

zero_point_config_t::zero_point_config_t(const primitive_attr_t &attr)
    : src_exists(!attr.zero_points_.has_default_values(DNNL_ARG_SRC))
    , dst_exists(!attr.zero_points_.has_default_values(DNNL_ARG_DST))
    , src_is_common(attr.zero_points_.common(DNNL_ARG_SRC)) {}

bool zero_point_config_t::zp_exists() const noexcept {
    return src_exists || dst_exists;
}

bool zero_points_valid(const primitive_attr_t *attr) noexcept {
    // mask for i/o-channel and ngroups
    static constexpr int c_mask = 0x1;
    static constexpr int g_mask = 0x3;

    int mask_src = 0, mask_dst = 0;
    attr->zero_points_.get(DNNL_ARG_SRC, nullptr, &mask_src, nullptr);
    attr->zero_points_.get(DNNL_ARG_DST, nullptr, &mask_dst, nullptr);
    return attr->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
            && utils::one_of(mask_src, 0, c_mask, g_mask) && mask_dst == 0;
}

void set_zp_src_comp_flags(memory_desc_t &weights_md, bool with_groups) {
    weights_md.extra.flags
            |= memory_extra_flags::compensation_conv_asymmetric_src;
    weights_md.extra.asymm_compensation_mask
            = (1 << 0) + (with_groups ? (1 << 1) : 0);
}

const int32_t *get_src_zp_comp(const int8_t *weights,
        const memory_desc_wrapper &weights_md, bool signed_input, dim_t ngroups,
        dim_t oc) {

    const auto comp_offset
            = weights_md.size() - weights_md.additional_buffer_size();
    const auto src_zp_com_offset = signed_input ? ngroups * oc : 0;
    return reinterpret_cast<const int32_t *>(&weights[comp_offset])
            + src_zp_com_offset;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
