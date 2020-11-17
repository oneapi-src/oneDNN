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

#ifndef CPU_ZERO_POINT_UTILS_HPP
#define CPU_ZERO_POINT_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct zero_point_config_t {
    zero_point_config_t() = default;
    zero_point_config_t(const primitive_attr_t &attr);

    bool src_exists = false;
    bool dst_exists = false;
    bool src_is_common = false;

    bool zp_exists() const noexcept;
};

bool zero_points_valid(const primitive_attr_t *attr) noexcept;
void set_zp_src_comp_flags(memory_desc_t &weights_md, bool with_groups);
const int32_t *get_src_zp_comp(const int8_t *weights,
        const memory_desc_wrapper &weights_md, bool signed_input, dim_t ngroups,
        dim_t oc);

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
