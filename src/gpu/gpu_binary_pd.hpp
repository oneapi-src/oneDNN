/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_GPU_BINARY_PD_HPP
#define GPU_GPU_BINARY_PD_HPP

#include "common/binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_binary_pd_t : public binary_pd_t {
    using binary_pd_t::binary_pd_t;

protected:
    bool post_ops_with_binary_ok(const primitive_attr_t *attr) const {
        const auto &p = attr->post_ops_;

        auto is_eltwise
                = [&](int idx) { return p.entry_[idx].is_eltwise(false); };
        auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(false); };
        auto is_binary = [&](int idx) { return p.entry_[idx].is_binary(); };

        bool is_po_ok = true;
        for (int po_idx = 0; po_idx < p.len(); ++po_idx) {
            is_po_ok &= is_eltwise(po_idx) | is_sum(po_idx) | is_binary(po_idx);

            if (is_sum(po_idx)) {
                if (p.entry_[po_idx].sum.dt != dnnl_data_type_undef
                        && types::data_type_size(p.entry_[po_idx].sum.dt)
                                != types::data_type_size(dst_md()->data_type))
                    return false;
            }
        }

        if (p.len() > 10) is_po_ok = false;

        return is_po_ok;
    }
};
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
