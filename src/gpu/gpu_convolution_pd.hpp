/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_GPU_CONVOLUTION_PD_HPP
#define GPU_GPU_CONVOLUTION_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_convolution_fwd_pd_t : public convolution_fwd_pd_t {
    using convolution_fwd_pd_t::convolution_fwd_pd_t;

protected:
    // TODO: consider either moving this method to primitive_conf.hpp or making
    //       it static, or removing the 'attr' argument accessible via attr()
    bool zero_points_ok(const primitive_attr_t *attr) const {
        const auto &zp = attr->zero_points_;

        using namespace data_type;
        bool ok = IMPLICATION(
                !utils::one_of(invariant_src_md()->data_type, s8, u8),
                zp.has_default_values());
        if (!ok) return false;

        if (!zp.has_default_values(DNNL_ARG_SRC)) {
            int mask_src = zp.get_mask(DNNL_ARG_SRC);
            ok = utils::one_of(mask_src, 0, (1 << 1));
            if (!ok) return false;
        }
        if (!zp.has_default_values(DNNL_ARG_WEIGHTS)) {
            int mask_wei = zp.get_mask(DNNL_ARG_WEIGHTS);
            ok = mask_wei == 0;
            if (!ok) return false;
        }
        if (!zp.has_default_values(DNNL_ARG_DST)) {
            int mask_dst = zp.get_mask(DNNL_ARG_DST);
            ok = utils::one_of(mask_dst, 0, (1 << 1));
            if (!ok) return false;
        }

        return true;
    }
};

struct gpu_convolution_bwd_data_pd_t : public convolution_bwd_data_pd_t {
    using convolution_bwd_data_pd_t::convolution_bwd_data_pd_t;

protected:
    // TODO: consider either moving this method to primitive_conf.hpp or making
    //       it static, or removing the 'attr' argument accessible via attr()
    bool zero_points_ok(const primitive_attr_t *attr) const {
        const auto &zp = attr->zero_points_;

        using namespace data_type;
        bool ok = IMPLICATION(
                !utils::one_of(invariant_dst_md()->data_type, s8, u8),
                zp.has_default_values());
        if (!ok) return false;

        if (!zp.has_default_values(DNNL_ARG_SRC)) {
            int mask_src = zp.get_mask(DNNL_ARG_SRC);
            ok = utils::one_of(mask_src, 0, (1 << 1));
            if (!ok) return false;
        }
        if (!zp.has_default_values(DNNL_ARG_DST)) {
            int mask_dst = zp.get_mask(DNNL_ARG_DST);
            ok = utils::one_of(mask_dst, 0, (1 << 1));
            if (!ok) return false;
        }

        return zp.has_default_values(DNNL_ARG_WEIGHTS);
    }

    // TODO: consider either moving this method to primitive_conf.hpp or making
    //       it static, or removing the 'attr' argument accessible via attr()
    bool post_ops_ok(const primitive_attr_t *attr) const {
        const auto &p = attr->post_ops_;

        auto is_eltwise
                = [&](int idx) { return p.entry_[idx].is_eltwise(false); };
        auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(false); };

        switch (p.len()) {
            case 0: return true; // no post_ops
            case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
            case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
            default: return false;
        }

        return false;
    }
};

struct gpu_convolution_bwd_weights_pd_t : public convolution_bwd_weights_pd_t {
    using convolution_bwd_weights_pd_t::convolution_bwd_weights_pd_t;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
