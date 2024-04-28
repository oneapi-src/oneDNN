/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/scale_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
constexpr size_t scales_simd_w = 16;
}

void book_precomputed_scales(memory_tracking::registrar_t &scratchpad,
        const arg_scales_t &attr_scales, size_t wei_scale_count,
        bool force_scales_book) {
    using namespace dnnl::impl::memory_tracking::names;

    const bool with_src_scales
            = !attr_scales.get(DNNL_ARG_SRC).has_default_values();
    const bool with_wei_scales
            = !attr_scales.get(DNNL_ARG_WEIGHTS).has_default_values();
    const auto wei_scales_dt = attr_scales.get(DNNL_ARG_WEIGHTS).data_type_;
    const auto wei_scale_groups_ndims
            = attr_scales.get(DNNL_ARG_WEIGHTS).ndims_;
    if ((with_src_scales && with_wei_scales) || force_scales_book
            || (wei_scales_dt != data_type::f32 && with_wei_scales)
            || (wei_scale_groups_ndims > 0 && with_wei_scales)) {
        const int wei_mask = attr_scales.get(DNNL_ARG_WEIGHTS).mask_;
        const size_t precomputed_scales_size = wei_mask == 0
                ? scales_simd_w
                : nstl::max(
                        static_cast<size_t>(wei_scale_count), scales_simd_w);
        scratchpad.template book<float>(
                memory_tracking::names::key_precomputed_scales,
                precomputed_scales_size);
    }
}

bool req_copy_scales(
        const primitive_attr_t *attr, const float scale_adjust_factor) {
    const auto &attr_scales = attr->scales_;
    const bool with_src_scales
            = !attr_scales.get(DNNL_ARG_SRC).has_default_values();
    const bool with_wei_scales
            = !attr_scales.get(DNNL_ARG_WEIGHTS).has_default_values();
    const auto wei_scales_dt = attr_scales.get(DNNL_ARG_WEIGHTS).data_type_;
    const auto wei_scale_groups_ndims
            = attr_scales.get(DNNL_ARG_WEIGHTS).ndims_;
    return (with_src_scales && with_wei_scales) || scale_adjust_factor != 1.0f
            || (wei_scales_dt != data_type::f32 && with_wei_scales)
            || (wei_scale_groups_ndims > 0 && with_wei_scales);
}

const float *precompute_scales(const memory_tracking::grantor_t &scratchpad,
        const float *src_scales, const float *wei_scales, dim_t oc,
        const primitive_attr_t *attr, float scale_adjust_factor) {
    // Note: per-ic-channel is no supported in default
    const int wei_scale_mask = attr->scales_.get(DNNL_ARG_WEIGHTS).mask_;
    return precompute_scales(scratchpad, src_scales, wei_scales, 1, oc, false,
            wei_scale_mask != 0, attr, scale_adjust_factor, false);
}

const float *precompute_scales(const memory_tracking::grantor_t &scratchpad,
        const float *src_scales, const float *wei_scales, dim_t IC, dim_t OC,
        const bool wei_scale_per_ic, const bool wei_scale_per_oc,
        const primitive_attr_t *attr, float scale_adjust_factor,
        bool req_transpose) {
    using namespace dnnl::impl::memory_tracking::names;

    const auto &attr_scales = attr->scales_;
    const bool with_src_scales
            = !attr_scales.get(DNNL_ARG_SRC).has_default_values();
    const auto wei_scale_count
            = (wei_scale_per_ic ? IC : 1) * (wei_scale_per_oc ? OC : 1);

    const float *scales = nullptr;
    if (req_copy_scales(attr, scale_adjust_factor)) {
        const int wei_scale_mask = attr_scales.get(DNNL_ARG_WEIGHTS).mask_;
        size_t size = 0;
        auto loc_scales
                = scratchpad.template get<float>(key_precomputed_scales, &size);
        if (wei_scale_mask == 0 || wei_scale_count == 1) {
            const size_t count = nstl::min(size / sizeof(float), scales_simd_w);
            utils::array_set(loc_scales,
                    src_scales[0] * wei_scales[0] * scale_adjust_factor, count);
        } else {
            const dim_t count = nstl::min(
                    static_cast<dim_t>(size / sizeof(float)), wei_scale_count);
            const auto wei_scale_dt
                    = attr_scales.get(DNNL_ARG_WEIGHTS).data_type_;
            const auto wei_scale_groups_ndims
                    = attr_scales.get(DNNL_ARG_WEIGHTS).ndims_;
            const auto wei_scale_groups_ic = wei_scale_groups_ndims > 0
                    ? attr_scales.get(DNNL_ARG_WEIGHTS).group_dims_[0]
                    : 1;
            // Note: per-ic-channel scales is only supported for
            // weights decompression for now
            if ((wei_scale_per_ic && wei_scale_groups_ic > 1)
                    || req_transpose) {
                const auto wei_scale_stride_ic
                        = wei_scale_per_ic ? wei_scale_per_oc ? OC : 1 : 0;
                const auto wei_scale_stride_oc = wei_scale_per_oc ? 1 : 0;
                assert(count == wei_scale_count);
                PRAGMA_OMP_SIMD()
                for_(int ic = 0; ic < IC; ic++)
                for (int oc = 0; oc < wei_scale_stride_ic; oc++) {
                    const auto wei_scale_idx = wei_scale_stride_oc * oc
                            + wei_scale_stride_ic * (ic / wei_scale_groups_ic);
                    const auto loc_scale_idx
                            = req_transpose ? oc * IC + ic : ic * OC + oc;
                    const float wei_scales_val = io::load_float_value(
                            wei_scale_dt, wei_scales, wei_scale_idx);
                    loc_scales[loc_scale_idx] = src_scales[0] * wei_scales_val
                            * scale_adjust_factor;
                }
            } else {
                PRAGMA_OMP_SIMD()
                for (dim_t c = 0; c < count; c++) {
                    const float wei_scales_val
                            = io::load_float_value(wei_scale_dt, wei_scales, c);
                    loc_scales[c] = src_scales[0] * wei_scales_val
                            * scale_adjust_factor;
                }
            }
        }
        scales = loc_scales;
    } else if (with_src_scales) {
        scales = src_scales;
    } else {
        scales = wei_scales;
    }

    return scales;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
