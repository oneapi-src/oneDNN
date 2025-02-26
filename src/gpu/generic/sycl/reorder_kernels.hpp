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

#ifndef GPU_GENERIC_SYCL_REORDER_KERNELS_HPP
#define GPU_GENERIC_SYCL_REORDER_KERNELS_HPP

#include "common/primitive_exec_types.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "xpu/sycl/memory_storage_base.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct reorder_kernel_t {
    static constexpr int vec_len = 8;
    static constexpr int max_supported_ndims = 6;

    reorder_kernel_t(const sycl_reorder_conf_t &conf, ::sycl::handler &cgh,
            const exec_ctx_t &ctx)
        : conf_(conf)
        , src_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_FROM))
        , dst_(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_TO))
        , src_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM))
        , dst_scale_(
                  CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO))
        , src_zero_points(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC))
        , dst_zero_points(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST))
        , scales_src_dt_(conf_.do_scale_src
                          ? ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , scales_dst_dt_(conf_.do_scale_dst
                          ? ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , src_zp_dt_(conf_.apply_src_zp
                          ? ctx.memory_mdw(DNNL_ARG_ATTR_ZERO_POINTS
                                       | DNNL_ARG_FROM)
                                    .data_type()
                          : data_type_t::dnnl_s32)
        , dst_zp_dt_(conf_.apply_dst_zp
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_TO)
                                    .data_type()
                          : data_type_t::dnnl_s32)
        , po_args_(cgh, ctx, conf.post_ops) {}

    void operator()(::sycl::nd_item<1> item) const {

        auto from_tensor = memory_tensor_t(src_, conf_.src_md);
        auto to_tensor = memory_tensor_t(dst_, conf_.dst_md);
        auto src_scales_tensor
                = memory_plain_t(src_scale_, scales_src_dt_); // 1D tensor
        auto dst_scales_tensor = memory_plain_t(dst_scale_, scales_dst_dt_);
        auto src_zp_tensor = memory_plain_t(src_zero_points, src_zp_dt_);
        auto dst_zp_tensor = memory_plain_t(dst_zero_points, dst_zp_dt_);

        const auto &src_dims = conf_.src_md.dims();
        const auto &dst_dims = conf_.dst_md.dims();
        int ndims = conf_.src_md.ndims();

        dims_t dst_padded_index;
        float src_scale = conf_.do_scale_src ? src_scales_tensor.load(0) : 1;
        float dst_scale = conf_.do_scale_dst ? dst_scales_tensor.load(0) : 1;
        float src_zp = conf_.apply_src_zp ? src_zp_tensor.load(0) : 0;
        float dst_zp = conf_.apply_dst_zp ? dst_zp_tensor.load(0) : 0;

        for (std::size_t flattened_index = item.get_global_id(0);
                flattened_index < conf_.num_elements;
                flattened_index += item.get_global_range(0)) {

            to_tensor.get_logical_index(
                    flattened_index, dst_padded_index, true);

            // and the only remaining case, both are within the data region
            float from_value = from_tensor.load_md(dst_padded_index, true);
            // apply src scale and zero points;
            // dequantized_x = scale * (quantized_x - zero_point);
            if (conf_.apply_src_zp) {
                if (conf_.src_zp_mask > 0) {
                    dim_t idx = get_quant_param_offset(dst_padded_index,
                            src_dims, conf_.src_zp_mask, ndims);
                    src_zp = src_zp_tensor.load(idx);
                }
                from_value = from_value - src_zp;
            }
            if (conf_.do_scale_src) {
                if (conf_.scale_src_mask > 0) {
                    dim_t idx = get_quant_param_offset(dst_padded_index,
                            src_dims, conf_.scale_src_mask, ndims);
                    src_scale = src_scales_tensor.load(idx);
                }
                from_value = from_value * src_scale;
            }

            auto dst_idx = conf_.dst_md.off_v(dst_padded_index, true);
            from_value = conf_.post_ops.apply(from_value, dst_, dst_idx);
            // during quanization, apply scale first
            // quantized value = dequan_value / scale + zero_point;
            if (conf_.do_scale_dst) {
                if (conf_.scale_dst_mask > 0) {
                    dim_t idx = get_quant_param_offset(dst_padded_index,
                            dst_dims, conf_.scale_dst_mask, ndims);
                    dst_scale = dst_scales_tensor.load(idx);
                }
                from_value = from_value / dst_scale;
            }
            if (conf_.apply_dst_zp) {
                if (conf_.dst_zp_mask > 0) {
                    dim_t idx = get_quant_param_offset(dst_padded_index,
                            dst_dims, conf_.dst_zp_mask, ndims);
                    dst_zp = dst_zp_tensor.load(idx);
                }
                from_value = from_value + dst_zp;
            }
            to_tensor.store_md(from_value, dst_padded_index, true);
        }
    }

private:
    // it may be worth worth making  these free functions such that other primitives can
    // use it as well.
    using sycl_dims_t = int32_t[6];

    inline dim_t get_quant_param_offset(const dims_t &logical_index,
            const sycl_dims_t &dims, int param_mask, int ndims) const {
        dim_t idx = 0;
        for (int32_t i = 0; i < ndims; i++) {
            bool ith_bit_set = (param_mask >> i) & 1;
            dim_t dimension_offset = 0;
            dim_t dimension_stride = 1;
            if (ith_bit_set) {
                dimension_offset = logical_index[i];
                dimension_stride = dims[i];
            }
            idx = idx * dimension_stride + dimension_offset;
        }
        return idx;
    }

    const xpu::sycl::md_t &src_md() const { return conf_.src_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }

    sycl_reorder_conf_t conf_;

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::inout_memory_arg_t dst_;
    xpu::sycl::in_memory_arg_t src_scale_;
    xpu::sycl::in_memory_arg_t dst_scale_;
    xpu::sycl::in_memory_arg_t src_zero_points;
    xpu::sycl::in_memory_arg_t dst_zero_points;
    data_type_t scales_src_dt_;
    data_type_t scales_dst_dt_;
    data_type_t src_zp_dt_;
    data_type_t dst_zp_dt_;
    post_op_input_args po_args_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
