/*******************************************************************************
* Copyright 2025 Intel Corporation
* Copyright 2025 Codeplay Software Limited
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

#ifndef GPU_GENERIC_SYCL_GROUP_NORMALIZATION_KERNEL_HPP
#define GPU_GENERIC_SYCL_GROUP_NORMALIZATION_KERNEL_HPP

#include <sycl/nd_item.hpp>

#include "common/primitive_exec_types.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"

namespace dnnl::impl::gpu::generic::sycl {
struct group_norm_fwd_t {
    using sycl_dims_t = int32_t[6];

    group_norm_fwd_t(const sycl_group_norm_conf_t &conf_,
            ::sycl::local_accessor<float, 1> &local_memory,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf_)
        , src(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , scale(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SCALE))
        , shift(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SHIFT))
        , dst(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , mean(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_MEAN))
        , variance(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_VARIANCE))
        , src_scale(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC))
        , dst_scale(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST))
        , po_args_(cgh, ctx, conf_.post_ops)
        , local_memory(local_memory) {}

    inline void operator()(::sycl::nd_item<2> it) const {
        auto batch = it.get_group(0);
        auto group_num = it.get_group(1);
        // Only one workgroup gets assigned to each group.
        // Purposefully not implementing optimal reduction / group_norm for the sake of
        // simplicity.

        auto src_wrapper = conf_.src_desc;
        auto dst_wrapper = conf_.dst_desc;
        dim_t num_elements_to_reduce = 1;
        const auto &dims = src_wrapper.dims();
        // As Batch and channels are dim 0 and 1 respectively
        for (int i = 2; i < src_wrapper.ndims(); i++) {
            num_elements_to_reduce *= dims[i];
        }
        dim_t num_spatial_elements = num_elements_to_reduce;
        num_elements_to_reduce *= conf_.num_channels_per_group;
        dims_t logical_index;
        for (auto i = group_num; i < static_cast<std::size_t>(conf_.num_groups);
                i += it.get_group_range(1)) {
            // as accumulation will always be in float.
            float accum = 0;
            float mean_value;
            float std_value;
            if (not conf_.use_global_stats) {
                for (dim_t j = it.get_local_linear_id();
                        j < num_elements_to_reduce;
                        j += static_cast<dim_t>(it.get_local_range(0))) {
                    get_logical_index(src_wrapper.ndims(), dims, batch,
                            group_num, num_spatial_elements, j, logical_index);
                    accum += load_float_value(src_wrapper.data_type(),
                            src.get_pointer(),
                            src_wrapper.off_v(logical_index));
                }

                workgroup_reduce(it, accum, it.get_local_range(0));
                // Divide by total elements to get the mean
                if (it.get_local_linear_id() == 0) {
                    local_memory[it.get_local_range(0)]
                            /= num_elements_to_reduce;
                    store_float_value(data_type::f32,
                            local_memory[it.get_local_range(0)],
                            mean.get_pointer(),
                            batch * conf_.num_groups + group_num);
                }
                ::sycl::group_barrier(it.get_group());
                //write each value to local memory;
                // start accum for standard deviation
                float mean = local_memory[it.get_local_range(0)];
                mean_value = mean;
                accum = 0;
                for (dim_t j = it.get_local_linear_id();
                        j < num_elements_to_reduce;
                        j += static_cast<dim_t>(it.get_local_range(0))) {
                    get_logical_index(src_wrapper.ndims(), dims, batch,
                            group_num, num_spatial_elements, j, logical_index);
                    accum += ::sycl::pown(
                            (load_float_value(src_wrapper.data_type(),
                                     src.get_pointer(),
                                     src_wrapper.off_v(logical_index))
                                    - mean),
                            2);
                }
                workgroup_reduce(it, accum, it.get_local_range(0) + 1);
                // calculate std + eta
                if (it.get_local_linear_id() == 0) {
                    float variance_val
                            = local_memory[it.get_local_range(0) + 1];
                    variance_val /= num_elements_to_reduce;
                    float std = ::sycl::sqrt(variance_val + conf_.eta);
                    local_memory[it.get_local_range(0) + 1] = std;
                    store_float_value(data_type::f32, variance_val,
                            variance.get_pointer(),
                            batch * conf_.num_groups + group_num);
                }
                ::sycl::group_barrier(it.get_group());
                std_value = local_memory[it.get_local_range(0) + 1];
            } else {
                mean_value
                        = load_float_value(data_type::f32, mean.get_pointer(),
                                batch * conf_.num_groups + group_num);
                std_value = load_float_value(data_type::f32,
                        variance.get_pointer(),
                        batch * conf_.num_groups + group_num);
                std_value = ::sycl::sqrt(std_value + conf_.eta);
            }

            //Now start the normalization
            for (dim_t j = it.get_local_linear_id(); j < num_elements_to_reduce;
                    j += static_cast<dim_t>(it.get_local_range(0))) {
                get_logical_index(src_wrapper.ndims(), dims, batch, group_num,
                        num_spatial_elements, j, logical_index);
                float value = load_float_value(src_wrapper.data_type(),
                        src.get_pointer(), src_wrapper.off_v(logical_index));
                float normalized_value = (value - mean_value) / std_value;
                int32_t channel_value = (j / num_spatial_elements)
                        % conf_.num_channels_per_group;
                if (conf_.use_scale) {
                    normalized_value *= load_float_value(data_type::f32,
                            scale.get_pointer(),
                            conf_.num_channels_per_group * group_num
                                    + channel_value);
                }
                if (conf_.use_shift) {
                    normalized_value += load_float_value(data_type::f32,
                            shift.get_pointer(),
                            conf_.num_channels_per_group * group_num
                                    + channel_value);
                }
                if (conf_.src_scaling) {
                    // Only one scaling factor per tensor is allowed,
                    // as per the spec. Scaling factor will also always be f32 as per spec.
                    normalized_value *= load_float_value(
                            data_type::f32, src_scale.get_pointer(), 0);
                }
                float prev_value = normalized_value;
                normalized_value = conf_.post_ops.apply(
                        normalized_value, prev_value, po_args_, logical_index);
                if (conf_.dst_scaling) {
                    // Only one scaling factor per tensor is allowed,
                    // as per the spec. Scaling factor will also always be f32 as per spec.
                    normalized_value *= (1.0f
                            / load_float_value(data_type::f32,
                                    dst_scale.get_pointer(), 0));
                }
                store_float_value(dst_wrapper.data_type(), normalized_value,
                        dst.get_pointer(), dst_wrapper.off_v(logical_index));
            }
        }
    }

private:
    inline void get_logical_index(int ndims, const sycl_dims_t &dims,
            dim_t batch, dim_t group_num, dim_t total_spacial_elements,
            dim_t flattened_index, dims_t &logical_index) const {

        logical_index[0] = batch;
        // Calculate Channel index
        logical_index[1] = group_num * conf_.num_channels_per_group
                + ((flattened_index / total_spacial_elements)
                        % conf_.num_channels_per_group);

        for (int i = ndims - 1; i >= 2; i--) {
            logical_index[i] = flattened_index % dims[i];
            flattened_index /= dims[i];
        }
    }

    inline void workgroup_reduce(::sycl::nd_item<2> &it,
            float workitem_accum_value, int32_t idx_to_write_red_value) const {
        local_memory[it.get_local_linear_id()] = workitem_accum_value;

        ::sycl::group_barrier(it.get_group());

        // Group leader accumulates;
        if (it.get_local_linear_id() == 0) {
            float total_sum = 0;
            for (auto i = std::size_t(0); i < it.get_local_range(0); i++) {
                total_sum += local_memory[i];
            }
            local_memory[idx_to_write_red_value] = total_sum;
        }
    }

    sycl_group_norm_conf_t conf_;
    xpu::sycl::in_memory_arg_t src;
    xpu::sycl::in_memory_arg_t scale;
    xpu::sycl::in_memory_arg_t shift;
    xpu::sycl::inout_memory_arg_t dst;
    xpu::sycl::inout_memory_arg_t mean;
    xpu::sycl::inout_memory_arg_t variance;
    xpu::sycl::in_memory_arg_t src_scale;
    xpu::sycl::in_memory_arg_t dst_scale;
    post_op_input_args po_args_;
    ::sycl::local_accessor<float, 1> local_memory;
};

struct group_norm_bwd_t {
    using sycl_dims_t = int32_t[6];
    group_norm_bwd_t(const sycl_gnorm_bwd_conf_t &conf_,
            ::sycl::local_accessor<float, 1> &local_memory,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf_)
        , src(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , diff_dst(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST))
        , mean(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MEAN))
        , variance(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_VARIANCE))
        , scales(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SCALE))
        , diff_src(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC))
        , diff_scales(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SCALE))
        , diff_bias(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SHIFT))
        , local_memory(local_memory) {}

    void operator()(::sycl::nd_item<1> it) const {
        auto channel_num = it.get_group(0);
        auto group_num = channel_num / conf_.num_channels_per_group;
        // each group handles a channel

        dim_t num_spatial_elements = 1;
        auto num_dims = conf_.src_desc.ndims();
        const auto &dims = conf_.src_desc.dims();

        for (int i = 2; i < num_dims; i++) {
            num_spatial_elements *= dims[i];
        }

        dims_t logical_index;
        logical_index[1] = channel_num;
        float gamma = conf_.scale_diff_required
                ? load_float_value(
                          data_type::f32, scales.get_pointer(), channel_num)
                : 1.0f;
        float diff_gamma = 0;
        float diff_beta = 0;
        for (dim_t batch = 0; batch < dims[0]; batch++) {
            logical_index[0] = batch;
            float mean_val = load_float_value(data_type::f32,
                    mean.get_pointer(), batch * conf_.num_groups + group_num);
            float variance_val
                    = load_float_value(data_type::f32, variance.get_pointer(),
                            batch * conf_.num_groups + group_num);
            float std = ::sycl::sqrt(variance_val + conf_.eta);
            for (dim_t spatial_index = it.get_local_id(0);
                    spatial_index < num_spatial_elements;
                    spatial_index += it.get_local_range(0)) {
                get_spatial_logical_index(
                        dims, logical_index, spatial_index, num_dims);
                float src_val = load_float_value(conf_.src_desc.data_type(),
                        src.get_pointer(), conf_.src_desc.off_v(logical_index));
                float diff_dst_val = load_float_value(
                        conf_.diff_dst_desc.data_type(), diff_dst.get_pointer(),
                        conf_.diff_dst_desc.off_v(logical_index));
                // mean and variance will always have memory format dnnl::format::ab
                float normalized_value = (src_val - mean_val) / std;
                // dL / dY = \sigma_{i = 0}^{C} dL/dy_{i} * \hat{x_{i}}
                diff_gamma += diff_dst_val * normalized_value;
                diff_beta += diff_dst_val;
            }
        }

        workgroup_reduce(it, diff_gamma);
        if (it.get_local_linear_id() == 0 && conf_.scale_diff_required) {
            // group leader writes back to global memory
            store_float_value(data_type::f32, local_memory[0],
                    diff_scales.get_pointer(), channel_num);
        }
        // update all threads with the final value of diff_gamma
        diff_gamma = local_memory[0];
        ::sycl::group_barrier(it.get_group());
        workgroup_reduce(it, diff_beta);
        if (it.get_local_linear_id() == 0 && conf_.bias_diff_required) {
            // group leader writes back to global memory
            store_float_value(data_type::f32, local_memory[0],
                    diff_bias.get_pointer(), channel_num);
        }
        // update all threads with the final value of diff_beta
        diff_beta = local_memory[0];

        // Calculate dL/dX
        for (dim_t batch = 0; batch < dims[0]; batch++) {
            logical_index[0] = batch;
            float mean_val = load_float_value(data_type::f32,
                    mean.get_pointer(), batch * conf_.num_groups + group_num);
            float variance_val
                    = load_float_value(data_type::f32, variance.get_pointer(),
                            batch * conf_.num_groups + group_num);
            float std = ::sycl::sqrt(variance_val + conf_.eta);
            for (dim_t spatial_index = it.get_local_id(0);
                    spatial_index < num_spatial_elements;
                    spatial_index += it.get_local_range(0)) {
                get_spatial_logical_index(
                        dims, logical_index, spatial_index, num_dims);
                float diff_src_value = load_float_value(
                        conf_.diff_dst_desc.data_type(), diff_dst.get_pointer(),
                        conf_.diff_dst_desc.off_v(logical_index));
                if (not conf_.used_global_stats) {
                    float x = load_float_value(conf_.src_desc.data_type(),
                            src.get_pointer(),
                            conf_.src_desc.off_v(logical_index));
                    float x_hat = (x - mean_val) * diff_gamma;
                    diff_src_value -= (diff_beta + (x_hat / std))
                            / (num_spatial_elements
                                    * conf_.num_channels_per_group);
                }
                diff_src_value = gamma * (diff_src_value / std);
                store_float_value(conf_.diff_src_desc.data_type(),
                        diff_src_value, diff_src.get_pointer(),
                        conf_.diff_src_desc.off_v(logical_index));
            }
        }
    }

private:
    inline void get_spatial_logical_index(const sycl_dims_t &dims,
            dims_t &logical_index, dim_t flattened_index,
            dim_t num_dims) const {
        for (dim_t i = num_dims - 1; i >= 2; i--) {
            logical_index[i] = flattened_index % dims[i];
            flattened_index /= dims[i];
        }
    }

    inline void workgroup_reduce(
            ::sycl::nd_item<1> &it, float wi_accum_value) const {
        local_memory[it.get_local_id(0)] = wi_accum_value;
        ::sycl::group_barrier(it.get_group());
        //group leader accumulates
        if (it.get_local_id(0) == 0) {
            float accum_value = 0;
            for (std::size_t i = 0; i < it.get_local_range(0); i++) {
                accum_value += local_memory[i];
            }
            local_memory[0] = accum_value;
        }
        ::sycl::group_barrier(it.get_group());
    }

    sycl_gnorm_bwd_conf_t conf_;
    xpu::sycl::in_memory_arg_t src;
    xpu::sycl::in_memory_arg_t diff_dst;
    xpu::sycl::in_memory_arg_t mean;
    xpu::sycl::in_memory_arg_t variance;
    xpu::sycl::in_memory_arg_t scales;
    xpu::sycl::inout_memory_arg_t diff_src;
    xpu::sycl::inout_memory_arg_t diff_scales;
    xpu::sycl::inout_memory_arg_t diff_bias;
    ::sycl::local_accessor<float, 1> local_memory;
};

} // namespace dnnl::impl::gpu::generic::sycl

#endif
