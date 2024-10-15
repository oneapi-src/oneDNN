/*******************************************************************************
* Copyright 2024 Intel Corporation
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
        , src_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC_0))
        , dst_(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , src_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0))
        , dst_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST))
        , scales_src_dt_(conf_.do_scale_src
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , scales_dst_dt_(conf_.do_scale_dst
                          ? ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)
                                    .data_type()
                          : data_type_t::dnnl_f32) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t src_mem(src_, conf_.src_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);
        memory_plain_t src_scale_mem(src_scale_, scales_src_dt_);
        memory_plain_t dst_scale_mem(dst_scale_, scales_dst_dt_);

        float scale_src = conf_.do_scale_src && conf_.scale_src_mask == 0
                ? src_scale_mem.load(0)
                : 1.f;
        float scale_dst = conf_.do_scale_dst && conf_.scale_dst_mask == 0
                ? dst_scale_mem.load(0)
                : 1.f;

        dims_t dims, off, strides;
        for (int i = 0; i < max_supported_ndims; i++) {
            dims[i] = (i < src_md().ndims()) ? src_md().dims()[i] : 1;
            strides[i]
                    = (i < src_md().ndims()) ? src_md().strides()[i] : INT_MAX;
        }
        dims_t dims_scales_src;
        if (conf_.scale_src_mask != 0) {
            for (int i = 0; i < max_supported_ndims; i++) {
                dims_scales_src[i]
                        = conf_.scale_src_mask >> i & 1 ? dims[i] : 1;
            }
        }
        dims_t dims_scales_dst;
        if (conf_.scale_dst_mask != 0) {
            for (int i = 0; i < max_supported_ndims; i++) {
                dims_scales_dst[i]
                        = conf_.scale_dst_mask >> i & 1 ? dims[i] : 1;
            }
        }

        for (int idx = item.get_global_id(0); idx < conf_.wk_size;
                idx += item.get_global_range(0)) {
            for (int i = 0; i < max_supported_ndims; i++) {
                off[i] = idx / strides[i] % dims[i];
            }

            int dst_idx = dst_md().off_v(off);
            auto src = src_mem.load(idx);

            if (conf_.do_scale_src) {
                if (conf_.scale_src_mask != 0) {
                    int scale_idx = 0;
                    for (int i = 0; i < max_supported_ndims; i++) {
                        if (i < src_md().ndims()) {
                            int off_scales_i = conf_.scale_src_mask >> i & 1
                                    ? off[i]
                                    : 0;
                            scale_idx = scale_idx * dims_scales_src[i]
                                    + off_scales_i;
                        }
                    }
                    scale_src = src_scale_mem.load(scale_idx);
                }
                src *= scale_src;
            }

            auto acc = src;
            acc = conf_.post_ops.apply(acc, dst_, dst_idx);
            if (conf_.do_scale_dst) {
                if (conf_.scale_dst_mask != 0) {
                    int scale_idx = 0;
                    for (int i = 0; i < max_supported_ndims; i++) {
                        if (i < src_md().ndims()) {
                            int off_scales_i = conf_.scale_dst_mask >> i & 1
                                    ? off[i]
                                    : 0;
                            scale_idx = scale_idx * dims_scales_dst[i]
                                    + off_scales_i;
                        }
                    }

                    scale_dst = dst_scale_mem.load(scale_idx);
                }
                acc /= scale_dst;
            }
            dst_mem.store(acc, dst_idx);
        }
    }

private:
    const xpu::sycl::md_t &src_md() const { return conf_.src_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }

    sycl_reorder_conf_t conf_;

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::inout_memory_arg_t dst_;
    xpu::sycl::in_memory_arg_t src_scale_;
    xpu::sycl::in_memory_arg_t dst_scale_;
    data_type_t scales_src_dt_;
    data_type_t scales_dst_dt_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
