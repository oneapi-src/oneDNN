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

#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct reorder_kernel_t {
    static constexpr int vec_len = 8;
    static constexpr int max_supported_ndims = 6;

    reorder_kernel_t(const sycl_reorder_conf_t &conf,
            xpu::sycl::in_memory_arg_t &src, xpu::sycl::out_memory_arg_t &dst,
            xpu::sycl::in_memory_arg_t &src_scale,
            xpu::sycl::in_memory_arg_t &dst_scale, data_type_t scales_src_dt,
            data_type_t scales_dst_dt)
        : conf_(conf)
        , src_(src)
        , dst_(dst)
        , src_scale_(src_scale)
        , dst_scale_(dst_scale)
        , scales_src_dt_(scales_src_dt)
        , scales_dst_dt_(scales_dst_dt) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;

        size_t base_idx = offset_t * conf_.block_size;

        float scale_src = conf_.do_scale_src && conf_.scale_src_mask == 0
                ? load_float_value(scales_src_dt_, src_scale_ptr(), 0)
                : 1.f;
        float scale_dst = conf_.do_scale_dst && conf_.scale_dst_mask == 0
                ? load_float_value(scales_dst_dt_, dst_scale_ptr(), 0)
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

        for (int i = 0; i < conf_.block_size; i++) {
            int idx = base_idx + i;
            if (idx < conf_.wk_size) {
                for (int i = 0; i < max_supported_ndims; i++) {
                    off[i] = idx / strides[i] % dims[i];
                }

                int dst_idx = dst_md().off_v(off);
                auto src = load_float_value(
                        src_md().data_type(), src_ptr(), idx);
                auto dst = load_float_value(
                        dst_md().data_type(), dst_ptr(), dst_idx);

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
                        scale_src = load_float_value(
                                scales_src_dt_, src_scale_ptr(), scale_idx);
                    }
                    src *= scale_src;
                }

                auto acc = src;
                acc = conf_.post_ops.apply(acc, dst);
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

                        scale_dst = load_float_value(
                                scales_dst_dt_, dst_scale_ptr(), scale_idx);
                    }
                    acc /= scale_dst;
                }
                store_float_value(
                        dst_md().data_type(), acc, dst_ptr(), dst_idx);
            }
        }
    }

private:
    const xpu::sycl::md_t &src_md() const { return conf_.src_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }

    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    float *src_scale_ptr() const {
        return static_cast<float *>(src_scale_.get_pointer());
    }
    float *dst_scale_ptr() const {
        return static_cast<float *>(dst_scale_.get_pointer());
    }

    sycl_reorder_conf_t conf_;

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
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
