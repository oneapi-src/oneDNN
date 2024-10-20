/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_SHUFFLE_KERNELS_HPP
#define GPU_GENERIC_SYCL_SHUFFLE_KERNELS_HPP

#include "common/dnnl_thread.hpp"
#include "common/primitive_exec_types.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "xpu/sycl/memory_storage_base.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct shuffle_kernel_vec1_t {
    shuffle_kernel_vec1_t(const sycl_shuffle_conf_t &conf,
            xpu::sycl::in_memory_arg_t &data, xpu::sycl::out_memory_arg_t &dst)
        : conf_(conf), data_(data), dst_(dst) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t data_mem(data_, conf_.src_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);

        auto sg = item.get_sub_group();
        dim_t blk_size = conf_.block_size;
        const dim_t stride_mb = conf_.stride_m;
        size_t base = ((item.get_group(0) * conf_.wg_size
                               + sg.get_group_id()[0] * sg.get_local_range()[0])
                        * blk_size
                + sg.get_local_id() * blk_size);

        size_t xaxis = (base / 1) % conf_.MB;
        dim_t yaxis = (base / conf_.MB) % conf_.C;

        int j = yaxis / conf_.transpose_col;
        int i = yaxis % conf_.transpose_col;

        const dim_t output_off = xaxis * stride_mb + yaxis * conf_.SP;
        const dim_t input_off
                = xaxis * stride_mb + (i * conf_.transpose_row + j) * conf_.SP;

        for (dim_t sp = 0; sp < conf_.SP; ++sp) {
            float dst = data_mem.load(input_off + sp);
            dst_mem.store(dst, output_off + sp);
        }
    }

private:
    const xpu::sycl::md_t &data_md() const { return conf_.src_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    const xpu::sycl::md_t &stat_md() const { return conf_.stat_md; }

    sycl_shuffle_conf_t conf_;
    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::out_memory_arg_t dst_;
};

struct shuffle_kernel_vec2_t {
    shuffle_kernel_vec2_t(const sycl_shuffle_conf_t &conf,
            xpu::sycl::in_memory_arg_t &data, xpu::sycl::out_memory_arg_t &dst)
        : conf_(conf), data_(data), dst_(dst) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t data_mem(data_, conf_.src_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);

        const dim_t stride_mb = conf_.stride_m;

        size_t ithr = item.get_global_id(0);
        dim_t sp_start {0}, sp_end {0};
        balance211(conf_.SP, conf_.nthr, ithr, sp_start, sp_end);

        for (dim_t mb = 0; mb < conf_.MB; mb++) {
            for (dim_t sp = sp_start; sp < sp_end; sp++) {
                const dim_t off = mb * stride_mb + sp * conf_.C;
                for (dim_t c = 0; c < conf_.C; ++c) {
                    dim_t i = c % conf_.transpose_col;
                    dim_t j = c / conf_.transpose_col;
                    float dst
                            = data_mem.load(off + i * conf_.transpose_row + j);
                    dst_mem.store(dst, off + c);
                }
            }
        }
    }

private:
    const xpu::sycl::md_t &data_md() const { return conf_.src_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    const xpu::sycl::md_t &stat_md() const { return conf_.stat_md; }

    sycl_shuffle_conf_t conf_;
    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::out_memory_arg_t dst_;
};

struct shuffle_kernel_vec3_t {
    shuffle_kernel_vec3_t(const sycl_shuffle_conf_t &conf,
            xpu::sycl::in_memory_arg_t &data, xpu::sycl::out_memory_arg_t &dst)
        : conf_(conf), data_(data), dst_(dst) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t data_mem(data_, conf_.src_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);

        size_t ithr = item.get_global_id(0);
        const dim_t outer_size = conf_.outer_size;
        const dim_t inner_size = conf_.inner_size;
        const dim_t dim = conf_.axis_size * inner_size;

        dim_t axis_size = conf_.axis_size;
        dim_t ax_start {0}, ax_end {0};
        balance211(axis_size, conf_.nthr, ithr, ax_start, ax_end);

        for (dim_t ou = 0; ou < outer_size; ou++) {
            for (dim_t iwork = ax_start; iwork < ax_end; ++iwork) {
                int j = iwork / conf_.transpose_col;
                int i = iwork % conf_.transpose_col;
                for (dim_t in = 0; in < inner_size; in++) {
                    const dim_t off = ou * dim + in;
                    float dst = data_mem.load(data_md().off_l(
                            off + (i * conf_.transpose_row + j) * inner_size));
                    dst_mem.store(
                            dst, data_md().off_l(off + iwork * inner_size));
                }
            }
        }
    }

private:
    const xpu::sycl::md_t &data_md() const { return conf_.src_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    const xpu::sycl::md_t &stat_md() const { return conf_.stat_md; }

    sycl_shuffle_conf_t conf_;
    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::out_memory_arg_t dst_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
