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

#ifndef GPU_GENERIC_SYCL_BINARY_KERNELS_HPP
#define GPU_GENERIC_SYCL_BINARY_KERNELS_HPP

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

struct binary_kernel_vec_t {
    static constexpr int vec_len = 8;
    static constexpr int max_supported_ndims = 6;

    binary_kernel_vec_t(const sycl_binary_conf_t &conf, ::sycl::handler &cgh,
            const exec_ctx_t &ctx)
        : conf_(conf)
        , src0_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC_0))
        , src1_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC_1))
        , dst_(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , src0_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0))
        , src1_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1))
        , scales_dt_((conf_.do_scale_src0) ? ctx.memory_mdw(DNNL_ARG_ATTR_SCALES
                                                        | DNNL_ARG_SRC_0)
                                                     .data_type()
                                           : data_type_t::dnnl_f32)
        , po_args_(cgh, ctx, conf_.post_ops) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t src0_mem(src0_, conf_.src0_md);
        memory_tensor_t src1_mem(src1_, conf_.src1_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);
        memory_plain_t src0_scale_mem(src0_scale_, scales_dt_);
        memory_plain_t src1_scale_mem(src1_scale_, scales_dt_);

        const float sm_0 = (conf_.do_scale_src0 ? src0_scale_mem.load(0) : 1.f);

        const float sm_1 = (conf_.do_scale_src1 ? src1_scale_mem.load(0) : 1.f);

        dims_t dims, strides, off_dst, off0, off1;
        bool any_broadcast = false;
        bool is_same_tag = true;
        for (int i = 0; i < max_supported_ndims; i++) {
            if (i < dst_mem.md().ndims()) {
                dims[i] = dst_mem.md().dims()[i];
                strides[i] = dst_mem.md().strides()[i];
                any_broadcast |= conf_.broadcast_dims0[i];
                any_broadcast |= conf_.broadcast_dims1[i];
            } else {
                dims[i] = 1;
                strides[i] = INT_MAX;
            }
            if (i < src0_mem.md().ndims()) {
                is_same_tag = is_same_tag
                        && (src0_mem.md().strides()[i]
                                == src1_mem.md().strides()[i]);
            }
        }

        const bool is_blocked_fmt = conf_.src0_md.inner_nblks() > 0
                || conf_.src1_md.inner_nblks() > 0
                || conf_.dst_md.inner_nblks() > 0;

        if (!any_broadcast && !is_blocked_fmt
                && conf_.post_ops.get_post_op() == 0
                && conf_.wk_size % vec_len == 0 && is_same_tag) {
            for (int vec_idx = item.get_global_id(0);
                    vec_idx < conf_.wk_size / vec_len;
                    vec_idx += item.get_global_range(0)) {
                auto src0_vec = src0_mem.load_vec<vec_len>(vec_idx);
                auto src1_vec = src1_mem.load_vec<vec_len>(vec_idx);

                if (conf_.do_scale_src0)
                    src0_vec *= ::sycl::vec<float, vec_len>(sm_0);
                if (conf_.do_scale_src1)
                    src1_vec *= ::sycl::vec<float, vec_len>(sm_1);

                auto acc_vec = compute_alg(src0_vec, src1_vec, conf_.alg_kind);
                // TODO: Adding post-ops seems to be interfering with compiler's
                // optimizations. Figure out how to make the compiler to generate
                // the right code.
                dst_mem.store_vec(acc_vec, vec_idx);
            }
        } else {
            for (int idx = item.get_global_id(0); idx < conf_.wk_size;
                    idx += item.get_global_range(0)) {
                auto l_offset = idx;
                for (int i = 0; i < conf_.ndims; i++) {
                    const int d = conf_.ndims - 1 - i;
                    const dim_t cur_dim = conf_.dst_md.dims()[d];
                    off_dst[d] = l_offset % cur_dim;
                    l_offset = l_offset / cur_dim;
                }

                for (int i = 0; i < max_supported_ndims; i++) {
                    off0[i] = conf_.broadcast_dims0[i] ? 0 : off_dst[i];
                    off1[i] = conf_.broadcast_dims1[i] ? 0 : off_dst[i];
                }

                auto src0 = src0_mem.load_md(off0);
                auto src1 = src1_mem.load_md(off1);

                if (conf_.do_scale_src0) src0 *= sm_0;
                if (conf_.do_scale_src1) src1 *= sm_1;

                auto acc = compute_alg_n(src0, src1, conf_.alg_kind);

                int dst_idx = dst_mem.md().off_v(off_dst);
                acc = conf_.post_ops.apply(
                        acc, dst_, dst_idx, po_args_, off_dst);
                dst_mem.store(acc, dst_idx);
            }
        }
    }

private:
    template <int width>
    ::sycl::vec<float, width> compute_alg(::sycl::vec<float, width> src0,
            ::sycl::vec<float, width> src1, alg_kind_t alg) const {
        switch (alg) {
            case alg_kind::binary_add: return src0 + src1;
            case alg_kind::binary_div: return src0 / src1;
            case alg_kind::binary_max: return ::sycl::fmax(src0, src1);
            case alg_kind::binary_min: return ::sycl::fmin(src0, src1);
            case alg_kind::binary_mul: return src0 * src1;
            case alg_kind::binary_sub: return src0 - src1;
            case alg_kind::binary_ge:
                return ((src0 >= src1) * -1).template convert<float>();
            case alg_kind::binary_gt:
                return ((src0 > src1) * -1).template convert<float>();
            case alg_kind::binary_le:
                return ((src0 <= src1) * -1).template convert<float>();
            case alg_kind::binary_lt:
                return ((src0 < src1) * -1).template convert<float>();
            case alg_kind::binary_eq:
                return ((src0 == src1) * -1).template convert<float>();
            case alg_kind::binary_ne:
                return ((src0 != src1) * -1).template convert<float>();
            default: return ::sycl::vec<float, width> {NAN};
        }
    }

    template <typename T>
    T compute_alg_n(T src0, T src1, alg_kind_t alg) const {
        switch (alg) {
            case alg_kind::binary_add: return src0 + src1;
            case alg_kind::binary_div: return src0 / src1;
            case alg_kind::binary_max: return ::sycl::max(src0, src1);
            case alg_kind::binary_min: return ::sycl::min(src0, src1);
            case alg_kind::binary_mul: return src0 * src1;
            case alg_kind::binary_sub: return src0 - src1;
            case alg_kind::binary_ge: return ((src0 >= src1));
            case alg_kind::binary_gt: return ((src0 > src1));
            case alg_kind::binary_le: return ((src0 <= src1));
            case alg_kind::binary_lt: return ((src0 < src1));
            case alg_kind::binary_eq: return ((src0 == src1));
            case alg_kind::binary_ne: return ((src0 != src1));
            default: return (T)(999);
        }
    }

    sycl_binary_conf_t conf_;

    xpu::sycl::in_memory_arg_t src0_;
    xpu::sycl::in_memory_arg_t src1_;
    xpu::sycl::inout_memory_arg_t dst_;
    xpu::sycl::in_memory_arg_t src0_scale_;
    xpu::sycl::in_memory_arg_t src1_scale_;
    data_type_t scales_dt_;
    post_op_input_args po_args_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
