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

#ifndef GPU_GENERIC_SYCL_SOFTMAX_KERNELS_HPP
#define GPU_GENERIC_SYCL_SOFTMAX_KERNELS_HPP

#include "common/compiler_workarounds.hpp"

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

struct softmax_fwd_kernel_vec_t {
    softmax_fwd_kernel_vec_t(const sycl_softmax_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , src_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , scale_src_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC))
        , scale_dst_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST))
        , dst_(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , po_args_(cgh, ctx, conf_.post_ops) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t src_mem(src_, conf_.src_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);
        memory_plain_t src_scale_mem(scale_src_, data_type::f32);
        memory_plain_t dst_scale_mem(scale_dst_, data_type::f32);

        dims_t dst_dims;
        for (int i = 0; i < xpu::sycl::md_t::max_dims; i++) {
            if (i < dst_mem.md().ndims()) {
                dst_dims[i] = dst_mem.md().dims()[i];
            } else {
                dst_dims[i] = 1;
            }
        }

        auto operation = [= WA_THIS_COPY_CAPTURE](
                                 dim_t &ou, dim_t &in) mutable {
            float space_denom = 0;
            float space_max = -FLT_MAX;
            dim_t ou_in_offset = ou * conf_.channels * conf_.inner_size + in;

            for (dim_t c = 0; c < conf_.channels; c++) {
                size_t off
                        = src_md().off_l(ou_in_offset + c * conf_.inner_size);
                float s = src_mem.load(off);
                space_max = nstl::max(space_max, s);
            }
            for (dim_t c = 0; c < conf_.channels; c++) {
                size_t src_off
                        = src_md().off_l(ou_in_offset + c * conf_.inner_size);
                float s = src_mem.load(src_off);
                float d = s - space_max;
                space_denom += ::sycl::exp((float)d);
            }
            if (conf_.alg_kind == alg_kind::softmax_log) {
                space_denom = ::sycl::log((float)space_denom);
            }
            for (dim_t c = 0; c < conf_.channels; c++) {
                size_t src_off
                        = src_md().off_l(ou_in_offset + c * conf_.inner_size);
                float s = src_mem.load(src_off);
                float d = s - space_max;
                if (conf_.alg_kind == alg_kind::softmax_accurate) {
                    d = ::sycl::exp((float)d);
                }
                float sd = space_denom;
                if (conf_.alg_kind == alg_kind::softmax_accurate) {
                    d /= sd;
                } else if (conf_.alg_kind == alg_kind::softmax_log) {
                    d -= sd;
                }

                size_t dst_off
                        = dst_md().off_l(ou_in_offset + c * conf_.inner_size);

                float scale = 1.0f;
                if (conf_.do_scale_src) {
                    scale = conf_.do_scale_src ? src_scale_mem.load(0) : scale;
                    d = d * scale;
                }

                dims_t off;
                utils::l_dims_by_l_offset(off,
                        ou_in_offset + c * conf_.inner_size, dst_dims,
                        dst_md().ndims());
                d = conf_.post_ops.apply(d, po_args_, off);

                if (conf_.do_scale_dst) {
                    scale = conf_.do_scale_dst ? dst_scale_mem.load(0) : scale;
                    d = d / scale;
                }

                dst_mem.store(d, dst_off);
            }
        };

        for (int idx = item.get_global_id(0); idx < conf_.wk_size;
                idx += item.get_global_range(0)) {
            dim_t in = (idx / (1)) % conf_.inner_size;
            dim_t ou = (idx / (conf_.inner_size)) % conf_.outer_size;
            operation(ou, in);
        }
    }

private:
    const xpu::sycl::md_t &src_md() const { return conf_.src_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }

    void *gen_ptr(xpu::sycl::in_memory_arg_t gen_) const {
        return gen_.get_pointer();
    }

    sycl_softmax_conf_t conf_;
    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::in_memory_arg_t scale_src_;
    xpu::sycl::in_memory_arg_t scale_dst_;
    xpu::sycl::inout_memory_arg_t dst_;
    post_op_input_args po_args_;
};

struct softmax_bwd_kernel_vec_t {
    softmax_bwd_kernel_vec_t(const sycl_softmax_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , dst_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , diff_dst_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST))
        , diff_src_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC)) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t dst_mem(dst_, conf_.dst_md);
        memory_tensor_t diff_src_mem(diff_src_, conf_.diff_src_md);
        memory_tensor_t diff_dst_mem(diff_dst_, conf_.diff_dst_md);

        auto operation = [= WA_THIS_COPY_CAPTURE](
                                 dim_t &ou, dim_t &in) mutable {
            dim_t ou_in_offset = ou * conf_.channels * conf_.inner_size + in;
            float sbr = 0;
            for (dim_t c = 0; c < conf_.channels; ++c) {
                auto diff_dst_off = diff_dst_md().off_l(
                        ou_in_offset + c * conf_.inner_size);
                float dd = diff_dst_mem.load(diff_dst_off);
                if (conf_.alg_kind == alg_kind::softmax_accurate) {
                    auto dst_off = dst_md().off_l(
                            ou_in_offset + c * conf_.inner_size);
                    float d = dst_mem.load(dst_off);
                    sbr += dd * d;
                } else if (conf_.alg_kind == alg_kind::softmax_log) {
                    sbr += dd;
                }
            }

            for (dim_t c = 0; c < conf_.channels; ++c) {
                auto diff_dst_off = diff_dst_md().off_l(
                        ou_in_offset + c * conf_.inner_size);
                auto dst_off
                        = dst_md().off_l(ou_in_offset + c * conf_.inner_size);

                float d = dst_mem.load(dst_off);
                float dd = diff_dst_mem.load(diff_dst_off);

                float val = 0;
                if (conf_.alg_kind == alg_kind::softmax_accurate) {
                    val = d * (dd - sbr);
                } else if (conf_.alg_kind == alg_kind::softmax_log) {
                    val = dd - ::sycl::exp(d) * sbr;
                }

                auto diff_src_off = diff_src_md().off_l(
                        ou_in_offset + c * conf_.inner_size);
                diff_src_mem.store(val, diff_src_off);
            }
        };

        for (int idx = item.get_global_id(0); idx < conf_.wk_size;
                idx += item.get_global_range(0)) {
            dim_t in = (idx / 1) % conf_.inner_size;
            dim_t ou = (idx / conf_.inner_size) % conf_.outer_size;
            operation(ou, in);
        }
    }

private:
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    const xpu::sycl::md_t &diff_dst_md() const { return conf_.diff_dst_md; }
    const xpu::sycl::md_t &diff_src_md() const { return conf_.diff_src_md; }

    sycl_softmax_conf_t conf_;
    xpu::sycl::in_memory_arg_t dst_;
    xpu::sycl::in_memory_arg_t diff_dst_;
    xpu::sycl::out_memory_arg_t diff_src_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
