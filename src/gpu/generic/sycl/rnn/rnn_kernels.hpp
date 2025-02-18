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

#ifndef SRC_GPU_GENERIC_SYCL_RNN_RNN_KERNELS_HPP
#define SRC_GPU_GENERIC_SYCL_RNN_RNN_KERNELS_HPP

#include "common/c_types_map.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_math_utils.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

inline int off_ker_bias(int dhc, int i0, int i1, int n_gates) {
    return i0 * dhc + i1;
}

inline int cell_ws_state(int states_ws_ld, int i, int j) {
    return i * states_ws_ld + j;
}

inline int cell_scratch_mem(
        int scratch_gates_ld, int dhc, int i, int n, int j) {
    return i * scratch_gates_ld + n * dhc + j;
}

struct ref_rnn_copy_t {
    ref_rnn_copy_t(const sycl_rnn_copy_conf_t &conf,
            const xpu::sycl::in_memory_arg_t &src,
            xpu::sycl::out_memory_arg_t &dst)
        : src_ {src}, dst_ {dst}, conf_ {conf} {}

    void operator()(::sycl::nd_item<3> item) const {
        const dim_t tl = item.get_global_id(0) / conf_.n_dir; // timestep/layer
        const dim_t dir = item.get_global_id(0) % conf_.n_dir; // direction
        const dim_t n = item.get_global_id(1); // batch
        const dim_t c = item.get_global_id(2); // channel

        if (dir >= conf_.n_dir || n >= conf_.batch || c >= conf_.range) return;

        dim_t src_offset = 0;
        dim_t dst_offset = 0;
        if (conf_.layer) { // layer
            if (tl >= conf_.n_iter) return;
            if (conf_.to_state) { // init
                src_offset = conf_.src_md.off(tl, n, c);
                dst_offset = conf_.dst_md.off(0, dir, tl, n, c);
            } else { // res
                src_offset = conf_.src_md.off(conf_.n_layer, dir, tl, n, c);
                dst_offset = conf_.dst_md.off(tl, n, dir * conf_.range + c);
            }
        } else { // iter
            if (tl >= conf_.n_layer) return;
            if (conf_.to_state) { // init
                src_offset = conf_.src_md.off(tl, dir, n, c);
                dst_offset = conf_.dst_md.off(tl, dir, conf_.n_iter, n, c);
            } else { // res
                src_offset
                        = conf_.src_md.off(tl + 1, dir, conf_.n_iter - 1, n, c);
                dst_offset = conf_.dst_md.off(tl, dir, n, c);
            }
        }
        if (src_ptr()) {
            auto src = load_float_value(
                    src_md().data_type(), src_ptr(), src_offset);
            if (dst_ptr()) {
                store_float_value(
                        src_md().data_type(), src, dst_ptr(), dst_offset);
            }
        } else {
            if (dst_ptr()) {
                store_float_value(
                        src_md().data_type(), 0.0f, dst_ptr(), dst_offset);
            }
        }
    }

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
    sycl_rnn_copy_conf_t conf_;

    const xpu::sycl::md_t &src_md() const { return conf_.src_md; }
    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
};

struct ref_rnn_bias {
    ref_rnn_bias(const sycl_rnn_bias_conf_t &conf,
            const xpu::sycl::inout_memory_arg_t &src_base,
            const xpu::sycl::in_memory_arg_t &bias,
            const xpu::sycl::out_memory_arg_t &dst_base)
        : src_ {src_base}, bias_ {bias}, dst_ {dst_base}, conf_ {conf} {}
    void operator()(::sycl::nd_item<3> item) const {

        const int b = item.get_global_id(1);
        const int c = item.get_global_id(0);

        if (b >= conf_.batch || c >= conf_.dhc) return;

        auto src = src_ptr();
        auto bias = bias_ptr();
        auto dst = dst_ptr();

        auto src_offset = src_data_offset(b, c);
        auto bias_offset = bias_data_offset(b, c);
        auto dst_offset = dst_data_offset(b, c);

        auto src_val
                = load_float_value(conf_.dst_md.data_type(), src, src_offset);
        auto bias_val = load_float_value(conf_.bias_type, bias, bias_offset);

        auto g = compute_gates(src_val, bias_val);

        store_float_value(conf_.dst_md.data_type(), g, dst, dst_offset);
        store_float_value(conf_.dst_md.data_type(), g, src, src_offset);
    }

    inline dim_t src_data_offset(int b, int c) const {
        return cell_scratch_mem(conf_.gates_ws_ld, conf_.dhc, b, 0, c);
    }

    inline dim_t bias_data_offset(int b, int c) const {
        return off_ker_bias(conf_.dhc, 0, c, 0);
    }

    inline dim_t dst_data_offset(int b, int c) const {
        return cell_ws_state(conf_.states_ws_ld, b, c);
    }

    float compute_gates(float in_val, float bias_val) const {
        switch (conf_.activation_kind) {
            case alg_kind::eltwise_relu:
                return (float)(math::relu_fwd(
                        (float)(in_val + bias_val), conf_.alpha));
            case alg_kind::eltwise_tanh:
                return (float)(math::tanh_fwd((float)(in_val + bias_val)));
            case alg_kind::eltwise_logistic:
                return (float)(math::logistic_fwd((float)(in_val + bias_val)));
            default: return 0;
        }
    }

    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *bias_ptr() const { return bias_.get_pointer(); }

    xpu::sycl::inout_memory_arg_t src_;
    xpu::sycl::in_memory_arg_t bias_;
    xpu::sycl::out_memory_arg_t dst_;
    sycl_rnn_bias_conf_t conf_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
