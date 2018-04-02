/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"

#include "ref_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace alg_kind;
using namespace math;

template <impl::data_type_t data_type>
void ref_eltwise_fwd_t<data_type>::execute_forward_generic() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());

    const int MB = conf_.MB();
    const int C = conf_.C();
    const int D = conf_.D();
    const int H = conf_.H();
    const int W = conf_.W();
    const auto alg_kind = conf_.desc()->alg_kind;
    const float alpha = conf_.desc()->alpha;
    const float beta = conf_.desc()->beta;
    const bool is_3d = conf_.desc()->data_desc.ndims == 5;

#   pragma omp parallel for collapse(5) schedule(static)
    for (int n = 0; n < MB; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int id = 0; id < D; ++id)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                auto d_off = is_3d
                    ? data_d.off(n, c, id, h, w) : data_d.off(n, c, h, w);
                data_t s = src[d_off];
                data_t &d = dst[d_off];
                switch (alg_kind) {
                case eltwise_relu: d = relu_fwd(s, alpha); break;
                case eltwise_tanh: d = tanh_fwd(s); break;
                case eltwise_elu: d = elu_fwd(s, alpha); break;
                case eltwise_square: d = square_fwd(s); break;
                case eltwise_abs: d = abs_fwd(s); break;
                case eltwise_sqrt: d = sqrt_fwd(s); break;
                case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
                case eltwise_bounded_relu:
                    d = bounded_relu_fwd(s, alpha); break;
                case eltwise_soft_relu: d = soft_relu_fwd(s); break;
                case eltwise_logistic: d = logistic_fwd(s); break;
                default: assert(!"unknown eltwise alg_kind");
                }
            }
        }
    }
}

template <impl::data_type_t data_type>
void ref_eltwise_fwd_t<data_type>::execute_forward_dense() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());

    const size_t nelems = data_d.nelems();
    const auto alg_kind = conf_.desc()->alg_kind;
    const float alpha = conf_.desc()->alpha;
    const float beta  = conf_.desc()->beta;

    src += data_d.blocking_desc().offset_padding;
    dst += data_d.blocking_desc().offset_padding;

    if (alg_kind == eltwise_relu) {
        // a fast path for relu as the most popular activation
#       pragma omp parallel for schedule(static)
        for (size_t e = 0; e < nelems; ++e)
            dst[e] = relu_fwd(src[e], alpha);
        return;
    }

#   pragma omp parallel for schedule(static)
    for (size_t e = 0; e < nelems; ++e) {
        const data_t s = src[e];
        data_t &d = dst[e];

        switch (alg_kind) {
        case eltwise_tanh: d = tanh_fwd(s); break;
        case eltwise_elu: d = elu_fwd(s, alpha); break;
        case eltwise_square: d = square_fwd(s); break;
        case eltwise_abs: d = abs_fwd(s); break;
        case eltwise_sqrt: d = sqrt_fwd(s); break;
        case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
        case eltwise_bounded_relu: d = bounded_relu_fwd(s, alpha); break;
        case eltwise_soft_relu: d = soft_relu_fwd(s); break;
        case eltwise_logistic: d = logistic_fwd(s); break;
        default: assert(!"unknown eltwise alg_kind");
        }
    }
}

template <impl::data_type_t data_type>
void ref_eltwise_bwd_t<data_type>::execute_backward_generic() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());
    const memory_desc_wrapper diff_data_d(conf_.diff_src_pd());

    const int MB = conf_.MB();
    const int C = conf_.C();
    const int D = conf_.D();
    const int H = conf_.H();
    const int W = conf_.W();
    const auto alg_kind = conf_.desc()->alg_kind;
    const float alpha = conf_.desc()->alpha;
    const float beta = conf_.desc()->beta;
    const bool is_3d = conf_.desc()->data_desc.ndims == 5;

#   pragma omp parallel for collapse(5) schedule(static)
    for (int n = 0; n < MB; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                auto data_off = is_3d
                    ? data_d.off(n, c, d, h, w) : data_d.off(n, c, h, w);
                auto diff_data_off = is_3d
                    ? diff_data_d.off(n, c, d, h, w)
                    : diff_data_d.off(n, c, h, w);
                data_t s = src[data_off];
                data_t dd = diff_dst[diff_data_off];
                data_t &ds = diff_src[diff_data_off];
                switch (alg_kind) {
                case eltwise_relu: ds = relu_bwd(dd, s, alpha); break;
                case eltwise_tanh: ds = tanh_bwd(dd, s); break;
                case eltwise_elu: ds = elu_bwd(dd, s, alpha); break;
                case eltwise_square: ds = square_bwd(dd, s); break;
                case eltwise_abs: ds = abs_bwd(dd, s); break;
                case eltwise_sqrt: ds = sqrt_bwd(dd, s); break;
                case eltwise_linear:
                    ds = linear_bwd(dd, s, alpha, beta); break;
                case eltwise_bounded_relu:
                    ds = bounded_relu_bwd(dd, s, alpha); break;
                case eltwise_soft_relu: ds = soft_relu_bwd(dd, s); break;
                case eltwise_logistic: ds = logistic_bwd(dd, s); break;
                default: assert(!"unknown eltwise alg_kind");
                }
            }
        }
    }
}

template <impl::data_type_t data_type>
void ref_eltwise_bwd_t<data_type>::execute_backward_dense() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());
    const memory_desc_wrapper diff_data_d(conf_.diff_src_pd());

    const size_t nelems = data_d.nelems();
    const auto alg_kind = conf_.desc()->alg_kind;
    const float alpha = conf_.desc()->alpha;
    const float beta = conf_.desc()->beta;

    src += data_d.blocking_desc().offset_padding;
    diff_dst += diff_data_d.blocking_desc().offset_padding;
    diff_src += diff_data_d.blocking_desc().offset_padding;

#   pragma omp parallel for schedule(static)
    for (size_t e = 0; e < nelems; ++e) {
        const data_t dd = diff_dst[e];
        const data_t s = src[e];
        data_t &ds = diff_src[e];

        switch (alg_kind) {
        case eltwise_relu: ds = relu_bwd(dd, s, alpha); break;
        case eltwise_tanh: ds = tanh_bwd(dd, s); break;
        case eltwise_elu: ds = elu_bwd(dd, s, alpha); break;
        case eltwise_square: ds = square_bwd(dd, s); break;
        case eltwise_abs: ds = abs_bwd(dd, s); break;
        case eltwise_sqrt: ds = sqrt_bwd(dd, s); break;
        case eltwise_linear: ds = linear_bwd(dd, s, alpha, beta); break;
        case eltwise_bounded_relu: ds = bounded_relu_bwd(dd, s, alpha); break;
        case eltwise_soft_relu: ds = soft_relu_bwd(dd, s); break;
        case eltwise_logistic: ds = logistic_bwd(dd, s); break;
        default: assert(!"unknown eltwise alg_kind");
        }
    }
}

template struct ref_eltwise_fwd_t<data_type::f32>;
template struct ref_eltwise_fwd_t<data_type::s32>;
template struct ref_eltwise_fwd_t<data_type::s16>;
template struct ref_eltwise_fwd_t<data_type::s8>;
template struct ref_eltwise_fwd_t<data_type::u8>;

template struct ref_eltwise_bwd_t<data_type::f32>;
template struct ref_eltwise_bwd_t<data_type::s32>;
template struct ref_eltwise_bwd_t<data_type::s16>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
