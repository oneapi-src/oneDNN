/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_JIT_CONV_PROBLEM_HPP
#define GPU_JIT_CONV_PROBLEM_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "gpu/jit/ir/core.hpp"
#include "gpu/jit/ir/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

inline const std::vector<prb_dim_t> &conv_layout_dims(
        tensor_kind_t tensor_kind, bool src_dst_with_group = false) {
    static const std::vector<prb_dim_t> src_dims({prb_dims::mb, prb_dims::ic,
            prb_dims::id, prb_dims::ih, prb_dims::iw});
    static const std::vector<prb_dim_t> src_g_dims({prb_dims::mb, prb_dims::g,
            prb_dims::ic, prb_dims::id, prb_dims::ih, prb_dims::iw});
    static const std::vector<prb_dim_t> wei_dims({prb_dims::g, prb_dims::oc,
            prb_dims::ic, prb_dims::kd, prb_dims::kh, prb_dims::kw});
    static const std::vector<prb_dim_t> dst_dims({prb_dims::mb, prb_dims::oc,
            prb_dims::od, prb_dims::oh, prb_dims::ow});
    static const std::vector<prb_dim_t> dst_g_dims({prb_dims::mb, prb_dims::g,
            prb_dims::oc, prb_dims::od, prb_dims::oh, prb_dims::ow});
    switch (tensor_kind) {
        case tensor_kind_t::src:
            return src_dst_with_group ? src_g_dims : src_dims;
        case tensor_kind_t::wei: return wei_dims;
        case tensor_kind_t::dst:
            return src_dst_with_group ? dst_g_dims : dst_dims;
        default: ir_error_not_expected();
    }
    return src_dims;
}

bool is_conv_index(const prb_dim_t &dim);
bool is_conv_index(const prb_dim_t &dim, prop_kind_t prop);
const std::vector<prb_dim_t> &conv_index_dims(prop_kind_t prop);

template <typename T>
T &&pick_a(prop_kind_t prop, T &&src, T &&wei, T &&dst) {
    return utils::one_of(prop, prop_kind::forward, prop_kind::backward_weights)
            ? std::forward<T>(src)
            : std::forward<T>(dst);
}

template <typename T>
T &&pick_b(prop_kind_t prop, T &&src, T &&wei, T &&dst) {
    return utils::one_of(prop, prop_kind::forward, prop_kind::backward_data)
            ? std::forward<T>(wei)
            : std::forward<T>(dst);
}

template <typename T>
T &&pick_c(prop_kind_t prop, T &&src, T &&wei, T &&dst) {
    bool is_fwd = (prop == prop_kind::forward);
    bool is_bwd_d = (prop == prop_kind::backward_data);
    return std::forward<T>(is_fwd ? dst : is_bwd_d ? src : wei);
}

inline tensor_kind_t to_abc(prop_kind_t prop, tensor_kind_t tensor) {
    bool is_bwd_d = (prop == prop_kind::backward_data);
    bool is_bwd_w = (prop == prop_kind::backward_weights);
    tensor_kind_t kinds[3]
            = {tensor_kind_t::a, tensor_kind_t::b, tensor_kind_t::c};
    if (is_bwd_d) std::swap(kinds[0], kinds[2]);
    if (is_bwd_w) std::swap(kinds[1], kinds[2]);
    switch (tensor) {
        case tensor_kind_t::src: return kinds[0];
        case tensor_kind_t::wei: return kinds[1];
        case tensor_kind_t::dst: return kinds[2];
        default: ir_error_not_expected();
    }
    return kinds[0];
}

prb_dim_t to_gemm(
        const prb_dim_t &d, prop_kind_t prop, bool is_transpose = false);
prb_tile_t to_gemm(
        const prb_tile_t &t, prop_kind_t prop, bool is_transpose = false);

inline const std::vector<prb_dim_t> &conv_stride_dims() {
    static std::vector<prb_dim_t> _stride_dims = [&]() {
        std::vector<prb_dim_t> ret;
        ret.push_back(prb_dims::sd);
        ret.push_back(prb_dims::sh);
        ret.push_back(prb_dims::sw);
        return ret;
    }();
    return _stride_dims;
}

inline const std::vector<prb_dim_t> &conv_dilation_dims() {
    static std::vector<prb_dim_t> _dilation_dims = [&]() {
        std::vector<prb_dim_t> ret;
        ret.push_back(prb_dims::dd);
        ret.push_back(prb_dims::dh);
        ret.push_back(prb_dims::dw);
        return ret;
    }();
    return _dilation_dims;
}

inline const std::vector<prb_dim_t> &conv_padding_dims() {
    static std::vector<prb_dim_t> _padding_dims = [&]() {
        std::vector<prb_dim_t> ret;
        ret.push_back(prb_dims::pd);
        ret.push_back(prb_dims::ph);
        ret.push_back(prb_dims::pw);
        return ret;
    }();
    return _padding_dims;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
