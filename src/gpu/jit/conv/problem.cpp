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

#include "gpu/jit/conv/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

bool is_conv_index(const prb_dim_t &dim) {
    for (auto prop : {prop_kind::forward, prop_kind::backward_data,
                 prop_kind::backward_weights}) {
        if (is_conv_index(dim, prop)) return true;
    }
    return false;
}

bool is_conv_index(const prb_dim_t &dim, prop_kind_t prop) {
    auto &dims = conv_index_dims(prop);
    for (auto &d : dims) {
        if (d == dim) return true;
    }
    return false;
}

const std::vector<prb_dim_t> &conv_index_dims(prop_kind_t prop) {
    auto get_dims = [&](prop_kind_t prop) {
        std::vector<prb_dim_t> ret;
        ret.push_back(prb_dims::mb);
        ret.push_back(prb_dims::g);
        ret.push_back(prb_dims::oc);
        ret.push_back(prb_dims::ic);
        ret.push_back(prb_dims::kd);
        ret.push_back(prb_dims::kh);
        ret.push_back(prb_dims::kw);
        if (prop != prop_kind::backward_data) {
            ret.push_back(prb_dims::od);
            ret.push_back(prb_dims::oh);
            ret.push_back(prb_dims::ow);
        } else {
            ret.push_back(prb_dims::id);
            ret.push_back(prb_dims::ih);
            ret.push_back(prb_dims::iw);
        }
        return ret;
    };
    static std::vector<prb_dim_t> fwd_dims = get_dims(prop_kind::forward);
    static std::vector<prb_dim_t> bwd_d_dims
            = get_dims(prop_kind::backward_data);
    static std::vector<prb_dim_t> bwd_w_dims
            = get_dims(prop_kind::backward_weights);
    switch (prop) {
        case prop_kind::forward: return fwd_dims;
        case prop_kind::backward_data: return bwd_d_dims;
        case prop_kind::backward_weights: return bwd_w_dims;
        default: ir_error_not_expected();
    }
    return fwd_dims;
}

prb_dim_t to_gemm(const prb_dim_t &d, prop_kind_t prop, bool is_transpose) {
    bool is_fwd = (prop == prop_kind::forward);
    bool is_bwd_d = (prop == prop_kind::backward_data);
    bool is_bwd_w = (prop == prop_kind::backward_weights);
    auto transpose_gemm = [](const prb_dim_t &d) {
        if (d == prb_dims::m) return prb_dims::n;
        if (d == prb_dims::n) return prb_dims::m;
        if (d == prb_dims::k) return prb_dims::k;
        ir_error_not_expected();
        return prb_dim_t();
    };
    auto pick = [&](const prb_dim_t &fwd, const prb_dim_t &bwd_d,
                        const prb_dim_t &bwd_w) {
        if (is_transpose) {
            if (is_fwd) return transpose_gemm(fwd);
            if (is_bwd_d) return transpose_gemm(bwd_d);
            if (is_bwd_w) return transpose_gemm(bwd_w);
        }
        if (is_fwd) return fwd;
        if (is_bwd_d) return bwd_d;
        if (is_bwd_w) return bwd_w;
        ir_error_not_expected();
        return prb_dim_t();
    };
    switch (d.kind()) {
        case prb_dim_kind_t::g: return prb_dims::b;
        case prb_dim_kind_t::mb:
            return pick(prb_dims::m, prb_dims::m, prb_dims::k);
        case prb_dim_kind_t::oc:
            return pick(prb_dims::n, prb_dims::k, prb_dims::n);
        case prb_dim_kind_t::ic:
            return pick(prb_dims::k, prb_dims::n, prb_dims::m);
        case prb_dim_kind_t::kd:
        case prb_dim_kind_t::kh:
        case prb_dim_kind_t::kw:
            return pick(prb_dims::k, prb_dims::k, prb_dims::m);
        case prb_dim_kind_t::od:
        case prb_dim_kind_t::oh:
        case prb_dim_kind_t::ow:
            return pick(prb_dims::m, prb_dim_t(), prb_dims::k);
        case prb_dim_kind_t::id:
        case prb_dim_kind_t::ih:
        case prb_dim_kind_t::iw:
            return pick(prb_dim_t(), prb_dims::m, prb_dim_t());
        default: return prb_dim_t();
    }
}

prb_tile_t to_gemm(const prb_tile_t &t, prop_kind_t prop, bool is_transpose) {
    prb_tile_t ret;
    ret[prb_dims::b] = 1;
    ret[prb_dims::m] = 1;
    ret[prb_dims::n] = 1;
    ret[prb_dims::k] = 1;
    for (auto &d : t) {
        auto gemm_d = to_gemm(d, prop, is_transpose);
        if (gemm_d.is_undef()) continue;
        ret[gemm_d] *= t[d];
    }
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
