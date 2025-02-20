/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"

#include "gpu/intel/jit/v2/conv/tensor_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

struct stride_t {
    dim_t factor = 0;
    std::vector<pvar_t> pvars;

    stride_t() = default;
    stride_t(dim_t factor) : factor(factor) {}
    bool is_zero() const { return factor == 0; }
    bool is_one() const { return factor == 1 && pvars.empty(); }

    stride_t &operator*=(const pvar_t &pvar) {
        pvars.push_back(pvar);
        return *this;
    }

    stride_t &operator*=(dim_t factor) {
        this->factor *= factor;
        if (is_zero()) pvars.clear();
        return *this;
    }

    template <typename BinaryFunc>
    expr_t binary_expr(dim_t rhs, const BinaryFunc &func) const {
        if (is_zero()) return func(0, rhs);
        if (is_one()) return func(1, rhs);
        expr_t lhs = pvars[0].var();
        for (size_t i = 1; i < pvars.size(); i++)
            lhs *= pvars[i].var();
        return func(lhs, ir_utils::safe_div(rhs, factor));
    }

    expr_t mod(dim_t rhs) const {
        return binary_expr(
                rhs, [](const expr_t &a, const expr_t &b) { return a % b; });
    }
    expr_t ge(dim_t rhs) const {
        return binary_expr(
                rhs, [](const expr_t &a, const expr_t &b) { return a >= b; });
    }
    expr_t le(dim_t rhs) const {
        return binary_expr(
                rhs, [](const expr_t &a, const expr_t &b) { return a <= b; });
    }

    void intersect(const stride_t &other) {
        if (is_zero()) {
            factor = other.factor;
            pvars = other.pvars;
            return;
        }
        factor = math::gcd(factor, other.factor);
        std::vector<pvar_t> new_pvars;
        for (auto &a : pvars) {
            bool found = false;
            for (auto &b : other.pvars) {
                if (a == b) { found = true; }
            }
            if (found) new_pvars.push_back(a);
        }
        pvars = new_pvars;
    }
};

struct block_2d_params_t {
    stride_t base_stride;
    dim_t x_stride;
    expr_t y_stride;
    pvar_t w_dim;
    pvar_t h_dim;
};

block_2d_params_t to_block_2d_params(const prop_kind_t &prop,
        const tensor_kind_t &tensor_kind, int type_size,
        const pvar_tile_t &tg_tile, const pvar_tile_t &iter_tile,
        const pvar_map_t<stride_t> &strides) {
    bool is_fwd = (prop == prop_kind::forward);
    bool is_bwd_d = (prop == prop_kind::backward_data);
    bool is_bwd_w = (prop == prop_kind::backward_weights);
    auto abc = to_abc(prop, tensor_kind);
    auto to_layout_dim = [&](const pvar_t &d) {
        if (strides.has(d)) return d;
        auto mnk = to_gemm(d, prop);
        if (mnk == pvars::m && abc == tensor_kind_t::b) return pvar_t();
        if (mnk == pvars::n && abc == tensor_kind_t::a) return pvar_t();
        if (mnk == pvars::k && abc == tensor_kind_t::c) return pvar_t();
        if ((is_fwd || is_bwd_w) && tensor_kind == tensor_kind_t::src
                && utils::one_of(d, pvars::ow, pvars::kw))
            return pvars::iw;
        if (is_bwd_d && tensor_kind == tensor_kind_t::dst
                && utils::one_of(d, pvars::iw, pvars::kw))
            return pvars::ow;
        gpu_error_not_expected() << "Unknown dim: " << d;
        return pvar_t();
    };
    block_2d_params_t params;
    for (auto &_d : iter_tile) {
        auto d = to_layout_dim(_d);
        if (d.is_undef()) continue;
        if (strides.at(d).is_one()) {
            gpu_assert(params.w_dim.is_undef());
            params.w_dim = d;
        } else {
            gpu_assert(params.h_dim.is_undef());
            params.h_dim = d;
        }
    }
    gpu_assert(!params.w_dim.is_undef());
    gpu_assert(!params.h_dim.is_undef());
    params.y_stride = expr_t(1);
    if ((is_fwd || is_bwd_w) && params.h_dim == pvars::iw) {
        params.y_stride = pvars::sw.var();
    }
    for (auto &d : strides) {
        if (utils::one_of(d, params.w_dim, params.h_dim)) continue;
        params.base_stride.intersect(strides[d]);
    }
    params.base_stride *= type_size;
    return params;
}

void generate_2d_reqs(const kernel_desc_t &desc, tensor_kind_t tensor_kind,
        prb_reqs_t &reqs) {
    using ir_utils::safe_div;
    if (to_abc(desc.prop, tensor_kind) == tensor_kind_t::c
            && desc.use_stream_k) {
        // No block 2D access with atomics.
        return;
    }
    bool is_fwd = (desc.prop == prop_kind::forward);
    bool is_bwd_w = (desc.prop == prop_kind::backward_weights);
    auto tag = append_groups(
            tensor_kind, desc.layout_tag(tensor_kind), desc.is_dw);
    pvar_map_t<stride_t> strides;
    stride_t stride(1);
    for (int i = tag.raw_tag().nentries() - 1; i >= 0; i--) {
        auto &e = tag.raw_tag().entries()[i];
        gpu_assert(!e.is_blocked);
        auto dim = tag.desc().prb_dim(e.index());
        strides[dim] = stride;
        stride *= dim;
    }
    int type_size = tag.type().size();
    auto params = to_block_2d_params(desc.prop, tensor_kind, type_size,
            desc.thread_group_tile, desc.iter_tile, strides);
    int base_align = block_2d_base_alignment(desc.hw_desc);
    auto W = params.w_dim.var();
    auto H = params.h_dim.var();
    auto P = strides.at(params.h_dim);
    if (!is_one(params.y_stride))
        P.pvars.push_back(pvar_t::from_var(params.y_stride));
    reqs.add_no_simplify(W >= safe_div(block_2d_min_dim(), type_size));
    reqs.add_no_simplify(W <= safe_div(block_2d_max_dim(), type_size));
    reqs.add_no_simplify(W % block_2d_w_alignment(type_size) == 0);
    if (is_one(params.y_stride)) {
        reqs.add_no_simplify(H <= block_2d_max_dim());
    } else {
        reqs.add_no_simplify(H % params.y_stride == 0);
        reqs.add_no_simplify(H / params.y_stride <= block_2d_max_dim());
    }
    reqs.add_no_simplify(P.ge(safe_div(block_2d_min_dim(), type_size)));
    reqs.add_no_simplify(P.le(safe_div(block_2d_max_dim(), type_size)));
    reqs.add_no_simplify(
            P.mod(safe_div(block_2d_pitch_alignment(desc.hw_desc), type_size))
            == 0);
    reqs.add_no_simplify(params.base_stride.mod(base_align) == 0);
    if ((is_fwd || is_bwd_w) && params.h_dim == pvars::iw) {
        reqs.add_no_simplify((params.y_stride == 1) | (pvars::pw.var() == 0));
        reqs.add_no_simplify((params.y_stride == 1) | (pvars::kw.var() == 1));
    }
}

prb_reqs_t generate_2d_reqs(const kernel_desc_t &desc) {
    prb_reqs_t reqs = desc.spec.reqs();
    if (!desc.use_2d_access) return reqs;
    generate_2d_reqs(desc, tensor_kind_t::src, reqs);
    generate_2d_reqs(desc, tensor_kind_t::wei, reqs);
    generate_2d_reqs(desc, tensor_kind_t::dst, reqs);
    reqs.simplify();
    return reqs;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
