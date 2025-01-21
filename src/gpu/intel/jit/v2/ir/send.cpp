/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/intel/jit/v2/ir/send.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {

bool process_coef_y_stride(
        plane_t &plane, expr_t &u, const expr_t &v, const prover_t &prover) {
    auto reduce_if_divisible = [](expr_t &a, const expr_t &b) {
        auto args = op_split(op_kind_t::_mul, a);
        for (auto &arg : args) {
            if (arg.is_equal(b)) {
                arg = expr_t();
                a = op_combine(op_kind_t::_mul, args);
                return true;
            }
        }
        return false;
    };
    if (reduce_if_divisible(u, plane.y_stride)) return true;
    if (auto *op = u.as_ptr<unary_op_t>()) {
        auto a = op->a;
        return process_coef_y_stride(plane, a, v, prover);
    }
    for (auto &size_val : {std::make_pair(u, 0), std::make_pair(v, 1)}) {
        auto dim = pvar_t::from_var(size_val.first);
        if (!dim.is_undef()) {
            if (!prover.require((plane.y_stride == 1)
                        | (size_val.first == size_val.second)))
                return false;
            return true;
        }
    }
    gpu_error_not_expected() << "Cannot make " << u << " x " << v
                             << " divisible by " << plane.y_stride;
    return false;
}

bool adjust_for_non_unit_y_stride(plane_t &plane,
        const pvar_coord_t<expr_t> &coord, const prover_t &prover) {
    if (is_one(plane.y_stride)) return true;
    auto y_stride_dim = pvar_t::from_var(plane.y_stride);
    if (y_stride_dim.is_undef()) return false;
    auto _y = to_linear(plane.y);
    auto &y = _y.as<linear_t>();
    auto to_size = [&](const expr_t &idx) {
        for (auto &d : coord) {
            if (coord[d].is_equal(idx)) return d.var();
        }
        return expr_t();
    };
    auto c = y.c;
    auto u_vec = y.u_vec;
    if (!process_coef_y_stride(plane, c, expr_t(), prover)) return false;
    for (int i = 0; i < y.nargs(); i++) {
        if (!process_coef_y_stride(
                    plane, u_vec[i], to_size(y.v_vec[i]), prover))
            return false;
    }
    if (!prover.require(plane.H % plane.y_stride == 0)) return false;
    plane.y = linear_t::to_expr(c, u_vec, y.v_vec);
    plane.H /= plane.y_stride;
    return true;
}

send_2d_desc_t::send_2d_desc_t(const view_t &view, const send_params_t &params,
        const prover_t &prover) {
    auto plane = view.plane();
    if (!params.hint_2d) return;
    if (!plane) return;
    if (!adjust_for_non_unit_y_stride(plane, view.coord(), prover)) return;

    auto &hint = params.hint_2d;
    hw = params.hw;
    address = params.address;
    op = params.op;
    type = view.type();
    transpose = hint.transpose;
    vnni = hint.vnni;
    W = plane.W;
    H = plane.H;
    P = plane.P;
    w = hint.width;
    h = hint.height;
    c = 1;
    w_rcount = ir_utils::safe_div(plane.w, w);
    h_rcount = ir_utils::safe_div(plane.h, h);
    w_dim = plane.w_dim;
    h_dim = plane.h_dim;
    base = get_2d_base(view);
    x_base = plane.x;
    y_base = plane.y;
    try_promote_count();
    is_valid = is_supported(view, prover);
}
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
