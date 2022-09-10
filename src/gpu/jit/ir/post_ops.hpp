/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef GPU_JIT_IR_POST_OPS_HPP
#define GPU_JIT_IR_POST_OPS_HPP

#include <string>
#include <vector>

#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class post_op_tensor_info_t {
public:
    post_op_tensor_info_t() = default;

    post_op_tensor_info_t(bool is_input, bool is_output, const view_t &view,
            const expr_t &buf, uint32_t mask, const expr_t &op_var, float scale)
        : is_input_(is_input)
        , is_output_(is_output)
        , view_(view)
        , buf_(buf)
        , mask_(mask)
        , op_var_(op_var)
        , scale_(scale) {
        if (op_var_.is_empty())
            op_var_ = var_t::make(type_t::f32(), make_op_var_name(buf));
        if (scale != 1)
            ir_assert(is_output_)
                    << "Scale is supported with output tensors only.";
    }

    bool is_input() const { return is_input_; }

    bool is_output() const { return is_output_; }

    bool needs_masked_update() const { return needs_masked_update_; }

    const view_t &view() const { return view_; }

    const expr_t &buf() const { return buf_; }

    const uint32_t &mask() const { return mask_; }

    const expr_t &op_var() const { return op_var_; }

    float scale() const { return scale_; }

    post_op_tensor_info_t create_sub_tensor(const tensor_t &tile) const {
        auto ret = *this;
        ret.view_ = ret.view_.create_sub_view(tile);
        return ret;
    }

    void require_masked_update() { needs_masked_update_ = true; }

private:
    static std::string make_op_var_name(const expr_t &buf) {
        auto *var = buf.as_ptr<var_t>();
        if (var) return var->name;

        auto *ptr = buf.as_ptr<ptr_t>();
        if (ptr) {
            auto prefix = make_op_var_name(ptr->base);
            ir_assert(is_const(ptr->off));
            int off = to_cpp<int>(ptr->off);
            return prefix + "_" + std::to_string(off);
        }

        ir_error_not_expected() << "Can't generate op var name: " << buf;
        return "unknown";
    }
    bool is_input_;
    bool is_output_;
    bool needs_masked_update_ = false;
    view_t view_;
    expr_t buf_;
    uint32_t mask_;
    expr_t op_var_;
    float scale_;
};

// There are two types of post-ops:
// - Eltwise:          lhs = eltwise(rhs) and rhs must be equal lhs
//   Eltwise is supported via special IR function eltwise_t
// - Generic post-op:  lhs = rhs
// Left-hand side (lhs) represents a single post-op tensor. Right-hand side
// tensor (rhs) is an IR expression over post-op tensors and constants.
//
// Post-op tensors support broadcast (when used from rhs) and reduction (when
// used from lhs) semantics.
//
// If lhs is (a x 1) tensor and rhs is (a x b) tensor then rhs is reduced:
//     lhs(i, 0) = sum over j rhs(i, j)
//
// If lhs is (a x b) tensor and rhs is (a x 1) tensor then rhs is broadcasted:
//     lhs(i, j) = rhs(i, 0)
class post_op_t {
public:
    post_op_t() = default;

    post_op_t(const expr_t &lhs, const expr_t &rhs,
            const func_t &eltwise = func_t())
        : lhs_(lhs), rhs_(simplify_rewrite(rhs)), eltwise_(eltwise) {}

    const expr_t &lhs() const { return lhs_; }

    const expr_t &rhs() const { return rhs_; }

    const func_t &eltwise() const { return eltwise_; }

    bool uses(const expr_t &op_var) const {
        if (contains_object(lhs_, op_var)) return true;
        if (contains_object(rhs_, op_var)) return true;
        return false;
    }

private:
    expr_t lhs_;
    expr_t rhs_;
    func_t eltwise_;
};

inline op_kind_t alg_kind_to_op_kind(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::binary_add: return op_kind_t::_add;
        case alg_kind::binary_sub: return op_kind_t::_sub;
        case alg_kind::binary_mul: return op_kind_t::_mul;
        case alg_kind::binary_div: return op_kind_t::_div;
        case alg_kind::binary_min: return op_kind_t::_min;
        case alg_kind::binary_max: return op_kind_t::_max;
        case alg_kind::binary_ge: return op_kind_t::_ge;
        case alg_kind::binary_gt: return op_kind_t::_gt;
        case alg_kind::binary_le: return op_kind_t::_le;
        case alg_kind::binary_lt: return op_kind_t::_lt;
        case alg_kind::binary_eq: return op_kind_t::_eq;
        case alg_kind::binary_ne: return op_kind_t::_ne;
        default: ir_error_not_expected();
    }
    return op_kind_t::undef;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
