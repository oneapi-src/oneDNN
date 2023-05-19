/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include <utility>

#include <compiler/jit/xbyak/ir/xbyak_visitor.hpp>
#include <util/array_ref.hpp>

#include "ir_data_initializer.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class ir_data_initializer_impl_t : public xbyak_visitor_t {
public:
    using xbyak_visitor_t::dispatch;
    using xbyak_visitor_t::visit;

    ir_data_initializer_impl_t() = default;

    // dispatch override
    func_c dispatch(func_c v) override {
        for (auto &p : v->params_) {
            initialize_expr_data(p, nullptr);
            if (p.isa<tensor>()) { update_spill_weight(p, 128); }
        }
        return xbyak_visitor_t::dispatch(std::move(v));
    }

    expr_c dispatch(expr_c v) override {
        initialize_expr_data(v, nullptr);
        update_spill_weight(v, 32 * loop_depth());
        return xbyak_visitor_t::dispatch(std::move(v));
    }

    stmt_c dispatch(stmt_c v) override {
        initialize_stmt_data(v);
        return xbyak_visitor_t::dispatch(std::move(v));
    }

    // visit override
    stmt_c visit(define_c v) override {
        initialize_expr_data(v->var_, current_scope());
        if (v->var_.isa<tensor>()) { update_spill_weight(v->var_, 128); }
        return xbyak_visitor_t::visit(std::move(v));
    }

    stmt_c visit(assign_c v) override {
        auto ret = xbyak_visitor_t::visit(std::move(v));
        assert(ret.isa<assign>());
        auto vv = ret.static_as<assign_c>();
        // Set specified xbyak_intrin operand to only use fp_reg_vex
        if (vv->value_.isa<xbyak_intrin>()) {
            auto intrin = vv->value_.static_as<xbyak_intrin>();
            switch (static_cast<xbyak_intrin_type>(intrin->type_)) {
                // vperm2f128 is an avx only intrinsic
                case xbyak_intrin_type::permute: {
                    force_fp_reg_vex({vv->var_});
                    force_fp_reg_vex(intrin->args_);
                } break;
                default: break;
            }
        }
        return vv;
    }

    stmt_c visit(for_loop_c v) override {
        initialize_expr_data(v->var_, current_scope());
        return xbyak_visitor_t::visit(std::move(v));
    }

    expr_c visit(xbyak_intrin_c v) override {
        auto &cond_mask = v->modifier_.cond_mask_;
        if (cond_mask.defined()) { initialize_expr_data(cond_mask, nullptr); }
        return xbyak_visitor_t::visit(std::move(v));
    }

    expr_c visit(tensorptr_c v) override {
        initialize_expr_data(v->base_, nullptr);
        return xbyak_visitor_t::visit(std::move(v));
    }

    expr_c visit(indexing_c v) override {
        auto ret = xbyak_visitor_t::visit(std::move(v));
        assert(ret.isa<indexing>());
        // Add more spill weight to ptr and index
        auto vv = ret.static_as<indexing_c>();
        assert(!(vv->idx_.empty()) && vv->idx_.back().defined()
                && vv->ptr_.defined());
        update_spill_weight(vv->ptr_, 64 * loop_depth());
        update_spill_weight(vv->idx_.back(), 64 * loop_depth());
        if (vv->mask_.defined()) {
            update_spill_weight(vv->mask_, 64 * loop_depth());
        }
        return vv;
    }

private:
    void initialize_expr_data(const expr_c &v, const stmt_base_t *def_scope) {
        if (!v->temp_data().isa<xbyak_expr_data_t>()) {
            v->temp_data() = xbyak_expr_data_t();
        }
        if (def_scope) { GET_EXPR_DATA(v).def_scope_ = def_scope; }
    }

    void initialize_stmt_data(const stmt_c &v) {
        if (!v->temp_data().isa<xbyak_stmt_data_t>()) {
            v->temp_data() = xbyak_stmt_data_t(loop_depth());
        }
    }

    void force_fp_reg_vex(array_ref<expr> ref) {
        for (auto &v : ref) {
            GET_VIRTUAL_REG(v).set_force_fp_vex();
        }
    }

    void update_spill_weight(const expr_c &v, const spill_weight_t weight) {
        GET_VIRTUAL_REG(v).add_weight(weight + 1);
    }
};

func_c ir_data_initializer_t::operator()(func_c v) {
    ir_data_initializer_impl_t ir_data_initializer;

    return ir_data_initializer.dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
