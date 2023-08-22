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
#include <vector>
#include <unordered_map>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/passlet/use_count_analysis.hpp>
#include <compiler/ir/ssa_data.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <compiler/ir/visitor.hpp>
#include <util/any_map.hpp>
#include <util/array_ref.hpp>

#include "intrinsics_combine.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

using namespace passlet;

struct use_count_t {
    size_t val_ = 0;
};

class use_count_analysis_viewer_t : public ssa_viewer_t {
public:
    using ssa_viewer_t::dispatch;
    using ssa_viewer_t::view;

    use_count_analysis_t uc_ana_;

    use_count_analysis_viewer_t()
        : uc_ana_ {temp_data_addresser<use_count_t, size_t,
                &use_count_t::val_>()} {}

    func_c dispatch(func_c f) override {
        uc_ana_.view(f, pass_phase::PRE_VISIT);
        ssa_viewer_t::dispatch(f->body_);
        uc_ana_.view(f, pass_phase::POST_VISIT);
        return f;
    }

    expr_c dispatch(expr_c v) override {
        uc_ana_.view(v, pass_phase::PRE_VISIT);
        uc_ana_.view(v, pass_phase::POST_VISIT);
        return v;
    }

    void view(define_c v) override {
        uc_ana_.view(v, pass_phase::PRE_VISIT);
        if (v->init_.defined()) { dispatch(v->init_); }
        uc_ana_.view(v, pass_phase::POST_VISIT);
    }

    void view(for_loop_c v) override {
        uc_ana_.view(v, pass_phase::PRE_VISIT);
        dispatch(v->iter_begin_);
        dispatch(v->iter_end_);
        dispatch(v->step_);
        dispatch(v->body_);
        uc_ana_.view(v, pass_phase::POST_VISIT);
    }
};

class intrinsics_combine_impl_t : public ssa_visitor_t {
public:
    using ssa_visitor_t::dispatch;
    using ssa_visitor_t::visit;

    std::unordered_map<stmt_c, std::vector<stmt_c> *> define_scope_map_;

    stmt_c visit(define_c v) override {
        define_scope_map_[v] = get_current_scope();
        return ssa_visitor_t::visit(std::move(v));
    }

    expr_c visit(add_c v) override {
        // combine (a * b) + c to fmadd(a, b, c)
        // if (a * b) only used in fmadd and defined in same scope
        auto combine_to_fmadd = [this](const expr &l, const expr &r) {
            stmt owner {l->ssa_data_->owner_.lock()};
            return owner.cast<define>()
                    .filter([this, l](const define &v) {
                        const auto iter = define_scope_map_.find(v);
                        return l->get_temp_data().get<use_count_t>().val_ == 1
                                && iter != define_scope_map_.end()
                                && iter->second == get_current_scope();
                    })
                    .map([](const define &v) { return v->init_; })
                    .map([](const expr &v) { return v.as<mul>(); })
                    .map([r](const mul &v) {
                        return builder::make_fmadd(v->l_, v->r_, r);
                    })
                    .get_or_else(expr());
        };
        if (v->dtype_.is_etype(sc_data_etype::F32)
                || v->dtype_.is_etype(sc_data_etype::F16)) {
            expr node;
            node = combine_to_fmadd(v->l_, v->r_);
            if (node.defined()) { return node; }
            node = combine_to_fmadd(v->r_, v->l_);
            if (node.defined()) { return node; }
        }
        return v;
    }

    expr_c visit(intrin_call_c v) override {
        // combine broadcast(A[i]) to broadcast_idx(A, i)
        // if A[i] only used in broadcast
        auto combine_to_broadcast = [](const expr &e, int dst_lanes) {
            stmt owner {e->ssa_data_->owner_.lock()};
            return owner.cast<define>()
                    .filter([e](const define &v) {
                        return e->get_temp_data().get<use_count_t>().val_ == 1;
                    })
                    .map([](const define &v) { return v->init_; })
                    .map([](const expr &v) { return v.as<indexing>(); })
                    .map([dst_lanes](const indexing &idxn) -> expr_c {
                        if (idxn->mask_.defined()) { return expr(); }
                        auto src_lanes = builder::make_constant(
                                (int)idxn->dtype_.lanes_);
                        src_lanes->ssa_data_ = utils::make_unique<ssa_data_t>();
                        assert(idxn->idx_.size() == 1);
                        return builder::make_x86_intrin(
                                x86_intrin_type::avx_broadcast_idx,
                                {idxn->ptr_, idxn->idx_.back(), src_lanes},
                                {{"lanes", dst_lanes}});
                    });
        };
        switch (v->type_) {
            case intrin_type::broadcast: {
                assert(v->args_.size() == 1);
                return combine_to_broadcast(v->args_[0], v->dtype_.lanes_)
                        .get_or_else(v);
            } break;
            default: break;
        }
        return v;
    }
};

func_c intrinsics_combine_t::operator()(func_c v) {
    use_count_analysis_viewer_t use_count_analysis;
    use_count_analysis.dispatch(v);
    intrinsics_combine_impl_t intrinsics_combine;
    return intrinsics_combine.top_level_dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
