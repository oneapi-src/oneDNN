/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include <unordered_map>
#include <unordered_set>

#include <utility>
#include <vector>
#include "../builder.hpp"
#include "../util_module_passes.hpp"
#include "../viewer.hpp"
#include "../visitor.hpp"
#include "simple_licm.hpp"
#include <compiler/ir/pass_dep_util.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(simple_loop_invariant_code_motion,
        SC_PASS_DEPENDS_ON(loop_merger, parallel_workload_dispatcher),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE(IR_SIMPLIFIED));

class tensor_def_viewer_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    bool volatile_ = false;
    void view(var_c v) override {
        if (!v->attr_ || !v->attr_->get_or_else(attr_key::const_attr, false)) {
            volatile_ = true;
        }
    }
    void view(indexing_c v) override { volatile_ = true; }
};

class tensor_def_hoister_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    tensor_def_viewer_t viewer_;
    // depth of non parallel loop in current scope.
    int non_parallel_loop_depth_ = 0;
    // define tensor with no related loop vars.
    std::vector<stmt_c> no_depend_defs_;
    bool can_not_parallel_ = false;
    // don't care about expr.
    expr_c dispatch(expr_c v) override { return v; }
    stmt_c visit(define_c v) override {
        if (non_parallel_loop_depth_ != 0 && !v->init_.defined()
                && (!v->var_.isa<tensor>()
                        || v->var_.checked_as<tensor>()->init_value_
                                == nullptr)) {
            bool do_hoist = true;
            if (v->var_.isa<tensor>()) {
                viewer_.dispatch(v->var_);
                if (viewer_.volatile_) { do_hoist = false; }
                viewer_.volatile_ = false;
            }
            if (do_hoist) {
                no_depend_defs_.emplace_back(v);
                return builder::make_stmts_unattached({});
            }
        }
        return v;
    }

    stmt_c visit(for_loop_c v) override {
        bool old_can_not_parallel = can_not_parallel_;
        if (v->kind_ != for_type::PARALLEL) {
            can_not_parallel_ = true;
            non_parallel_loop_depth_++;
        } else {
            // if parallel for locate at non parallel for ignore that, else
            // dispatch inside.
            if (can_not_parallel_) { return v; }
        }
        auto newv = ir_visitor_t::visit(v);
        if (v->kind_ != for_type::PARALLEL) {
            can_not_parallel_ = old_can_not_parallel;
            non_parallel_loop_depth_--;
            if (non_parallel_loop_depth_ == 0 && !no_depend_defs_.empty()) {
                no_depend_defs_.emplace_back(newv);
                auto ret = builder::make_stmts_unattached(no_depend_defs_);
                no_depend_defs_.clear();
                return ret;
            }
        }
        return newv;
    }
};

func_c simple_loop_invariant_code_motion_t::operator()(func_c f) {
    tensor_def_hoister_t pass;
    auto ret = pass.dispatch(std::move(f));
    return ret;
}

stmt_c simple_loop_invariant_code_motion_t::operator()(stmt_c f) {
    tensor_def_hoister_t pass;
    return pass.dispatch(std::move(f));
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
