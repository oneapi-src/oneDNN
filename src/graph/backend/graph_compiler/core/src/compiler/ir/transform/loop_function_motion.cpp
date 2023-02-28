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

#include <utility>
#include <vector>
#include "../builder.hpp"
#include "../util_module_passes.hpp"
#include "../viewer.hpp"
#include "../visitor.hpp"
#include "loop_function_motion.hpp"
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(simple_loop_function_motion,
        SC_PASS_DEPENDS_ON(loop_merger, parallel_workload_dispatcher,
                ir_simplifier, closurizer_cpu),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE(IR_SIMPLIFIED));

class pure_func_hoister_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    int index_ = 0;
    // Currently all pure funcs will be promoted to outermost loop
    int loop_depth_ = 0;
    // Each pure func ret value will assign to a var
    std::vector<stmt_c> call_var_defs_;
    expr_c visit(call_c v) override {
        // Currently only promote pure funcs with no arg
        if (is_pure_func_call(v) && v->args_.empty() && loop_depth_ > 0) {
            assert(v->dtype_ != datatypes::void_t);
            func_t callee = v->get_prototype();
            auto var_name = "call_var_" + std::to_string(index_++) + "_"
                    + callee->name_;
            auto call_var = builder::make_var(v->dtype_, var_name);
            call_var_defs_.emplace_back(builder::make_var_tensor_def_unattached(
                    call_var, linkage::local, v));
            return call_var;
        }
        return v;
    }
    stmt_c visit(for_loop_c v) override {
        loop_depth_++;
        auto vv = ir_visitor_t::visit(std::move(v));
        loop_depth_--;
        if (loop_depth_ == 0 && !call_var_defs_.empty()) {
            call_var_defs_.emplace_back(vv);
            auto ret = builder::make_stmts_unattached(call_var_defs_);
            call_var_defs_.clear();
            return ret;
        }
        return vv;
    }
};

func_c simple_loop_function_motion_t::operator()(func_c v) {
    pure_func_hoister_t pure_func_hoister;

    return pure_func_hoister.dispatch(std::move(v));
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
