/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#include "loop_unroll.hpp"
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/visitor.hpp>
#include <util/any_map.hpp>

namespace sc {

SC_DECL_PASS_INFO(loop_unroller, SC_PASS_DEPENDS_ON(tensor_init),
        SC_PASS_REQUIRE_STATE(CONST_FOLDED), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE(IR_SIMPLIFIED));

class loop_unroller_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    expr_c dispatch(expr_c v) override { return v; }
    stmt_c visit(stmts_c v) override {
        auto ret = ir_visitor_t::visit(v);
        std::vector<for_loop> to_unroll;
        for (auto &s : ret.static_as<stmts>()->seq_) {
            if (s.isa<for_loop>()) {
                auto &attr = s.static_as<for_loop>()->attr_;
                if (attr && attr->has_key(stmt_attr_key::unroll_loop)) {
                    to_unroll.emplace_back(s.static_as<for_loop>());
                }
            }
        }
        if (to_unroll.empty()) { return ret; }
        stmts writable
                = (ret.ptr_same(v) ? ret->remake() : ret).static_as<stmts>();
        for (auto &f : to_unroll) {
            f->unroll(f->attr_->get<int>(stmt_attr_key::unroll_loop), writable);
        }
        return constant_folder_t()(writable);
    }
};

func_c loop_unroller_t::operator()(func_c f) {
    loop_unroller_impl_t impl;
    return impl.dispatch(f);
};

stmt_c loop_unroller_t::operator()(stmt_c f) {
    loop_unroller_impl_t impl;
    return impl.dispatch(std::move(f));
};

} // namespace sc
