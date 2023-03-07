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
#include <util/any_map.hpp>

#include "ir_indexer.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class ir_indexer_impl_t : public xbyak_visitor_t {
public:
    using xbyak_visitor_t::dispatch;

    stmt_index_t ir_index_;

    func_c dispatch(func_c f) override {
        ir_index_ = 0;
        return xbyak_visitor_t::dispatch(std::move(f));
    }

    stmt_c dispatch(stmt_c s) override {
        stmt_c ret;

        if (s->node_type_ == sc_stmt_type::for_loop
                || s->node_type_ == sc_stmt_type::if_else
                || s->node_type_ == sc_stmt_type::stmts) {
            ir_index_ += stmt_index_const::increment;
            auto &stmt_data = GET_STMT_DATA(s);
            stmt_data.init_index_ = ir_index_;
        }

        ret = xbyak_visitor_t::dispatch(std::move(s));

        ir_index_ += stmt_index_const::increment;
        auto &stmt_data = GET_STMT_DATA(ret);
        stmt_data.set_index(ir_index_);

        return ret;
    }

    expr_c dispatch(expr_c v) override { return v; }
};

func_c ir_indexer_t::operator()(func_c v) {
    ir_indexer_impl_t ir_indexer;

    return ir_indexer.dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
