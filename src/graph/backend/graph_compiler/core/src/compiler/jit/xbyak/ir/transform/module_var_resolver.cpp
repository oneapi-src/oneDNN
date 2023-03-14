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
#include <unordered_map>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/visitor.hpp>
#include <util/any_map.hpp>

#include "module_var_resolver.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class module_var_resolver_t_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    std::unordered_map<expr_c, expr> module_vars_;
    expr_c module_data_;

    func_c dispatch(func_c v) override {
        if (v->params_.size() > 1 && v->params_[1].isa<tensor>()) {
            auto module_tensor = v->params_[1].static_as<tensor>();
            if (module_tensor->name_ == "__module_data") {
                module_data_ = module_tensor;
            }
        }
        return ir_visitor_t::dispatch(std::move(v));
    }

    expr_c dispatch(expr_c v) override {
        if (module_vars_.find(v) != module_vars_.end()) {
            auto new_expr = module_vars_[v];
            return copy_attr(*v, std::move(new_expr));
        } else {
            return ir_visitor_t::dispatch(std::move(v));
        }
    }

    stmt_c visit(define_c v) override {
        if (v->var_.isa<var>()) {
            auto thevar = v->var_.static_as<var>();
            if (module_data_.defined() && thevar->attr_
                    && thevar->attr_->has_key(
                            attr_keys::module_global_offset)) {
                // TODO(XXX): module_var to indexing, maybe define just before
                // use?
                auto &offset = thevar->attr_->get_any(
                        attr_keys::module_global_offset);
                expr module_idx_ptr;
                if (auto absptr = offset.get_or_null<void *>()) {
                    module_idx_ptr = make_expr<constant_node>(
                            reinterpret_cast<uint64_t>(*absptr),
                            datatypes::s8.get_pointerof());
                } else {
                    // Get var_ptr = &module_data[offset]
                    auto index = builder::make_constant(offset.get<size_t>());
                    module_idx_ptr = builder::tensor_ptr(module_data_, {index});
                }
                // var_ptr->elem_dtype_ is not module_data_->elem_dtype_
                // var_ptr->elem_dtype_ should be var->dtype_
                auto var_ptr = builder::make_tensor(
                        thevar->name_ + "_ptr", {1}, thevar->dtype_);
                // Replace future module_var use with var_ptr[0]
                auto module_var = builder::make_indexing(
                        var_ptr, builder::make_constant(UINT64_C(0)));
                module_vars_[thevar] = module_var;
                // Will replace define with module_var's tensor define
                auto var_ptr_define = make_stmt<define_node_t>(
                        var_ptr, linkage::local, module_idx_ptr);
                return std::move(var_ptr_define);
            }
        }
        return ir_visitor_t::visit(std::move(v));
    }
};

func_c module_var_resolver_t::operator()(func_c v) {
    module_var_resolver_t_impl_t module_var_resolver;

    return module_var_resolver.dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
