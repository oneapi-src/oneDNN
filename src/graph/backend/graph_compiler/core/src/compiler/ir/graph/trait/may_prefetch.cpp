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

#include <string>
#include <utility>
#include <vector>
#include "may_prefetch.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/visitor.hpp>
#include <unordered_map>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace op_traits {

// replace all var/tensors in the tensor with remade ones, because we are
// creating a new function
class remake_args_visitor_t : public ir_visitor_t {
public:
    using ir_visitor_t::visit;
    std::vector<expr> args_;
    std::vector<expr> old_args_;
    std::unordered_map<expr_c, expr> mapping_;

    expr get_mapped(const expr_c &v) {
        auto itr = mapping_.find(v);
        if (itr != mapping_.end()) { return itr->second; }
        auto ret = v->remake();
        mapping_[v] = ret;
        args_.emplace_back(ret);
        old_args_.emplace_back(v.remove_const());
        return ret;
    }

    expr_c visit(var_c v) override { return get_mapped(v); }
    expr_c visit(tensor_c v) override { return get_mapped(v); }
};

void may_prefetch_t::generate_prefetcher_body_for_slice(const context_ptr &ctx,
        const std::vector<expr> &func_args,
        const std::vector<tensor_slice> &ins, const std::vector<int> &indices) {
    COMPILE_ASSERT(false, "not yet implemented");
}

func_t may_prefetch_t::generate_prefetcher_and_set_idle(const context_ptr &ctx,
        bool is_global, const std::vector<tensor_slice> &ins,
        const std::vector<int> &indices, std::vector<stmt> &out_set_idle_code) {
    std::vector<tensor_slice> new_ins;
    new_ins.reserve(ins.size());
    // remake the vars and tensors
    remake_args_visitor_t vis;
    sc_data_type_t type_trigger = datatypes::s32;
    vis.args_.emplace_back(builder::make_tensor("state", {1}, type_trigger));
    vis.args_.emplace_back(builder::make_var(type_trigger, "expected"));
    // the tid in func arg. It may not be valid (-1), we need to check it and
    // reassign the real tid if necessary
    auto tid_in_arg = builder::make_var(datatypes::s32, "tid");
    vis.args_.emplace_back(tid_in_arg);
    size_t tid_in_arg_idx = vis.args_.size() - 1;

    size_t num_standard_args = vis.args_.size();
    vis.old_args_.emplace_back();

    std::vector<expr> generator_args = vis.args_;

    vis.args_.front()->attr()["volatile"] = true;
    // remake the tensor slice
    for (auto &idx : indices) {
        auto &v = ins.at(idx);
        auto base = vis.visit(v.tptr_).checked_as<tensorptr>();
        std::vector<expr> shapes;
        for (auto &s : v.shape_) {
            shapes.emplace_back(vis.dispatch(s).remove_const());
        }
        new_ins.emplace_back();
        new_ins.back().tptr_ = std::move(base);
        new_ins.back().shape_ = std::move(shapes);
    }

    auto func_body = make_stmt<stmts_node_t>(std::vector<stmt>());
    std::vector<expr> func_args;
    for (size_t i = 0; i < num_standard_args; i++) {
        func_args.emplace_back(vis.args_[i]);
    }

    expr general_args;
    if (num_standard_args + 1 == vis.args_.size()) {
        general_args = vis.args_[num_standard_args];
    } else {
        general_args = builder::make_tensor(
                "__args", {vis.args_.size() - 1}, datatypes::generic);
        // extract the arguments in new prefetch function from general arg array
        for (size_t i = num_standard_args; i < vis.args_.size(); i++) {
            auto &arg = vis.args_[i];
            func_body->seq_.emplace_back(
                    builder::make_var_tensor_def_unattached(arg, linkage::local,
                            builder::make_cast(arg->dtype_,
                                    general_args[i - num_standard_args])));
        }
    }

    func_args.emplace_back(general_args);

    // call the user func to generate the func body
    builder::ir_builder_t builder;
    builder.push_scope();

    // the real tid
    auto realtid = builder::make_var(datatypes::s32, "realtid");
    builder.push_var_tensor_def(realtid, linkage::local,
            builder::make_select(tid_in_arg < 0,
                    builtin::get_thread_id_func()(), tid_in_arg));
    generator_args[tid_in_arg_idx] = realtid;

    std::vector<expr> args_ins;
    for (auto &t : new_ins) {
        COMPILE_ASSERT(t.is_full(),
                "Cannot generate prefetcher for tensor slice yet.");
        args_ins.emplace_back(t.get_real_tensor());
    }
    generate_prefetcher_body_for_tensor(ctx, generator_args, args_ins, indices);

    auto body = builder.pop_scope();
    for (auto &s : body.checked_as<stmts>()->seq_) {
        func_body->seq_.emplace_back(std::move(s));
    }

    auto op = dynamic_cast<sc_op *>(this);
    std::string func_name = op->op_name_;
    func_name += "_";
    func_name += std::to_string(op->logical_op_id_);
    func_name += "_prefetch";
    auto retfunc = builder::make_func(
            func_name, func_args, func_body, datatypes::index);
    retfunc->attr()[function_attrs::low_level] = true;
    retfunc->attr()[function_attrs::private_] = true;
    retfunc->decl_->attr()[function_attrs::private_] = true;

    vis.old_args_[0] = builder::make_func_addr(retfunc);
    out_set_idle_code.emplace_back(builder::make_evaluate_unattached(
            make_expr<intrin_call_node>(intrin_type::set_thread_idle_func,
                    vis.old_args_, any_map_t {})));
    return retfunc;
}
} // namespace op_traits
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
