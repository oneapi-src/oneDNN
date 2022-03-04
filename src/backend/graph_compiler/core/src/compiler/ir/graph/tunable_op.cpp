/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>

namespace sc {

tunable_op_t::tunable_op_t(const std::string &op_name,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : sc_op(op_name, ins, outs, attrs) {}

sc_op_ptr tunable_op_t::copy(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr) {
    auto ret = mgr.make(op_name_, ins, outs, attrs_);
    auto tune_ret = ret->stc_cast<tunable_op_t>();
    tune_ret->op_name_ = op_name_;
    tune_ret->config_data_ = config_data_;
    tune_ret->is_quantized_ = is_quantized_;
    tune_ret->need_compensation_ = need_compensation_;
    tune_ret->should_quantized_ = should_quantized_;
    return ret;
}

bool tunable_op_t::is_valid(const context_ptr &ctx) {
    if (!config_data_
            || !create_generator()->is_valid_config(ctx, config_data_.get())) {
        return false;
    }
    return true;
}

ir_module_ptr tunable_op_t::get_func(context_ptr ctx) {
    auto ret = std::make_shared<ir_module_t>(ctx);
    auto gen_ptr = create_generator();
    set_config_if_empty(ctx, gen_ptr.get());
    std::vector<expr> ins;
    std::vector<expr> outs;
    auto func = graph::create_func_decl_for_op(this, ins, outs);

    builder::ir_builder_t bld;
    bld.push_scope();
    std::vector<for_loop> loops;
    bool status = gen_ptr->generate(
            ctx, config_data_.get(), nullptr, ins, outs, loops);
    assert(status);
    bld.push_returns(true);
    auto body = bld.pop_scope();
    gen_ptr->schedule_loops(ctx, config_data_.get(), body, loops);
    auto args = outs;
    args.insert(args.end(), ins.begin(), ins.end());

    func->body_ = std::move(body);
    ret->add_func({func});
    ret->set_entry_func_idx(0);
    return ret;
}

void tunable_op_t::set_config(const std::shared_ptr<void> &config) {
    config_data_ = config;
}

void tunable_op_t::set_config_if_empty(
        context_ptr ctx, body_generator_base_t *p) {
    if (!config_data_) { set_config(p->get_default_config(std::move(ctx))); }
}

} // namespace sc
