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

#include "lowering.hpp"
#include <algorithm>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include "fusible_op.hpp"
#include "graph.hpp"
#include "pass/pass.hpp"
#include "visitor.hpp"
#include <compiler/ir/builder.hpp>
#include <microkernel/builtin.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <unordered_map>
#include <util/scoped_timer.hpp>

namespace sc {

SC_MODULE(graph.lowering)

struct result_dump_config_t {
    bool enabled_ = false;
    std::vector<std::string> filter_;
    std::string path_ = "./dump";
    bool binary_format_ = false;
    size_t bytes_per_dump_ = 0;

    bool should_function_dump(const std::string &name) {
        if (filter_.empty()) { return true; }
        for (auto &f : filter_) {
            if (utils::string_startswith(name, f)) { return true; }
        }
        return false;
    }

    result_dump_config_t(const std::string &cfg) {
        if (cfg.empty()) { return; }
        auto configs = utils::string_split(cfg, ",");
        for (auto &c : configs) {
            auto kv = utils::string_split(c, "=");
            if (kv.size() != 2) {
                SC_MODULE_WARN << "Bad graph result dump config: " << c;
                continue;
            }
            if (kv[0] == "filter") {
                enabled_ = true;
                filter_ = utils::string_split(kv[1], ":");
            } else if (kv[0] == "path") {
                enabled_ = true;
                path_ = kv[1];
            } else if (kv[0] == "format") {
                enabled_ = true;
                binary_format_ = std::stoi(kv[1]);
            } else if (kv[0] == "bytes") {
                enabled_ = true;
                bytes_per_dump_ = std::stoull(kv[1]);
            } else {
                SC_MODULE_WARN << "Bad dump config key name " << kv[0];
                continue;
            }
        }
        if (enabled_) {
            SC_MODULE_WARN << "The generated code will dump tensor results to "
                           << path_
                           << ", filter=" << utils::print_vector(filter_)
                           << ", binaryformat=" << binary_format_
                           << ", byteslimit=" << bytes_per_dump_;
        }
    }
};

static expr make_global_string(
        const ir_module_ptr &mod, const std::string &v, int &counter) {
    std::string name = "__gstring";
    name += std::to_string(counter++);
    auto contents = std::make_shared<static_data_t>(v.c_str(), v.size() + 1);
    auto ret = builder::make_tensor(name, {v.size() + 1}, datatypes::s8,
            address_space::automatic, contents);
    auto def = builder::make_var_tensor_def_unattached(
            ret, linkage::private_global);
    mod->add_global_var(def.checked_as<define>());
    return ret;
}

static void make_dump_tensor_call(const std::vector<expr> &outs,
        const sc_op_ptr &node, const ir_module_ptr &ret_mod,
        const func_t &callee, int &global_str_counter,
        result_dump_config_t &dump_config, const expr &dump_out_path,
        stmts_node_t *target_body) {
    for (size_t i = 0; i < outs.size(); i++) {
        auto &out = outs[i];
        auto &graph_tsr = node->get_outputs()[i];
        if (!out.isa<tensor>()) continue;
        auto tsr = out.checked_as<tensor>();
        std::stringstream tensor_name;
        tensor_name << callee->name_ << '.' << tsr->name_ << '.'
                    << graph_tsr->details_.get_format();
        auto namestr = make_global_string(
                ret_mod, tensor_name.str(), global_str_counter);
        std::stringstream shape_name;
        size_t total_shape1 = utils::get_sizeof_type(tsr->elem_dtype_);
        for (auto &dimv : tsr->dims_) {
            auto dim = get_const_as_int(dimv.checked_as<constant_c>());
            total_shape1 *= dim;
            shape_name << dim << ',';
        }
        auto shapestr = make_global_string(
                ret_mod, shape_name.str(), global_str_counter);
        auto the_call = builtin::call_dump_tensor(out, namestr, shapestr,
                total_shape1, dump_config.bytes_per_dump_, dump_out_path,
                dump_config.binary_format_,
                static_cast<uint64_t>(tsr->elem_dtype_));
        target_body->seq_.emplace_back(
                builder::make_evaluate_unattached(the_call));
    }
}

static void make_value_check_call(const std::vector<expr> &outs,
        const ir_module_ptr &ret_mod, const func_t &callee,
        int &global_str_counter, stmts_node_t *target_body) {
    for (auto &out : outs) {
        auto tsr = out.checked_as<tensor>();
        if (tsr->elem_dtype_.type_code_ != sc_data_etype::F32) { continue; }
        auto namestr = make_global_string(
                ret_mod, callee->name_ + "." + tsr->name_, global_str_counter);
        size_t total_shape1 = utils::get_sizeof_type(tsr->elem_dtype_);
        for (auto &dimv : tsr->dims_) {
            total_shape1 *= get_const_as_int(dimv.checked_as<constant_c>());
        }
        auto the_call = builtin::call_value_check(out, namestr, total_shape1);
        target_body->seq_.emplace_back(
                builder::make_evaluate_unattached(the_call));
    }
}

static graph_tensor_ptr get_linked_output_tsr(const graph_tensor_ptr &ltensor) {
    if (!ltensor->uses_.empty()) {
        for (size_t i = 0; i < ltensor->uses_.size(); i++) {
            if (ltensor->uses_[i].second->isa<tensor_view_op_t>()) {
                auto reshape = ltensor->uses_[i].second;
                for (auto &cld : reshape->get_outputs()[0]->uses_) {
                    if (cld.second->isa<output_op>()) {
                        return cld.second->get_inputs()[cld.first];
                    }
                }
            }
        }
    }
    return nullptr;
}

struct lowering_visitor_state_t {
    std::unordered_map<graph_tensor_ptr, size_t> tensor_pending_refcount_;
    op_visitor_t::updater_func topo_sorter_;
    std::vector<size_t> op_exec_tick_;
    std::vector<bool> op_visited_;
    //  need to visit the input outs in reversed order to align to old lowering
    //  input argument order (like pop_back_selector). Our visitor must visit
    //  the input ops first
    std::list<sc_op_ptr>::iterator input_op_itr;
    size_t cur_tick_ = 0;
    size_t max_tensor_size_;

    lowering_visitor_state_t(sc_graph_t &g)
        : topo_sorter_ {op_visitor_t::create_DAG_updater(g.ops_.size())}
        , op_exec_tick_(g.ops_.size())
        , op_visited_(g.ops_.size()) {
        max_tensor_size_ = 0;
        for (auto &op : g.ops_) {
            for (auto &tsr : op->get_outputs()) {
                max_tensor_size_
                        = std::max(max_tensor_size_, tsr->details_.size());
            }
        }
    }

    size_t &get_tensor_pending_refcount(const graph_tensor_ptr &p) {
        auto itr = tensor_pending_refcount_.find(p);
        if (itr == tensor_pending_refcount_.end()) {
            auto ret = tensor_pending_refcount_.insert(
                    std::make_pair(p, p->uses_.size()));
            return ret.first->second;
        }
        return itr->second;
    }

    op_visitor_t::updater_func get_updater() {
        auto ths = this;
        return [ths](op_visitor_t *vis, const sc_op_ptr &op) {
            for (auto &in : op->get_inputs()) {
                ths->get_tensor_pending_refcount(in)--;
            }
            auto tick = ths->cur_tick_++;
            if (op->isa<output_op>() || op->isa<constant_op_t>()) {
                ths->op_exec_tick_[op->logical_op_id_] = 0;
            } else {
                ths->op_exec_tick_[op->logical_op_id_] = tick;
            }
            ths->op_visited_[op->logical_op_id_] = true;
            ths->topo_sorter_(vis, op);
        };
    }

    // find the distance of an op to the visited ops
    int get_op_distance_to_visited_set(sc_op *op, std::vector<int> &d) {
        auto id = op->logical_op_id_;
        if (op_visited_[id]) { return 0; }
        if (d[id] != 0) { return d[id]; }
        if (op->isa<output_op>()) {
            d[id] = 0;
            return 0;
        }
        int ret = -1;
        for (auto &v : op->get_inputs()) {
            int cur_d
                    = get_op_distance_to_visited_set(v->producer_owner_, d) + 1;
            ret = std::max(ret, cur_d);
        }
        d[id] = ret;
        return ret;
    }

    static constexpr float distance_factor = 2.0f;
    // for each input tensor, check if the refcount=1. If so, it means that
    // after the Op is visited, the input tensor is no longer needed compute the
    // score of each visitable candidate op. the score is "SUM_{each input
    // tensor}(normalized_sizeof(tensor)/ref_count_modifier*heat_modifier) -
    // SUM_{each output tensor}(normalized_sizeof(tensor)+ distance_modifier)"
    float evaluate_op_score(sc_op *op, std::vector<int> &distance_to_visited) {
        float cur_score = 0;

        for (auto &in : op->get_inputs()) {
            // if the input tensor is input_op, there is no temp buffer to be
            // free'd
            if (!in->producer_owner_->isa<input_op>()) {
                // compute the heat modifier of the tensor. The hotter
                // the tensor is (computed lately), the larger the
                // modifier.
                auto owner = in->producer_owner_;
                auto tick_diff
                        = cur_tick_ - op_exec_tick_[owner->logical_op_id_];
                assert(cur_tick_ > op_exec_tick_[owner->logical_op_id_]);
                float heat_modifier;
                switch (tick_diff) {
                    case 0:
                    case 1: heat_modifier = 2.5f; break;
                    case 2: heat_modifier = 1.5f; break;
                    default: heat_modifier = 1.0f;
                }
                // if it is last use, ref_count_modifier=1. If not,
                // ref_count_modifier=number of uses
                size_t ref_count_modifier;
                if (this->get_tensor_pending_refcount(in) == 1) {
                    ref_count_modifier = 1;
                } else {
                    ref_count_modifier = in->uses_.size();
                }
                float cur_tsr = float(in->details_.size()) / ref_count_modifier
                        / max_tensor_size_ * heat_modifier;
                cur_score += cur_tsr;
            }
        }
        for (auto &out : op->get_outputs()) {
            // if this output is connected to output op, it is not a temp
            // buffer, and we don't need to count its size
            if (out->uses_.size() == 1UL
                    && out->uses_[0].second->isa<output_op>()) {
                continue;
            }
            int distance = 1;
            for (auto &use : out->uses_) {
                distance = std::max(distance,
                        get_op_distance_to_visited_set(
                                use.second.get(), distance_to_visited));
            }
            float cur_tsr = (distance - 1) * distance_factor
                    + float(out->details_.size()) / max_tensor_size_;
            cur_score -= cur_tsr;
        }
        return cur_score;
    }

    using queue_iterator_t = std::list<sc_op_ptr>::iterator;
    op_visitor_t::selector_func get_selector() {
        auto ths = this;
        return [ths](op_visitor_t *vis) -> sc_op_ptr {
            if (ths->cur_tick_ == 0) {
                ths->input_op_itr = vis->to_visit_.end();
                --ths->input_op_itr;
            }
            if (ths->input_op_itr != vis->to_visit_.end()) {
                // if there is input ops, return and advance the input_op_itr
                auto ret = *ths->input_op_itr;
                auto to_remove = ths->input_op_itr;
                if (ths->input_op_itr == vis->to_visit_.begin()) {
                    ths->input_op_itr = vis->to_visit_.end();
                } else {
                    --ths->input_op_itr;
                }
                vis->to_visit_.erase(to_remove);

                SC_MODULE_INFO << "Scheduling const/input: iter "
                               << ths->cur_tick_ << ", Op " << ret->op_name_
                               << "_" << ret->logical_op_id_;
                return ret;
            }
            // fast path: if there is only one op, just pop it
            if (vis->to_visit_.size() == 1) {
                auto ret = vis->to_visit_.back();
                vis->to_visit_.pop_back();
                return ret;
            }
            float best_score = std::numeric_limits<float>::lowest();
            std::list<sc_op_ptr>::reverse_iterator to_remove;

            std::vector<int> distance(ths->op_visited_.size());
            // visit the queue in reversed order to align to old lowering input
            // argument order (like pop_back_selector)
            for (auto itr = vis->to_visit_.rbegin();
                    itr != vis->to_visit_.rend(); ++itr) {
                auto &op = *itr;
                assert(!op->isa<input_op>() && !op->isa<constant_op_t>());
                float cur_score = ths->evaluate_op_score(op.get(), distance);
                SC_MODULE_INFO << "Scheduling score: iter " << ths->cur_tick_
                               << ", Op " << op->op_name_ << "_"
                               << op->logical_op_id_ << " = " << cur_score;
                if (cur_score > best_score) {
                    best_score = cur_score;
                    to_remove = itr;
                }
            }
            auto ret = *to_remove;
            SC_MODULE_INFO << "Scheduling selects: iter " << ths->cur_tick_
                           << ", Op " << ret->op_name_ << "_"
                           << ret->logical_op_id_;
            vis->to_visit_.erase(std::next(to_remove).base());
            return ret;
        };
    }
};

namespace graph {
std::string get_tensor_name(graph_tensor *t, sc_op *linked_output) {
    std::string tensor_name;
    if (t->producer_owner_->get_outputs().size() == 1UL) {
        tensor_name = t->producer_owner_->attrs_.get_or_else(
                "temp.name", tensor_name);
    }
    if (tensor_name.empty() && linked_output
            && linked_output->get_inputs().size() == 1UL) {
        tensor_name
                = linked_output->attrs_.get_or_else("temp.name", tensor_name);
    }
    return tensor_name;
}
} // namespace graph

ir_module_ptr lower_graph(context_ptr ctx, sc_graph_t &graph,
        const std::vector<sc_op_ptr> &args) {
    auto timer = SC_SCOPED_TIMER_INFO("graph.driver.time.lowering", "");
    if (!ctx->flags_.dump_graph_.empty()) {
        SC_INFO << "visualize graph to a dot file and a json file";
        visualize(ctx->flags_.dump_graph_, graph);
    }
    result_dump_config_t dump_config {ctx->flags_.graph_dump_results_};
    lowering_visitor_state_t visiter_state(graph);
    op_visitor_t vis {
            visiter_state.get_selector(), visiter_state.get_updater()};
    visiter_state.input_op_itr = vis.to_visit_.end();
    std::vector<expr> params;
    stmts func_body = make_stmt<stmts_node_t>(std::vector<stmt>());
    stmts init_body = make_stmt<stmts_node_t>(std::vector<stmt>());
    // todo: use graph-id to generate name
    auto func = builder::make_func(
            graph.attrs_.get_or_else<std::string>("temp.name", "main_entry"),
            params, func_body, datatypes::void_t);
    // todo: logical tensor should also have an unique id
    std::unordered_map<graph_tensor_ptr, expr> ltsr_rtsr;
    std::unordered_map<expr, expr> ltsr_gtsr;
    int tensor_counter = 0;
    int global_tensor_counter = 0;
    auto ret_mod = ir_module_t::from_entry_func(ctx, func);

    expr dump_out_path;
    int global_str_counter = 0;
    if (dump_config.enabled_) {
        dump_out_path = make_global_string(
                ret_mod, dump_config.path_, global_str_counter);
    }

    if (graph.attrs_.get_or_else("folded_input", false)) {
        ret_mod->attr_.set("folded_input", true);
    }
    auto get_or_create_tensor = [&](const graph_tensor_ptr &t, bool is_arg,
                                        int const_type) -> expr {
        auto itr = ltsr_rtsr.find(t);
        if (itr != ltsr_rtsr.end()) { return itr->second; }
        sc_op *linked_output = nullptr;
        if (!is_arg) {
            for (auto &use : t->uses_) {
                // finds if any of the use of the tensor is marked output
                if (use.second->isa<output_op>()) {
                    is_arg = true;
                    linked_output = use.second.get();
                    break;
                }
            }
        }

        std::vector<expr> dims = dims_to_expr(t->details_.get_blocking_dims());
        std::vector<expr> strides = dims_to_expr(t->details_.get_strides());
        std::string tensor_name
                = graph::get_tensor_name(t.get(), linked_output);
        if (tensor_name.empty()) {
            tensor_name
                    = std::string("buffer_") + std::to_string(tensor_counter);
        }
        expr tsr = builder::make_stensor(
                tensor_name, dims, strides, t->details_.dtype_);
        tensor_counter++;
        ltsr_rtsr.insert(std::make_pair(t, tsr));

        if (!is_arg) {
            if (const_type != const_kind::not_const) {
                if (const_type == const_kind::global_const) {
                    tsr = ret_mod->make_global_stensor(
                            tsr.checked_as<tensor>()->elem_dtype_,
                            "folded_const_"
                                    + std::to_string(global_tensor_counter++),
                            tsr.checked_as<tensor>()->dims_,
                            tsr.checked_as<tensor>()->strides_);
                    if (auto const_node
                            = t->producer_owner_->dyn_cast<constant_op_t>()) {
                        auto const_value = const_node->get_constant_values();
                        tsr.checked_as<tensor>()->init_value_ = const_value;
                    }
                    ltsr_rtsr[t] = tsr;
                } else {
                    init_body->seq_.emplace_back(
                            builder::make_var_tensor_def_unattached(tsr));
                }
            } else {
                func_body->seq_.emplace_back(
                        builder::make_var_tensor_def_unattached(tsr));
            }
        }
        return tsr;
    };
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        std::vector<expr> ins;
        std::vector<expr> outs;
        // special kinds of Ops that we need to take care of
        enum op_kinds {
            other = 0,
            input,
            output,
            constant,
            reshape,
        } kind = other;
        if (node->isa<input_op>()) {
            kind = input;
        } else if (node->isa<output_op>()) {
            kind = output;
        } else if (node->isa<constant_op_t>()) {
            kind = constant;
            if (node->attrs_.get_or_else("constant", const_kind::not_const)
                    == const_kind::not_const) {
                node->attrs_.set("constant", const_kind::global_const);
            }
        } else if (node->isa<tensor_view_op_t>()) {
            kind = reshape;
        }
        auto get_reshape_tptr = [&](const graph_tensor_ptr &old_tsr,
                                        const graph_tensor_ptr &new_tsr,
                                        int const_type, op_kinds kind) {
            auto base_tsr
                    = get_or_create_tensor(old_tsr, kind == input, const_type);
            auto ndims = old_tsr->details_.get_blocking_dims().size();
            std::vector<expr_c> base_idx(ndims, expr(0));
            auto &new_int_shape = new_tsr->details_.get_blocking_dims();
            std::vector<expr_c> new_shape = dims_to_expr_c(new_int_shape);
            return builder::tensor_ptr(base_tsr, base_idx, new_shape);
        };
        int const_type
                = node->attrs_.get_or_else("constant", const_kind::not_const);
        for (auto &ltensor : node->get_inputs()) {
            ins.emplace_back(get_or_create_tensor(ltensor, false, const_type));
        }
        for (auto &ltensor : node->get_outputs()) {
            if (kind == constant) {
                get_or_create_tensor(ltensor, false, const_kind::global_const);
            } else if (kind == reshape) {
                COMPILE_ASSERT(node->get_inputs().size() == 1,
                        "Reshape should have 1 input");
                if (ltsr_rtsr.find(ltensor) != ltsr_rtsr.end()) { break; }
                ltsr_rtsr[ltensor] = get_reshape_tptr(
                        node->get_inputs()[0], ltensor, const_type, kind);
            } else {
                graph_tensor_ptr out_tsr;
                // for pattern like node->reshape->output
                if (auto out_tsr = get_linked_output_tsr(ltensor)) {
                    ltsr_rtsr[ltensor] = get_reshape_tptr(
                            out_tsr, ltensor, const_type, kind);
                    outs.emplace_back(ltsr_rtsr[ltensor]);
                } else {
                    outs.emplace_back(get_or_create_tensor(
                            ltensor, kind == input, const_type));
                }
            }
        }
        switch (kind) {
            case input: {
                for (auto &v : outs) {
                    params.emplace_back(v);
                }
                break;
            }
            case output: {
                for (auto &v : ins) {
                    params.emplace_back(v);
                }
                break;
            }
            case constant:
            case reshape: {
                break;
                // nothing to do.
            }
            default: {
                std::vector<expr> exprargs;
                exprargs.insert(exprargs.end(), outs.begin(), outs.end());
                exprargs.insert(exprargs.end(), ins.begin(), ins.end());
                auto mod = node->get_func(ctx);
                ret_mod->merge(*mod);
                auto callee = mod->get_entry_func();
                stmts_node_t *target_body
                        = (const_type != const_kind::not_const)
                        ? init_body.get()
                        : func_body.get();
                target_body->seq_.emplace_back(
                        builder::make_evaluate_unattached(
                                builder::make_call(callee, exprargs)));

                if (ctx->flags_.value_check_) {
                    make_value_check_call(outs, ret_mod, callee,
                            global_str_counter, target_body);
                }
                if (dump_config.enabled_
                        && dump_config.should_function_dump(callee->name_)) {
                    make_dump_tensor_call(outs, node, ret_mod, callee,
                            global_str_counter, dump_config, dump_out_path,
                            target_body);
                }
            }
        }
    });
    if (!args.empty()) {
        std::vector<expr> new_param;
        for (auto &v : args) {
            if (auto inop = v->dyn_cast<input_op>()) {
                for (auto &in : inop->get_outputs()) {
                    auto itr = ltsr_rtsr.find(in);
                    COMPILE_ASSERT(itr != ltsr_rtsr.end(),
                            "Cannot find the input op in the generated "
                            "function");
                    new_param.emplace_back(itr->second);
                }
            } else if (auto outop = v->dyn_cast<output_op>()) {
                for (auto &out : outop->get_inputs()) {
                    auto itr = ltsr_rtsr.find(out);
                    COMPILE_ASSERT(itr != ltsr_rtsr.end(),
                            "Cannot find the output op in the generated "
                            "function");
                    new_param.emplace_back(itr->second);
                }
            } else {
                COMPILE_ASSERT(false,
                        "The Op given in the args is not input or output");
            }
        }
        COMPILE_ASSERT(new_param.size() == params.size(),
                "The args count does not match the count of in/out "
                "tensors");
        params = std::move(new_param);
    }
    if (!init_body->seq_.empty()) {
        expr is_init_var = ret_mod->make_global_var(datatypes::boolean,
                "is_init", linkage::private_global,
                graph.attrs_.get_or_else("folded_input", false));
        init_body->seq_.emplace_back(
                builder::make_assign_unattached(is_init_var, true));
        sc::func_t init_func = builder::make_func(
                "__init_const_globals", params, init_body, datatypes::void_t);
        init_func->attr()["private"] = true;
        ret_mod->add_func({init_func});
        stmt const_init = builder::make_if_else_unattached(
                builder::make_logic_not(is_init_var),
                builder::make_stmts_unattached(
                        {builder::make_evaluate_unattached(
                                builder::make_call(init_func, params))}),
                stmts());
        func_body->seq_.insert(func_body->seq_.begin(), const_init);
    }
    func->params_ = std::move(params);
    func->decl_->params_ = func->params_;
    if (utils::compiler_configs_t::get().print_pass_result_) {
        SC_MODULE_INFO << ret_mod;
    }
    return ret_mod;
}
} // namespace sc
