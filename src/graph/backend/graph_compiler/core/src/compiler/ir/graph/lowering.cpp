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
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/dynamic_internal_info.hpp>
#include <compiler/ir/graph/dynamic_lower_info.hpp>
#include <compiler/ir/graph/dynamic_utils.hpp>
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/pass/graph_constant_cache.hpp>
#include <compiler/ir/graph/trait/may_prefetch.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/pass/ir_copy_internal.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/concat_memory_planning.hpp>
#include <compiler/ir/transform/cpu/local_tensor_lower.hpp>
#include <compiler/ir/transform/dead_write_eliminate.hpp>
#include <compiler/ir/transform/dyn_tsr_transform.hpp>
#include <compiler/ir/transform/index2var.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <compiler/ir/transform/tensor_inplace_info.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/managed_matmul_core.hpp>
#include <ops/matmul_core.hpp>
#include <ops/reshape.hpp>
#include <runtime/config.hpp>
#include <runtime/const_cache_wrapper.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/scoped_timer.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(graph.lowering)

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

static graph_tensor_ptr get_linked_output_tsr(const graph_tensor_ptr &ltensor) {
    if (ltensor->producer_owner_->isa<input_op>()) { return nullptr; }
    if (!ltensor->uses_.empty()) {
        for (size_t i = 0; i < ltensor->uses_.size(); i++) {
            if (ltensor->uses_[i].second->isa<tensor_view_op_t>()
                    || ltensor->uses_[i]
                               .second->isa<ops::dynamic_reshape_op>()) {
                auto reshape = ltensor->uses_[i].second;
                auto next_ltensor = reshape->get_outputs()[0];
                for (auto &cld : next_ltensor->uses_) {
                    if (cld.second->isa<output_op>()) {
                        return cld.second->get_inputs()[cld.first];
                    } else if (cld.second->isa<tensor_view_op_t>()
                            || cld.second->isa<ops::dynamic_reshape_op>()) {
                        auto cur_linked_out
                                = get_linked_output_tsr(next_ltensor);
                        if (cur_linked_out) { return cur_linked_out; }
                    }
                }
            }
        }
    }
    return nullptr;
}

static bool has_output_uses(const graph_tensor_ptr &ltensor) {
    if (!ltensor->uses_.empty()) {
        for (size_t i = 0; i < ltensor->uses_.size(); i++) {
            if (ltensor->uses_[i].second->isa<output_op>()) { return true; }
        }
    }
    return false;
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
    bool is_dynamic_;

    lowering_visitor_state_t(sc_graph_t &g)
        : topo_sorter_ {op_visitor_t::create_DAG_updater(g.ops_.size())}
        , op_exec_tick_(g.ops_.size())
        , op_visited_(g.ops_.size()) {
        max_tensor_size_ = 0;
        is_dynamic_ = g.is_dynamic();
        if (!is_dynamic_) {
            for (auto &op : g.ops_) {
                for (auto &tsr : op->get_outputs()) {
                    max_tensor_size_ = std::max(max_tensor_size_,
                            tsr->details_.get_blocking_byte_size());
                }
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
                float cur_tsr = is_dynamic_
                        ? heat_modifier
                        : float(in->details_.get_blocking_byte_size())
                                / ref_count_modifier / max_tensor_size_
                                * heat_modifier;
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
                    + (is_dynamic_ ? 1.f
                                   : float(out->details_
                                                     .get_blocking_byte_size())
                                            / max_tensor_size_);
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

class tv_tsr_replacer_t : public ir_copier_impl_t {
public:
    using ir_copier_impl_t::dispatch;
    using ir_copier_impl_t::view;
    tv_tsr_replacer_t(std::unordered_map<expr_c, expr> &replace_map,
            bool create_var_tensor = false)
        : ir_copier_impl_t(replace_map, create_var_tensor) {}
    void view(define_c v) override {
        if (replace_map_.find(v->var_) != replace_map_.end()) {
            returned_stmt_ = builder::make_stmts_unattached({});
        } else {
            ir_copier_impl_t::view(v);
        }
    }
};

static void add_def_comments(const stmt &def_node, graph_tensor *t) {
    std::stringstream ss;
    t->details_.to_string(ss);
    if (auto old_comments
            = def_node->attr().get_or_null<std::vector<std::string>>(
                    "comments")) {
        old_comments->emplace_back(ss.str());
    } else {
        def_node->attr()["comments"] = std::vector<std::string> {ss.str()};
    }
}

enum op_kinds : int {
    kother = 0,
    kinput,
    koutput,
    kconstant,
    kreorder,
    kreshape,
};

struct general_lower_params_t {
    ir_module_ptr ret_mod;
    std::unordered_map<graph_tensor_ptr, tsr_info_t> &ltsr_rtsr;
    sc_graph_t &graph;
    stmts func_body;
    stmts init_body;
    int &tensor_counter;
    int &global_tensor_counter;
    bool is_graph_dynamic;
    // the number of lazy initialzied constant tensors in shared const cache
    size_t num_lazy_init_shared_const_tsr_;
    // the number of compile-time initialzied constant tensors in shared const
    // cache
    size_t num_inited_shared_const_tsr_;
    std::unordered_set<sc_dim> external_dyn_vars;
    std::vector<expr> &shared_consts;
};

expr get_or_create_tensor(general_lower_params_t &gp, const graph_tensor_ptr &t,
        bool is_arg, int const_type,
        info_etype_t type = info_etype_t::real_tensor) {
    bool tsr_is_dynamic = t->details_.is_dynamic();
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
    if (is_arg || const_type != const_kind::not_const) {
        // input/output and const tsr don't need placeholder
        if (gp.is_graph_dynamic) {
            COMPILE_ASSERT(t->details_.get_format_candidates().size() <= 1,
                    "Input/output/constant tsr should have only empty or "
                    "one format candidate");
        }
        if (type == info_etype_t::placeholder) {
            type = info_etype_t::real_tensor;
        }
    }
    auto itr = gp.ltsr_rtsr.find(t);
    if (itr == gp.ltsr_rtsr.end()) {
        gp.ltsr_rtsr[t] = tsr_info_t();
        itr = gp.ltsr_rtsr.find(t);
        itr->second.count_ = gp.tensor_counter++;
    } else {
        if (type == info_etype_t::real_tensor
                && itr->second.tensor_.defined()) {
            if (gp.graph.is_dynamic()) {
                itr->second.tensor_->attr().set(attr_keys::always_trans, true);
            }
            return itr->second.tensor_;
        }
        if (type == info_etype_t::placeholder
                && itr->second.placeholder_.defined()) {
            return itr->second.placeholder_;
        }
        if (type == info_etype_t::format && itr->second.format_.defined()) {
            return itr->second.format_;
        }
        if (type == info_etype_t::out_size && itr->second.size_.defined()) {
            return itr->second.size_;
        }
    }

    std::vector<expr> dims, strides;
    sc_data_type_t tsr_dtype;
    expr tsr;

    std::string tensor_name = graph::get_tensor_name(t.get(), linked_output);
    if (tensor_name.empty()) {
        tensor_name
                = std::string("buffer_") + std::to_string(itr->second.count_);
    }
    if (type == info_etype_t::real_tensor) {
        bool is_size_dynamic = !is_arg && gp.is_graph_dynamic
                && (t->details_.get_format_candidates().size() > 1
                        || can_op_query_output(
                                t->producer_owner_->shared_from_this()));
        expr dyn_tsr_size;
        if (is_size_dynamic) {
            assert(itr->second.size_.defined());
            dyn_tsr_size = builder::make_indexing(itr->second.size_, {0});
            dyn_tsr_size->attr().set(attr_keys::no_index2var, true);
        }

        dims = is_size_dynamic ? std::vector<expr> {dyn_tsr_size}
                               : t->details_.get_blocking_dims_expr(gp.graph);
        strides = is_size_dynamic ? std::vector<expr> {UINT64_C(1)}
                                  : t->details_.get_strides_expr(gp.graph);
        tsr_dtype = t->details_.dtype_;
        tsr = builder::make_stensor(tensor_name, dims, strides, tsr_dtype);
        tsr->attr()[attr_keys::plain_dims]
                = gp.graph.dims_to_expr(t->details_.get_plain_dims());
        if (itr->second.placeholder_.defined()) {
            // for dynamic tensor transform
            tsr->attr()["temp.dyn_placeholder"] = itr->second.placeholder_;
        }
        itr->second.tensor_ = tsr;
        if (is_arg || const_type != const_kind::not_const) {
            itr->second.placeholder_ = tsr;
        }
        if (gp.graph.is_dynamic()) {
            tsr->attr().set(attr_keys::always_trans, true);
        }
        // this tensor is an input to a standalone concat op
        if (t->attrs_.has_key(concat_optim_attr_keys::graph_memory_offset)) {
            tsr->attr()[concat_optim_attr_keys::pass_memory_offset]
                    = t->attrs_.get<std::vector<expr>>(
                            concat_optim_attr_keys::graph_memory_offset);
        }
    } else if (type == info_etype_t::placeholder) {
        if (itr->second.tensor_.defined()) {
            // first check if the real tensor exist
            tsr = itr->second.tensor_;
            itr->second.placeholder_ = tsr;
            tsr->attr().set(attr_keys::always_trans, true);
        } else {
            tensor_name += "_placeholder";
            dims = std::vector<expr> {sizeof(runtime::dynamic_tensor_t)};
            tsr_dtype = datatypes::u8;
            tsr = builder::make_tensor(tensor_name, dims, tsr_dtype);
            itr->second.placeholder_ = tsr;
        }
    } else if (type == info_etype_t::format) {
        tensor_name += "_format";
        tsr = builder::make_tensor(
                tensor_name, {UINT64_C(1)}, datatypes::index);
        itr->second.format_ = tsr;
    } else {
        assert(type == info_etype_t::out_size);
        tensor_name += "_size";
        tsr = builder::make_tensor(
                tensor_name, {UINT64_C(1)}, datatypes::index);
        itr->second.size_ = tsr;
    }
    if (type == info_etype_t ::real_tensor) {
        stmt def_node;
        if (!is_arg) {
            if (const_type != const_kind::not_const) {
                if (const_type == const_kind::global_const) {
                    std::string folded_name;
                    auto ownerop = t->producer_owner_;
                    std::shared_ptr<cached_const_graph_tensor> cached;
                    if (auto cache
                            = ownerop->attrs_
                                      .get_or_null<std::vector<std::shared_ptr<
                                              cached_const_graph_tensor>>>(
                                              op_attr_key::const_input_cache)) {
                        auto idx = std::find(ownerop->get_outputs().begin(),
                                           ownerop->get_outputs().end(), t)
                                - ownerop->get_outputs().begin();
                        cached = cache->at(idx);
                        if (cached->buf_base_->is_lazy_) {
                            gp.num_lazy_init_shared_const_tsr_++;
                        } else {
                            gp.num_inited_shared_const_tsr_++;
                        }
                        std::stringstream ss;
                        ss << "shared_const_" << gp.global_tensor_counter++;
                        folded_name = ss.str();
                        tsr.static_as<tensor>()->name_ = folded_name;
                    } else {
                        folded_name = "folded_const_"
                                + std::to_string(gp.global_tensor_counter++);
                        tsr = copy_attr(*tsr,
                                gp.ret_mod->make_global_stensor(
                                        tsr.checked_as<tensor>()->elem_dtype_,
                                        folded_name,
                                        tsr.checked_as<tensor>()->dims_,
                                        tsr.checked_as<tensor>()->strides_,
                                        linkage::private_global, &def_node));
                    }
                    if (cached) {
                        tsr->attr()[attr_keys::shared_const] = cached;
                        // const cached tensors are lowered to "local tensors"
                        // with special marks
                        def_node = builder::make_var_tensor_def_unattached(tsr);
                        def_node->attr()["comments"]
                                = std::vector<std::string> {
                                        "The tensor is cached in global "
                                        "constant cache"};
                        // they are not real local tensors, don't schedule
                        // buffers
                        def_node->attr()[attr_keys::tsr_dont_buf_sched] = true;
                        gp.func_body->seq_.insert(
                                gp.func_body->seq_.begin(), def_node);
                        gp.shared_consts.emplace_back(tsr);
                    }
                    // global tensor does not need cached dynamic var
                    tsr->attr_->set("temp.dyn_placeholder", expr());
                    if (auto const_node
                            = t->producer_owner_->dyn_cast<constant_op_t>()) {
                        auto const_value = const_node->get_constant_values();
                        tsr.checked_as<tensor>()->init_value_ = const_value;
                    }
                    if (gp.graph.is_dynamic()) {
                        tsr->attr().set(attr_keys::always_trans, true);
                    }
                    itr->second.tensor_ = tsr;
                    itr->second.placeholder_ = tsr;
                } else {
                    def_node = builder::make_var_tensor_def_unattached(tsr);
                    gp.init_body->seq_.emplace_back(def_node);
                }
            } else {
                def_node = builder::make_var_tensor_def_unattached(tsr);
                gp.func_body->seq_.emplace_back(def_node);
            }
        }
        if (def_node.defined()) { add_def_comments(def_node, t.get()); }
    } else if (type == info_etype_t::placeholder) {
        // placeholder
        // if use tensor as plhd, do nothing.
        if (!itr->second.tensor_.defined()) {
            gp.func_body->seq_.emplace_back(
                    builder::make_var_tensor_def_unattached(tsr));
            std::string name;
            if (tsr.isa<tensor>()) {
                name = tsr.checked_as<tensor>()->name_;
            } else {
                assert(tsr.isa<tensorptr>());
                name = tsr.checked_as<tensorptr>()
                                ->base_->ptr_.checked_as<tensor>()
                                ->name_
                        + "_tptr";
            }
            auto shape_tsr = builder::make_tensor(
                    std::string("dyn_shape_") + tsr.checked_as<tensor>()->name_,
                    {t->details_.get_plain_dims().size()}, datatypes::index);
            shape_tsr->attr().set(attr_keys::no_dead_write, true);
            shape_tsr->attr().set(attr_keys::no_tensor2var, true);
            tsr->attr().set("temp.dyn_shape_of_placeholder", shape_tsr);
            gp.func_body->seq_.emplace_back(
                    builder::make_var_tensor_def_unattached(shape_tsr));
            gp.func_body->seq_.emplace_back(builder::make_evaluate_unattached(
                    builder::make_write_struct(tsr, shape_tsr,
                            dyn_tsr_struct_t::name,
                            dyn_tsr_struct_t::fields::dim_ptr)));
            gp.func_body->seq_.emplace_back(builder::make_evaluate_unattached(
                    builder::make_write_struct(tsr,
                            builder::make_constant(
                                    {t->details_.get_plain_dims().size()},
                                    datatypes::s32),
                            dyn_tsr_struct_t::name,
                            dyn_tsr_struct_t::fields::ndims)));
            uint64_t etype = t->details_.dtype_.is_etype_pointer()
                    ? t->details_.dtype_.get_pointer_element().as_etype_int()
                    : t->details_.dtype_.as_etype_int();
            gp.func_body->seq_.emplace_back(builder::make_evaluate_unattached(
                    builder::make_write_struct(tsr,
                            builder::make_constant({etype}, datatypes::u32),
                            dyn_tsr_struct_t::name,
                            dyn_tsr_struct_t::fields::dtype)));
            auto plain_shapes_int = t->details_.get_plain_dims();
            auto plain_shapes = gp.graph.dims_to_expr(plain_shapes_int);
            uint64_t dyn_mask_int = 0;
            for (size_t i = 0; i < plain_shapes.size(); i++) {
                if (!is_dynamic_dim(plain_shapes_int[i])
                        || gp.external_dyn_vars.find(plain_shapes_int[i])
                                != gp.external_dyn_vars.end()) {
                    gp.func_body->seq_.emplace_back(
                            builder::make_assign_unattached(
                                    builder::make_indexing(shape_tsr, {i}),
                                    plain_shapes[i]));
                }
                dyn_mask_int
                        |= (uint64_t(!plain_shapes[i].isa<constant>()) << i);
            }
            gp.func_body->seq_.emplace_back(builder::make_evaluate_unattached(
                    builder::make_write_struct(tsr,
                            builder::make_constant(
                                    {dyn_mask_int}, datatypes::u8),
                            dyn_tsr_struct_t::name,
                            dyn_tsr_struct_t::fields::dyn_mask)));
        }
    } else if (type == info_etype_t::format) {
        // placeholder can be replaced by tensor while format can't
        gp.func_body->seq_.emplace_back(
                builder::make_var_tensor_def_unattached(tsr));
        uint64_t init_format = 0;
        if (t->details_.get_format_candidates().size() <= 1) {
            init_format = uint64_t(t->details_.get_format().to_runtime());
        }
        gp.func_body->seq_.emplace_back(builder::make_assign_unattached(
                builder::make_indexing(tsr, {0}), init_format));
    } else if (type == info_etype_t::out_size) {
        if (const_type == const_kind::not_const) {
            gp.func_body->seq_.emplace_back(
                    builder::make_var_tensor_def_unattached(tsr));
            gp.func_body->seq_.back()->attr().set(
                    attr_keys::tsr_dont_buf_sched, true);
        }
    }
    return tsr;
};

expr create_op_query_func(const context_ptr &ctx, general_lower_params_t &gp,
        std::vector<expr> &op_dispatch_kernel, const sc_op_ptr &node) {
    std::vector<expr> plhd_ins, fmt_ins;
    std::vector<expr> plhd_outs, fmt_outs, size_outs;
    bool need_dispatch = can_op_be_queried(node);
    // current input
    for (auto &ltensor : node->get_inputs()) {
        auto const_type = ltensor->producer_owner_->attrs_.get_or_else(
                "constant", const_kind::not_const);
        plhd_ins.emplace_back(get_or_create_tensor(
                gp, ltensor, false, const_type, info_etype_t::placeholder));
        fmt_ins.emplace_back(get_or_create_tensor(
                gp, ltensor, false, const_type, info_etype_t::format));
    }
    // input before reorder
    if (node->isa<tunable_op_t>()
            || (node->isa<fused_op_t>()
                    && !node->stc_cast<fused_op_t>()->main_op_.empty())
            || (node->isa<mixed_fuse_op_t>()
                    && !node->stc_cast<mixed_fuse_op_t>()
                                ->get_internal_tunable_input_indices()
                                .empty())) {
        auto &inputs = node->get_inputs();
        std::vector<size_t> query_idxs;
        if (node->isa<fused_op_t>() || node->isa<tunable_op_t>()) {
            size_t sz;
            if (node->isa<fused_op_t>()) {
                sz = node->stc_cast<fused_op_t>()
                             ->main_op_.ops_[1]
                             ->get_inputs()
                             .size();
            } else {
                sz = inputs.size();
            }
            query_idxs.reserve(sz);
            for (size_t i = 0; i < sz; i++) {
                query_idxs.emplace_back(i);
            }
        } else if (node->isa<mixed_fuse_op_t>()) {
            query_idxs = node->stc_cast<mixed_fuse_op_t>()
                                 ->get_internal_tunable_input_indices();
        }
        for (auto i : query_idxs) {
            auto ltensor = node->get_inputs()[i];
            auto node_before = ltensor->producer_owner_;
            auto const_type_before = node_before->attrs_.get_or_else(
                    "constant", const_kind::not_const);
            // find the buffer before reorder.
            if (node_before->isa<reorder_op_t>()
                    && (node_before->attrs_.as_map().empty()
                            || node_before->attrs_.get_or_else(
                                       "constant", const_kind::not_const)
                                    == const_kind::not_const)) {
                ltensor = node_before->get_inputs()[0];
            }
            plhd_ins.emplace_back(get_or_create_tensor(gp, ltensor, false,
                    const_type_before, info_etype_t::placeholder));
            fmt_ins.emplace_back(get_or_create_tensor(gp, ltensor, false,
                    const_type_before, info_etype_t::format));
        }
    }
    auto const_type
            = node->attrs_.get_or_else("constant", const_kind::not_const);
    for (auto &ltensor : node->get_outputs()) {
        expr plhd, fmt, size;
        if (node->isa<input_op>()) {
            // use real tensor instead of placeholder.
            plhd = get_or_create_tensor(
                    gp, ltensor, true, const_type, info_etype_t::real_tensor);
            fmt = get_or_create_tensor(
                    gp, ltensor, true, const_type, info_etype_t::format);
        } else if (node->isa<constant_op_t>()) {
            plhd = get_or_create_tensor(gp, ltensor, false,
                    const_kind::global_const, info_etype_t::real_tensor);
            fmt = get_or_create_tensor(gp, ltensor, false,
                    const_kind::global_const, info_etype_t::format);
        } else {
            plhd = get_or_create_tensor(
                    gp, ltensor, false, const_type, info_etype_t::placeholder);
            // expect for output tsr
            if (!plhd.defined()) {
                plhd = get_or_create_tensor(gp, ltensor, false, const_type,
                        info_etype_t::real_tensor);
            }
            fmt = get_or_create_tensor(
                    gp, ltensor, false, const_type, info_etype_t::format);
            size = get_or_create_tensor(
                    gp, ltensor, false, const_type, info_etype_t::out_size);
        }
        plhd_outs.emplace_back(plhd);
        fmt_outs.emplace_back(fmt);
        size_outs.emplace_back(size);
    }
    // Pruning, because the format propagation is broken after reorder,
    // so it doesn't need query to deliver formats. Notes that only
    // reorder could, other ops should propagate their format even does
    // not need dispatch.
    if ((node->isa<reorder_op_t>() && !need_dispatch)
            || const_type != const_kind::not_const) {
        return expr();
    }
    expr dyn_ker_ptr;
    // update dynamic query format
    if (!op_dispatch_kernel[node->logical_op_id_].defined()) {
        auto &table_map = gp.ret_mod->get_op_table_map();
        auto func_name = node->op_name_ + "__"
                + std::to_string(node->logical_op_id_) + "_ptr";
        auto table_name = func_name + "_table";
        auto table_it = table_map.find(table_name);
        auto table_var = builder::make_var(datatypes::pointer, table_name);
        auto table_ptr = table_it != table_map.end()
                ? table_it->second
                : std::make_shared<op_dispatch_tables_t>();
        int internal_func_num = get_num_of_internal_funcs(node);
        // kernel pointer vector the first is outer function.
        dyn_ker_ptr = builder::make_tensor(func_name,
                {static_cast<uint64_t>(1 + internal_func_num)},
                datatypes::index);
        std::vector<expr> query_func_args;
        query_func_args.emplace_back(table_var);
        query_func_args.insert(
                query_func_args.end(), plhd_outs.begin(), plhd_outs.end());
        query_func_args.insert(
                query_func_args.end(), plhd_ins.begin(), plhd_ins.end());
        query_func_args.insert(
                query_func_args.end(), fmt_outs.begin(), fmt_outs.end());
        query_func_args.insert(
                query_func_args.end(), fmt_ins.begin(), fmt_ins.end());
        query_func_args.insert(
                query_func_args.end(), size_outs.begin(), size_outs.end());
        query_func_args.push_back(dyn_ker_ptr);
        expr query_call; // call node
        if (node->isa<fused_op_t>()) {
            auto fused_node = node->stc_cast<fused_op_t>();
            auto query_mod = fused_node->get_dynamic_query_func(ctx);
            gp.ret_mod->merge(*query_mod);
            assert(table_ptr);
            query_call = builder::make_call(
                    query_mod->get_entry_func(), query_func_args);
        } else if (node->isa<mixed_fuse_op_t>()) {
            auto fused_node = node->stc_cast<mixed_fuse_op_t>();
            auto query_mod = fused_node->get_dynamic_query_func(ctx);
            gp.ret_mod->merge(*query_mod);
            assert(table_ptr);
            query_call = builder::make_call(
                    query_mod->get_entry_func(), query_func_args);
        } else {
            auto table_ptr = std::make_shared<op_dispatch_tables_t>();
            gp.ret_mod->add_op_table(std::make_pair(table_name, table_ptr));
            initialize_dispatch_table_with_op(ctx, node, table_ptr);
            query_call = call_op_dynamic_query_function(node, query_func_args);
        }
        stmts_node_t *target_body = gp.func_body.get();
        if (table_it == table_map.end()) {
            auto table_def = builder::make_var_tensor_def_unattached(
                    table_var, linkage::private_global);
            gp.ret_mod->add_global_var(table_def.checked_as<define>());
        }
        target_body->seq_.emplace_back(
                builder::make_var_tensor_def_unattached(dyn_ker_ptr));
        target_body->seq_.emplace_back(
                builder::make_evaluate_unattached(query_call));
        op_dispatch_kernel[node->logical_op_id_] = builder::make_reinterpret(
                builder::make_indexing(dyn_ker_ptr, 0), datatypes::pointer);
        op_dispatch_kernel[node->logical_op_id_]->attr().set(
                attr_keys::no_index2var, true);
    }
    return dyn_ker_ptr;
}

std::pair<expr, expr> get_reshape_tptr(general_lower_params_t &gp,
        const graph_tensor_ptr &old_tsr, const graph_tensor_ptr &new_tsr,
        int const_type, op_kinds kind) {
    auto base_tsr
            = get_or_create_tensor(gp, old_tsr, kind == kinput, const_type);
    size_t ndims;
    if (base_tsr.isa<tensorptr>()) {
        ndims = base_tsr.static_as<tensorptr>()->shape_.size();
    } else {
        assert(base_tsr.isa<tensor>());
        ndims = base_tsr.static_as<tensor>()->dims_.size();
    }
    std::vector<expr_c> base_idx(ndims, expr(0));
    std::vector<expr> new_shape_tmp
            = new_tsr->details_.get_blocking_dims_expr(gp.graph);
    std::vector<expr_c> new_shape(new_shape_tmp.begin(), new_shape_tmp.end());
    auto new_tptr = builder::tensor_ptr(base_tsr, base_idx, new_shape);
    new_tptr->attr().set(attr_keys::plain_dims,
            gp.graph.dims_to_expr(new_tsr->details_.get_plain_dims()));
    return std::make_pair(base_tsr, new_tptr);
}

void create_op_tensors(general_lower_params_t &gp, std::vector<expr> &ins,
        std::vector<expr> &outs, const sc_op_ptr &node, op_kinds kind) {
    int const_type
            = node->attrs_.get_or_else("constant", const_kind::not_const);
    for (auto &ltensor : node->get_inputs()) {
        // As the traversal is not in order, so the constant type of
        // tensor should be decided by the node before.
        ins.emplace_back(get_or_create_tensor(gp, ltensor, false, const_type));
    }
    for (auto &ltensor : node->get_outputs()) {
        if (kind == kconstant) {
            get_or_create_tensor(gp, ltensor, false, const_kind::global_const);
        } else if (kind == kreshape) {
            COMPILE_ASSERT(node->get_inputs().size() == 1
                            || node->get_inputs().size() == 2,
                    "Reshape should have 1 or 2(dynamic_reshape) input");
            // If the output of tensor view is output of graph
            if (gp.ltsr_rtsr.find(ltensor) != gp.ltsr_rtsr.end()
                    && has_output_uses(ltensor)) {
                break;
            }
            auto out_tsr_pair = get_reshape_tptr(
                    gp, node->get_inputs()[0], ltensor, const_type, kind);
            auto it = gp.ltsr_rtsr.find(ltensor);
            if (it != gp.ltsr_rtsr.end() && it->second.tensor_.defined()) {
                COMPILE_ASSERT(gp.is_graph_dynamic,
                        "If output tsr of tensor view is defined, it "
                        "should in dynamic mode.");
                // the tsr replace map for tensor view op. Because in
                // dynamic mode, the output of tensor view may be
                // traversed first.
                std::unordered_map<expr_c, expr> tv_replace_map;
                tv_replace_map.insert(std::make_pair(
                        it->second.tensor_, out_tsr_pair.second));
                tv_tsr_replacer_t cpy(tv_replace_map, false);
                gp.func_body = cpy.dispatch(gp.func_body)
                                       .remove_const()
                                       .checked_as<stmts>();
                gp.init_body = cpy.dispatch(gp.init_body)
                                       .remove_const()
                                       .checked_as<stmts>();
            }
            gp.ltsr_rtsr[ltensor].tensor_ = out_tsr_pair.second;
        } else {
            graph_tensor_ptr out_tsr;
            // for pattern like node->reshape->output, node != input
            if (auto out_tsr = get_linked_output_tsr(ltensor)) {
                gp.ltsr_rtsr[ltensor].tensor_ = get_reshape_tptr(
                        gp, out_tsr, ltensor, const_type, kind)
                                                        .second;
                outs.emplace_back(gp.ltsr_rtsr[ltensor].tensor_);
            } else {
                outs.emplace_back(get_or_create_tensor(
                        gp, ltensor, kind == kinput, const_type));
            }
        }
    }
}

static std::string get_dispatch_callee_name(const expr &kernel) {
    assert(kernel.isa<indexing>());
    return kernel.checked_as<indexing>()->ptr_.checked_as<tensor>()->name_;
}

static void dynamic_reshape_var_assignment(general_lower_params_t &gp,
        std::unordered_set<expr> &dynamic_var_set, const sc_op_ptr &node) {
    auto &graph = gp.graph;
    auto out_plain_dims = node->get_outputs()[0]->details_.get_plain_dims();
    auto shape_tsr = gp.ltsr_rtsr[node->get_inputs()[1]].tensor_;
    for (size_t i = 0; i < out_plain_dims.size(); i++) {
        auto &dyn_dim = out_plain_dims[i];
        // may be inferred by dynamic shape binding.
        if (!is_dynamic_dim(dyn_dim)) { continue; }
        auto var = graph.dim_to_expr(dyn_dim);
        if (dynamic_var_set.find(var) == dynamic_var_set.end()) {
            gp.func_body->seq_.emplace_back(
                    builder::make_var_tensor_def_unattached(
                            graph.dim_to_expr(dyn_dim), linkage::local,
                            builder::make_indexing(shape_tsr, {i})));
            dynamic_var_set.insert(var);
        }
    }
}
static func_t insert_prefetch(const context_ptr &ctx,
        const std::vector<std::pair<sc_op_ptr, stmt>> &op_execution_log,
        sc_op *node, const std::vector<expr> &input_tsr, ir_module_t &mod,
        std::vector<stmt> &outbody) {
    if (!ctx->flags_.prefetch_) { return func_t(); }
    if (op_execution_log.empty()) { return func_t(); }
    auto prefetch_op = node->dyn_cast<op_traits::may_prefetch_t>();
    if (!prefetch_op) { return func_t(); }
    std::vector<tensor_slice> ins;
    ins.reserve(input_tsr.size());
    for (auto &in : input_tsr) {
        ins.emplace_back(in);
    }
    auto can_prefetch_index = prefetch_op->query_prefetch(ctx, true, ins);
    if (can_prefetch_index.empty()) { return func_t(); }
    // now find the position where we insert prefetch
    size_t pos = std::string::npos;
    // how many Ops are executed between "pos" and current Op
    size_t pos_diff = 0;
    for (int64_t i = op_execution_log.size() - 1; i >= 0; i--) {
        if (op_execution_log[i].second.defined()) {
            // if the op is executed in main body (not in cached const/input)
            auto gflop = op_execution_log[i].first->get_gflop();
            // todo(yijie): currently gflop>0 means the op has a complex op like
            // matmul so we only need to compare gflop with 0. We should later
            // set a reasonable threshold
            bool is_good_for_insert = gflop > 0;
            if (is_good_for_insert) {
                auto func = op_execution_log[i]
                                    .second.checked_as<evaluate>()
                                    ->value_.checked_as<call>()
                                    ->func_;
                if (func->attr_
                        && func->attr_->get_or_else(
                                function_attrs::has_idle_func, false)) {
                    // if we want to insert to a position that is already doing
                    // prefetch, skip
                    return func_t();
                }
                pos = i;
                break;
            }
            pos_diff++;
        }
    }
    if (pos == std::string::npos || pos_diff > 3) {
        // if there is no large ops, or the position is too far away from the
        // current op, don't insert prefetch
        return func_t();
    }

    std::vector<int> filtered_prefetch_index;
    auto cache_thres = ctx->machine_.cpu_flags_.getDCacheSize(2)
            * runtime_config_t::get().get_num_threads();
    // we need to check at this point (at "pos"), which input of the Op ("node")
    // is ready to prefetch
    for (auto in_idx : can_prefetch_index) {
        auto &tsr = node->get_inputs().at(in_idx);
        // todo (yijie): add more detailed cache size management
        // skip prefetch on this tensor if it is too large
        if (tsr->details_.get_blocking_byte_size() * 2 > cache_thres) {
            continue;
        }
        auto producer_op = tsr->producer_owner_;
        size_t producer_idx = op_execution_log.size();
        for (size_t i = 0; i < op_execution_log.size(); i++) {
            if (op_execution_log[i].first.get() == producer_op) {
                if (op_execution_log[i].second.defined()) {
                    producer_idx = i;
                } else {
                    // if the producer op is in cached const/input, it is always
                    // ready to be prefetched
                    producer_idx = 0;
                }
                break;
            }
        }
        if (producer_idx <= pos) {
            if (!input_tsr.at(in_idx).isa<tensor>()) {
                SC_MODULE_WARN << "Cannot prefetch tensor_view"
                               << input_tsr.at(in_idx);
            } else {
                // if the input is already computed before "pos"
                filtered_prefetch_index.push_back(in_idx);
            }
        }
    }
    if (filtered_prefetch_index.empty()) { return func_t(); }
    std::vector<stmt> out;
    auto ret = prefetch_op->generate_prefetcher_and_set_idle(
            ctx, true, ins, filtered_prefetch_index, out);
    // insert the thread pool manipulation code before pos
    bool found = false;
    for (auto itr = outbody.begin(); itr != outbody.end(); ++itr) {
        if (itr->ptr_same(op_execution_log[pos].second)) {
            for (auto &set_idle_body : out) {
                itr = outbody.insert(itr, set_idle_body);
            }
            auto func = op_execution_log[pos]
                                .second.checked_as<evaluate>()
                                ->value_.checked_as<call>()
                                ->func_;
            func->attr()[function_attrs::has_idle_func] = true;
            found = true;
            break;
        }
    }
    assert(found);
    SC_UNUSED(found);
    mod.add_func({ret});
    return ret;
}

static void add_func_doc_string(const func_t &f) {
    std::vector<std::string> comments;
    comments.reserve(f->params_.size() + 1);
    comments.emplace_back(f->name_);
    for (auto &p : f->params_) {
        std::stringstream ss;
        ss << "@param " << p << ' '
           << p->attr().get<std::string>("temp.comments");
        comments.emplace_back(ss.str());
        p->attr().remove("temp.comments");
    }
    f->attr()["comments"] = std::move(comments);
}

// check if an op can be marked no_post_barrier
stmt_base_t *has_no_dep_with_prev(const sc_op_ptr &node,
        const op_dep_matrix_t &dep_mat,
        const std::vector<std::pair<sc_op_ptr, stmt>> &op_execution_log) {
    // mark no_post_barrier on eveluate-call
    if (op_execution_log.empty()) { return nullptr; }
    auto itr = op_execution_log.rbegin();
    for (; itr != op_execution_log.rend(); ++itr) {
        if (!itr->second.defined()) continue;
        if (dep_mat.lookup(node, itr->first) != 0) { return nullptr; }
        break;
    }
    if (itr == op_execution_log.rend()) { return nullptr; }
    // itr now points to the previously executed op
    auto ret = itr->second.get();
    // search from the previous executed op to find if the current op depends on
    // any of the op in the no_post_barrier section
    for (itr = itr + 1; itr != op_execution_log.rend(); ++itr) {
        if (!itr->second.defined()) continue;
        // if the prev op is already marked no_dep, need to continue to check to
        // ensure all consequent ops with no_dep does not depend on each other
        if (itr->second->attr_
                && itr->second->attr_->has_key(attr_keys::no_post_barrier)) {
            // if current op does not depend on the previous op
            if (dep_mat.lookup(node, itr->first) == 0) {
                continue;
            } else {
                return nullptr;
            }
        } else {
            return ret;
        }
    }
    return ret;
}

ir_module_ptr lower_graph(context_ptr ctx, sc_graph_t &graph,
        const std::vector<sc_op_ptr> &args, bool mark_as_main) {
    auto timer = SC_SCOPED_TIMER_INFO("graph.driver.time.lowering", "");
    lowering_visitor_state_t visiter_state(graph);
    op_visitor_t vis {
            visiter_state.get_selector(), visiter_state.get_updater(), true};
    visiter_state.input_op_itr = vis.to_visit_.end();
    std::vector<expr> params;
    stmts func_body = make_stmt<stmts_node_t>(std::vector<stmt>());
    stmts init_body = make_stmt<stmts_node_t>(std::vector<stmt>());
    constexpr const char *default_graph_name = "main_entry";
    auto graph_name = graph.attrs_.get_or_else<std::string>(
            "temp.name", default_graph_name);
    // todo: use graph-id to generate name
    auto func = builder::make_func(
            graph_name, params, func_body, datatypes::void_t);
    // todo: logical tensor should also have an unique id
    // tsr_info_t include dynamic placeholder(dynamic tensor with empty
    // datapointer) and runtime format.
    std::unordered_map<graph_tensor_ptr, tsr_info_t> ltsr_rtsr;
    // external dyn vars set.
    std::unordered_set<sc_dim> external_dyn_vars
            = graph.get_external_dynamic_vars();
    // function pointer
    std::vector<expr> op_dispatch_kernel(graph.ops_.size());
    int tensor_counter = 0;
    int global_tensor_counter = 0;
    auto ret_mod = ir_module_t::from_entry_func(ctx, func);
    auto use_managed_tp = std::make_shared<bool>(false);

    expr dump_out_path;
    if (graph.attrs_.get_or_else("folded_input", false)) {
        ret_mod->attr_.set("folded_input", true);
    }
    bool is_graph_dynamic = graph.is_dynamic()
            && !graph.attrs_.get_or_else("temp.force_static", false);
    // the shared constant tensors. They need to be passed to the init_globals()
    // func
    std::vector<expr> shared_consts;
    general_lower_params_t gp {ret_mod, ltsr_rtsr, graph, func_body, init_body,
            tensor_counter, global_tensor_counter, is_graph_dynamic,
            /*num_lazy_init_const_tsr*/ 0, /*num_inited_shared_const_tsr_*/ 0,
            external_dyn_vars, shared_consts};
    // the set of dynamic var defined in func body.(dynamic reshape)
    std::unordered_set<expr> dyn_var_set;
    // record the node, index is op id.
    std::vector<bool> query_visited(graph.ops_.size(), false);
    op_dep_matrix_t dep_mat {graph};
    std::vector<std::pair<sc_op_ptr, stmt>> op_execution_log;
    int num_ops = 0;
    // internal function related
    bool need_extra_internal_func_arg
            = !mark_as_main && graph.need_dynamic_internal_query();
    int total_num_of_internal_funcs = get_num_of_internal_funcs(graph);
    expr extra_internal_func_arg = builder::make_tensor("extra_internal_funcs",
            {total_num_of_internal_funcs}, datatypes::index);
    extra_internal_func_arg->attr()["temp.comments"] = std::string(
            "Extra tensor arg contains pointer to internal funcs");
    int cur_internal_idx = 0;

    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        std::vector<expr> ins, outs;
        // special kinds of Ops that we need to take care of
        op_kinds kind = kother;
        if (node->isa<input_op>()) {
            kind = kinput;
        } else if (node->isa<output_op>()) {
            kind = koutput;
        } else if (node->isa<constant_op_t>()) {
            kind = kconstant;
            if (node->attrs_.get_or_else("constant", const_kind::not_const)
                    == const_kind::not_const) {
                node->attrs_.set("constant", const_kind::global_const);
            }
        } else if (node->isa<reorder_op_t>()) {
            // todo: assume reorder is fused break in dynamic now.
            kind = kreorder;
        } else if (node->isa<tensor_view_op_t>()
                || node->isa<ops::dynamic_reshape_op>()) {
            kind = kreshape;
        }

        // if the node is reorder or has tail reorder in its internal graph,
        // query its uses op first.
        if (is_graph_dynamic && can_op_be_queried(node)) {
            auto create_op_query_func_wrapper = std::bind(create_op_query_func,
                    ctx, std::ref(gp), std::ref(op_dispatch_kernel),
                    std::placeholders::_1);
            lower_query_function(
                    query_visited, node, create_op_query_func_wrapper);
        }
        // tensor decl should put after query functions.
        create_op_tensors(gp, ins, outs, node, kind);
        if (is_graph_dynamic && kind == kreorder
                && node->attrs_.get_or_else("constant", const_kind::not_const)
                        == const_kind::not_const) {
            outs[0]->attr().set("temp.may_inplace", true);
        }
        int const_type
                = node->attrs_.get_or_else("constant", const_kind::not_const);
        bool executed_in_main_body = false;
        switch (kind) {
            case kinput: {
                auto &op_outs = node->get_outputs();
                for (size_t i = 0; i < outs.size(); i++) {
                    auto &v = outs[i];
                    params.emplace_back(v);
                    std::stringstream ss;
                    op_outs[i]->details_.to_string(ss);
                    v->attr()["temp.comments"] = ss.str();
                }
                break;
            }
            case koutput: {
                auto &op_ins = node->get_inputs();
                for (size_t i = 0; i < ins.size(); i++) {
                    auto &v = ins[i];
                    params.emplace_back(v);
                    std::stringstream ss;
                    op_ins[i]->details_.to_string(ss);
                    v->attr()["temp.comments"] = ss.str();
                }
                break;
            }
            case kconstant:
            case kreshape: {
                if (node->isa<ops::dynamic_reshape_op>()) {
                    dynamic_reshape_var_assignment(gp, dyn_var_set, node);
                }
                break;
                // nothing to do.
            }
            default: {
                std::vector<expr> exprargs;
                exprargs.insert(exprargs.end(), outs.begin(), outs.end());
                exprargs.insert(exprargs.end(), ins.begin(), ins.end());
                expr kernel_call;
                std::string callee_name;
                if (is_graph_dynamic && can_op_be_dispatched(node)) {
                    auto &base_kernel = op_dispatch_kernel[node->logical_op_id_]
                                                .checked_as<intrin_call>()
                                                ->args_[0];
                    assert(is_graph_dynamic);
                    assert(base_kernel.defined());
                    callee_name = get_dispatch_callee_name(base_kernel);
                    std::string table_name = callee_name + "_table";
                    int dyn_idx = 0;

                    node->get_dispatch_key_set()->for_each_key_process(
                            std::bind(create_dispatch_funcs_by_keys, ctx,
                                    std::ref(ret_mod), table_name, node,
                                    std::placeholders::_1,
                                    std::ref(op_dispatch_kernel
                                                    [node->logical_op_id_]),
                                    std::ref(dyn_idx), use_managed_tp,
                                    /*internal*/ false));
                    if (node->need_dynamic_internal_query()) {
                        node->info_.internal_info_->parti_in_ltsrs_
                                = graph::extract_detail_from_tensors(
                                        node->get_inputs());
                        node->info_.internal_info_->parti_out_ltsrs_
                                = graph::extract_detail_from_tensors(
                                        node->get_outputs());
                        create_internal_dispatch_funcs_by_node(
                                ctx, ret_mod, table_name, node, use_managed_tp);
                        assert(base_kernel.isa<indexing>());
                        auto ker_tsr = base_kernel.static_as<indexing>()->ptr_;
                        // index 0 is the outer dispatch kernel, elements from
                        // index 1 are the internal dispatch kernels.
                        exprargs.emplace_back(
                                builder::tensor_ptr(ker_tsr, {1}));
                    }
                    kernel_call = make_expr<call_node>(
                            op_dispatch_kernel[node->logical_op_id_], exprargs);
                } else {
                    // no dispatch
                    auto mod = node->get_func(ctx);
                    auto inp_node = node->dyn_cast<op_traits::may_inplace_t>();
                    if (inp_node && ctx->flags_.tensor_inplace_) {
                        auto inp_hint = inp_node->get_inplace_map();
                        if (!inp_hint.empty()) {
                            auto func = mod->get_entry_func();
                            std::vector<std::pair<int,
                                    std::vector<tensor_inplace_info_t>>>
                                    out_hint;
                            for (auto &kv : inp_hint) {
                                int output_id = kv.first;
                                out_hint.emplace_back(output_id,
                                        std::vector<tensor_inplace_info_t> {});
                                for (auto &info : kv.second) {
                                    auto input_id = info.used_arg_idx_;
                                    // convert input id to argument index
                                    int new_idx = input_id
                                            + node->get_outputs().size();
                                    out_hint.back().second.emplace_back(
                                            tensor_inplace_info_t {
                                                    new_idx, info.kind_});
                                }
                            }
                            func->attr()[function_attrs::inplace_hint]
                                    = std::move(out_hint);
                        }
                    }
                    ret_mod->merge(*mod);
                    auto callee = mod->get_entry_func();
                    if (need_extra_internal_func_arg
                            && node->need_dynamic_internal_query()) {
                        int cur_internal_funcs
                                = get_num_of_internal_funcs(node);
                        exprargs.emplace_back(builder::tensor_ptr(
                                extra_internal_func_arg, {cur_internal_idx}));
                        cur_internal_idx += cur_internal_funcs;
                    }
                    callee_name = callee->name_;
                    kernel_call = builder::make_call(callee, exprargs);
                    if (node->isa<concat_op_t>()) {
                        kernel_call->attr()
                                [concat_optim_attr_keys::is_standalone_concat]
                                = true;
                    }
                    if (mark_as_main && const_type == const_kind::not_const) {
                        insert_prefetch(ctx, op_execution_log, node.get(), ins,
                                *ret_mod, func_body->seq_);
                    }
                }
                stmts_node_t *target_body
                        = (const_type != const_kind::not_const)
                        ? init_body.get()
                        : func_body.get();
                target_body->seq_.emplace_back(
                        builder::make_evaluate_unattached(kernel_call));
                if (const_type == const_kind::not_const) {
                    executed_in_main_body = true;
                    if (auto marked_stmt = has_no_dep_with_prev(
                                node, dep_mat, op_execution_log)) {
                        marked_stmt->attr()[attr_keys::no_post_barrier]
                                = callee_name;
                    }
                    op_execution_log.emplace_back(
                            node, target_body->seq_.back());
                    num_ops++;
                }
            }
        }
        if (!executed_in_main_body) {
            op_execution_log.emplace_back(node, stmt());
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
                    new_param.emplace_back(itr->second.tensor_);
                }
            } else if (auto outop = v->dyn_cast<output_op>()) {
                for (auto &out : outop->get_inputs()) {
                    auto itr = ltsr_rtsr.find(out);
                    COMPILE_ASSERT(itr != ltsr_rtsr.end(),
                            "Cannot find the output op in the generated "
                            "function");
                    new_param.emplace_back(itr->second.tensor_);
                }
            } else {
                COMPILE_ASSERT(false,
                        "The Op given in the args is not input or output");
            }
        }
        COMPILE_ASSERT(new_param.size() == params.size(),
                "The args count does not match the count of in/out "
                "tensors, new_param.size="
                        << new_param.size()
                        << ", param.size()=" << params.size() << ".");
        params = std::move(new_param);
    }
    if (need_extra_internal_func_arg) {
        assert(cur_internal_idx == total_num_of_internal_funcs);
        params.emplace_back(extra_internal_func_arg);
    }
    if (!init_body->seq_.empty()) {
        expr is_init_var;
        const size_t share_const_is_init_num_stmts = 2;
        std::vector<stmt> to_insert;
        if (gp.num_lazy_init_shared_const_tsr_ == 0) {
            // if there are no lazy inited buffer in shared const
            is_init_var = ret_mod->make_global_var(datatypes::boolean,
                    "is_init", linkage::private_global,
                    graph.attrs_.get_or_else("folded_input", false));
            init_body->seq_.emplace_back(
                    builder::make_assign_unattached(is_init_var, true));
        } else {
            // need special __is_init local tensor for lazy inited buffer
            auto is_init_tensor
                    = builder::make_tensor("__is_init", {1}, datatypes::s32);
            is_init_tensor->attr()[attr_keys::is_init_for_const_cache] = true;
            is_init_tensor->attr()[attr_keys::no_index2var] = true;
            to_insert.emplace_back(
                    builder::make_var_tensor_def_unattached(is_init_tensor));
            to_insert.back()->attr()["comments"] = std::vector<std::string> {
                    "the element of it is auto-filled based on states of "
                    "shared_const tensors"};
            to_insert.back()->attr()[attr_keys::is_shared_const_init_stmt]
                    = true;
            to_insert.emplace_back(
                    builder::make_assign_unattached(is_init_tensor[0], 1));
            to_insert.back()->attr()[attr_keys::is_shared_const_init_stmt]
                    = true;
            is_init_var = (is_init_tensor[0] != 0);
            assert(share_const_is_init_num_stmts == to_insert.size());
            func_body->seq_.insert(func_body->seq_.begin(), to_insert.begin(),
                    to_insert.end());
        }
        auto params_for_init = params;
        params_for_init.insert(params_for_init.end(), shared_consts.begin(),
                shared_consts.end());
        func_t init_func = builder::make_func("__init_const_globals",
                params_for_init, init_body, datatypes::void_t);
        init_func->attr()[function_attrs::private_] = true;
        ret_mod->add_func({init_func});
        stmt const_init = builder::make_if_else_unattached(
                builder::make_logic_not(is_init_var),
                builder::make_stmts_unattached(
                        {builder::make_evaluate_unattached(builder::make_call(
                                init_func, params_for_init))}),
                stmts());
        // insert the stmt "if (!is_init) {__init_const_globals(...);}"
        if (gp.num_inited_shared_const_tsr_ + gp.num_lazy_init_shared_const_tsr_
                == 0) {
            // if there are no shared const buffers
            func_body->seq_.insert(func_body->seq_.begin(), const_init);
        } else {
            // find the first position in the body after the definition of
            // shared consts
            for (auto itr = func_body->seq_.begin() + to_insert.size();;
                    itr++) {
                if (itr == func_body->seq_.end()
                        || !(*itr).cast<define_c>()
                                    .map([](const define_c &v) {
                                        return v->var_.as<tensor>();
                                    })
                                    .filter([](const tensor &v) {
                                        return v->attr_
                                                && v->attr_->has_key(attr_keys::
                                                                shared_const);
                                    })
                                    .has_value()) {
                    func_body->seq_.insert(itr, const_init);
                    break;
                }
            }
        }
    }
    func->params_ = std::move(params);
    func->decl_->params_ = func->params_;
    func->body_ = std::move(func_body);
    func->attr()[function_attrs::top_level] = true;
    add_func_doc_string(func);
    if (mark_as_main) func->attr()[function_attrs::is_main] = true;
    if (utils::compiler_configs_t::get().print_pass_result_) {
        SC_MODULE_INFO << ret_mod;
    }
    auto gflop = graph.attrs_.get_or_else(sc_graph_t::attr_key_t::gflop, 0.0f);
    ret_mod->attr_[ir_module_t::attr_key_t::GFLOP] = gflop;

    // if the workload is too small, directly use thread pool backend instead of
    // managed thread pool. if gflop per thread is large enough, or there is
    // only one single thread, enable managed thread pool. For MLP workload on
    // 24-core cascade lake, 1.6Gflop is turning point of choosing
    // managed/native thread pool
    auto &rtl_cfg = runtime_config_t::get();
    bool use_managed_thread_pool = false;
    if (rtl_cfg.managed_thread_pool_) {
        if (num_ops > 2 || rtl_cfg.get_num_threads() == 1
                || gflop / rtl_cfg.get_num_threads() > 0.0666f) {
            use_managed_thread_pool = true;
        } else if (ctx->use_amx()) {
            use_managed_thread_pool = true;
        }
    }
    ret_mod->attr_[ir_module_t::attr_key_t::MANAGED_THREAD_POOL]
            = use_managed_thread_pool;
    *use_managed_tp = use_managed_thread_pool;
    if (graph_name != default_graph_name) {
        ret_mod->attr_[ir_module_t::attr_key_t::NAME] = graph_name;
    }
    if (graph.attrs_.has_key("shared_const_bases")) {
        ret_mod->attr_[ir_module_t::attr_key_t::SHARED_CONST_BASES]
                = graph.attrs_["shared_const_bases"];
    }

    if (ctx->flags_.graph_default_private_) {
        for (auto &f : ret_mod->get_contents()) {
            f->attr()[function_attrs::private_] = true;
        }

        ret_mod->get_entry_func()->attr().remove(function_attrs::private_);
        for (auto &table : ret_mod->get_op_table_map()) {
            for (auto &kv : table.second->kernel_table_) {
                if (kv.second.already_compiled()) {
                    if (auto f
                            = ret_mod->get_func(kv.second.name_or_postfix_)) {
                        f->attr().remove(function_attrs::private_);
                    }
                }
            }
        }
    }
    return ret_mod;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
