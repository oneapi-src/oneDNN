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
#include <atomic>
#include <utility>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/dynamic_internal_info.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/transform/dead_func_eliminate.hpp>
#include <compiler/ir/transform/dyn_tsr_transform.hpp>
#include <compiler/ir/transform/func_inline.hpp>
#include <compiler/ir/transform/index2var.hpp>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
// distiguish overloaded functions.
static std::atomic<int> internal_idx = {0};
tunable_op_t::tunable_op_t(const std::string &op_name,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : sc_op(op_name, ins, outs, attrs) {}

sc_op_ptr tunable_op_t::copy(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr) {
    auto ret = mgr.make(op_name_, ins, outs, attrs_);
    ret->copy_dispatch_key_set_from_op(shared_from_this());
    ret->info_.internal_info_ = info_.internal_info_; // shadow copy.
    auto tune_ret = ret->stc_cast<tunable_op_t>();
    tune_ret->op_name_ = op_name_;
    tune_ret->config_data_ = config_data_;
    tune_ret->dyn_config_candidates_ = dyn_config_candidates_;
    tune_ret->is_quantized_ = is_quantized_;
    tune_ret->need_compensation_ = need_compensation_;
    tune_ret->should_quantized_ = should_quantized_;
    return ret;
}

bool tunable_op_t::is_valid(const context_ptr &ctx) {
    if (!config_data_
            || !create_generator()->is_valid_config(
                    ctx, config_data_.data_.get())) {
        return false;
    }
    return true;
}

ir_module_ptr tunable_op_t::get_func(context_ptr ctx) {
    auto ret = std::make_shared<ir_module_t>(ctx);
    auto gen_ptr = create_generator();
    bool need_inner_query = need_dynamic_internal_query();
    set_config_if_empty(ctx, gen_ptr.get());
    std::vector<expr> ins;
    std::vector<expr> outs;
    auto func = graph::create_func_decl_for_op(this, ins, outs);
    auto gen_body = [&](const body_generator_ptr &gen_ptr) {
        builder::ir_builder_t bld;
        bld.push_scope();
        std::vector<for_loop> loops;
        bool status = gen_ptr->generate(
                ctx, config_data_.data_.get(), nullptr, ins, outs, loops);
        assert(status);
        bld.push_returns(true);
        auto body = bld.pop_scope();
        gen_ptr->schedule_loops(ctx, config_data_.data_.get(), body, loops);
        return body;
    };
    stmt body;
    if (need_inner_query) {
        std::vector<for_loop> loops;
        func->params_.emplace_back(
                builder::make_tensor("internal_func", {1}, datatypes::index));
        func->params_.back()->attr().set(attr_keys::no_index2var, true);
        func->decl_->params_ = func->params_;
        func_t single_core_func = gen_ptr->get_single_core_func(
                ctx, config_data_.data_.get(), nullptr, ins, outs, loops);
        auto common_args = outs;
        common_args.insert(common_args.end(), ins.begin(), ins.end());
        auto common_params = common_args;
        common_params.emplace_back(
                builder::make_var(datatypes::pointer, "single_core_func"));
        common_args.emplace_back(builder::make_func_addr(single_core_func));
        auto internal_func = builder::make_func(
                std::string("internal_func") + std::to_string(internal_idx++),
                common_params, builder::make_returns_unattached(true),
                datatypes::boolean);
        internal_func->attr().set(attr_keys::keep_func, true);
        auto inter_ptr = builder::make_reinterpret(
                builder::make_indexing(func->params_.back(), {0}),
                datatypes::pointer);
        inter_ptr->attr().set("prototype", internal_func);
        body = builder::make_stmts_unattached(
                {builder::make_evaluate_unattached(
                         make_expr<call_node>(inter_ptr, common_args)),
                        builder::make_returns_unattached(true)});
        ret->add_func({func, internal_func, single_core_func});
    } else {
        body = gen_body(gen_ptr);
        ret->add_func({func});
    }

    func->body_ = std::move(body);
    ret->set_entry_func_idx(0);
    return ret;
}

func_t tunable_op_t::get_func(mixed_parti_t *parti,
        const std::vector<expr> &ins, const std::vector<expr> &outs) {
    bool need_inner_query = need_dynamic_internal_query();
    auto copy_from_fmgr = [&](fusion_manager &fmgr) {
        // record output anchor
        extract_anchor_from_fmgr_to_parti(&fmgr, parti, outs, get_outputs(),
                parti->ready_for_op(this) ? parti->lookup_anchor_map(this)
                                          : nullptr);
        // record input anchor if necessary
        if (parti->empty()) {
            extract_anchor_from_fmgr_to_parti(
                    &fmgr, parti, ins, get_inputs(), nullptr);
        }
    };
    auto gen_body = [&](const body_generator_ptr &gen_ptr) {
        set_config_if_empty(parti->ctx_, gen_ptr.get());
        fusion_manager fmgr;
        builder::ir_builder_t bld;
        bld.push_scope();
        std::vector<for_loop> loops;
        bool status = gen_ptr->generate(
                parti->ctx_, config_data_.data_.get(), &fmgr, ins, outs, loops);
        assert(status);

        auto body = bld.pop_scope();
        copy_from_fmgr(fmgr);
        // bind outer_loop with axis
        if (!need_inner_query && !loops.empty() && (loops[0].get())
                && loops[0]->attr().has_key(stmt_attr_key::loop_axis_hint)) {
            auto bd_axis = loops[0]->attr().get<bound_axis>(
                    stmt_attr_key::loop_axis_hint);
            loops[0]->attr().remove(stmt_attr_key::loop_axis_hint);
            // init axis binder
            parti->ax_binder_.init(get_outputs()[0], bd_axis);
        }
        return body;
    };

    auto gen_ptr = create_generator();
    stmt body;
    // single core func is assicated with format so it needs to be query first.
    if (need_inner_query) {
        if (!parti->dyn_inter_) {
            parti->dyn_inter_
                    = std::make_shared<mixed_dyn_internal_info_t>(parti->ctx_);
        }
        fusion_manager fmgr;
        std::vector<for_loop> loops;
        func_t single_core_func = gen_ptr->get_single_core_func(
                parti->ctx_, config_data_.data_.get(), &fmgr, ins, outs, loops);
        parti->dyn_inter_->single_core_func_ = single_core_func;
        parti->dyn_inter_->single_core_func_extra_args_
                = gen_ptr->get_extra_args_from_func(single_core_func);
        copy_from_fmgr(fmgr);
        auto common_args = outs;
        common_args.insert(common_args.end(), ins.begin(), ins.end());
        std::for_each(
                common_args.begin(), common_args.end(), [](const expr &arg) {
                    arg->attr().set(attr_keys::always_trans, true);
                });
        parti->dyn_inter_->inter_func_extra_args_ = std::vector<expr> {
                builder::make_var(datatypes::pointer, "single_core_func")};
        parti->dyn_inter_->inter_call_extra_args_
                = std::vector<expr> {builder::make_func_addr(single_core_func)};
        // only for function signature.
        auto internal_func = builder::make_func(
                std::string("internal_func") + std::to_string(internal_idx++),
                common_args, builder::make_returns_unattached(true),
                datatypes::boolean);
        internal_func->attr().set(attr_keys::keep_func, true);
        parti->dyn_inter_->inter_func_ = internal_func;
        parti->dyn_inter_->mod_->add_func({single_core_func, internal_func});
        if (!parti->dyn_inter_->inter_funcs_param_.defined()) {
            parti->dyn_inter_->inter_funcs_param_ = builder::make_tensor(
                    "internal_funcs", {1}, datatypes::index);
            parti->dyn_inter_->num_func_ = 1;
            parti->dyn_inter_->inter_funcs_param_->attr().set(
                    attr_keys::no_index2var, true);
        } else {
            assert(parti->dyn_inter_->inter_funcs_param_.isa<tensor>());
            auto inter_tsr
                    = parti->dyn_inter_->inter_funcs_param_.static_as<tensor>();
            inter_tsr->dims_[0] = ++parti->dyn_inter_->num_func_;
        }
        auto inter_ptr = builder::make_reinterpret(
                builder::make_indexing(parti->dyn_inter_->inter_funcs_param_,
                        {parti->dyn_inter_->num_func_ - 1}),
                datatypes::pointer);
        inter_ptr->attr().set("prototype", internal_func);
        auto inter_call = make_expr<call_node>(inter_ptr, common_args);
        parti->dyn_inter_->inter_call_ = inter_call;
        body = builder::make_stmts_unattached(
                {builder::make_evaluate_unattached(inter_call)});
        // auto ret_params = outs;
        // ret_params.emplace_back(parti->dyn_inter_->inter_funcs_param_);
    } else {
        body = gen_body(gen_ptr);
    }
    auto func = builder::make_func(std::string(""), std::vector<expr> {},
            std::move(body), datatypes::boolean);
    return func;
}

void tunable_op_t::create_mixed_partition(mixed_parti_t *parti) {
    parti->buf_alloc_.allocate_buffer(this);
    std::vector<expr> ins, outs;
    std::tie(ins, outs) = parti->buf_alloc_.get_buffer(this);
    parti->func_ = get_func(parti, ins, outs);
}

void tunable_op_t::append_mixed_partition(mixed_parti_t *parti) {
    COMPILE_ASSERT(parti->ready_for_op(this),
            "No suitable anchor found for " << op_name_ << "_"
                                            << logical_op_id_);
    parti->buf_alloc_.allocate_buffer(this);
    parti->buf_alloc_.update_input_buffer_info(this);

    commit_into_anchor(parti->lookup_anchor_map(this).get());
    // update output buffer info after inner anchor created
    parti->buf_alloc_.update_output_buffer_info(this);
}

void tunable_op_t::search_anchor(mixed_parti_t *parti) {
    search_op_anchor_in_parti(this, parti);
}

void tunable_op_t::commit_into_anchor(fuse_anchor_map_t *committed_anchor) {
    auto parti = committed_anchor->get_binded_mxp();
    std::vector<expr> ins, outs;
    std::tie(ins, outs) = parti->buf_alloc_.get_buffer(this);
    // prepare slice
    std::vector<slice_range> ins_slice(get_inputs().size()),
            outs_slice(get_outputs().size());

    std::transform(get_inputs().begin(), get_inputs().end(), ins_slice.begin(),
            [&committed_anchor](const graph_tensor_ptr &gt) {
                auto slice_list = committed_anchor->fsmap_.get(gt);
                COMPILE_ASSERT(slice_list.size() == 1,
                        "multi-slice is not expected to tunable op");
                return slice_list[0];
            });
    std::transform(get_outputs().begin(), get_outputs().end(),
            outs_slice.begin(),
            [&committed_anchor](const graph_tensor_ptr &gt) {
                auto slice_list = committed_anchor->fsmap_.get(gt);
                COMPILE_ASSERT(slice_list.size() == 1,
                        "multi-slice is not expected to tunable op");
                return slice_list[0];
            });

    // prepare tptr for function call
    std::vector<expr> tptr_ins(ins.size()), tptr_outs(outs.size());
    std::transform(ins.begin(), ins.end(), ins_slice.begin(), tptr_ins.begin(),
            [&](const expr &tsr, const slice_range &range) {
                return transform_tsr2tptr_with_range(tsr, range);
            });
    std::transform(outs.begin(), outs.end(), outs_slice.begin(),
            tptr_outs.begin(), [&](const expr &tsr, const slice_range &range) {
                return transform_tsr2tptr_with_range(tsr, range);
            });

    // prepare strided tsr for function definition
    std::vector<expr> strd_ins(ins.size()), strd_outs(outs.size());
    std::transform(ins.begin(), ins.end(), ins_slice.begin(), strd_ins.begin(),
            [&](const expr &tsr, const slice_range &range) {
                return transform_tsr2stsr_with_range(tsr, range);
            });
    std::transform(outs.begin(), outs.end(), outs_slice.begin(),
            strd_outs.begin(), [&](const expr &tsr, const slice_range &range) {
                return transform_tsr2stsr_with_range(tsr, range);
            });

    node_ptr_map def_to_call_map;
    for (size_t i = 0; i < ins.size(); i++) {
        def_to_call_map[strd_ins[i].impl] = tptr_ins[i].impl;
    }
    for (size_t i = 0; i < outs.size(); i++) {
        def_to_call_map[strd_outs[i].impl] = tptr_outs[i].impl;
    }
    auto func = get_func(parti, strd_ins, strd_outs);

    // replace strided tensor with tensorptr
    mxp_replacer_t(def_to_call_map).replace_func(func);
    committed_anchor->commit_stmt(func->body_);

    // commit content id to anchor
    committed_anchor->append_content(static_cast<sc_op *>(this));
}

void tunable_op_t::set_config(const config_ptr &config) {
    config_data_ = config;
}

void tunable_op_t::set_config_if_empty(
        context_ptr ctx, body_generator_base_t *p) {
    if (!config_data_) { set_config(p->get_default_config(std::move(ctx))); }
}

config_ptr tunable_op_t::get_default_config(context_ptr ctx) {
    auto gen = this->create_generator();
    return gen->get_default_config(ctx);
}

config_ptr_vec tunable_op_t::get_dynamic_config_candidates(
        const context_ptr &ctx) {
    if (dyn_config_candidates_.empty()) {
        if (auto gen = create_generator()) {
            dyn_config_candidates_ = gen->get_dynamic_config_candidates(ctx);
        }
    }
    return dyn_config_candidates_;
}

impl_kind_map tunable_op_t::convert_config_candidates_to_impl_map(
        const config_ptr_vec &configs) {
    if (configs.empty()) { return impl_kind_map(); }
    auto gen = create_generator();
    impl_kind_map ret;
    ret.reserve(configs.size());
    for (int i = 0; i < static_cast<int>(configs.size()); i++) {
        auto &cfg = configs[i];
        ret.insert(std::make_pair(gen->convert_config_to_keys(cfg), i));
    }
    return ret;
}

std::vector<int> tunable_op_t::get_impl_dispatch_candidates(
        const context_ptr &ctx) {
    return get_dynamic_impl_dispatch_candidates(this, ctx);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
