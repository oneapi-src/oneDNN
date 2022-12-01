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
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/transform/func_inline.hpp>
#include <unordered_map>

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
            || !create_generator()->is_valid_config(
                    ctx, config_data_.data_.get())) {
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
            ctx, config_data_.data_.get(), nullptr, ins, outs, loops);
    assert(status);
    bld.push_returns(true);
    auto body = bld.pop_scope();
    gen_ptr->schedule_loops(ctx, config_data_.data_.get(), body, loops);
    auto args = outs;
    args.insert(args.end(), ins.begin(), ins.end());

    func->body_ = std::move(body);
    ret->add_func({func});
    ret->set_entry_func_idx(0);
    return ret;
}

func_t tunable_op_t::get_func(mixed_parti_t *parti,
        const std::vector<expr> &ins, const std::vector<expr> &outs) {
    auto gen_ptr = create_generator();
    set_config_if_empty(parti->ctx_, gen_ptr.get());
    fusion_manager fmgr;
    builder::ir_builder_t bld;
    bld.push_scope();
    std::vector<for_loop> loops;
    bool status = gen_ptr->generate(
            parti->ctx_, config_data_.data_.get(), &fmgr, ins, outs, loops);
    assert(status);

    auto body = bld.pop_scope();

    auto func = builder::make_func(std::string(""), std::vector<expr> {},
            std::move(body), datatypes::boolean);
    // record anchor
    extract_anchor_from_fmgr_to_parti(&fmgr, parti, outs, get_outputs(),
            parti->ready_for_op(this) ? parti->lookup_anchor_map(this)
                                      : nullptr);
    // bind outer_loop with axis
    if (!loops.empty() && (loops[0].get())
            && loops[0]->attr().has_key(stmt_attr_key::loop_axis_hint)) {
        auto bd_axis = loops[0]->attr().get<bound_axis>(
                stmt_attr_key::loop_axis_hint);
        loops[0]->attr().remove(stmt_attr_key::loop_axis_hint);
        // init axis binder
        parti->ax_binder_.init(get_outputs()[0], bd_axis);
    }

    return func;
}

void tunable_op_t::create_mixed_partition(mixed_parti_t *parti) {
    parti->buf_alloc_.allocate_buffer(this);
    std::vector<expr> ins, outs;
    std::tie(ins, outs) = parti->buf_alloc_.get_buffer(this);
    parti->func_ = get_func(parti, ins, outs);
}

void tunable_op_t::append_mixed_partition(mixed_parti_t *parti) {
    search_anchor(parti);
    COMPILE_ASSERT(parti->ready_for_op(this),
            "No suitable anchor found for " << op_name_ << "_"
                                            << logical_op_id_);
    parti->buf_alloc_.allocate_buffer(this);
    commit_into_anchor(parti);
}

void tunable_op_t::search_anchor(mixed_parti_t *parti) {
    search_op_anchor_in_parti(this, parti);
}

void tunable_op_t::commit_into_anchor(mixed_parti_t *parti) {
    std::vector<expr> ins, outs;
    std::tie(ins, outs) = parti->buf_alloc_.get_buffer(this);
    // prepare slice
    std::vector<slice_range> ins_slice(get_inputs().size()),
            outs_slice(get_outputs().size());

    auto committed_anchor = parti->lookup_anchor_map(this);
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

    std::unordered_map<expr, expr> def_to_call_map;
    for (size_t i = 0; i < ins.size(); i++) {
        def_to_call_map[strd_ins[i]] = tptr_ins[i];
    }
    for (size_t i = 0; i < outs.size(); i++) {
        def_to_call_map[strd_outs[i]] = tptr_outs[i];
    }
    // commit content id to anchor
    committed_anchor->append_content(static_cast<sc_op *>(this));
    parti->buf_alloc_.update_input_buffer_info(this);
    auto func = get_func(parti, strd_ins, strd_outs);
    // update output buffer info after inner anchor created
    parti->buf_alloc_.update_output_buffer_info(this);

    // replace strided tensor with tensorptr
    mxp_replacer_t(def_to_call_map).replace_func(func);
    committed_anchor->commit_stmt(func->body_);
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

} // namespace sc
