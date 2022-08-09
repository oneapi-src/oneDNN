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

#include <assert.h>
#include <atomic>
#include <numeric>
#include <unordered_map>

#include <algorithm>
#include <utility>
#include "fused_op.hpp"
#include "fusible_op.hpp"
#include "fusion_mgr.hpp"
#include "outer_loop_generator.hpp"
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/transform/parallel_workload_dispatch.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

namespace sc {

ir_module_ptr fusible_op_t::get_func(context_ptr ctx) {
    int base_idx = 0;
    if (auto binary_node = this->dyn_cast<binary_elementwise_op_t>()) {
        // if bc side (smaller side) is the lhs, we need to set base_idx to 1
        if (!binary_node->get_broadcast_input()) { base_idx = 1; }
    }
    outer_loop_generator_t gen(base_idx);
    return fusible_op_get_func(this, gen, ctx, true);
}

void fusible_op_t::create_mixed_partition(mixed_parti_t *parti) {
    parti->buf_alloc_.allocate_buffer(this);
    std::vector<expr> ins, outs;
    std::tie(ins, outs) = parti->buf_alloc_.get_buffer(this);
    fusion_manager fmgr;
    builder::ir_builder_t bld;
    bld.push_scope();
    std::vector<for_loop> loops;
    int base_idx = 0;
    if (auto binary_node = this->dyn_cast<binary_elementwise_op_t>()) {
        // if bc side (smaller side) is the lhs, we need to set base_idx to 1
        if (!binary_node->get_broadcast_input()) { base_idx = 1; }
    }
    bool use_output_mode = false;
    if (auto reo_op = this->dyn_cast<reorder_op_t>()) {
        use_output_mode = reo_op->support_output_loop();
        if (use_output_mode) {
            reo_op->attrs_.set(op_attr_key::break_pre_fuse, true);
        }
    }
    outer_loop_generator_t gen(base_idx, use_output_mode);
    if (attrs_.has_key("temp.mixed_partition_hint.sub_graph_ptr")) {
        fmgr.bind_graph(attrs_.get<sc_graph_t *>(
                "temp.mixed_partition_hint.sub_graph_ptr"));
    }
    bool status = gen.generate(parti->ctx_, nullptr, &fmgr, ins, outs, loops);
    COMPILE_ASSERT(status, "generate outer loops failed, please check");
    auto body = bld.pop_scope();
    parti->func_ = builder::make_func(std::string(""), std::vector<expr> {},
            std::move(body), datatypes::boolean);
    extract_anchor_from_fmgr_to_parti(&fmgr, parti,
            use_output_mode ? outs : ins,
            use_output_mode ? get_outputs() : get_inputs());
    append_mixed_partition(parti);
    // if last fanchor used, mark break post fusion, otherwise remove the last
    // fanchor
    if (parti->lookup_anchor_map(this) == parti->fanchors_.back()) {
        attrs_.set(op_attr_key::break_post_fuse, true);
        for (auto iter = parti->fanchors_.begin();
                iter < parti->fanchors_.end(); iter++) {
            if (iter == (parti->fanchors_.end() - 1)) continue;
            parti->clear_fanchor(*iter);
        }
        auto last_anchor = parti->fanchors_.back();
        parti->fanchors_.clear();
        parti->fanchors_ = {last_anchor};
    } else {
        parti->clear_fanchor(parti->fanchors_.back());
        parti->fanchors_.pop_back();
    }
}

void fusible_op_t::append_mixed_partition(mixed_parti_t *parti) {
    search_anchor(parti);
    COMPILE_ASSERT(parti->ready_for_op(this),
            "Not suitable anchor found for " << op_name_);

    if (!parti->empty()) {
        parti->buf_alloc_.allocate_buffer(this);
        parti->buf_alloc_.update_input_buffer_info(this, parti);
    }

    if (!parti->empty()) {
        int base_idx = 0;
        if (auto binary_node = this->dyn_cast<binary_elementwise_op_t>()) {
            // if bc side (smaller side) is the lhs, we need to set base_idx to
            // 1
            if (!binary_node->get_broadcast_input()) { base_idx = 1; }
        }
        auto base_gt = get_inputs()[base_idx];
        auto committed_anchor = parti->lookup_anchor_map(this);
        auto &fsmap = committed_anchor->fsmap_;
        auto slice_list = fsmap.get(base_gt);
        if (slice_list.size() == 1) {
            builder::ir_builder_t bld;
            bld.push_scope();
            anchor_loop_generator_t gen(base_gt, committed_anchor);
            // create_inner_anchor
            auto inner_anchor_num = gen.create_inner_anchor(parti->fanchors_);
            auto inner_ss = bld.pop_scope().checked_as<stmts>();
            // search inner anchor again
            search_anchor(parti);
            if (committed_anchor != parti->lookup_anchor_map(this)) {
                committed_anchor->commit_stmts(inner_ss);
            } else {
                // erase unused inner anchor
                parti->fanchors_.erase(
                        parti->fanchors_.end() - inner_anchor_num,
                        parti->fanchors_.end());
            }
        }
    }
    // update output buffer info after inner anchor created
    parti->buf_alloc_.update_output_buffer_info(this, parti);
    commit_into_anchor(parti);
}

void fusible_op_t::search_anchor(mixed_parti_t *parti) {
    search_op_anchor_in_parti(this, parti);
}

void fusible_op_t::commit_into_anchor(mixed_parti_t *parti) {
    std::vector<expr> in_tsrs, out_tsrs;
    std::tie(in_tsrs, out_tsrs) = parti->buf_alloc_.get_buffer(this);
    std::vector<std::vector<tensor_slice>> inputs(in_tsrs.size()),
            outputs(out_tsrs.size());

    auto committed_anchor = parti->lookup_anchor_map(this);
    std::transform(get_inputs().begin(), get_inputs().end(), in_tsrs.begin(),
            inputs.begin(),
            [&committed_anchor](const graph_tensor_ptr &gt, const expr &tsr) {
                std::vector<tensor_slice> multi_tsl;
                for (auto &range : committed_anchor->fsmap_.get(gt)) {
                    multi_tsl.emplace_back(
                            tensor_slice(tsr, slice_range(range)));
                }
                return multi_tsl;
            });
    std::transform(get_outputs().begin(), get_outputs().end(), out_tsrs.begin(),
            outputs.begin(),
            [&committed_anchor](const graph_tensor_ptr &gt, const expr &tsr) {
                std::vector<tensor_slice> multi_tsl;
                if (!committed_anchor->fsmap_.get(gt).empty()) {
                    for (auto &range : committed_anchor->fsmap_.get(gt)) {
                        multi_tsl.emplace_back(
                                tensor_slice(tsr, slice_range(range)));
                    }
                } else {
                    COMPILE_ASSERT(gt->producer_owner_->isa<reorder_op_t>(),
                            "only reorder op support this case")
                    multi_tsl.emplace_back(tensor_slice(tsr));
                }
                return multi_tsl;
            });
    auto in_slice_size = inputs[0].size();
    COMPILE_ASSERT(in_slice_size, "No input slice found for " << op_name_);

    // generate IR
    builder::ir_builder_t bld;
    bld.push_scope();
    // unwrapper tensor slice, for compute_block, it just accpet single
    // tensor_slice
    for (size_t i = 0; i < in_slice_size; i++) {
        std::vector<const tensor_slice *> new_inputs_ptr(inputs.size());
        std::vector<tensor_slice *> new_outputs_ptr(outputs.size());
        std::transform(inputs.begin(), inputs.end(), new_inputs_ptr.begin(),
                [&i](std::vector<tensor_slice> &ins) { return &ins[i]; });

        std::transform(outputs.begin(), outputs.end(), new_outputs_ptr.begin(),
                [&i](std::vector<tensor_slice> &out) { return &out[i]; });
        compute_block(parti->ctx_, new_outputs_ptr, new_inputs_ptr);
    }
    auto ss = bld.pop_scope().checked_as<stmts>();

    // commit into anchor
    committed_anchor->commit_stmts(ss);
}

void fusible_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    if (this->isa<constant_op_t>()) {
        out_formats.push_back({info_.outputs_[0]->details_.get_format()});
    } else {
        out_formats.push_back({info_.inputs_[0]->details_.get_format()});
    }
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

size_t fusible_op_t::compute_workload(const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    size_t wkld = 0UL;
    auto accumulate_workload
            = [&wkld](size_t weight, const shape_dtype_pair &v) {
                  auto &dtype = v.second;
                  wkld += utils::get_sizeof_type(dtype) * weight;
              };
    std::for_each(ins.begin(), ins.end(),
            std::bind(accumulate_workload,
                    static_cast<size_t>(
                            op_traits::workload_computable_t::read_weight),
                    std::placeholders::_1));
    std::for_each(outs.begin(), outs.end(),
            std::bind(accumulate_workload,
                    static_cast<size_t>(
                            op_traits::workload_computable_t::write_weight),
                    std::placeholders::_1));
    return wkld;
}

size_t fusible_op_t::compute_fusible_workload(const context_ptr &ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    if (is_dynamic()) { return memory_access_threshold_per_thread; }
    std::vector<shape_dtype_pair> wkld_ins, wkld_outs;
    wkld_ins.resize(inputs.size());
    wkld_outs.resize(dst.size());
    auto get_shape_dtype_pair = [](const tensor_slice *v) {
        return std::make_pair(get_expr_to_dims(v->shape_), v->get_base_dtype());
    };
    std::transform(inputs.begin(), inputs.end(), wkld_ins.begin(),
            get_shape_dtype_pair);
    std::transform(
            dst.begin(), dst.end(), wkld_outs.begin(), get_shape_dtype_pair);
    return compute_workload(wkld_ins, wkld_outs);
}

input_op::input_op(const sc_dims &dims, sc_data_type_t dtype) {
    op_name_ = "input";
    info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
            this, sc_data_format_t(), dims, dtype));
}

input_op::input_op(const logical_tensor_t &lt) {
    op_name_ = "input";
    info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this, lt));
}

input_op::input_op(const std::vector<graph_tensor_ptr> &outs) {
    info_.outputs_ = outs;
    for (auto &out : info_.outputs_) {
        out->producer_owner_ = this;
    }
    op_name_ = "input";
}

void input_op::initialize_dynamic_placeholder() {
    for (auto &out : info_.outputs_) {
        auto plain_dims = out->details_.get_plain_dims();
        for (auto &it : plain_dims) {
            if (it == dimensions::dynamic_any) {
                it = get_owner_graph().get_next_dynamic_placeholder();
            }
        }
        out->details_.set_plain_dims(plain_dims);
    }
}

void input_op::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
}

output_op::output_op(const graph_tensor_ptr &v) {
    info_.inputs_.emplace_back(v);
    op_name_ = "output";
}

output_op::output_op(const std::vector<graph_tensor_ptr> &in) {
    info_.inputs_ = in;
    op_name_ = "output";
}

void output_op::prepare_fusion_data(fdata_map &fdmap) {
    assert(info_.outputs_.empty() && "Wrong op output size.\n");
    auto &inputs = info_.inputs_[0];
    auto &outdetail = fdmap.get(inputs);
    outdetail.need_alloc_ = false;
}

// special handling for union values
bool constant_op_t::compare_contents(const sc_op *other) const {
    COMPILE_ASSERT(attrs_.has_key("values") && attrs_.has_key("dtype"),
            "expecting values and dtype in attr");
    COMPILE_ASSERT(
            other->attrs_.has_key("values") && other->attrs_.has_key("dtype"),
            "expecting values and dtype in attr");
    auto dtype = attrs_.get<sc_data_type_t>("dtype");
    if (other->attrs_.get<sc_data_type_t>("dtype") != dtype) { return false; }
    if (attrs_.has_key("format")) {
        if (!other->attrs_.has_key("format")) { return false; }
        if (other->attrs_.get<sc_data_format_t>("format")
                != attrs_.get<sc_data_format_t>("format")) {
            return false;
        }
    }
    auto &vals = attrs_.get<std::shared_ptr<static_data_t>>("values");
    auto &vals2 = other->attrs_.get<std::shared_ptr<static_data_t>>("values");
    if (vals->size_ != vals2->size_) { return false; }

    switch (get_type_category_nothrow(dtype)) {
        case CATE_FLOAT:
            for (size_t i = 0; i < vals->size_ / 4; i++) {
                if (static_cast<float *>(vals->data_)[i]
                        != static_cast<float *>(vals2->data_)[i]) {
                    return false;
                }
            }
            break;
        case CATE_INT:
        case CATE_UINT:
            for (size_t i = 0; i < vals->size_ / 4; i++) {
                if (static_cast<uint32_t *>(vals->data_)[i]
                        != static_cast<uint32_t *>(vals2->data_)[i]) {
                    return false;
                }
            }
            break;
        default:
            throw std::runtime_error("Met unexpected dtype for constant");
            break;
    }
    return true;
}

size_t constant_op_t::hash_contents() const {
    size_t seed = 0;
    COMPILE_ASSERT(attrs_.has_key("values") && attrs_.has_key("dtype"),
            "expecting values and dtype in attr");
    if (attrs_.has_key("format")) {
        hash_combine(seed, attrs_.get<sc_data_format_t>("format"));
    }
    auto &vals = attrs_.get<std::shared_ptr<static_data_t>>("values");

    for (size_t i = 0; i < vals->size_; i++) {
        hash_combine(seed, static_cast<char *>(vals->data_)[i]);
    }

    return seed;
}

void constant_op_t::reset_const_values() {
    if (attrs_.has_key("temp.var") && attrs_.has_key("temp.val/var")) {
        int K = static_cast<int>(
                attrs_.get<std::shared_ptr<VConst>>("temp.var")->var_);
        int base_val = attrs_.get<int>("temp.val/var");
        // update private member
        const_values_ = std::make_shared<static_data_t>(
                std::vector<int> {base_val * K});
        // update attr
        attrs_.set("values", const_values_);
    }
}

constant_op_t::constant_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT(ins.empty(), "No op input.\n");
    COMPILE_ASSERT(attrs.has_key("values") && attrs.has_key("dtype")
                    && attrs.has_key("plain_dims"),
            "expecting values, format and dtype in attr");
    op_name_ = "constant";
    sc_data_format_t format
            = attrs.get_or_else("format", sc_data_format_t(format_kinds::A));
    attrs_ = attrs;
    const_values_ = attrs.get<std::shared_ptr<static_data_t>>("values");
    sc_data_type_t dtype = attrs.get<sc_data_type_t>("dtype");
    sc_dims plain_dims = attrs.get<sc_dims>("plain_dims");

    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, logical_tensor_t {format, plain_dims, dtype}));
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
    }
}

// todo: support tensor expr
constant_op_t::constant_op_t(std::shared_ptr<static_data_t> v,
        sc_data_type_t dtype, const sc_dims &plain_dims,
        const sc_data_format_t &format) {
    const_values_ = std::move(v);
    info_.outputs_.emplace_back(
            std::make_shared<graph_tensor>(this, format, plain_dims, dtype));
    info_.outputs_[0]->details_.dtype_ = dtype;
    info_.outputs_[0]->details_.set_plain_dims(plain_dims);
    attrs_.set("dtype", dtype);
    attrs_.set("values", const_values_);
    attrs_.set("plain_dims", plain_dims);
    attrs_.set("format", format);
    op_name_ = "constant";
}

void constant_op_t::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
    auto &output = info_.outputs_[0];
    auto &outdetail = fdmap.get(output);
    auto blocking_dims = get_constant_blocking_dims();
    outdetail.need_alloc_ = true;
}

OP_REGISTER(constant_op_t, constant)

} // namespace sc
