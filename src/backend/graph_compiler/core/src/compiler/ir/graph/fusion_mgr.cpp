/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#include "fusion_mgr.hpp"
#include <assert.h>
#include <atomic>
#include <iostream>
#include <string>
#include <utility>
#include "../easy_build.hpp"
#include "fusible_op.hpp"
#include "fusion_mgr_utils.hpp"
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <compiler/ir/visitor.hpp>
#include <microkernel/builtin.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <util/any_map.hpp>

namespace sc {

static bool slice_full_on_axes(
        const sc_dims &dim, slice_range ranges, const std::vector<int> &axes) {
    for (auto &ax : axes) {
        if (!ranges[ax].first.isa<constant>()
                || !ranges[ax].second.isa<constant>()) {
            return false;
        }
        if (get_const_as_int(ranges[ax].first.checked_as<constant>()) != 0
                || get_const_as_int(ranges[ax].second.checked_as<constant>())
                        != dim[ax]) {
            return false;
        }
    }
    return true;
}

bool fusion_data::is_contiguous() {
    if (tsr_slice_list_.size() > 1) return false;
    auto &tsr_slice = tsr_slice_list_[0];
    return tsr_slice.is_full();
}

fusion_data &fusion_anchor_data::get(graph_tensor *v) {
    auto itr = this->datamap_.find(v);
    if (itr != this->datamap_.end()) { return itr->second; }
    auto &ret = this->datamap_[v];
    ret.shape_ = dims_to_expr(v->details_.get_blocking_dims());
    return ret;
}

int fusion_anchor_data::get_input_idx(sc_op *v) const {
    assert(input_idx_map_);
    auto itr = input_idx_map_->find(v);
    assert(itr != input_idx_map_->end());
    return itr->second;
}
int fusion_anchor_data::get_output_idx(sc_op *v) const {
    assert(output_idx_map_);
    auto itr = output_idx_map_->find(v);
    assert(itr != output_idx_map_->end());
    return itr->second;
}

fusion_data &fusion_anchor_data::get(const graph_tensor_ptr &v) {
    return get(v.get());
}

template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>() {
    auto ret = std::make_shared<input_op>();
    graph_.add(ret);
    input_idx_map_[ret.get()] = input_op_count_++;
    return ret;
}
template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(
        sc_dims &&dims, const sc_data_type_t &dtype) {
    auto ret = std::make_shared<input_op>(dims, dtype);
    graph_.add(ret);
    input_idx_map_[ret.get()] = input_op_count_++;
    return ret;
}

template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(
        const logical_tensor_t &lt) {
    auto ret = std::make_shared<input_op>(lt);
    graph_.add(ret);
    input_idx_map_[ret.get()] = input_op_count_++;
    return ret;
}

template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(logical_tensor_t &lt) {
    return make<input_op>(const_cast<const logical_tensor_t &>(lt));
}

sc_op_ptr fusion_manager::make_input(const std::vector<graph_tensor_ptr> &tsr) {
    auto ret = std::make_shared<input_op>(tsr);
    graph_.add(ret);
    input_idx_map_[ret.get()] = input_op_count_++;
    return ret;
}

template <>
std::shared_ptr<input_op> fusion_manager::make<input_op>(
        logical_tensor_t &&lt) {
    return make<input_op>(static_cast<const logical_tensor_t &>(lt));
}

template <>
std::shared_ptr<output_op> fusion_manager::make<output_op>(
        const graph_tensor_ptr &arg) {
    auto ret = std::make_shared<output_op>(arg);
    graph_.add(ret);
    output_idx_map_[ret.get()] = output_op_count_++;
    return ret;
}

void fusion_manager::add(fusion_op_ptr node) {
    graph_.add(std::move(node));
}

void fusion_manager::put_input_first(input_op *inp) {
    COMPILE_ASSERT(input_idx_map_.find(inp) != input_idx_map_.end(),
            "Cound not found given input op in current fusion manager graph");
    int inp_idx = input_idx_map_.find(inp)->second;
    for (auto &cur_inp : graph_.get_input_ops()) {
        if (input_idx_map_[cur_inp.get()] < inp_idx) {
            input_idx_map_[cur_inp.get()]++;
        } else if (cur_inp.get() == inp) {
            input_idx_map_[cur_inp.get()] = 0;
        }
    }
}

fusion_manager::fusion_manager(fusion_manager &&other)
    : input_op_count_(other.input_op_count_)
    , output_op_count_(other.output_op_count_)
    , alloc_tensor_count_(other.alloc_tensor_count_)
    , graph_(std::move(other.graph_))
    , allocated_tensors_(std::move(other.allocated_tensors_))
    , input_idx_map_(std::move(other.input_idx_map_))
    , output_idx_map_(std::move(other.output_idx_map_)) {}

/**
 * This function pre-aloocate tensor, whether used indeedly is decided in
 * schedule tensor pass
 * */
void fusion_manager::allocate_tensor(
        graph_tensor_ptr output, fusion_anchor_data &fdata) {
    auto &fdetail = fdata.get(output);
    fusible_op_t *fop = output->producer_owner_->dyn_cast<fusible_op_t>();
    tensor tsr;
    auto allocate_ =
            [&](const std::string &name, const std::vector<expr> &shapes,
                    sc::address_space addrspace = sc::address_space::automatic,
                    const std::shared_ptr<static_data_t> &init_value = nullptr,
                    bool global = false) {
                bool forbidden_shrink = false;
                // This situation can not shrink output
                if (fop->isa<reorder_op_t>()) {
                    if (fop->dyn_cast<reorder_op_t>()->check_padding()
                            || fdetail.original_ranges_list_.size() > 1) {
                        forbidden_shrink = true;
                    }
                }
                for (auto &range : fdetail.original_ranges_list_) {
                    if (!(forbidden_shrink && tsr.defined())) {
                        tsr = builder::make_tensor(
                                name + std::to_string(alloc_tensor_count_++),
                                shapes, output->details_.dtype_, addrspace,
                                init_value)
                                      .checked_as<tensor>();
                        if (global) {
                            auto def = builder::make_var_tensor_def_unattached(
                                    tsr, linkage::private_global)
                                               .static_as<define>();
                            global_defines_.emplace_back(def);
                        } else {
                            allocated_tensors_.emplace_back(tsr);
                        }
                    }
                    if (!forbidden_shrink && !tsr->init_value_) {
                        // The tensor to shrink should not have init value
                        tsr->attr()[tensor_shrinker_attrs::should_shrink]
                                = tensor_shrinker_t::shrink_info_t {
                                        /*base*/ get_slice_idx(range),
                                        /*shape*/ get_slice_shape(range),
                                        stmts()};
                    }
                    fdetail.tsr_slice_list_.emplace_back(
                            tensor_slice(tsr, slice_range(range)));
                }
            };
    // TODO(xxx): remove this reorder judgement
    if (output->producer_owner_->dyn_cast<reorder_op_t>()) {
        allocate_("_reorder_buf_",
                dims_to_expr(output->details_.get_blocking_dims()));
    } else if (auto const_op
            = output->producer_owner_->dyn_cast<constant_op_t>()) {
        auto const_value = const_op->get_constant_values();
        allocate_("_const_buf_", fdetail.shape_, address_space::automatic,
                const_value, true);
    } else if (output->producer_owner_->dyn_cast<unary_elementwise_op_t>()
            || output->producer_owner_->dyn_cast<binary_elementwise_op_t>()
            || output->producer_owner_->dyn_cast<reduce_op_t>()) {
        allocate_("_fuse_buf_",
                dims_to_expr(output->details_.get_blocking_dims()));
    }
    // TODO(xxx): adapte to other movement type. As the workaround, we just
    // maintain current style
    else {
        tsr = builder::make_tensor(std::string("_fuse_buf_")
                        + std::to_string(alloc_tensor_count_++),
                fdetail.shape_, output->details_.dtype_)
                      .checked_as<tensor>();
        allocated_tensors_.emplace_back(tsr);
        fdetail.tsr_slice_list_ = {tensor_slice(tsr)};
    }
}

std::vector<sc_op_ptr> fusion_manager::dispatch_fusion_anchor(
        const context_ptr &ctx, std::vector<fusion_anchor_data> &fdata_list,
        const std::vector<expr> &outs, const std::vector<expr> &inargs) {
    // to be decided op list contains ops that have not found suitable anchor
    std::vector<sc_op_ptr> tbd_op_list;
    if (graph_.empty()) { return tbd_op_list; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");
    COMPILE_ASSERT(
            !output_anchor_position_.empty() && !output_anchor_slice_.empty(),
            "no output anchor found, please create them before commit");
    auto dispatcher = [&](const sc_op_ptr &cur, int anchor_id) -> bool {
        if (cur->isa<reduce_op_t>()) {
            auto rd_axes = cur->dyn_cast<reduce_op_t>()->get_rd_axis();
            auto original_ranges = fdata_list[anchor_id]
                                           .get(cur->get_inputs()[0])
                                           .original_ranges_list_;
            auto &src_dim = cur->get_inputs()[0]->details_.get_blocking_dims();

            // check the slice range whether meet the least demand of reduce op
            for (auto &src_range : original_ranges) {
                if (!slice_full_on_axes(src_dim, src_range, rd_axes)) {
                    if (!(std::find(tbd_op_list.begin(), tbd_op_list.end(), cur)
                                != tbd_op_list.end())) {
                        cur->attrs_.set(op_attr_key::fused_mode_hint,
                                op_attr_key::break_pre_fuse);
                        tbd_op_list.emplace_back(cur);
                    }
                    return false;
                }
            }
            cur->dyn_cast<fusible_op_t>()->anchor_id = anchor_id;
        } else {
            cur->dyn_cast<fusible_op_t>()->anchor_id = 0;
        }
        return true;
    };

    for (size_t anchor_id = 0; anchor_id < output_anchor_position_.size();
            anchor_id++) {
        // skip
        if (anchor_id > 0) do_prepare_fusion_data(ctx, fdata_list, anchor_id);
        std::vector<sc_op_ptr> failed_infer_ops;
        do_infer_slice_ranges(fdata_list, failed_infer_ops, anchor_id);
        if (!failed_infer_ops.empty()) {
            tbd_op_list.insert(tbd_op_list.end(), failed_infer_ops.begin(),
                    failed_infer_ops.end());
            return tbd_op_list;
        }
        do_allocate_tensor(fdata_list, anchor_id, outs, inargs);
        if (anchor_id == 0) {
            for (auto &cur : sorted_ops_) {
                dispatcher(cur, anchor_id);
            }
        } else {
            for (auto iter = tbd_op_list.begin(); iter != tbd_op_list.end();) {
                bool is_success = dispatcher(*iter, anchor_id);
                if (is_success) {
                    // clear hint if necessary
                    if ((*iter)->attrs_.has_key(op_attr_key::fused_mode_hint)) {
                        (*iter)->attrs_.remove(op_attr_key::fused_mode_hint);
                    }
                    iter = tbd_op_list.erase(iter);
                    continue;
                }
                iter++;
            }
        }
        // record max anchor fusion manager use.
        max_anchor_ = anchor_id;
        if (tbd_op_list.empty()) break;
    }
    return tbd_op_list;
}

void fusion_manager::do_query(const context_ptr &ctx,
        std::vector<fusion_anchor_data> &fdata_list, bool legacy_mode,
        int anchor_id) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");
    for (auto &cur : sorted_ops_) {
        auto src = output_anchor_slice_[anchor_id].first;
        auto dst = output_anchor_slice_[anchor_id].second;
        cur->dyn_cast<fusible_op_t>()->prepare_fusion_data(
                ctx, src, dst, fdata_list[anchor_id]);
    }
}

void fusion_manager::do_infer_slice_ranges(
        std::vector<fusion_anchor_data> &fdata_list,
        std::vector<sc_op_ptr> &failed_ops, int anchor_id) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");

    auto validate_slice_range = [&fdata_list, &failed_ops, &anchor_id](
                                        const sc_op_ptr &fop) -> bool {
        if (fop->isa<input_op>() || fop->isa<output_op>()
                || fop->isa<constant_op_t>())
            return true;
        // check legalize, except for reorder_op, all slice of inputs and
        // outputs should be equal
        size_t common_slice_size = fdata_list[anchor_id]
                                           .get(fop->get_inputs()[0])
                                           .original_ranges_list_.size();
        if (common_slice_size == 0) {
            // usually occurs in pre-op fusion
            if (fop->isa<tensor_view_op_t>() || fop->isa<reorder_op_t>()) {
                failed_ops.emplace_back(fop);
                return false;
            } else {
                COMPILE_ASSERT(common_slice_size > 0,
                        "input slice size is expected larger than 0 for "
                                << fop->op_name_ << " op")
            }
        }
        if (common_slice_size > 1 && fop->isa<reorder_op_t>()) {
            failed_ops.emplace_back(fop);
            return false;
        }
        for (auto &ins : fop->get_inputs()) {
            if (common_slice_size
                    != fdata_list[anchor_id]
                               .get(ins)
                               .original_ranges_list_.size()) {
                failed_ops.emplace_back(fop);
                return false;
            }
        }
        for (auto &out : fop->get_outputs()) {
            if (common_slice_size
                    != fdata_list[anchor_id]
                               .get(out)
                               .original_ranges_list_.size()) {
                // Currently, only reorder op will generate multi-slice
                if (!fop->isa<reorder_op_t>()) {
                    failed_ops.emplace_back(fop);
                    return false;
                } else if (fdata_list[anchor_id]
                                   .get(out)
                                   .original_ranges_list_.empty()) {
                    for (auto &user : fop->get_outputs()[0]->uses_) {
                        if (user.second->isa<output_op>()) {
                            continue;
                        } else {
                            user.second->attrs_.set(
                                    op_attr_key::fused_mode_hint,
                                    op_attr_key::break_pre_fuse);
                            failed_ops.emplace_back(user.second);
                        }
                    }
                    if (!failed_ops.empty()) return false;
                }
            }
        }
        return true;
    };

    // To enable pre-op fusion, it is allowed that infer ops list maybe not a
    // topology sequence.
    auto infer_ops_list = op_sorting_visitor_t::sort_by_rules(graph_,
            fdata_list, {op_sorting_visitor_t::sort_rule::preop_fusion});
    for (auto &cur : infer_ops_list) {
        cur->dyn_cast<fusible_op_t>()->infer_slice_ranges(
                fdata_list[anchor_id]);
        if (!validate_slice_range(cur)) return;
    }
}

void fusion_manager::do_prepare_fusion_data(const context_ptr &ctx,
        std::vector<fusion_anchor_data> &fdata_list, int anchor_id) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");
    for (auto &cur : sorted_ops_) {
        auto src = output_anchor_slice_[anchor_id].first;
        auto dst = output_anchor_slice_[anchor_id].second;
        cur->dyn_cast<fusible_op_t>()->prepare_fusion_data(
                ctx, src, dst, fdata_list[anchor_id]);
    }
}

// reset first access of buffer reuse hint for special cases like the input of
// tunable op and connection tensors between two anchors.
void reset_buffer_reuse_first_access_hint(const expr &tsr) {
    const int64_t reset_first_access_tick = 0;
    if (tsr.defined()
            && tsr->attr().has_key(attr_keys::hint_first_access_tick)) {
        tsr->attr().set(
                attr_keys::hint_first_access_tick, reset_first_access_tick);
    }
}

// Set buffer reuse hint(hint_first_access_tick and hint_last_access_tick) in
// tensor attribute for buffer schedule based on graph of fusion manager. When
// hint_first_access_tick and hint_last_access_tick are set, the `tsr` can be
// reused and its lifetime is between [hint_first_access_tick,
// hint_last_access_tick]. first_access of a tensor is defined as tick of tensor
// create. last_access of a tensor is defined as maximum tick of its uses.
// access_has_updated is a boolean, used when tensor_slice_list.size() > 1
void set_buffer_reuse_hint(int64_t &hint_tick, fusion_anchor_data &fdata,
        const sc_op_ptr &node, const expr &tsr,
        bool access_has_updated = false) {
    // tick self add, if node is input op, last access == 1
    if (!access_has_updated) { hint_tick++; }
    // update inp tsrs' last access to maximum last access
    for (auto &in : node->get_inputs()) {
        auto &in_detail = fdata.get(in);
        if (!in_detail.tsr_slice_list_.empty()) {
            for (auto &tsr_slice : in_detail.tsr_slice_list_) {
                auto in_tsr = tsr_slice.tptr_->base_->ptr_;
                in_tsr->attr().set(attr_keys::hint_last_access_tick, hint_tick);
            }
        }
    }
    if (node->isa<reduce_op_t>() || node->isa<movement_op_t>()) { return; }
    if (tsr.defined()) {
        // current tsr's first access == last access
        tsr->attr().set(attr_keys::hint_first_access_tick, hint_tick);
        tsr->attr().set(attr_keys::hint_last_access_tick, hint_tick);
    }
}

void fusion_manager::do_allocate_tensor(
        std::vector<fusion_anchor_data> &fdata_list, int anchor_id,
        const std::vector<expr> &outs, const std::vector<expr> &inargs) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");
    auto ths = this;
    auto alloc_func = [ths, &fdata_list, &outs, &inargs](
                              const sc_op_ptr &cur_node,
                              const std::vector<sc::tensor_slice> &src,
                              const std::vector<sc::tensor_slice> &dst,
                              int anchor_id, int64_t &hint_tick) {
        auto cur = cur_node->dyn_cast<fusible_op_t>();
        auto &fdata = fdata_list[anchor_id];
        if (dynamic_cast<input_op *>(cur)) {
            // if it is the input node

            // if there is only one input node and only one output node, we need
            // to copy the src tensor to dst tensor, not inplemented
            assert(ths->graph_.ops_.size() > 2);
            auto input_cur = static_cast<input_op *>(cur);
            auto &input_cur_out_detail = fdata.get(input_cur->get_outputs()[0]);
            int input_idx = fdata.get_input_idx(input_cur);
            int arg_idx = input_idx - static_cast<int>(src.size());
            // for input node, use the input tensor
            if (arg_idx >= 0) {
                expr tsr;
                // query if inargs is given by outside
                if (!inargs.empty()) {
                    tsr = inargs[arg_idx];
                } else {
                    auto arg_tsr = builder::make_tensor(
                            std::string("arg_tsr_") + std::to_string(arg_idx),
                            dims_to_expr(
                                    input_cur->get_outputs()[0]
                                            ->details_.get_blocking_dims()),
                            input_cur->get_outputs()[0]->details_.dtype_);
                    input_cur->info_.args_.emplace_back(arg_tsr);
                    tsr = arg_tsr;
                }
                for (size_t i = 0;
                        i < input_cur_out_detail.original_ranges_list_.size();
                        i++) {
                    auto &range = input_cur_out_detail.original_ranges_list_[i];
                    input_cur_out_detail.tsr_slice_list_.emplace_back(
                            tensor_slice(tsr, slice_range(range)));
                }
            } else {
                // TODO(xxx): need to consider multi-slice from create_anchor
                // stage
                input_cur_out_detail.tsr_slice_list_ = {src[input_idx]};
                // may reuse input buffer, e.g. originalout
                set_buffer_reuse_hint(hint_tick, fdata, cur_node,
                        src[input_idx].tptr_->base_->ptr_);
                reset_buffer_reuse_first_access_hint(
                        src[input_idx].tptr_->base_->ptr_);
            }
            return;
        } else if (dynamic_cast<output_op *>(cur)) {
            // if it is the output node

            // if there is only one input node and only one output node, we need
            // to copy the src tensor to dst tensor, not inplemented
            assert(ths->graph_.ops_.size() > 2);
            auto output_cur = static_cast<output_op *>(cur);
            auto &output_cur_in_detail = fdata.get(output_cur->get_inputs()[0]);
            int output_idx = fdata.get_output_idx(output_cur);

            // for output node, use the output tensor
            if (!dst.empty()) {
                // TODO(xxx): need to consider multi-slice from create_anchor
                // stage
                output_cur_in_detail.tsr_slice_list_ = {dst[output_idx]};
            } else {
                expr tsr;
                // query if outs is given by outside
                if (!outs.empty()) {
                    tsr = outs[fdata.get_output_idx(output_cur)];
                } else {
                    // Here, it will not be pushed into allocated_tensors_,
                    // because
                    // it will be replaced in final IR generation
                    auto arg_tsr = builder::make_tensor(
                            std::string("output") + std::to_string(output_idx),
                            dims_to_expr(
                                    cur->get_inputs()[0]
                                            ->details_.get_blocking_dims()),
                            cur->get_inputs()[0]->details_.dtype_);
                    cur->info_.args_ = {arg_tsr};
                    tsr = arg_tsr;
                }

                // output will inplace the last buffer, so we need to clear it
                // firstly.
                output_cur_in_detail.tsr_slice_list_.clear();
                if (output_cur->get_inputs()[0]
                                ->producer_owner_->isa<reorder_op_t>()
                        && output_cur_in_detail.original_ranges_list_.empty()) {
                    output_cur_in_detail.tsr_slice_list_.emplace_back(
                            tensor_slice(tsr));
                } else {
                    for (size_t i = 0; i
                            < output_cur_in_detail.original_ranges_list_.size();
                            i++) {
                        auto &range
                                = output_cur_in_detail.original_ranges_list_[i];
                        output_cur_in_detail.tsr_slice_list_.emplace_back(
                                tensor_slice(tsr, slice_range(range)));
                    }
                }
            }
            // update tensors' last_access
            set_buffer_reuse_hint(hint_tick, fdata, cur_node, expr());
            return;
        } else if (dynamic_cast<tensor_view_op_t *>(cur)) {
            auto &cur_in_detail = fdata.get(cur->get_inputs()[0]);
            auto &cur_out_detail = fdata.get(cur->get_outputs()[0]);
            auto reshape_cur = static_cast<tensor_view_op_t *>(cur);
            cur_out_detail.need_alloc_ = false;
            cur_out_detail.tsr_slice_list_ = cur_in_detail.tsr_slice_list_;
            COMPILE_ASSERT(cur_out_detail.tsr_slice_list_.size()
                            == cur_out_detail.original_ranges_list_.size(),
                    "Unexpected size found");
            for (size_t i = 0; i < cur_out_detail.tsr_slice_list_.size(); i++) {
                auto &tsr_slice = cur_out_detail.tsr_slice_list_[i];
                auto &range = cur_out_detail.original_ranges_list_[i];
                // get based tensor
                auto based_tsr = tsr_slice.get_real_tensor();
                // get reshaped base tensor
                auto reshaped_base_tsr = builder::tensor_ptr(based_tsr,
                        std::vector<expr>(based_tsr->dims_.size(), 0),
                        dims_to_expr(reshape_cur->get_shapes()));
                // attach shrink info for reshaped base
                // tensor(tensorptr) if necessary
                if (based_tsr->attr().has_key(
                            tensor_shrinker_attrs::should_shrink)
                        || based_tsr->attr().has_key(
                                tensor_shrinker_attrs::may_shrink)) {
                    reshaped_base_tsr
                            ->attr()[tensor_shrinker_attrs::should_shrink]
                            = tensor_shrinker_t::shrink_info_t {
                                    /*base*/ get_slice_idx(range),
                                    /*shape*/ get_slice_shape(range), stmts()};
                }
                tsr_slice = tensor_slice(reshaped_base_tsr, std::move(range));
                // TODO(xxx): can be removed in the future
                cur_out_detail.shape_ = tsr_slice.get_shape();
            }
            // set buffer reuse hint for loop lifetime analysis
            if (!cur_out_detail.tsr_slice_list_.empty()) {
                bool access_has_updated = false;
                for (auto &tsr_slice : cur_out_detail.tsr_slice_list_) {
                    set_buffer_reuse_hint(hint_tick, fdata, cur_node,
                            cur_out_detail.need_alloc_
                                    ? tsr_slice.tptr_->base_->ptr_
                                    : expr(),
                            access_has_updated);
                    access_has_updated = true;
                }
            }
            return;
        } else if (dynamic_cast<constant_op_t *>(cur)) {
            auto &cur_out_detail = fdata.get(cur->get_outputs()[0]);
            if (cur_out_detail.shape_.empty()) {
                // TODO(xxx): default use index-0
                for (auto &r : cur_out_detail.original_ranges_list_[0]) {
                    cur_out_detail.shape_.emplace_back(r.second);
                }
            }
            ths->allocate_tensor(cur->get_outputs()[0], fdata);
            return;
        } else {
            // for every sub-node that the current node can share buffer
            // with
            auto share_map = cur->get_info().tensor_share_info_;
            auto &outputs = cur->get_outputs();
            auto &inputs = cur->get_inputs();
            for (unsigned i = 0; i < outputs.size(); i++) {
                auto &out_i_detail = fdata.get(outputs[i]);
                for (auto j : share_map[i]) {
                    auto &in_j_detail = fdata.get(inputs[j]);
                    if (in_j_detail.use_count_ == 1
                            && !dynamic_cast<constant_op_t *>(
                                    inputs[j]->producer_owner_)) {
                        if (auto inp = inputs[j]
                                               ->producer_owner_
                                               ->dyn_cast<input_op>()) {
                            if (fdata.is_arg_input(inp)) continue;
                        }
                        // if the subnode is used only once, we can reuse
                        // its buffer
                        out_i_detail.need_alloc_ = false;
                        in_j_detail.use_count_++;
                        // inplace tsr_slice_list_
                        out_i_detail.tsr_slice_list_
                                = in_j_detail.tsr_slice_list_;
                        break;
                    }
                }
                // no node can share buffer with the current node
                if (out_i_detail.need_alloc_) {
                    ths->allocate_tensor(outputs[i], fdata);
                }
                // set buffer reuse hint for loop lifetime analysis
                // out_i_detail.tsr_slice_list may be empty when next op is
                // output op
                if (!out_i_detail.tsr_slice_list_.empty()) {
                    bool access_has_updated = false;
                    for (auto &tsr_slice : out_i_detail.tsr_slice_list_) {
                        set_buffer_reuse_hint(hint_tick, fdata, cur_node,
                                out_i_detail.need_alloc_
                                        ? tsr_slice.tptr_->base_->ptr_
                                        : expr(),
                                access_has_updated);
                        access_has_updated = true;
                    }
                }
            }
            return;
        }
    };
    // reset hint_tick before next schedule
    // start hint tick is 0, and when encounter an op, tick increases by 1.
    int64_t hint_tick = 0;
    for (auto &cur : sorted_ops_) {
        auto src = output_anchor_slice_[anchor_id].first;
        auto dst = output_anchor_slice_[anchor_id].second;
        alloc_func(cur, src, dst, anchor_id, hint_tick);
    }
    for (auto &op : sorted_ops_) {
        if (dynamic_cast<constant_op_t *>(op.get())) { continue; }
        for (auto &tsr_slices : ths->get_output_tsr_slices_list(
                     op->dyn_cast<fusible_op_t>(), fdata_list[anchor_id])) {
            for (auto &buf : tsr_slices) {
                assert(buf->tptr_.defined());
            }
        }
    }
}

// This function can find the parent node in IR, if the node has no parent node,
// return itself.
static stmt get_parent_node(stmt node) {
    if (!node->attr().has_key("builder.parent_node")) return node;
    stmt parent {node->attr()["builder.parent_node"]
                         .get<std::weak_ptr<stmt_base_t>>()
                         .lock()};
    COMPILE_ASSERT(parent.defined(), "parent node should not be null");
    return parent;
}

/** This function will find the nearest parent 'for_loop' node body for tensor
 *  declaration. If no loop found, it will just push statements in outer-most
 *  stmt.
 *  @param anchor: tensor should belong to which anchor
 * */
static stmt get_parent_loop_body(stmt anchor) {
    auto node = std::move(anchor);
    while (node->attr().has_key("builder.parent_node")) {
        if (node.isa<for_loop>()) {
            auto ret = node.static_as<for_loop>();
            return ret->body_;
        }
        node = get_parent_node(node);
    }
    if (node.isa<for_loop>()) {
        auto ret = node.static_as<for_loop>();
        return ret->body_;
    }
    return node;
}

void fusion_manager::do_declare_tensor(
        std::vector<fusion_anchor_data> &fdata_list) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");
    // declare tptr_ of real tensor in fdata, and put it at the beginning of ss
    auto declare_ = [&](fusion_data &fdetail, std::vector<stmt> &ss) {
        for (auto &tsr_slice : fdetail.tsr_slice_list_) {
            auto tsr = tsr_slice.get_real_tensor();
            if (tsr->attr_ && tsr->attr_->has_key("declared")) break;
            // Only temp buffer need to be declared
            std::vector<sc::tensor>::iterator tensor_iter = std::find_if(
                    allocated_tensors_.begin(), allocated_tensors_.end(),
                    [tsr](sc::tensor &t) { return t.ptr_same(tsr); });
            if (tensor_iter != allocated_tensors_.end()) {
                ss.emplace(ss.begin(),
                        builder::make_var_tensor_def_unattached(tsr));
                tsr->attr()["declared"] = true;
            }
        }
    };

    for (int i = static_cast<int>(sorted_ops_.size()) - 1; i >= 0; i--) {
        auto cur_op = sorted_ops_[i]->dyn_cast<fusible_op_t>();
        // TODO(xxx): when constant op support tensor mode, it can be removed
        // here.
        if (cur_op->isa<output_op>() || cur_op->isa<constant_op_t>()) continue;
        int anchor_id = cur_op->anchor_id;
        fusion_anchor_data &fdata = fdata_list[anchor_id];

        if (cur_op->isa<input_op>()) {
            auto &out = cur_op->get_outputs()[0];
            auto &fdetail = fdata.get(out);
            // shrink tensor from the input of fusion manager may also need to
            // move its definition place, we need to provide an anchor to it.

            auto &tsr_slice = fdetail.tsr_slice_list_[0];
            auto tsr = tsr_slice.get_real_tensor();
            if (tsr->attr_
                    && tsr->attr_->has_key(tensor_shrinker_attrs::may_shrink)) {
                COMPILE_ASSERT(fdetail.original_ranges_list_.size() == 1,
                        "shrink tensor is expected to have one range");
                // search real input anchor
                int real_anchor = anchor_id;
                for (int64_t j = static_cast<int64_t>(sorted_ops_.size()) - 1;
                        j >= 0; j--) {
                    auto tmp_op = sorted_ops_[j]->dyn_cast<fusible_op_t>();
                    if (tmp_op->isa<input_op>() || tmp_op->isa<output_op>()
                            || tmp_op->isa<constant_op_t>())
                        continue;
                    int tmp_anchor = tmp_op->anchor_id;
                    auto &tmp_fdetail = fdata_list[tmp_anchor];
                    bool found = false;
                    for (auto &ins : tmp_op->get_inputs()) {
                        auto &tmp_fdetail = fdata.get(ins);
                        for (auto &tmp_tsr_slice :
                                tmp_fdetail.tsr_slice_list_) {
                            if (tmp_tsr_slice.get_real_tensor().ptr_same(tsr)) {
                                real_anchor = tmp_anchor;
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                    if (found) break;
                    for (auto &outs : tmp_op->get_outputs()) {
                        auto &tmp_fdetail = fdata.get(outs);
                        for (auto &tmp_tsr_slice :
                                tmp_fdetail.tsr_slice_list_) {
                            if (tmp_tsr_slice.get_real_tensor().ptr_same(tsr)) {
                                real_anchor = tmp_anchor;
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                    if (found) break;
                }

                // set input tensor shrink info
                auto range = fdata_list[real_anchor]
                                     .get(out)
                                     .tsr_slice_list_[0]
                                     .get_ranges();
                tsr->attr()[tensor_shrinker_attrs::should_shrink]
                        = tensor_shrinker_t::shrink_info_t {
                                /*base*/ get_slice_idx(range),
                                /*shape*/ get_slice_shape(range), stmts()};

                // set declare info
                auto &decl_anchor_stmt
                        = output_anchor_position_.at(real_anchor);
                stmts decl_body = get_parent_loop_body(decl_anchor_stmt)
                                          .checked_as<stmts>();
                stmts place_holder = builder::make_stmts_unattached({})
                                             .checked_as<stmts>();
                place_holder
                        ->attr()[tensor_shrinker_attrs::tensor_for_placerholder]
                        = std::weak_ptr<expr_base>(tsr.impl);
                decl_body->seq_.emplace(decl_body->seq_.begin(), place_holder);
                auto &shrink_info
                        = tsr->attr_->get<tensor_shrinker_t::shrink_info_t>(
                                tensor_shrinker_attrs::should_shrink);
                shrink_info.move_def_ = place_holder;
            }
            continue;
        }

        auto &anchor = output_anchor_position_[anchor_id];
        // get decl_body, if anchor_id=0, just put it in
        // beginning of the first anchor. Otherwise, we need to get current
        // anchor nearest parent loop body.
        stmts decl_body = anchor_id == 0
                ? anchor
                : get_parent_loop_body(anchor).checked_as<stmts>();
        // get parent loop sequence, tensor declaration will be placed at the
        // beginning of this sequence.
        auto &ss = decl_body->seq_;
        for (auto &ins : cur_op->get_inputs()) {
            auto &fdetail = fdata.get(ins);
            declare_(fdetail, ss);
        }

        for (auto &out : cur_op->get_outputs()) {
            auto &fdetail = fdata.get(out);
            declare_(fdetail, ss);
        }
    }
}

void fusion_manager::do_compute_block(
        const context_ptr &ctx, std::vector<fusion_anchor_data> &fdata_list) {
    if (graph_.empty()) { return; }

    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");

    /** this wrapper function is expected to deal with multi-slice condition
     * TODO(xxx): Theoretically, if one op correctly implement its
     * 'infer_slice_range' function and write 'compute_block' function from
     * the view of tensor_slice, it will be able to deal with multi-slice
     * condition by this 'compute_wrapper'. As the result, although movement op
     * share the common interface to deal with multi-slice, they only can
     * address single slice. (@note: reorder can produce multi slice outputs,
     * but not inputs)
     * */
    auto compute_wrapper_ = [&](fusible_op_t *cur, const sc::context_ptr &ctx,
                                    sc::fusion_anchor_data &result) {
        // Here is the op which does not overwrite compute block virtual
        // function
        if (cur->isa<input_op>() || cur->isa<output_op>()
                || cur->isa<constant_op_t>()) {
            return;
        }

        auto dst = get_output_tsr_slices_list(cur, result);
        auto inputs = get_input_tsr_slices_list(cur, result);
        COMPILE_ASSERT(
                !inputs.empty(), "Op " << cur->op_name_ << "has no input");

        auto in_slice_size = inputs[0].size();
        for (auto &in : inputs) {
            COMPILE_ASSERT(in.size() == in_slice_size,
                    "[" << cur->op_name_
                        << "]: inputs slice size should be equal but got "
                        << in.size() << " and " << in_slice_size);
        }

        // elementwise op and reduce op should ensure their output have same
        // slice size with input
        if (!cur->isa<reorder_op_t>()) {
            for (auto &out : dst) {
                COMPILE_ASSERT(out.size() == in_slice_size,
                        "slice size of output "
                        "should be equal to "
                        "Inputs, except for movement op, but got "
                                << out.size() << " and " << in_slice_size);
            }
        }

        // unwrapper tensor slice, for compute_block, it just accpet single
        // tensor_slice
        for (size_t i = 0; i < in_slice_size; i++) {
            std::vector<const tensor_slice *> new_inputs_ptr;
            new_inputs_ptr.reserve(inputs.size());

            for (auto &in : inputs) {
                new_inputs_ptr.emplace_back(in[i]);
            }
            std::vector<tensor_slice *> new_outputs_ptr;
            new_outputs_ptr.reserve(dst.size());

            for (auto &out : dst) {
                new_outputs_ptr.emplace_back(out[i]);
            }
            cur->compute_block(ctx, new_outputs_ptr, new_inputs_ptr, result);
        }
    };

    // automatically skip input/output/constant
    for (auto &cur : sorted_ops_) {
        fusible_op_t *fop = cur->dyn_cast<fusible_op_t>();
        int anchor_id = fop->anchor_id;
        builder::ir_builder_t dummy_builder;
        dummy_builder.push_scope();
        compute_wrapper_(fop, ctx, fdata_list[anchor_id]);
        auto s = dummy_builder.pop_scope().checked_as<stmts>();
        auto anchor_pos = output_anchor_position_[anchor_id];
        anchor_pos->seq_.insert(
                anchor_pos->seq_.end(), s->seq_.begin(), s->seq_.end());
    }
}

void fusion_manager::create_output_fusion_anchor(
        const std::vector<tensor_slice> &src,
        const std::vector<tensor_slice> &dst) {
    COMPILE_ASSERT(!src.empty(), "No src tensor slice is found");

    // append to output_anchor_position
    auto bld = builder::get_current_builder();
    auto s = bld->push_anchor();
    output_anchor_position_.emplace_back(std::move(s));
    // append to output_anchor_slice
    output_anchor_slice_.emplace_back(std::make_pair(src, dst));
}

void fusion_manager::clear_anchor() {
    for (auto &anchor : output_anchor_position_) {
        stmt parent = get_parent_node(anchor);
        auto &ss_parent = parent.checked_as<stmts>()->seq_;
        // find anchor iter
        std::vector<sc::stmt>::iterator anchor_iter
                = std::find_if(ss_parent.begin(), ss_parent.end(),
                        [anchor](stmt &s) { return s.ptr_same(anchor); });
        COMPILE_ASSERT(anchor_iter != ss_parent.end(),
                "Could not found anchor in current parent stmts");
        // move all stmts out of anchor in avoid of blank bracket
        auto &ss_in_anchor = anchor->seq_;
        ss_parent.insert(anchor_iter, ss_in_anchor.begin(), ss_in_anchor.end());
        // re-find anchor iter
        anchor_iter = std::find_if(ss_parent.begin(), ss_parent.end(),
                [anchor](stmt &s) { return s.ptr_same(anchor); });
        COMPILE_ASSERT(anchor_iter != ss_parent.end(),
                "Could not found anchor in current parent stmts");
        // remove anchor
        ss_parent.erase(anchor_iter);
    }
    // clear anchor status
    output_anchor_position_.clear();
    output_anchor_slice_.clear();
}

/**
 * this function is designed to dispatch suitable tensor_slice for each
 * graph_tensor. For instance:
 *
 * Anchor 1         |  Anchor 0          |  Fusible Op | Anchor id
 * outA_1 = inA_1   |  outA_0 = inA_0    |  Op1        |     0
 * outB_1 = outA_1  |  outB_0 = outA_0   |  Op2        |     0
 * outC_1 = outB_1  |  outC_0 = outB_0   |  Op3        |     1
 *
 * @note: outX_i represents different tensor X on anchor i
 *
 * For two ops (Op2 and Op3) like above, they should be rescheduled to below:
 * Anchor 1         |  Anchor 0          |  Fusible Op | Anchor id
 *                  |  outA_0 = inA_0    |  Op1        |     0
 *                  |  outB_1 = outA_0   |  Op2        |     0
 * outC_1 = outB_1  |                    |  Op3        |     1
 *
 * We will find all outB_0 in previous anchor and then replace them with outB_1
 * by tensor_ptr.
 *
 * @note: replacement only occurs in cross-anchor condition.
 * */
void fusion_manager::do_shedule_tensor(
        std::vector<fusion_anchor_data> &fdata_list) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");

    /**
     * replace_: suppport multi-slice mode (except movement type op).
     * For non-movement type op, if they are dealing with multi-slice, one
     * certain slice on different anchor should be similar, which means they
     * should have same numbers of tensor one by one.
     * For movement type op, currently, if they generate multi-slice,
     * use_one_anchor should be turned on in reschedule anchor pass. as the
     * result, this function will never be called.
     * */
    auto replace_ = [](fusion_data &prev_data, fusion_data &cur_data, int i) {
        COMPILE_ASSERT(prev_data.tsr_slice_list_.size()
                        == cur_data.tsr_slice_list_.size(),
                "It seems that one movement type op generate multi-slice, "
                "please turn on 'use_one_anchor' flag during reschedule anchor "
                "pass");
        auto &prev_tptr = prev_data.tsr_slice_list_[i].tptr_;
        auto cur_tsr = cur_data.tsr_slice_list_[i].get_real_tensor();
        // use current tensorptr original tensor and previous tensorptr offset
        prev_tptr
                = builder::tensor_ptr(cur_tsr, prev_tptr->base_->idx_, {}, true)
                          .static_as<tensorptr>();
        // reset boundary tensor's first access hint
        reset_buffer_reuse_first_access_hint(cur_tsr);
    };

    for (size_t i = 0; i < sorted_ops_.size(); i++) {
        // get current fusible op
        auto cur_op = sorted_ops_[i]->dyn_cast<fusible_op_t>();
        // get anchor id
        int cur_op_anchor = cur_op->anchor_id;
        // auto skip
        if (cur_op_anchor == 0) continue;
        for (auto &cur_op_ins : cur_op->get_inputs()) {
            // get cur op tsr
            for (auto &tsr_slice :
                    fdata_list[cur_op_anchor].get(cur_op_ins).tsr_slice_list_) {
                auto cur_op_tsr = tsr_slice.get_real_tensor();
                // visitor all previous anchor
                for (int anchor_id = cur_op_anchor - 1; anchor_id >= 0;
                        anchor_id--) {
                    // check all op before cur op
                    for (size_t j = 0; j < i; j++) {
                        auto prev_op = sorted_ops_[j]->dyn_cast<fusible_op_t>();
                        // get previous anchor
                        int prev_op_anchor = prev_op->anchor_id;
                        // auto skip
                        if (prev_op_anchor != anchor_id) continue;
                        // check its inputs and outputs, whether exist tensor
                        // which need to be reppalced by new tensor in
                        // cur_anchor, to ensure buffer sharing.
                        for (auto &prev_op_ins : prev_op->get_inputs()) {
                            auto &prev_op_data_in_cur_anchor
                                    = fdata_list[cur_op_anchor].get(
                                            prev_op_ins);
                            for (size_t i = 0;
                                    i < prev_op_data_in_cur_anchor
                                                .tsr_slice_list_.size();
                                    i++) {
                                auto &tsr_slice = prev_op_data_in_cur_anchor
                                                          .tsr_slice_list_[i];
                                auto prev_op_tsr_in_cur_anchor
                                        = tsr_slice.get_real_tensor();
                                // if cur_op and prev_op share the same buffer
                                // in cur anchor, we ned to use the shared
                                // buffer to replace prev op tensor in prev
                                // anchor
                                if (cur_op_tsr.ptr_same(
                                            prev_op_tsr_in_cur_anchor)) {
                                    auto &prev_op_data_in_prev_anchor
                                            = fdata_list[prev_op_anchor].get(
                                                    prev_op_ins);
                                    replace_(prev_op_data_in_prev_anchor,
                                            prev_op_data_in_cur_anchor, i);
                                }
                            }
                        }
                        for (auto &prev_op_out : prev_op->get_outputs()) {
                            auto &prev_op_data_in_cur_anchor
                                    = fdata_list[cur_op_anchor].get(
                                            prev_op_out);
                            for (size_t i = 0;
                                    i < prev_op_data_in_cur_anchor
                                                .tsr_slice_list_.size();
                                    i++) {
                                auto &tsr_slice = prev_op_data_in_cur_anchor
                                                          .tsr_slice_list_[i];
                                auto prev_op_tsr_in_cur_anchor
                                        = tsr_slice.get_real_tensor();
                                // if cur_op and prev_op share the same buffer
                                // in cur anchor, we ned to use the shared
                                // buffer to replace prev op tensor in prev
                                // anchor
                                if (cur_op_tsr.ptr_same(
                                            prev_op_tsr_in_cur_anchor)) {
                                    auto &prev_op_data_in_prev_anchor
                                            = fdata_list[prev_op_anchor].get(
                                                    prev_op_out);
                                    replace_(prev_op_data_in_prev_anchor,
                                            prev_op_data_in_cur_anchor, i);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // push all tensor definition to suitable anchor
    do_declare_tensor(fdata_list);
    // clear allocated_tensors
    allocated_tensors_.clear();
}

void fusion_manager::init_sorted_ops() {
    if (graph_.empty()) return;
    // reset
    sorted_ops_.clear();
    // default sorted by dfs_topology_sort
    op_visitor_t::dfs_topology_sort(graph_.ops_.size())
            .visit_graph(graph_, [&](const sc_op_ptr &cur) {
                sorted_ops_.emplace_back(cur);
            });
}

void fusion_manager::do_reshedule_anchor(
        std::vector<fusion_anchor_data> &fdata_list, bool use_one_anchor) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");

    for (size_t i = 0; i < sorted_ops_.size(); i++) {
        auto cur_op = sorted_ops_[i]->dyn_cast<fusible_op_t>();
        int cur_anchor = cur_op->anchor_id;
        /**
         * TODO(xxx): For those moememnt ops, they may generate completely
         * different multi-slice in *different anchor*, it is hard to replace
         * their each tensor_slice in tensor_slice_list_ across different
         * anchors. As the workaround, we just turn on use_one_anchor to avoid
         * cross-anchor condition.
         * */
        if (!use_one_anchor && cur_op->isa<movement_op_t>()) {
            for (auto &out : cur_op->get_outputs()) {
                bool is_last_op = true;
                for (auto &user : out->uses_) {
                    if (!user.second->isa<output_op>()) { is_last_op = false; }
                }
                if (is_last_op) continue;
                if (fdata_list[cur_anchor].get(out).original_ranges_list_.size()
                        > 1) {
                    use_one_anchor = true;
                    break;
                }
            }
        }
    }

    int tmp_anchor = sorted_ops_[0]->dyn_cast<fusible_op_t>()->anchor_id;
    for (size_t i = 0; i < sorted_ops_.size(); i++) {
        auto cur_op = sorted_ops_[i]->dyn_cast<fusible_op_t>();
        // if use_one_anchor is on, just set max_anchor to each op anchor
        if (use_one_anchor)
            cur_op->anchor_id = max_anchor_;
        else {
            if (cur_op->anchor_id > tmp_anchor)
                tmp_anchor = cur_op->anchor_id;
            else if (cur_op->anchor_id < tmp_anchor)
                cur_op->anchor_id = tmp_anchor;
        }
    }
}

/**
 * (fp32) (fp32)   (fp32 bc_arg) (fp32)    (int 8) (fp32 bc_arg)
 *  Op1    Op2          Op1     Op2           Op1   Op2
 *   \    /  \           \      /              |    /
 *    \  /    \           \    /              Op3  /
 *     Op3     \            Op3                |  /
 *      \      /             |                 Op4
 *       \    /              |                  |
 *        \  /               |                  |
 *         Op4 (fp32)       Op4(fp32)        Op5(fp32)
 *      Case 1              Case 2            Case 3
 * For Case 1, output_op i have dependence with input_op j.
 * For Case 2, output_op i have different dims with input_op j.
 * For Case 3, output_op i have different dtype with input_op j.
 * TODO(xxx): this is just first stage fir inplace propagation pass. During the
 * second stage, all fusible op inside fusion pattern should be visited and
 * queried to ensure whether inplace really occurs.
 * */
std::vector<std::vector<int>> fusion_manager::query_inplace() {
    // auto skip
    if (graph_.empty()) { return {}; }
    auto output_op_list = graph_.get_output_ops();
    for (auto &op : graph_.ops_) {
        if (op->isa<reorder_op_t>() || op->isa<cast_op_t>()) {
            return std::vector<std::vector<int>>(output_op_list.size());
        }
    }
    auto input_op_list = graph_.get_input_ops();
    size_t input_op_size = input_op_list.size();
    std::vector<std::vector<int>> inplace_list;
    // firstly we will check each op in outputs op
    for (size_t i = 0; i < output_op_list.size(); i++) {
        auto cur_out = output_op_list[i]->dyn_cast<output_op>();
        std::vector<int> inp_idx = {};
        for (size_t j = 0; j < input_op_size; j++) {
            auto cur_in = input_op_list[j]->dyn_cast<input_op>();
            bool can_replace = true;
            // Case 1: output_op i have dependence with input_op j.
            if (cur_out->get_inputs()[0] == cur_in->get_outputs()[0]) {
                can_replace = false;
            }
            // Case 2: output_op i have different dims with input_op j.
            else if (cur_out->get_inputs()[0]->details_.get_blocking_dims()
                    != cur_in->get_outputs()[0]->details_.get_blocking_dims()) {
                can_replace = false;
            }
            // Case 3: output_op i have different dtype with input_op j.
            else if (cur_out->get_inputs()[0]->details_.dtype_
                    != cur_in->get_outputs()[0]->details_.dtype_) {
                can_replace = false;
            }
            if (can_replace) inp_idx.emplace_back(j);
        }
        inplace_list.emplace_back(inp_idx);
    }
    return inplace_list;
}

std::vector<sc_op_ptr> fusion_manager::prepare_and_check(const context_ptr &ctx,
        std::vector<fusion_anchor_data> &fdata_list,
        const std::vector<expr> &outs, const std::vector<expr> &inargs) {
    if (graph_.empty()) return {};
    COMPILE_ASSERT(
            !output_anchor_position_.empty() && !output_anchor_slice_.empty(),
            "no output anchor found, please create them firstly");
    fdata_list
            = std::vector<fusion_anchor_data>(output_anchor_position_.size());

    // check all output_anchor_slice have same src size
    size_t common_src_size = output_anchor_slice_[0].first.size();
    for (auto &slice : output_anchor_slice_) {
        COMPILE_ASSERT(slice.first.size() == common_src_size,
                "all output_anchor_slice should have same src size");
    }

    for (auto &fdata : fdata_list) {
        fdata.input_idx_map_ = &input_idx_map_;
        fdata.output_idx_map_ = &output_idx_map_;
        fdata.num_commit_src_ = common_src_size;
    }

    // Init sorted_ops sequence
    init_sorted_ops();
    // The old query, should be removed in future
    do_query(ctx, fdata_list, false);
    // dispatch suitable commit anchor for each fusible op
    return dispatch_fusion_anchor(ctx, fdata_list, outs, inargs);
};

void fusion_manager::commit(const ir_module_ptr &modu,
        std::vector<fusion_anchor_data> &fdata_list) {
    if (graph_.empty()) { return; }
    COMPILE_ASSERT(!sorted_ops_.empty(),
            "sorted ops are expected to be ready, please initilize it first");

    // sorted by rules
    sorted_ops_ = op_sorting_visitor_t::sort_by_rules(graph_, fdata_list,
            {op_sorting_visitor_t::sort_rule::same_kind,
                    op_sorting_visitor_t::sort_rule::fusion_anchor});

    // reschedule anchor
    do_reshedule_anchor(fdata_list);
    // shedule tensor
    do_shedule_tensor(fdata_list);
    // generate real code in IR
    do_compute_block(modu->ctx_, fdata_list);
    add_to_module(modu);
    // remove empty anchor in avoid to loop fuse/reorder error
    clear_anchor();
}

void fusion_manager::add_to_module(const ir_module_ptr &mod) {
    for (auto &def : global_defines_) {
        mod->add_global_var(def);
    }
}

std::vector<std::vector<const tensor_slice *>>
fusion_manager::get_input_tsr_slices_list(
        fusible_op_t *op, fusion_anchor_data &fdata) const {
    std::vector<std::vector<const tensor_slice *>> result;
    for (auto &input : op->get_inputs()) {
        std::vector<const tensor_slice *> tmp;
        for (auto &tsr_slice : fdata.get(input).tsr_slice_list_) {
            tmp.emplace_back(&tsr_slice);
        }
        result.emplace_back(tmp);
    }
    return result;
}

std::vector<std::vector<tensor_slice *>>
fusion_manager::get_output_tsr_slices_list(
        fusible_op_t *op, fusion_anchor_data &fdata) const {
    std::vector<std::vector<tensor_slice *>> result;
    for (auto &output : op->get_outputs()) {
        std::vector<tensor_slice *> tmp;
        for (auto &tsr_slice : fdata.get(output).tsr_slice_list_) {
            tmp.emplace_back(&tsr_slice);
        }
        result.emplace_back(tmp);
    }
    return result;
}

} // namespace sc
