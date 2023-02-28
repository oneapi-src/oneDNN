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

#include "fused_op.hpp"
#include "fusible_op.hpp"
#include "fusion_data.hpp"
#include "tunable_op.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace op_traits {

// could be publicly used
void op_traits::batchwise_shrinkable_t::collect_shrinked_lt_map(
        int bw_size, gt2gt_map &bw_lt_map) {
    auto ths = dynamic_cast<sc_op *>(this);
    COMPILE_ASSERT(!dynamic_cast<fused_op_t *>(ths),
            "fused op " << ths->op_name_ << " must override this function")
    std::vector<graph_tensor_ptr> new_ins, new_out;
    auto old_ins = ths->get_inputs(), old_out = ths->get_outputs();
    std::for_each(old_ins.begin(), old_ins.end(),
            [&bw_lt_map, &bw_size](const graph_tensor_ptr &gt) {
                op_traits::batchwise_shrinkable_t::record_shrinked_gt(
                        bw_lt_map, gt, bw_size);
            });
    std::for_each(old_out.begin(), old_out.end(),
            [&bw_lt_map, &bw_size](const graph_tensor_ptr &gt) {
                op_traits::batchwise_shrinkable_t::record_shrinked_gt(
                        bw_lt_map, gt, bw_size);
            });
}

sc_op_ptr op_traits::batchwise_shrinkable_t::bw_shrinked_copy(
        gt2gt_map &bw_lt_map, sc_graph_t &shrinked_graph) {
    return op_traits::batchwise_shrinkable_t::bw_shrinked_copy(
            bw_lt_map, shrinked_graph, any_map_t {});
}

sc_op_ptr op_traits::batchwise_shrinkable_t::bw_shrinked_copy(
        gt2gt_map &bw_lt_map, sc_graph_t &shrinked_graph,
        const any_map_t &changed_attr) {
    auto ths = dynamic_cast<sc_op *>(this);
    COMPILE_ASSERT(!dynamic_cast<fused_op_t *>(ths),
            "fused op " << ths->op_name_ << " must override this function")
    std::vector<graph_tensor_ptr> new_ins, new_out;
    auto old_ins = ths->get_inputs(), old_out = ths->get_outputs();
    new_ins.resize(old_ins.size());
    new_out.resize(old_out.size());
    std::transform(old_ins.begin(), old_ins.end(), new_ins.begin(),
            [&bw_lt_map, &ths](const graph_tensor_ptr &gt) {
                COMPILE_ASSERT(bw_lt_map.haskey(gt),
                        ths->op_name_
                                << ": new input graph tensor not found in map")
                return bw_lt_map.get(gt);
            });
    std::transform(old_out.begin(), old_out.end(), new_out.begin(),
            [&bw_lt_map, &ths](const graph_tensor_ptr &gt) {
                COMPILE_ASSERT(bw_lt_map.haskey(gt),
                        ths->op_name_
                                << ": new input graph tensor not found in map")
                return bw_lt_map.get(gt);
            });

    auto new_attr = ths->attrs_;
    auto attr_map = changed_attr.as_map();
    for (auto &att : attr_map) {
        new_attr[att.first] = att.second;
    }
    new_attr[op_attr_key::bwise_fuse] = true;
    auto new_op
            = shrinked_graph.make(ths->op_name_, new_ins, new_out, new_attr);

    if (ths->isa<brgemm_fusion_acceptable_t>()) {
        new_op->dyn_cast<brgemm_fusion_acceptable_t>()->copy_from(
                ths->dyn_cast<brgemm_fusion_acceptable_t>());
    }
    if (ths->isa<tunable_op_t>()) {
        auto tune_ret = new_op->stc_cast<tunable_op_t>();
        auto tune_ths = ths->stc_cast<tunable_op_t>();
        tune_ret->op_name_ = tune_ths->op_name_;
        tune_ret->set_config(tune_ths->get_config());
        tune_ret->is_quantized_ = tune_ths->is_quantized_;
        tune_ret->need_compensation_ = tune_ths->need_compensation_;
        tune_ret->should_quantized_ = tune_ths->should_quantized_;
    }
    return new_op;
}

graph_tensor_ptr op_traits::batchwise_shrinkable_t::shrink_gt(
        const graph_tensor_ptr &orig_gt, int shrink_offset) {
    auto blocking_fmt = orig_gt->details_.get_format();
    auto blocking_dims = orig_gt->details_.get_blocking_dims();
    sc_dims new_plain_dims;
    if (blocking_fmt.is_any()) {
        new_plain_dims = blocking_dims;
        std::transform(new_plain_dims.begin(),
                new_plain_dims.begin() + shrink_offset, new_plain_dims.begin(),
                [](const sc_dim &d) { return 1; });
    } else {
        auto plain_dims = orig_gt->details_.get_plain_dims();
        new_plain_dims = plain_dims;
        int real_shrink = shrink_offset;
        if (real_shrink) {
            for (int block_i = 0; block_i < real_shrink; block_i++) {
                sc_dim cur_dim = blocking_dims[block_i];
                int plain_i = blocking_fmt.format_code_.get(block_i);
                new_plain_dims[plain_i] /= cur_dim;
            }
        }
    }
    auto new_fmt = shrink_format_by_plain_dims(orig_gt, new_plain_dims);
    return std::make_shared<graph_tensor>(
            nullptr, new_fmt, new_plain_dims, orig_gt->details_.dtype_);
}

int op_traits::batchwise_shrinkable_t::get_shrinkable_offset(
        const graph_tensor_ptr &gt, bool aggresive_mode) {
    auto fmt = gt->details_.get_format();
    auto blocking_dims = gt->details_.get_blocking_dims();
    // if fmt is not blocking, avoid vectorized axis
    if (!fmt.is_blocking()) return blocking_dims.size() - 1;
    int bs_size = blocking_dims.size() - fmt.format_code_.ndims();
    auto p2b_map = fmt.format_code_.collect_p2b_mapping();
    int offset = 0;
    auto plain_dims = gt->details_.get_plain_dims();
    // avoid padding
    auto check_padding
            = [&blocking_dims, &plain_dims, &bs_size](
                      const std::vector<int> &blocks, int plain_pos) {
                  sc_dim acc_dim = 1;
                  for (auto &block_pos : blocks) {
                      acc_dim *= blocking_dims.at(bs_size + block_pos);
                  }
                  return acc_dim != plain_dims[bs_size + plain_pos];
              };
    std::vector<int> mask(fmt.format_code_.norig_dims(), 0);
    for (; offset < fmt.format_code_.ndims(); offset++) {
        auto plain_pos = fmt.format_code_.get(offset);
        mask[plain_pos]++;
        // no blocking found in current plain axis, skip.
        if (p2b_map[plain_pos].size() == 1)
            continue;
        else if (mask[plain_pos] > (aggresive_mode ? static_cast<int>(
                                            p2b_map[plain_pos].size() - 1)
                                                   : 1)
                || check_padding(p2b_map[plain_pos], plain_pos))
            break;
    }
    return bs_size + offset;
}

sc_data_format_t op_traits::batchwise_shrinkable_t::shrink_format_by_plain_dims(
        const graph_tensor_ptr &orig_gt, const sc_dims &plain_dims) {
    auto old_plain_dims = orig_gt->details_.get_plain_dims();
    COMPILE_ASSERT(plain_dims.size() == old_plain_dims.size(),
            "plain dims size should not be different");
    auto old_fmt = orig_gt->details_.get_format();
    if (!old_fmt.is_blocking()) { return old_fmt; }
    auto old_blocks = old_fmt.blocks_;
    auto new_blocks = old_blocks;
    for (int i = 0; i < static_cast<int>(old_plain_dims.size()); i++) {
        if (plain_dims[i] != old_plain_dims[i]) {
            auto old_blocks_i = old_fmt.format_code_.collect_blocking_index(i);
            std::reverse(old_blocks_i.begin(), old_blocks_i.end());
            bool reset_new_block = false;
            for (auto &j : old_blocks_i) {
                if (reset_new_block) {
                    new_blocks[j] = plain_dims[i];
                } else {
                    if (old_blocks[j] >= plain_dims[i]) {
                        reset_new_block = true;
                        new_blocks[j] = plain_dims[i];
                    } else {
                        COMPILE_ASSERT(plain_dims[i] % old_blocks[j] == 0,
                                "Unexpected strided/padding case found, and "
                                "could "
                                "not "
                                "infer new format")
                    }
                }
            }
        }
    }
    return sc_data_format_t(old_fmt.format_code_, new_blocks);
}

void op_traits::batchwise_shrinkable_t::record_shrinked_gt(
        gt2gt_map &bw_lt_map, const graph_tensor_ptr &gt, int shrink_offset) {
    if (bw_lt_map.haskey(gt)) return;
    bw_lt_map.get(gt)
            = op_traits::batchwise_shrinkable_t::shrink_gt(gt, shrink_offset);
}

void op_traits::batchwise_shrinkable_t::record_shrinked_gt(gt2gt_map &bw_lt_map,
        const graph_tensor_ptr &gt, const sc_dims &plain_dims) {
    if (bw_lt_map.haskey(gt)) return;
    auto new_fmt = shrink_format_by_plain_dims(gt, plain_dims);
    bw_lt_map.get(gt) = std::make_shared<graph_tensor>(
            nullptr, new_fmt, plain_dims, gt->details_.dtype_);
}

void op_traits::batchwise_shrinkable_t::record_shrinked_axis(
        gt2axis_map &bw_axis_map, const graph_tensor_ptr &gt, int bw_size) {
    auto generate_vector_by_num = [](int num) {
        std::vector<int> ret;
        ret.reserve(num);
        for (int i = 0; i < num; i++)
            ret.emplace_back(i);
        return ret;
    };
    if (bw_axis_map.haskey(gt)) return;
    bool scaler = gt->producer_owner_->isa<constant_op_t>()
            || (gt->details_.get_blocking_dims().size() == 1
                    && gt->details_.get_blocking_dims()[0] == 1);
    bw_axis_map.get(gt) = scaler ? std::vector<int>(bw_size, -1)
                                 : generate_vector_by_num(bw_size);
}

void op_traits::batchwise_shrinkable_t::record_shrinked_axis(
        gt2axis_map &bw_axis_map, const graph_tensor_ptr &gt,
        const std::vector<int> &axis) {
    if (bw_axis_map.haskey(gt)) return;
    bool scaler = gt->producer_owner_->isa<constant_op_t>()
            || (gt->details_.get_blocking_dims().size() == 1
                    && gt->details_.get_blocking_dims()[0] == 1);
    bw_axis_map.get(gt) = scaler ? std::vector<int>(axis.size(), -1) : axis;
}

void op_traits::batchwise_shrinkable_t::collect_shrinked_axis_map(
        int bw_size, gt2axis_map &bw_axis_map) {
    auto ths = dynamic_cast<sc_op *>(this);
    COMPILE_ASSERT(!dynamic_cast<fused_op_t *>(ths),
            "fused op " << ths->op_name_ << " must override this function")

    for (auto &ins : ths->get_inputs()) {
        record_shrinked_axis(bw_axis_map, ins, bw_size);
    }
    for (auto &out : ths->get_outputs()) {
        record_shrinked_axis(bw_axis_map, out, bw_size);
    }
}

} // namespace op_traits
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
