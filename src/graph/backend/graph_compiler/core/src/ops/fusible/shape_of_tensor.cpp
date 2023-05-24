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

#include "shape_of_tensor.hpp"
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <runtime/dynamic_dispatch/ops/config.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

shape_of_tensor_op_t::shape_of_tensor_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "shape_of_tensor";
    COMPILE_ASSERT(ins.size() == 1, "Shape of tensor op takes 1 input");
    info_.inputs_ = ins;
    attrs_ = attrs;
    auto &shape_idx = attrs_.get<int>("shape_idx");
    shape_type_ = padding_shape_etype_t(
            attrs_.get<int>(attr_keys::padding_shape_type));
    auto shape_size
            = static_cast<int>(ins[0]->details_.get_plain_dims().size());
    COMPILE_ASSERT(shape_idx < shape_size,
            "Shape index: " << shape_idx
                            << " should not be large than input shape "
                               "size: "
                            << shape_size << "!");

    shape_idx_ = shape_idx;
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        // shrinked from sc_dim
        info_.outputs_[0]->details_.dtype_ = datatypes::s32;
        info_.outputs_[0]->details_.set_plain_dims({1});
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
    }
}

void shape_of_tensor_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    out_formats.reserve(info_.outputs_.size());
    out_formats.push_back({sc_data_format_t(format_kinds::A)});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void shape_of_tensor_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // judge if it is the first op in partition then use the outmost anchor
    if (fsmap.datamap_.size() == 1) {
        auto src_dim = get_inputs()[0]->details_.get_blocking_dims();
        std::vector<int> required_axis(src_dim.size(), 0);
        for (size_t i = 0; i < required_axis.size(); i++) {
            required_axis[i] = i;
        }
        // check the slice range whether meet the outmost anchor
        for (auto &src_range : fsmap.get(get_inputs()[0])) {
            if (!slice_full_on_axis(src_dim, src_range, required_axis)) {
                stat_map.append_ops_by_status(this, infer_status_code::RETRY);
                return;
            }
        }
    }
    fsmap.get(get_outputs()[0])
            = slice_range_list {{std::make_pair(expr(0), expr(1))}};
}

static int get_constant_padded_shape(
        int shape, padding_shape_etype_t type, bool is_batch) {
    switch (type) {
        case padding_shape_etype_t::without_padding: return shape; break;
        case padding_shape_etype_t::matmul_padding: {
            int block = get_matmul_dyn_cfg_single(shape, /*is_batch*/ is_batch);
            return utils::divide_and_ceil(shape, block) * block;
            break;
        }
        case padding_shape_etype_t::conv_padding:
            COMPILE_ASSERT(false, "Unimplement of conv padding shape!");
            break;
        default: break;
    }
    return 0;
}

static expr get_expr_padded_shape(builder::builder_impl_t *bld,
        const expr &shape, padding_shape_etype_t type, bool is_batch) {
    switch (type) {
        case padding_shape_etype_t::without_padding:
            return builder::make_cast(datatypes::s32, shape);
            break;
        case padding_shape_etype_t::matmul_padding: {
            auto block = builder::make_var(datatypes::s32, "matmul_block");
            bld->push_var_tensor_def(block, linkage::local,
                    builtin::call_get_matmul_dyn_cfg_single(
                            builder::make_cast(datatypes::s32, shape),
                            is_batch));
            return builder::make_cast(
                    datatypes::s32, divide_and_ceil(shape, block) * block);
            break;
        }
        case padding_shape_etype_t::conv_padding:
            COMPILE_ASSERT(false, "Unimplement of conv padding shape!");
            break;
        default: break;
    }
    return expr();
}

void shape_of_tensor_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    auto bld = builder::get_current_builder();
    auto &graph = get_owner_graph();
    auto input_plain_shapes_expr
            = graph.dims_to_expr(get_inputs()[0]->details_.get_plain_dims());
    auto shape = input_plain_shapes_expr[shape_idx_];
    bool is_batch
            = attrs_.get_or_else(attr_keys::shape_of_tensor_is_batch, false);
    expr padded_shape;
    if (shape.isa<constant>()) {
        padded_shape = get_constant_padded_shape(
                static_cast<int>(get_expr_as_int(shape)), shape_type_,
                is_batch);
    } else {
        padded_shape = get_expr_padded_shape(bld, shape, shape_type_, is_batch);
    }
    assert(padded_shape.defined());
    bld->push_assign(builder::make_indexing(dst[0]->tptr_, {0}), padded_shape);
}

OP_REGISTER(shape_of_tensor_op_t, shape_of_tensor);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
