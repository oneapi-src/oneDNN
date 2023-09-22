/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#include <algorithm>
#include <vector>

#include "may_broadcast.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace op_traits {

std::vector<int> may_broadcast_t::get_auto_broadcast_bc_axis(
        const sc_dims &input_shape, const sc_dims &output_shape) {
    if (input_shape.size() == 1 && input_shape[0] == 1) { return {-1}; }
    // following auto_broadcast semantics
    const size_t input_rank = input_shape.size();
    const size_t output_rank = output_shape.size();
    COMPILE_ASSERT(output_rank >= input_rank,
            "Incorrect input or output shape for broadcast op.");
    const size_t offset = output_rank - input_rank;
    std::vector<int> bc_axis;
    for (size_t i = 0; i < input_rank; ++i) {
        // TODO(yifei): consider whether input_shape[i] != 1 is
        // necessary here
        if (input_shape[i] == output_shape[i + offset]) {
            bc_axis.emplace_back(i + offset);
        }
    }
    if (bc_axis.empty()) { bc_axis.emplace_back(-1); }
    return bc_axis;
}

bool may_broadcast_t::broadcastable_shape_equal(
        const sc_dims &shape1, const sc_dims &shape2) {
    COMPILE_ASSERT(shape1.size() <= shape2.size(),
            "broadcastable shape equal function shall have input shape1 "
            "smaller than input shape2.");
    if (shape1.size() != shape2.size()) return false;
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (!is_dynamic_dim(shape1[i]) && !is_dynamic_dim(shape2[i])
                && shape1[i] != shape2[i]) {
            return false;
        } else if (shape1[i] == 1 && is_dynamic_dim(shape2[i])) {
            return false;
        }
    }
    return true;
}

sc_dims may_broadcast_t::infer_auto_broadcast_output_shape(
        const sc_dims &lhs, const sc_dims &rhs) {
    const size_t lhs_rank = lhs.size();
    const size_t rhs_rank = rhs.size();
    const size_t max_rank = std::max(lhs_rank, rhs_rank);

    const size_t lhs_offset = max_rank - lhs_rank;
    const size_t rhs_offset = max_rank - rhs_rank;
    bool auto_broadcastable = true;
    sc_dims ret(max_rank, 1);
    for (size_t i = 0; i < max_rank; ++i) {
        sc_dim l = 1, r = 1;
        if (i >= lhs_offset) l = lhs[i - lhs_offset];
        if (i >= rhs_offset) r = rhs[i - rhs_offset];
        if (l == 1 || r == 1) {
            ret[i] = (l == 1 ? r : l);
        } else if (is_dynamic_dim(l) && is_dynamic_dim(r)) {
            // TODO(xxx): correctly handle dynamic case
            ret[i] = l;
        } else if (is_dynamic_dim(l) || is_dynamic_dim(r)) {
            ret[i] = (is_dynamic_dim(l) ? r : l);
        } else {
            if (l != r) {
                auto_broadcastable = false;
                break;
            } else {
                ret[i] = l;
            }
        }
    }
    COMPILE_ASSERT(auto_broadcastable,
            "The given input shapes do not follow auto_broadcast "
            "semantics. "
            "Please recheck the input shapes, or consider specifying "
            "detailed "
            "bc_axis.");
    return ret;
}

sc_data_format_t may_broadcast_t::infer_broadcast_format(
        const logical_tensor_t &target_lt, const logical_tensor_t &bc_lt) {
    COMPILE_ASSERT(
            bc_lt.get_plain_dims().size() == target_lt.get_plain_dims().size(),
            "infer_blocking_format only support plain dimension aligned "
            "cases");
    sc_data_format_kind_t target_lt_format_code
            = target_lt.get_format().format_code_;
    sc_data_format_t::blocking_t blocks = target_lt.get_format().blocks_;
    sc_data_format_kind_t bc_lt_format_code = bc_lt.get_format().format_code_;
    // start infer the blocks
    sc_dims bc_plain_dim = bc_lt.get_plain_dims();
    sc_dims target_plain_dim = target_lt.get_plain_dims();
    int target_batch_dim = target_lt.get_plain_dims().size()
            - target_lt_format_code.norig_dims();
    for (int i = 0; i < target_lt_format_code.norig_dims(); ++i) {
        if (bc_plain_dim[target_batch_dim + i] == 1
                && target_plain_dim[target_batch_dim + i] != 1) {
            // if bc_plain_dim is 1 and this axis is with broadcast
            // semantics
            auto axis = target_lt_format_code.collect_blocking_index(i);
            for (auto ax : axis) {
                blocks[ax] = 1;
            }
        }
    }
    // start infer the format code
    // if both batch OR both non-batch
    // smaller side's format code == larger side's format code
    COMPILE_ASSERT(target_lt_format_code.norig_dims()
                    == bc_lt_format_code.norig_dims(),
            "Unsupported case for broadcastable op query format.");
    return sc_data_format_t(target_lt.get_format().format_code_, blocks);
}

} // namespace op_traits
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
