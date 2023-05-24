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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TENSOR_SLICE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TENSOR_SLICE_HPP

#include <utility> //for std::pair
#include <vector>
#include <compiler/ir/sc_expr.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
using slice_range = std::vector<std::pair<expr, expr>>;
/**
 * A slice of the tensor.
 * @param tptr_ the base tensor_ptr
 * @param shape_ the slice shape
 * */
struct tensor_slice {
    tensorptr tptr_;
    std::vector<expr> shape_;
    tensor_slice() = default;

    tensor_slice(const expr &tsr);

    tensor_slice(const expr &tsr, slice_range &&ranges);

    // Gets the start address of the tensor slice
    expr get_tensor_ptr() const { return tptr_; }

    // Gets the shape of the sliced tensor
    const std::vector<expr> &get_shape() const { return shape_; }

    int64_t nslice_dims() const { return static_cast<int64_t>(shape_.size()); }
    int64_t nbase_dims() const {
        return static_cast<int64_t>(get_base_dims().size());
    }

    // Gets the offset of the sliced tensor
    const std::vector<expr> &get_offset() const { return tptr_->base_->idx_; }

    // Gets the ranges of the sliced tensor
    slice_range get_ranges() const;

    // Gets the real shape of base tensor (const version)
    const std::vector<expr> &get_base_dims() const;

    // Gets the dtype of base tensor
    sc_data_type_t get_base_dtype() const;

    // Gets the real tensor of tensor slice, not the tensor_ptr
    tensor get_real_tensor() const;

    // check whether slice is full on specific axis
    bool full_on_axis(const std::vector<int> &axis) const;

    // is_full
    bool is_full() const;

    // is_const
    bool is_const() const;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
