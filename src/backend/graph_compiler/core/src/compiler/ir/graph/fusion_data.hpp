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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_DATA_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_DATA_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph_map.hpp"
#include <compiler/ir/sc_expr.hpp>
#include <unordered_map>

namespace sc {

using slice_range = std::vector<std::pair<expr, expr>>;
using slice_range_list = std::vector<slice_range>;
using slice_range_map = std::unordered_map<int, slice_range_list>;

enum class infer_status_code : int {
    OK = 0, // Successful
    RETRY, // Need retry another anchor
    FAIL, // Could not infer
    END,
};

struct infer_status_map_t {
    std::vector<std::vector<sc_op_ptr>> inf_stat_map_;

    infer_status_map_t() {
        inf_stat_map_.resize(static_cast<int>(infer_status_code::END));
    }

    std::vector<sc_op_ptr> &get_ops_by_status(infer_status_code code) {
        COMPILE_ASSERT(code != infer_status_code::END, "END code found");
        return inf_stat_map_[static_cast<int>(code)];
    }

    const bool is_ok() {
        return get_ops_by_status(infer_status_code::RETRY).empty()
                && get_ops_by_status(infer_status_code::FAIL).empty();
    }

    const bool is_fail() {
        return !get_ops_by_status(infer_status_code::FAIL).empty();
    }

    const bool is_retry() {
        return !get_ops_by_status(infer_status_code::RETRY).empty();
    }

    void append_ops_by_status(sc_op *cur, infer_status_code code) {
        COMPILE_ASSERT(code != infer_status_code::END, "END code found");
        inf_stat_map_[static_cast<int>(code)].emplace_back(
                cur->shared_from_this());
    }

    void clear() {
        for (auto &ops : inf_stat_map_)
            ops.clear();
    }
};

inline std::vector<expr> get_slice_idx(const slice_range &range) {
    std::vector<expr> ret;
    for (auto &r : range) {
        ret.emplace_back(r.first);
    }
    return ret;
}

inline std::vector<expr> get_slice_shape(const slice_range &range) {
    std::vector<expr> ret;
    for (auto &r : range) {
        ret.emplace_back(r.second);
    }
    return ret;
}

bool share_with_output(const graph_tensor_ptr &gt);

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

    // check whether slice is full on specific axes
    bool full_on_axes(const std::vector<int> &axes) const;

    // is_full
    bool is_full() const;
};

struct buffer_reuse_identity;
using tsr_reuse_vec = std::vector<expr>;
using buffer_identity_count
        = std::unordered_map<buffer_reuse_identity, tsr_reuse_vec>;

// fusion_data_t is related to buffer and slice range info of fusible op
struct fusion_data_t {
    // the number of uses of the tensor in the graph. TODO: Should be replaced
    // by uses_.size()
    int use_count_ = 0;
    bool need_alloc_ = true;
    fusion_data_t() = default;
    fusion_data_t(const fusion_data_t &) = delete;
    const expr &get_buffer() const { return buffer_; };
    const tensor get_allocated_tensor() const {
        auto buf = buffer_;
        while (!buf.isa<tensor>()) {
            COMPILE_ASSERT(buf.isa<tensorptr>(),
                    "tensor_slice only accepts a tensor or tensorptr, got: "
                            << buf);
            auto base = buf.static_as<tensorptr>()->base_;
            COMPILE_ASSERT(base.isa<indexing>(),
                    "tensor_ptr base should be indexing, but got: " << base);
            buf = base.checked_as<indexing>()->ptr_;
        }
        COMPILE_ASSERT(buf.isa<tensor>(), "Tensor type is expected")
        return buf.static_as<tensor>();
    };

private:
    expr buffer_;
    friend class fusion_manager;
    friend void set_buffer_reuse_hint(buffer_identity_count &, int64_t &,
            fdata_map &, const sc_op_ptr &, const expr &, bool);
};

struct fuse_state_t {
    fdata_map fdmap_;
    std::vector<fslice_map> fsmap_list_;
    fuse_state_t() = default;
};

struct fuse_anchor_t {
    stmts anchor_position_;
    std::pair<std::vector<tensor_slice>, std::vector<tensor_slice>>
            anchor_slice_;
    fuse_anchor_t() = default;
    fuse_anchor_t(stmts pos,
            std::pair<std::vector<tensor_slice>, std::vector<tensor_slice>>
                    slice)
        : anchor_position_(std::move(pos)), anchor_slice_(std::move(slice)) {};
};

} // namespace sc

namespace std {
template <>
struct hash<sc::buffer_reuse_identity> {
    std::size_t operator()(const sc::buffer_reuse_identity &in) const;
};
} // namespace std
#endif
