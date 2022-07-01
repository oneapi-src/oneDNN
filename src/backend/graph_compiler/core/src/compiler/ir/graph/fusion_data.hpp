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
#include <unordered_set>

namespace sc {

using slice_range = std::vector<std::pair<expr, expr>>;
using slice_range_list = std::vector<slice_range>;
using slice_range_map = std::unordered_map<int, slice_range_list>;

enum class infer_status_code : int {
    OK = 0, // Successful
    RETRY, // Need retry another anchor
    FAIL, // Could not infer
    UNKNOWN, // Unknown
    END,
};

struct infer_status_map_t {
    std::vector<std::unordered_set<sc_op_ptr>> inf_stat_map_;

    infer_status_map_t() {
        inf_stat_map_.resize(static_cast<int>(infer_status_code::END));
    }

    std::unordered_set<sc_op_ptr> &get_ops_by_status(infer_status_code code) {
        COMPILE_ASSERT(code != infer_status_code::END, "END code found");
        return inf_stat_map_[static_cast<int>(code)];
    }

    const bool is_ok() {
        return get_ops_by_status(infer_status_code::RETRY).empty()
                && get_ops_by_status(infer_status_code::FAIL).empty()
                && get_ops_by_status(infer_status_code::UNKNOWN).empty();
    }

    const bool is_fail() {
        return !get_ops_by_status(infer_status_code::FAIL).empty();
    }

    const bool is_retry() {
        return !get_ops_by_status(infer_status_code::RETRY).empty();
    }

    const bool is_unknown() {
        return !get_ops_by_status(infer_status_code::UNKNOWN).empty();
    }

    void append_ops_by_status(sc_op *cur, infer_status_code code) {
        COMPILE_ASSERT(code != infer_status_code::END, "END code found");
        inf_stat_map_[static_cast<int>(code)].insert(cur->shared_from_this());
    }

    void remove_ops_by_status(sc_op *cur, infer_status_code code) {
        COMPILE_ASSERT(code == infer_status_code::UNKNOWN,
                "remove_ops_by_status temporarily only supports remove "
                "unknown_status to avoid potential misuse.");
        auto pos = inf_stat_map_[static_cast<int>(code)].find(
                cur->shared_from_this());
        COMPILE_ASSERT(pos != inf_stat_map_[static_cast<int>(code)].end(),
                "Op not found in unknown_status map.");
        inf_stat_map_[static_cast<int>(code)].erase(pos);
    }

    void clear() {
        for (auto &ops : inf_stat_map_)
            ops.clear();
    }

    static std::vector<sc_op_ptr> stat_map_to_vector(
            const std::unordered_set<sc_op_ptr> &stat_map) {
        std::vector<sc_op_ptr> result;
        result.reserve(stat_map.size());
        for (auto itr : stat_map) {
            result.push_back(itr);
        }
        return result;
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
    bool buffer_allocated() const { return buffer_.defined(); }
    void set_buffer(const expr &buf);
    const expr &get_buffer() const { return buffer_; };
    tensor get_real_tensor() const;

private:
    expr buffer_; /*tensor or tensorptr*/
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
