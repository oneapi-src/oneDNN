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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_DATA_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_DATA_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph_map.hpp"
#include "tensor_slice.hpp"
#include <compiler/ir/sc_expr.hpp>
#include <unordered_map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

using slice_range_list = std::vector<slice_range>;
using slice_range_map = std::unordered_map<int, slice_range_list>;

enum class infer_status_code : int {
    OK = 0, // Successful
    RETRY, // Need retry another anchor
    FAIL, // Could not infer
    UNKNOWN, // Unknown
    END,
};

expr do_cast_and_fold(const expr &);
struct infer_status_map_t {
private:
    context_ptr ctx_;
    bool recursive_mode_;
    std::vector<std::unordered_set<sc_op_ptr>> inf_stat_map_;

public:
    infer_status_map_t(context_ptr ctx = nullptr, bool recursive_mode = true)
        : ctx_(ctx), recursive_mode_(recursive_mode) {
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

    const bool is_recursive_mode() const { return recursive_mode_; }

    const context_ptr get_context() const { return ctx_; }

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
        ret.emplace_back(do_cast_and_fold(r.first));
    }
    return ret;
}

inline std::vector<expr> get_slice_shape(const slice_range &range) {
    std::vector<expr> ret;
    for (auto &r : range) {
        ret.emplace_back(do_cast_and_fold(r.second));
    }
    return ret;
}

inline slice_range gen_slice_by_dims(const sc_dims &dims) {
    slice_range ret;
    for (auto &r : dims) {
        ret.emplace_back(std::make_pair(expr(0), dim2unsigned(r)));
    }
    return ret;
}

inline slice_range gen_slice_by_dims_expr(const std::vector<expr> &dims) {
    slice_range ret;
    for (auto &r : dims) {
        ret.emplace_back(std::make_pair(dim2unsigned(0), r));
    }
    return ret;
}

bool is_reshaped_tensor(const expr &tsr);

expr transform_tsr2stsr_with_range(const expr &tsr, const slice_range &range);

expr transform_tsl2stsr(const tensor_slice &tsl);

expr transform_tsr2tptr_with_range(const expr &tsr, const slice_range &range);

expr transform_tptr2stsr(const expr &tptr);

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

    fusion_data_t(fusion_data_t &&moved)
        : use_count_(moved.use_count_)
        , need_alloc_(moved.need_alloc_)
        , buffer_(moved.buffer_) {}
    bool buffer_allocated() const { return buffer_.defined(); }
    void set_buffer(bool is_dynamic, const expr &buf);
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

// common fusion anchor
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

// iterated fusion anchor
struct iter_fuse_anchor_t {
    stmts anchor_position_;
    expr iter_;
    expr tsr_;
    slice_range_list slice_list_;
    stmt dispatch_helper_;
    iter_fuse_anchor_t() = default;
    iter_fuse_anchor_t(stmts pos, expr iter, expr tsr,
            slice_range_list slice_list, stmt dispatch_helper)
        : anchor_position_(pos)
        , iter_(iter)
        , tsr_(tsr)
        , slice_list_(slice_list)
        , dispatch_helper_(dispatch_helper) {}
};

// grouped fusion anchor
struct grouped_fuse_anchor_t {
    stmts anchor_position_;
    std::unordered_map<expr, slice_range_list> expr_slice_map_;
    grouped_fuse_anchor_t() = default;
    grouped_fuse_anchor_t(stmts pos,
            std::unordered_map<expr, slice_range_list> expr_slice_map)
        : anchor_position_(std::move(pos))
        , expr_slice_map_(std::move(expr_slice_map)) {};
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::graph::gc::buffer_reuse_identity> {
    std::size_t operator()(
            const dnnl::impl::graph::gc::buffer_reuse_identity &in) const;
};
} // namespace std
#endif
