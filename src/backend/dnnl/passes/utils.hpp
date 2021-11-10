/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
#ifndef BACKEND_DNNL_PASSES_UTILS_HPP
#define BACKEND_DNNL_PASSES_UTILS_HPP

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/op.hpp"
#include "interface/value.hpp"
#include "utils/utils.hpp"

#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

void insert_op_before(std::shared_ptr<impl::op_t> &inserted_op,
        std::shared_ptr<impl::op_t> &base_op, size_t offset);

void insert_op_before(op_t *inserted_op, op_t *base_op, size_t offset);

void insert_op_after(std::shared_ptr<impl::op_t> &inserted_op,
        std::shared_ptr<impl::op_t> &base_op, size_t offset);

void fuse_op_to_successor(
        op_t *op, std::vector<std::shared_ptr<op_t>> &subgraph);

void fuse_op_to_predecessor(op_t *op,
        std::vector<std::shared_ptr<op_t>> &subgraph, size_t in_offset = 0);

status_t set_given_inputs_outputs(std::vector<std::shared_ptr<op_t>> &subgraph,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs);

void set_all_layout_to_any(std::vector<std::shared_ptr<op_t>> &subgraph);

void set_weight_bias_constant(std::vector<std::shared_ptr<op_t>> &subgraph);

inline bool is_preprocess_op(impl::op_t &op) {
    static std::set<impl::op_kind_t> preprocess_ops = {op_kind::permute,
            op_kind::to_group, op_kind::expand, op_kind::squeeze};
    return preprocess_ops.count(op.get_kind()) != 0;
}

inline bool is_inplace(op_t &op) {
    if (is_preprocess_op(op)) return true;

    const static std::set<impl::op_kind_t> ops {
            op_kind::mul_scales, op_kind::add_zps};
    return ops.count(op.get_kind()) != 0;
}

class subgraph_visualizer_t {
public:
    subgraph_visualizer_t(size_t partition_id)
        : partition_id_(partition_id), enabled_(false), index_(0) {
        // Set DNNL_GRAPH_DUMP=2 to enable dump subgraph
        enabled_ = impl::utils::getenv_int("DNNL_GRAPH_DUMP", 0) > 1;
    }

    status_t run(const std::vector<std::shared_ptr<op_t>> &subgraph,
            const std::string &name_suffix, bool is_layout_sensitive,
            bool is_memory_sensitive = false,
            const std::function<std::string(const value_t *)> &mem_info_func
            = {});

private:
    size_t partition_id_;
    bool enabled_;
    size_t index_;
};

void replace_op(std::shared_ptr<op_t> &org_op, std::shared_ptr<op_t> &new_op);

using pd_cache_t = std::unordered_map<op_t *, dnnl::primitive_desc>;

inline const std::map<op_kind_t, dnnl::algorithm> &get_eltwise_alg_map() {
    static const std::map<op_kind_t, dnnl::algorithm> &eltwise_alg_map
            = {{impl::op_kind::Abs, dnnl::algorithm::eltwise_abs},
                    {impl::op_kind::Elu, dnnl::algorithm::eltwise_elu},
                    {impl::op_kind::Exp, dnnl::algorithm::eltwise_exp},
                    {impl::op_kind::GELU, dnnl::algorithm::eltwise_gelu_erf},
                    {impl::op_kind::HardTanh, dnnl::algorithm::eltwise_clip},
                    {impl::op_kind::Log, dnnl::algorithm::eltwise_log},
                    {impl::op_kind::ReLU, dnnl::algorithm::eltwise_relu},
                    {impl::op_kind::Round, dnnl::algorithm::eltwise_round},
                    {impl::op_kind::Sigmoid, dnnl::algorithm::eltwise_logistic},
                    {impl::op_kind::Sqrt, dnnl::algorithm::eltwise_sqrt},
                    {impl::op_kind::Square, dnnl::algorithm::eltwise_square},
                    {op_kind::dnnl_swish, dnnl::algorithm::eltwise_swish},
                    {impl::op_kind::Tanh, dnnl::algorithm::eltwise_tanh}};
    return eltwise_alg_map;
}

inline bool is_eltwise_kind(op_kind_t kind) {
    const std::set<op_kind_t> eltwise_kinds {impl::op_kind::Abs,
            impl::op_kind::Elu, impl::op_kind::Exp, impl::op_kind::GELU,
            impl::op_kind::HardTanh, impl::op_kind::Log, impl::op_kind::ReLU,
            impl::op_kind::Round, impl::op_kind::Sigmoid, impl::op_kind::Sqrt,
            impl::op_kind::Square, op_kind::dnnl_swish, impl::op_kind::Tanh};
    return eltwise_kinds.find(kind) != eltwise_kinds.end();
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
