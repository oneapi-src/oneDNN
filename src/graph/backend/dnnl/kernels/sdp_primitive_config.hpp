/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_PRIMITIVE_CONFIG_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_PRIMITIVE_CONFIG_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/primitive.hpp"
#include "common/sdpa_utils.hpp"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/value.hpp"

#include "graph/backend/dnnl/subgraph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_ptr = std::shared_ptr<op_t>;
using ltw = logical_tensor_wrapper_t;

struct sdp_primitive_config_t {
public:
    sdp_primitive_config_t() = default;

    std::shared_ptr<op_t> mm1_ = nullptr;
    std::shared_ptr<op_t> mm2_ = nullptr;

    // for paged attention
    std::shared_ptr<op_t> k_paged_cache_load_ = nullptr;
    std::shared_ptr<op_t> v_paged_cache_load_ = nullptr;

    std::shared_ptr<value_t> q_ = nullptr;
    std::shared_ptr<value_t> k_ = nullptr;
    std::shared_ptr<value_t> v_ = nullptr;
    std::shared_ptr<value_t> dst_ = nullptr;
    std::shared_ptr<value_t> scale_ = nullptr;
    std::shared_ptr<value_t> attn_mask_ = nullptr;

    std::shared_ptr<value_t> k_scale_ = nullptr;
    std::shared_ptr<value_t> v_scale_ = nullptr;

    std::shared_ptr<value_t> k_zero_points_ = nullptr;
    std::shared_ptr<value_t> v_zero_points_ = nullptr;

    // for paged attention
    std::shared_ptr<value_t> k_cache_ = nullptr;
    std::shared_ptr<value_t> v_cache_ = nullptr;
    std::shared_ptr<value_t> block_table_ = nullptr;

    bool invert_scale_ = false;
    bool quantized_ = false;
    bool causal_mask_ = false;
    bool page_attention_enabled_ = false;
    dim_t kv_head_number_;

    // SDP pd and primitive.
    std::shared_ptr<primitive_desc_t> sdpa_pd_;
    std::shared_ptr<primitive_t> sdpa_prim_;

    // // paged attention SDP pd and primitive.
    std::shared_ptr<dnnl::sdpa_micro::primitive_desc> paged_sdpa_pd_;
    std::shared_ptr<dnnl::sdpa_micro::primitive> paged_sdpa_prim_;
    memory prompt_lens;
    memory subsequence_begins;
    memory block_indices_begin;

private:
    op_ptr get_post_op(const op_ptr &op) const;
    op_ptr get_paged_cache_load_op(const op_ptr &op) const;

public:
    status_t locate_io(std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs);

    // The function is used to check if the configuration of SDP is supported by
    // current implementation of micro kernel. Refer to the following limitation:
    // 1. only support limited pattern, variants with select op are not supported
    // 2. only support fp16 data type
    // 3. only support 4-dims tensor
    status_t initial_check(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs);

    // Initialize parameters and primitive.
    status_t init(std::shared_ptr<subgraph_t> &sg, const dnnl::engine &p_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs);
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
