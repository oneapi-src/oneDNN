/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#ifndef BACKEND_DNNL_SUBGRAPH_HPP
#define BACKEND_DNNL_SUBGRAPH_HPP

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/op.hpp"
#include "interface/value.hpp"
#include "utils/utils.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/fusion_info.hpp"
#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

struct op_executable_t;

// The subgraph_t class is a subclass of graph_t, which is used as the only
// parameter of transformation passes. Each transformation pass will process the
// subgraph_t object, and after that, the content of subgraph_t object will be
// changed.
class subgraph_t : public impl::graph_t {
public:
    subgraph_t(const std::vector<op_ptr> &ops, const dnnl::engine &eng,
            bool reset_layout = true);

    subgraph_t(const std::vector<op_ptr> &ops, const dnnl::engine &eng,
            impl::fpmath_mode_t fpm_mode, bool reset_layout);

    subgraph_t(const std::vector<op_ptr> &ops, bool reset_layout = true);

    std::vector<op_ptr> &get_mutable_ops() {
        return const_cast<std::vector<op_ptr> &>(get_ops());
    }

    // The inputs and outputs logical tensors given by users at compilation
    // stage
    std::vector<impl::logical_tensor_t> ins_;
    std::vector<impl::logical_tensor_t> outs_;

    // The engine that the subgraph is compiled for
    const dnnl::engine *p_engine_;

    // This manager holds each op's fusion information
    fusion_info_mgr_t fusion_info_mgr_;

    // The custom cache to store the created primitive desc
    pd_cache_t pd_cache_;

    // The vector to tell which op in the subgraph is constant and will only run
    // once
    std::vector<bool> is_constant_;

    // The executable for each op in subgraph
    std::vector<std::shared_ptr<op_executable_t>> execs_;
};

class subgraph_visualizer_t {
public:
    subgraph_visualizer_t() = default;

    subgraph_visualizer_t(size_t partition_id,
            const std::function<std::string(const value_t *)> &mem_info_func
            = {})
        : enabled_(false)
        , mem_info_func_(mem_info_func)
#ifdef DNNL_GRAPH_ENABLE_DUMP
        , partition_id_(partition_id)
        , index_(0)
#endif
    {
        MAYBE_UNUSED(partition_id);
        // Set _DNNL_GRAPH_BACKEND_SUBGRAPH_DUMP=1 to enable dump subgraph
        enabled_ = impl::utils::getenv_int_internal("BACKEND_SUBGRAPH_DUMP", 0)
                > 0;
    }

    status_t run(const std::shared_ptr<subgraph_t> &sg,
            const std::string &name_suffix, bool is_layout_sensitive,
            bool is_memory_sensitive = false);

private:
    bool enabled_;
    std::function<std::string(const value_t *)> mem_info_func_;
#ifdef DNNL_GRAPH_ENABLE_DUMP
    size_t partition_id_;
    size_t index_;
#endif
};

class subgraph_validator_t {
public:
    subgraph_validator_t() = default;
    status_t run(const std::shared_ptr<subgraph_t> &sg);
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
