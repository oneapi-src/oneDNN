/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "graph/backend/dnnl/kernels/large_partition.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t larger_partition_kernel_t::compile_impl(
        const dnnl_partition_impl_t *part, const engine_t *g_engine,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    p_engine_ = make_dnnl_engine(*g_engine);
    g_alloc_
            = reinterpret_cast<graph::allocator_t *>(g_engine->get_allocator());

    // get subgraph from the deep copied partition
    subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_,
            part->get_fpmath_mode(), part->get_use_blocked_layout(), true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    // Populate the transform passes into the pipeline
    // Note: `std::call_once` should be kept in a single translation unit since
    // GCC 11.
    std::call_once(once_flag_, [&, this]() {
        vis_ = subgraph_visualizer_t(part->id(), [this](const value_t *val) {
            return this->memory_planner_.get_memory_info(val);
        });
        pipeline_ = pass_pipeline_t(vis_);
        setup_pipeline(pipeline_, memory_planner_, enabled_constant_cache());
    });

    // Run the added passes
    BACKEND_DNNL_CHECK(pipeline_.run(subgraph_));

    // fill information for inputs logical tensors
    for (size_t i = 0; i < inputs.size(); i++) {
        auto &in = const_cast<logical_tensor_t &>(inputs[i]);
        in = subgraph_->ins_[i];
    }

    // fill information for outputs logical tensors
    for (size_t i = 0; i < outputs.size(); i++) {
        auto &out = const_cast<logical_tensor_t &>(outputs[i]);
        out = subgraph_->outs_[i];
    }

    resource_ctor_ = [this]() {
        return this->memory_planner_.get_exec_args_set().clone();
    };

    constant_key_ = generate_constant_cache_key(part->id(),
            memory_planner_.get_exec_args_set().get_persistent_mem_desc_list());

    return status::success;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
