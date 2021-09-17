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

#ifndef BACKEND_DNNL_KERNELS_INT8_POOL_HPP
#define BACKEND_DNNL_KERNELS_INT8_POOL_HPP

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/passes/compile_ops.hpp"
#include "backend/dnnl/passes/infer_type.hpp"
#include "backend/dnnl/passes/insert_ops.hpp"
#include "backend/dnnl/passes/layout_propagation.hpp"
#include "backend/dnnl/passes/lower_down.hpp"
#include "backend/dnnl/passes/memory_planning.hpp"
#include "backend/dnnl/passes/op_executable.hpp"
#include "backend/dnnl/thread_local_cache.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

struct quantized_pooling : public kernel_base {
private:
    dnnl::engine p_engine_;
    impl::allocator_t *g_alloc_;

    primitive_attr_mgr prm_attr_mgr_;
    memory_planner_t memory_planner_;
    executable_mgr exec_mgr_;

    std::vector<op_executable *> execs_;
    std::vector<exec_args> exec_args_;

    std::vector<std::shared_ptr<impl::op_t>> opt_subgraph_;

    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;

    pd_cache_t pd_cache_;

public:
    virtual ~quantized_pooling() {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
    }

    virtual impl::status_t compile_impl(const dnnl_partition_impl_t *part,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        // TODO(wuxun): since oneDNN pooling primitive only support u8u8 or
        // s8s8 on CPU device for now, we need to check whether the data types
        // between input and output are compatible. If we enable this check in
        // op schema or primitive supports u8s8/s8u8, then this check can be
        // safely removed.
        if (inputs[0].data_type != outputs[0].data_type)
            return status::unsupported;

        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = g_engine->get_allocator();

        std::vector<std::shared_ptr<op_t>> subgraph = part->get_ops();

        set_all_layout_to_any(subgraph);

        // have to set the given inputs and outputs before infer shape and
        // compile
        set_given_inputs_outputs(subgraph, inputs, outputs);

        // for those primitive ops like pooling, it requires the scales and zps
        // between input tensor and output tensor are the same. So here, we
        // don't need to split Dequant and Quant ops firstly, it should be okay
        // to directly fuse Dequant and Quant into int8 pool op.
        fuse_to_int8_pool(subgraph);

        insert_permute(subgraph);
        insert_reorder(subgraph);

        subgraph_visualizer_t vis(part->id());
        vis.run(subgraph, "after_lower_down", false);

        impl::graph_t agraph(subgraph);
        BACKEND_DNNL_CHECK(agraph.infer_shape());
        BACKEND_DNNL_CHECK(infer_type(agraph));

        vis.run(subgraph, "after_infer_shape_infer_type", true);

        BACKEND_DNNL_CHECK(layout_propagation(
                subgraph, p_engine_, prm_attr_mgr_, pd_cache_));

        vis.run(subgraph, "after_layout_propagation", true);

        // fill layout information for outputs logical tensors
        for (size_t i = 0; i < outputs.size(); i++) {
            for (auto out_val : impl::graph_t(subgraph).get_output_values()) {
                auto compiled_lt = out_val->get_logical_tensor();
                if (compiled_lt.id == outputs[i].id) {
                    auto lt = const_cast<impl::logical_tensor_t *>(&outputs[i]);
                    auto md = make_dnnl_memory_desc(compiled_lt);
                    lt->ndims = compiled_lt.ndims;
                    impl::utils::array_copy(
                            lt->dims, compiled_lt.dims, DNNL_GRAPH_MAX_NDIMS);
                    impl::utils::array_copy(lt->layout.strides,
                            compiled_lt.layout.strides, DNNL_GRAPH_MAX_NDIMS);
                    fill_layout_info(lt, md);
                }
            }
        }

        // bind the memory for each op
        BACKEND_DNNL_CHECK(memory_planner_.run(
                subgraph, inputs, outputs, p_engine_, prm_attr_mgr_));

        vis.run(subgraph, "after_memory_planning", true, true,
                [this](const value_t *val) {
                    return this->memory_planner_.get_memory_info(val);
                });

        BACKEND_DNNL_CHECK(compile_ops(
                subgraph, p_engine_, prm_attr_mgr_, exec_mgr_, pd_cache_));

        // topologically sort the executables
        impl::topo_order_visit(impl::graph_t(subgraph).get_output_ops(),
                [this](impl::op_t *op) {
                    auto exec_key = op->get_attr<int64_t>("executable_key");
                    auto &exec = exec_mgr_.get_executable(exec_key);
                    execs_.emplace_back(exec.get());

                    return impl::status::success;
                });

        opt_subgraph_ = subgraph;

        // generate a hash key for exec_args_mgr
        resource_ctor_ = [this]() {
            return this->memory_planner_.get_exec_args_set().clone();
        };

        return impl::status::success;
    }

    virtual impl::status_t execute_impl(const dnnl_partition_impl_t *part,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(part);
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        // each thread's own local resource
        thread_local_cache_t<execution_args_set_t> res_cache;
        execution_args_set_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
            mem_idx.first.set_data_handle(
                    inputs[mem_idx.second].get_data_handle());
        }
        for (const auto &mem_idx : res->get_mems_use_external_outputs()) {
            mem_idx.first.set_data_handle(
                    outputs[mem_idx.second].get_data_handle());
        }

        temporary_scratchpad_t scratchpad(
                memory_planner_.total_internal_temporary_size(), p_engine_,
                *g_alloc_);
        assertm(scratchpad.size()
                        >= memory_planner_.total_internal_temporary_size(),
                "no enough scratchpad memory");
        grantor_t var_grantor = memory_planner_.internal_temporary_grantor(
                scratchpad.get_buffer());

        registry_t::key_t key = 0;
        for (auto &mem_offkey : res->get_mems_use_internal_temporary()) {
            mem_offkey.first.set_data_handle(
                    var_grantor.get(mem_offkey.second));
        }

        for (size_t i = 0; i < execs_.size(); i++) {
            execs_[i]->execute(p_stream, res->get_exec_args()[i]);
        }

        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
