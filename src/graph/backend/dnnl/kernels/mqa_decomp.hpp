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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_MQA_DECOMP_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_MQA_DECOMP_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/backend/dnnl/kernels/kernel_base.hpp"
#include "graph/backend/dnnl/kernels/mqa_decomp_config.hpp"

#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"
#include "graph/backend/dnnl/dnnl_partition_impl.hpp"
#include "graph/backend/dnnl/op_executable.hpp"
#include "graph/backend/dnnl/scratchpad.hpp"
#include "graph/backend/dnnl/subgraph.hpp"
#include "graph/backend/dnnl/thread_local_cache.hpp"

#include "graph/backend/dnnl/passes/memory_planning.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// The second template param dt is used to indicate the internal data type of
// int8 mqa pattern. It doesn't take any effect if quantized param is false.
template <bool quantized = false, memory::data_type dt = memory::data_type::f32>
struct mqa_decomp_kernel_t : public kernel_base_t {
private:
    allocator_t *g_alloc_ = nullptr;
    // used for mqa internal memory planning
    registry_t mqa_registry_;
    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;
    subgraph_visualizer_t vis_;

    // MQA-related params
    mqa_decomp_config_t mqa_cfg_;

public:
    mqa_decomp_kernel_t() {
        thread_local_cache_t<mqa_args_set_t> res_cache;
        res_cache.retain();
    }

    ~mqa_decomp_kernel_t() override {
        thread_local_cache_t<mqa_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
        res_cache.release();
    }

    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    void prepare_sub_args(const grantor_t &var_grantor, const int id,
            const size_t block_size,
            std::unordered_map<dnnl_memory_t, std::vector<memory>> &mem_map);

    status_t execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override;

    class mqa_args_set_t {
    public:
        mqa_args_set_t(mqa_decomp_kernel_t<quantized, dt> *mqa_kernel) {
            int nthr = mqa_kernel->mqa_cfg_.nthr;
            //construct new args
            auto args_ctor = [this, nthr](const std::unordered_map<int, memory>
                                                  &ori_args,
                                     std::vector<std::unordered_map<int,
                                             memory>> &args) {
                args.resize(nthr);
                for (auto iter : ori_args) {
                    memory ori_mem = iter.second;
                    if (mem_map.count(ori_mem.get()) == 0) {
                        //construct new memory
                        mem_map[ori_mem.get()] = std::vector<memory>(nthr);
                        for (int tid = 0; tid < nthr; tid++) {
                            mem_map[ori_mem.get()][tid]
                                    = memory(ori_mem.get_desc(),
                                            ori_mem.get_engine(), nullptr);
                            if (iter.first >= DNNL_ARG_ATTR_SCALES
                                    && iter.first <= DNNL_ARG_ATTR_POST_OP_DW) {
                                mem_map[ori_mem.get()][tid].set_data_handle(
                                        ori_mem.get_data_handle());
                            }
                        }
                    }
                    for (int tid = 0; tid < nthr; tid++) {
                        args[tid].insert(
                                {iter.first, mem_map[ori_mem.get()][tid]});
                    }
                }
            };
            args_ctor(
                    mqa_kernel->mqa_cfg_.sub_reorder0_args, sub_reorder0_args);
            args_ctor(
                    mqa_kernel->mqa_cfg_.sub_reorder1_args, sub_reorder1_args);
            args_ctor(mqa_kernel->mqa_cfg_.sub_mm1_args, sub_mm1_args);
            args_ctor(mqa_kernel->mqa_cfg_.sub_softmax_args, sub_softmax_args);
            args_ctor(
                    mqa_kernel->mqa_cfg_.sub_reorder2_args, sub_reorder2_args);
            args_ctor(mqa_kernel->mqa_cfg_.sub_mm2_args, sub_mm2_args);
            args_ctor(
                    mqa_kernel->mqa_cfg_.sub_reorder3_args, sub_reorder3_args);
        }
        std::unordered_map<dnnl_memory_t, std::vector<memory>> mem_map;
        // execution args for each op in the subgraph
        std::vector<std::unordered_map<int, memory>> sub_reorder0_args,
                sub_reorder1_args, sub_mm1_args, sub_softmax_args,
                sub_reorder2_args, sub_mm2_args, sub_reorder3_args;
    };

    std::function<std::shared_ptr<mqa_args_set_t>()> resource_ctor_;

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        UNUSED(g_stream);
        UNUSED(inputs);
        UNUSED(outputs);
        UNUSED(sycl_deps);
        UNUSED(sycl_event);
        return status::unimplemented;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &cl_deps,
            cl_event *ret_event) override {
        UNUSED(g_stream);
        UNUSED(inputs);
        UNUSED(outputs);
        UNUSED(cl_deps);
        UNUSED(ret_event);
        return status::unimplemented;
    }
#endif

    DEF_KERNEL_METHOD_STR(mqa_decomp_kernel_t)
    DNNL_DISALLOW_COPY_AND_ASSIGN(mqa_decomp_kernel_t)
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
