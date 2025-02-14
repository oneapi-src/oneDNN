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

#include "graph/backend/dnnl/kernels/sdp_primitive.hpp"

#include "common/sdpa_pd.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/stream.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "gpu/intel/sycl/stream.hpp"
#endif

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#include "graph/backend/dnnl/op_executable.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

template <bool quantized>
status_t sdp_primitive_kernel_t<quantized>::compile_impl(
        const dnnl_partition_impl_t *part, const engine_t *g_engine,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    p_engine_ = make_dnnl_engine(*g_engine);
    g_alloc_
            = reinterpret_cast<graph::allocator_t *>(g_engine->get_allocator());

    // First, dry run on a deep copy
    subgraph_
            = std::make_shared<subgraph_t>(graph_t::deep_copy(part->get_ops()),
                    p_engine_, part->get_fpmath_mode(), false, true);
    CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    CHECK(cfg_.initial_check(subgraph_, inputs));

    subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
        return this->memory_planner_.get_memory_info(val);
    });
    pass_pipeline_t pipeline = pass_pipeline_t(vis);

    BACKEND_DNNL_ADD_PASS(pipeline, lower_down);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_implicit_causal_mask);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_reshape_for_gqa);
    if (quantized) {
        BACKEND_DNNL_ADD_PASS(pipeline, lift_up_typecast);
        BACKEND_DNNL_ADD_PASS(pipeline, lift_up_quantize);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_typecast_to_matmul_or_conv);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_typecast_to_predecessor);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_src_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_src_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, insert_runtime_u8_to_s8_for_matmul);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_zero_points);
    }
    BACKEND_DNNL_ADD_PASS(pipeline, binary_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_matmul);
    if (quantized) {
        BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);
    }

    pipeline.reset_visualize_arg(true, false);
    BACKEND_DNNL_ADD_PASS(pipeline, infer_shape);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_transpose_to_matmul);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_transpose_to_matmul);
    BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

    // bind the memory for each op
    auto memory_plan = [&](std::shared_ptr<subgraph_t> &sg) {
        return memory_planner_.run(sg);
    };
    pipeline.reset_visualize_arg(true, true);
    BACKEND_DNNL_ADD_PASS(pipeline, memory_plan);

    auto modify_subgraph = [&] {
        // Run the added passes
        CHECK(pipeline.run(subgraph_));

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

        return status::success;
    };

    resource_ctor_ = [this]() {
        return this->memory_planner_.get_exec_args_set().clone();
    };

    CHECK(modify_subgraph());

    cfg_.quantized_ = quantized;
    CHECK(cfg_.init(subgraph_, p_engine_, inputs, outputs));

    return status::success;
}

template <bool quantized>
void sdp_primitive_kernel_t<quantized>::prepare_args_set(
        const execution_args_set_t *res, const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const scratchpad_t &scratchpad) {
    // update the data of partition in/outputs args
    for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
        mem_idx.first.set_data_handle(inputs[mem_idx.second].get_data_handle());
    }
    for (const auto &mem_idx : res->get_mems_use_external_outputs()) {
        mem_idx.first.set_data_handle(
                outputs[mem_idx.second].get_data_handle());
    }
}

template <bool quantized>
status_t sdp_primitive_kernel_t<quantized>::get_prim_exec_args(
        exec_args_t &args, memory (&mem_storage)[10],
        const execution_args_set_t *res) const {
    bool ok = res->find_value_mem_map(cfg_.q_.get(), mem_storage[0])
            && res->find_value_mem_map(cfg_.k_.get(), mem_storage[1])
            && res->find_value_mem_map(cfg_.v_.get(), mem_storage[2])
            && res->find_value_mem_map(cfg_.dst_.get(), mem_storage[3]);

    if (cfg_.scale_)
        ok = ok && res->find_value_mem_map(cfg_.scale_.get(), mem_storage[4]);
    if (cfg_.attn_mask_)
        ok = ok
                && res->find_value_mem_map(
                        cfg_.attn_mask_.get(), mem_storage[5]);
    if (quantized && !(cfg_.k_scale_ || cfg_.v_scale_))
        return status::invalid_arguments;
    if (cfg_.k_scale_)
        ok = ok && res->find_value_mem_map(cfg_.k_scale_.get(), mem_storage[6]);
    if (cfg_.v_scale_)
        ok = ok && res->find_value_mem_map(cfg_.v_scale_.get(), mem_storage[7]);

    if (cfg_.k_zero_points_)
        ok = ok
                && res->find_value_mem_map(
                        cfg_.k_zero_points_.get(), mem_storage[8]);
    if (cfg_.v_zero_points_)
        ok = ok
                && res->find_value_mem_map(
                        cfg_.v_zero_points_.get(), mem_storage[9]);

    VCONDCHECK(graph, exec, check, sdp_primitive_kernel, ok,
            status::runtime_error,
            "sdp_primitive_kernel get_prim_exec_args failed");

    memory_arg_t mem_arg_q = {mem_storage[0].get(), true};
    memory_arg_t mem_arg_k = {mem_storage[1].get(), true};
    memory_arg_t mem_arg_v = {mem_storage[2].get(), true};
    memory_arg_t mem_arg_dst = {mem_storage[3].get(), false};
    memory_arg_t mem_arg_scale = {mem_storage[4].get(true), true};
    memory_arg_t mem_arg_mask = {mem_storage[5].get(true), true};
    memory_arg_t mem_arg_k_scale = {mem_storage[6].get(true), true};
    memory_arg_t mem_arg_v_scale = {mem_storage[7].get(true), true};
    memory_arg_t mem_arg_k_zero_points = {mem_storage[8].get(true), true};
    memory_arg_t mem_arg_v_zero_points = {mem_storage[9].get(true), true};

    args.clear();
    args[DNNL_ARG_QUERIES] = mem_arg_q;
    args[DNNL_ARG_KEYS] = mem_arg_k;
    args[DNNL_ARG_VALUES] = mem_arg_v;
    args[DNNL_ARG_DST] = mem_arg_dst;
    args[DNNL_ARG_SCALE] = mem_arg_scale;
    args[DNNL_ARG_ATTN_MASK] = mem_arg_mask;
    args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS] = mem_arg_k_scale;
    args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES] = mem_arg_v_scale;
    args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS] = mem_arg_k_zero_points;
    args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES] = mem_arg_v_zero_points;

    return status::success;
}

template <bool quantized>
status_t sdp_primitive_kernel_t<quantized>::execute_impl(
        const stream_t *g_stream, const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs) {
    dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    // Micro kernel doesn't use scratchpad memory, here we force-set size as
    // zero to avoid redundant memory allocation and deallocation.
    temporary_scratchpad_t scratchpad(0, p_engine_, *g_alloc_);
    prepare_args_set(res, inputs, outputs, scratchpad);

    memory mem_storage[10];
    exec_args_t args;
    CHECK(get_prim_exec_args(args, mem_storage, res));
    exec_ctx_t ctx(p_stream.get(), std::move(args));

    return cfg_.sdpa_prim_->execute(ctx);
}

#ifdef DNNL_WITH_SYCL
template <bool quantized>
status_t sdp_primitive_kernel_t<quantized>::sycl_execute_impl(
        const stream_t *g_stream, const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs,
        const std::vector<::sycl::event> &sycl_deps,
        ::sycl::event *sycl_event) {

    dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    // Micro kernel doesn't use scratchpad memory, here we force-set size as
    // zero to avoid redundant memory allocation and deallocation.
    temporary_scratchpad_t scratchpad(0, p_engine_, *g_alloc_);
    prepare_args_set(res, inputs, outputs, scratchpad);

    memory mem_storage[10];
    exec_args_t args;
    CHECK(get_prim_exec_args(args, mem_storage, res));
    exec_ctx_t ctx(p_stream.get(), std::move(args));

    // Relying on the library's internals here. Since graph API is currently
    // enabled only for the Intel vendor it is fine to cast stream to
    // gpu::intel::sycl::stream_t unconditionally.
    auto *sycl_stream = dnnl::impl::utils::downcast<
            dnnl::impl::gpu::intel::sycl::stream_t *>(p_stream.get());

    sycl_stream->before_exec_hook();

    if (!sycl_deps.empty()) sycl_stream->sycl_ctx().set_deps(sycl_deps);

    auto status = cfg_.sdpa_prim_->execute(ctx);

    auto return_event = sycl_stream->get_output_event();

    scratchpad.set_deps(return_event);
    if (sycl_event) *sycl_event = return_event;

    sycl_stream->after_exec_hook();

    return status;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
template <bool quantized>
status_t sdp_primitive_kernel_t<quantized>::ocl_execute_impl(
        const stream_t *g_stream, const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs,
        const std::vector<cl_event> &cl_deps, cl_event *ret_event) {

    dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    // Micro kernel doesn't use scratchpad memory, here we force-set size as
    // zero to avoid redundant memory allocation and deallocation.
    temporary_scratchpad_t scratchpad(0, p_engine_, *g_alloc_);
    prepare_args_set(res, inputs, outputs, scratchpad);

    memory mem_storage[10];
    exec_args_t args;
    CHECK(get_prim_exec_args(args, mem_storage, res));
    exec_ctx_t ctx(p_stream.get(), std::move(args));

    // TODO (pc): refactor
    auto *ocl_stream = dnnl::impl::utils::downcast<gpu::intel::ocl::stream_t *>(
            p_stream.get());

    ocl_stream->before_exec_hook();

    if (!cl_deps.empty()) {
        std::vector<xpu::ocl::wrapper_t<cl_event>> events(cl_deps.size());
        for (size_t i = 0; i < cl_deps.size(); i++)
            events[i] = xpu::ocl::wrapper_t<cl_event>(cl_deps[i], true);
        ocl_stream->ocl_ctx().set_deps(events);
    }

    auto status = cfg_.sdpa_prim_->execute(ctx);

    cl_event return_event = nullptr;
    if ((ocl_stream->flags() & stream_flags::in_order) == 0) {
        auto last = ocl_stream->get_output_event();
        return_event = last.release();
    }

    scratchpad.set_deps(return_event);
    if (ret_event) *ret_event = return_event;

    ocl_stream->after_exec_hook();

    return status;
}
#endif

template struct sdp_primitive_kernel_t<true>;
template struct sdp_primitive_kernel_t<false>;

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
