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

#include "graph/backend/dnnl/kernels/decomp_kernel/sdp_decomp.hpp"
#include "graph/backend/dnnl/kernels/utils.hpp"

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#include "graph/backend/dnnl/op_executable.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "cpu/cpu_stream.hpp"
#include "oneapi/dnnl/dnnl_threadpool.h"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
template <bool quantized, memory::data_type dt>
status_t sdp_decomp_kernel_t<quantized, dt>::compile_impl(
        const dnnl_partition_impl_t *part, const engine_t *g_engine,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    if (g_engine->kind() != engine_kind::cpu || !enable_decomp_kernel())
        return status::unimplemented;
    p_engine_ = make_dnnl_engine(*g_engine);
    g_alloc_
            = reinterpret_cast<graph::allocator_t *>(g_engine->get_allocator());

    // get subgraph from the deep copied partition
    subgraph_ = std::make_shared<subgraph_t>(
            part->get_ops(), p_engine_, part->get_fpmath_mode(), false, true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    // Check if it's supported by decomposition kernel
    if (!sdp_cfg_.initial_check(subgraph_, inputs))
        return status::unimplemented;

    subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
        return this->memory_planner_.get_memory_info(val);
    });
    pass_pipeline_t pipeline = pass_pipeline_t(vis);
    pass_pipeline_t select_pipeline = pass_pipeline_t(vis);
    BACKEND_DNNL_ADD_PASS(pipeline, lower_down);
    // Decompose select to binary ops if necessary
    BACKEND_DNNL_ADD_PASS(pipeline, decompose_select_to_binary_ops);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_reshape_for_gqa);
    // Fusion and canonicalization passes begin
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
    }
    BACKEND_DNNL_ADD_PASS(pipeline, binary_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_matmul);
    if (quantized) {
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);
    }
    pipeline.reset_visualize_arg(true, false);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_transpose_to_matmul);
    BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

    // Run the added passes
    BACKEND_DNNL_CHECK(pipeline.run(subgraph_));

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

    resource_ctor_
            = [this]() { return std::make_shared<sdp_args_set_t>(this); };

    // Initialize and construct kernel params
    sdp_cfg_.construct_params<quantized, dt>(
            subgraph_, sdp_registry_, p_engine_, inputs);

    // Create a new subgraph for select. the select_out_ops is the new
    // subgraph's output ops. The out values of these out_ops are the
    // connection between two graphs
    std::vector<op_ptr> select_out_ops;
    if (sdp_cfg_.has_select) {
        sdp_cfg_.record_select_ops(subgraph_, select_out_ops);
        select_subgraph_ = std::make_shared<subgraph_t>(sdp_cfg_.select_op,
                p_engine_, part->get_fpmath_mode(),
                part->get_use_blocked_layout(), false);

        const std::vector<logical_tensor_t> select_inputs
                = {inputs[sdp_cfg_.graph_inport[5]],
                        inputs[sdp_cfg_.graph_inport[6]]};

        select_subgraph_->ins_ = select_inputs;
        BACKEND_DNNL_ADD_PASS(select_pipeline, replace_select_values);

        // do constant propagation again since layout propagation may
        // insert/delete operators
        if (enabled_constant_cache()) {
            BACKEND_DNNL_ADD_PASS(select_pipeline, constant_propagation);
        }

        // bind the memory for each op
        auto memory_plan = [&](std::shared_ptr<subgraph_t> &sg) {
            return memory_planner_.run(sg);
        };
        select_pipeline.reset_visualize_arg(true, true);
        BACKEND_DNNL_ADD_PASS(select_pipeline, memory_plan);
        BACKEND_DNNL_ADD_PASS(select_pipeline, compile_ops);

        BACKEND_DNNL_CHECK(select_pipeline.run(select_subgraph_));

        sdp_cfg_.record_select_out_index(select_subgraph_, select_out_ops);
        select_resource_ctor_ = [this]() {
            return this->memory_planner_.get_exec_args_set().clone();
        };
    }

    return status::success;
}

template <bool quantized, memory::data_type dt>
void sdp_decomp_kernel_t<quantized, dt>::prepare_sub_args(
        const grantor_t &var_grantor, const int id, const size_t block_size,
        std::unordered_map<dnnl_memory_t, std::vector<memory>> &mem_map) {
    auto size_offset = id * block_size;
    mem_map[sdp_cfg_.sub_mm1_wei.get()][id].set_data_handle(
            var_grantor.get(sdp_cfg_.mem_key_map[sdp_cfg_.sub_mm1_wei.get()])
            + size_offset);
    // mm1
    mem_map[sdp_cfg_.sub_mm1_src.get()][id].set_data_handle(
            var_grantor.get(
                    sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_src1_src2.get()])
            + size_offset);
    mem_map[sdp_cfg_.sub_mm1_dst.get()][id].set_data_handle(
            var_grantor.get(
                    sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_dst1_wei2.get()])
            + size_offset);
    // softmax
    mem_map[sdp_cfg_.sub_softmax_dst.get()][id].set_data_handle(
            var_grantor.get(
                    sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_src1_src2.get()])
            + size_offset);
    // mm2
    mem_map[sdp_cfg_.sub_mm2_wei.get()][id].set_data_handle(
            var_grantor.get(
                    sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_dst1_wei2.get()])
            + size_offset);
    mem_map[sdp_cfg_.sub_mm2_dst.get()][id].set_data_handle(
            var_grantor.get(sdp_cfg_.mem_key_map[sdp_cfg_.sub_mm2_dst.get()])
            + size_offset);
    // scratchpad, each thread will have a largest scratchpad.
    mem_map[sdp_cfg_.sub_scratchpad.get()][id].set_data_handle(
            var_grantor.get(sdp_cfg_.mem_key_map[sdp_cfg_.sub_scratchpad.get()])
            + size_offset);
}

template <bool quantized, memory::data_type dt>
void sdp_decomp_kernel_t<quantized, dt>::prepare_args_set(
        const execution_args_set_t *res, const std::vector<tensor_t> &inputs,
        const scratchpad_t &scratchpad) {
    // update the data of partition in/outputs args
    for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
        mem_idx.first.set_data_handle(inputs[mem_idx.second].get_data_handle());
    }

    grantor_t var_grantor = memory_planner_.internal_temporary_grantor(
            scratchpad.get_buffer());

    for (auto &mem_offkey : res->get_mems_use_internal_temporary()) {
        mem_offkey.first.set_data_handle(var_grantor.get(mem_offkey.second));
    }
}

template <bool quantized, memory::data_type dt>
status_t sdp_decomp_kernel_t<quantized, dt>::execute_impl(
        const stream_t *g_stream, const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs) {
    dnnl::stream strm = make_dnnl_stream(p_engine_, *g_stream);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    auto *tp_stream
            = dnnl::impl::utils::downcast<dnnl::impl::cpu::cpu_stream_t *>(
                    const_cast<stream_t *>(g_stream));
    tp_stream->before_exec_hook();
    int thread_num = 1;
    dnnl_threadpool_interop_get_max_concurrency(&thread_num);
    sdp_cfg_.nthr = thread_num;
#endif

    // each thread's own local resource
    thread_local_cache_t<execution_args_set_t> select_res_cache;
    execution_args_set_t *select_res = nullptr;
    if (sdp_cfg_.has_select)
        select_res = select_res_cache.get_or_add(
                reinterpret_cast<size_t>(this), select_resource_ctor_);
    thread_local_cache_t<sdp_args_set_t> res_cache;
    sdp_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    int MBO = sdp_cfg_.batch_size, MBI = sdp_cfg_.num_head_q;

    char *src1_user_pointer = static_cast<char *>(
            inputs[sdp_cfg_.graph_inport[0]].get_data_handle());
    char *wei1_user_pointer = static_cast<char *>(
            inputs[sdp_cfg_.graph_inport[1]].get_data_handle());
    char *wei2_user_pointer = static_cast<char *>(
            inputs[sdp_cfg_.graph_inport[4]].get_data_handle());
    char *dst2_user_pointer = static_cast<char *>(outputs[0].get_data_handle());

    // allocate the select internal memory
    temporary_scratchpad_t select_scratchpad(
            memory_planner_.total_internal_temporary_size(), p_engine_,
            *g_alloc_);
    assertm(select_scratchpad.size()
                    >= memory_planner_.total_internal_temporary_size(),
            "no enough scratchpad memory");
    if (sdp_cfg_.has_select) {
        const std::vector<tensor_t> select_inputs
                = {inputs[sdp_cfg_.graph_inport[5]],
                        inputs[sdp_cfg_.graph_inport[6]]};
        prepare_args_set(select_res, select_inputs, select_scratchpad);
    }
    size_t block_size = sdp_registry_.size();
    temporary_scratchpad_t scratchpad(
            block_size * sdp_cfg_.nthr, p_engine_, *g_alloc_);
    assertm(scratchpad.size() >= sdp_registry_.size(),
            "no enough scratchpad memory");
    grantor_t var_grantor = sdp_registry_.grantor(scratchpad.get_buffer());

    const auto get_mem_dt_size = [](const memory &m) -> size_t {
        return memory::data_type_size(m.get_desc().get_data_type());
    };

    const auto loop = [&](int tid, int nthr, dim_t bo, dim_t bi) {
        // prepare execution args and allocate real memory
        prepare_sub_args(var_grantor, tid, block_size, res->mem_map);

        // reorder0
        auto &sub_src1_tid = res->mem_map[sdp_cfg_.sub_src1.get()][tid];
        // reorder1:
        auto &sub_wei1_user_tid
                = res->mem_map[sdp_cfg_.sub_wei1_user.get()][tid];

        // matmul1
        //post op index offset.
        size_t start_index = 0;
        if (sdp_cfg_.has_scale) {
            auto &sub_mm1_post_scale_tid
                    = res->mem_map[sdp_cfg_.sub_mm1_post_mem[start_index++]
                                           .get()][tid];
            sub_mm1_post_scale_tid.set_data_handle(
                    inputs[sdp_cfg_.graph_inport[2]].get_data_handle());
        }
        if (sdp_cfg_.has_attention_mask) {
            auto &sub_mm1_post_add_tid
                    = res->mem_map[sdp_cfg_.sub_mm1_post_mem[start_index++]
                                           .get()][tid];
            const auto &mask_input = inputs[sdp_cfg_.graph_inport[3]];
            const auto mask_strides
                    = ltw(mask_input.get_logical_tensor()).vstrides();
            const auto mask_dims = ltw(mask_input.get_logical_tensor()).vdims();
            size_t mask_offset = 0;
            if (mask_dims.size() == 4) {
                if (mask_dims[0] != 1) mask_offset += bo * mask_strides[0];
                if (mask_dims[1] != 1) mask_offset += bi * mask_strides[1];
            }
            sub_mm1_post_add_tid.set_data_handle(
                    static_cast<char *>(mask_input.get_data_handle())
                    + mask_offset * get_mem_dt_size(sub_mm1_post_add_tid));
        }
        if (sdp_cfg_.has_select) {
            //connect select_graph and sdp_graph
            for (size_t i = start_index; i < sdp_cfg_.sub_mm1_post_mem.size();
                    i++) {
                auto &sub_mm1_post_tid
                        = res->mem_map[sdp_cfg_.sub_mm1_post_mem[i].get()][tid];
                const auto &select_res_args = select_res->get_exec_args();
                auto out_mem
                        = select_res_args[sdp_cfg_.select_outop_index[i - 1]]
                                  .at(DNNL_ARG_DST);
                auto out_strides = out_mem.get_desc().get_strides();
                sub_mm1_post_tid.set_data_handle(
                        static_cast<char *>(out_mem.get_data_handle())
                        + bo * out_strides[0]
                                * get_mem_dt_size(sub_mm1_post_tid));
            }
        }
        // reorder2:
        auto &sub_wei2_user_tid
                = res->mem_map[sdp_cfg_.sub_wei2_user.get()][tid];

        //reorder3
        auto &sub_dst_user_tid = res->mem_map[sdp_cfg_.sub_dst_user.get()][tid];

        // matmul2
        auto &sub_mm2_dst_tid = res->mem_map[sdp_cfg_.sub_mm2_dst.get()][tid];

        const size_t sub_src1_offset = (bo * sdp_cfg_.src1_strides[0]
                                               + bi * sdp_cfg_.src1_strides[1])
                * get_mem_dt_size(sub_src1_tid);
        const size_t group_head = sdp_cfg_.num_head_q / sdp_cfg_.num_head_kv;
        size_t wei_head_offset = bi / group_head;
        const size_t sub_wei1_offset
                = (bo * sdp_cfg_.wei1_strides[0]
                          + wei_head_offset * sdp_cfg_.wei1_strides[1])
                * get_mem_dt_size(sub_wei1_user_tid);
        const size_t sub_wei2_offset
                = (bo * sdp_cfg_.wei2_strides[0]
                          + wei_head_offset * sdp_cfg_.wei2_strides[1])
                * get_mem_dt_size(sub_wei2_user_tid);
        const size_t sub_dst_user_offset
                = (bo * sdp_cfg_.dst_strides[0] + bi * sdp_cfg_.dst_strides[1])
                * get_mem_dt_size(sub_dst_user_tid);

        sub_wei1_user_tid.set_data_handle(wei1_user_pointer + sub_wei1_offset);
        sub_src1_tid.set_data_handle(src1_user_pointer + sub_src1_offset);
        sub_wei2_user_tid.set_data_handle(wei2_user_pointer + sub_wei2_offset);
        sub_dst_user_tid.set_data_handle(
                dst2_user_pointer + sub_dst_user_offset);

        // If the last reorder is inplace, it means we don't have to do
        // extra reorder, thus we should set matmul's output to the user's
        // output directly.
        if (sdp_cfg_.sub_reorder3.get_inplace()) {
            sub_mm2_dst_tid.set_data_handle(
                    dst2_user_pointer + sub_dst_user_offset);
        }

        // in parallel region - these primitives should use single thread.
        sdp_cfg_.sub_reorder0.execute(strm, res->sub_reorder0_args[tid]);
        sdp_cfg_.sub_reorder1.execute(strm, res->sub_reorder1_args[tid]);
        sdp_cfg_.sub_mm1_prim.execute(strm, res->sub_mm1_args[tid]);

        sdp_cfg_.sub_softmax_prim.execute(strm, res->sub_softmax_args[tid]);

        sdp_cfg_.sub_reorder2.execute(strm, res->sub_reorder2_args[tid]);

        sdp_cfg_.sub_mm2_prim.execute(strm, res->sub_mm2_args[tid]);
        sdp_cfg_.sub_reorder3.execute(strm, res->sub_reorder3_args[tid]);
    };
    if (sdp_cfg_.has_select) {
        for (size_t i = 0; i < select_subgraph_->execs_.size(); i++) {
            select_subgraph_->execs_[i]->execute(
                    strm, select_res->get_exec_args()[i]);
        }
    }
    parallel_nd_ext(sdp_cfg_.nthr, MBO, MBI, loop);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    tp_stream->after_exec_hook();
#endif
    return status::success;
}

template struct sdp_decomp_kernel_t<false, dnnl::memory::data_type::f32>;
template struct sdp_decomp_kernel_t<true, dnnl::memory::data_type::bf16>;
template struct sdp_decomp_kernel_t<true, dnnl::memory::data_type::f32>;
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
