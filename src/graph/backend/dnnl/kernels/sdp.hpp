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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/dnnl_thread.hpp"
#include "graph/interface/backend.hpp"
#include "graph/interface/graph.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"
#include "graph/backend/dnnl/dnnl_partition_impl.hpp"
#include "graph/backend/dnnl/op_executable.hpp"
#include "graph/backend/dnnl/scratchpad.hpp"
#include "graph/backend/dnnl/thread_local_cache.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using ltw = logical_tensor_wrapper_t;
using op_ptr = std::shared_ptr<op_t>;
using registry_key = size_t;

struct sdp_decomp_config_t {
public:
    sdp_decomp_config_t() = default;

    // SDP input dimension
    memory::dim batch_size, num_head, seq_len, size_per_head;

    // Thread nums during the workflow
    int nthr;

    // Used to record the exact input offset of mm1_src,mm1_wei,mm1_div,mm1_add,
    // mm2_wei in subgraph
    std::vector<int> input_id;

    // Primitives that actually perform calculations
    primitive sub_reorder0_prim, sub_reorder1_prim, sub_mm1_prim,
            sub_softmax_prim, sub_reorder2_prim, sub_mm2_prim,
            sub_reorder3_prim;

    // Args used in the execution of primitives
    std::unordered_map<int, memory> sub_reorder0_args, sub_reorder1_args,
            sub_mm1_args, sub_softmax_args, sub_reorder2_args, sub_mm2_args,
            sub_reorder3_args;

    // A map from memory to registry key, used to record the internal memories
    // location inside of the whole buffer.
    std::unordered_map<dnnl_memory_t, registry_key> mem_key_map;

    /// Internal memory objects for each primitive in each threads.
    // reorder0
    memory sub_src1;
    // reorder1
    memory sub_wei1_user, sub_wei1_zp;
    //mm1
    memory sub_mm1_src, sub_mm1_wei, sub_mm1_dst, sub_mm1_post_div,
            sub_mm1_post_add;
    //softmax
    memory sub_softmax_dst;
    //reorder2
    memory sub_wei2_user, sub_wei2_zp;
    //mm2
    memory sub_mm2_wei, sub_mm2_dst;
    //reorder3
    memory sub_dst_user;
    //scratchped
    memory sub_scratchpad;
    // shared memory
    memory sub_max_src1_src2, sub_max_dst1_wei2;

private:
    // Used to record the ops contained in SDP
    std::vector<std::shared_ptr<op_t>> sdp_op;

    // Zero point values used to do shift from u8 to s8
    int32_t reorder1_zp = -128, reorder2_zp = -128;

public:
    // The function is used to check if the configuration of SDP is supported by
    // current implementation of decomp kernel. Currently, this implementation
    // can only handle 4-dims tensor and limits the layout of the SDP's input.
    // For better performance, we also limit the numerical relationship between
    // batch size and thread num.
    // If the check passes, initialize few members according to inputs
    // If no, return unimplemented status directly and fallback to large kernel
    // TODOs: we have follow to-do tasks in the future:
    //   1. The batch_size and max_threads conditions need to be further checked
    //   2. Enable the latency scenario with batch_size = 1
    bool initial_check(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs) {
        // Initialize nthr with current threads num
        nthr = dnnl_get_current_num_threads();

        // The order of input logical tensors in inputs is not certain, we need
        // to record the input offset in a certain order of ops.
        record_input_offset(sg, inputs);

        memory::dims src1_user_dims = ltw(inputs[input_id[0]]).vdims();
        if (src1_user_dims.size() != 4) return false;

        // Initialize SDP input dimension according to the src of mm1
        batch_size = src1_user_dims[0];
        num_head = src1_user_dims[1];
        seq_len = src1_user_dims[2];
        size_per_head = src1_user_dims[3];

        bool mm1_src_format
                = is_format(make_dnnl_memory_desc(inputs[input_id[0]]),
                        memory::format_tag::acbd);
        bool mm1_wei_format
                = is_format(make_dnnl_memory_desc(inputs[input_id[1]]),
                        memory::format_tag::adbc);
        bool mm2_wei_format
                = is_format(make_dnnl_memory_desc(inputs[input_id[4]]),
                        memory::format_tag::acbd);
        return (batch_size * num_head) % nthr == 0 && mm1_src_format
                && mm1_wei_format && mm2_wei_format;
    }

    // Used to construct all params that SDP need
    template <bool quantized = false,
            memory::data_type dt = memory::data_type::f32>
    impl::status_t construct_params(std::shared_ptr<subgraph_t> &sg,
            registry_t &sdp_registry, const dnnl::engine &p_engine,
            const std::vector<logical_tensor_t> &inputs) {

        // Record the ops inside of SDP pattern for later usage
        record_sdp_ops(sg, quantized);

        // Acquire the data type from input param for later primitive creation.
        // The src and wei dt of both quantized sdp and float sdp are the same.
        memory::data_type dt_in_user = static_cast<memory::data_type>(
                ltw(inputs[input_id[0]]).data_type());
        memory::data_type dt_wei
                = quantized ? memory::data_type::s8 : dt_in_user;
        memory::data_type dt_inter = quantized ? dt : dt_in_user;
        memory::data_type dt_zp = memory::data_type::s32;

        ////////////////////////////////////////////////////////////////////////
        ////////////// Start Creating primitives ///////////////////////////////
        ////////////////////////////////////////////////////////////////////////
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
        // TODO: Here we create primitive with single thread, no exact reason,
        // pending on primitive investigation and fix
        omp_set_num_threads(1);
#endif
        // intermediate md used to create primitives
        memory::desc sub_src1_md, sub_wei1_user_md, sub_wei1_md, sub_wei1_zp_md,
                sub_mm1_src_md, sub_mm1_wei_md, sub_mm1_dst_md,
                sub_mm1_post_div_md, sub_mm1_post_add_md, sub_softmax_dst_md,
                sub_wei2_user_md, sub_wei2_zp_md, sub_mm2_wei_md,
                sub_mm2_dst_md, sub_dst_md, sub_dst_user_md;

        // must use user mode to support concurrent execution
        primitive_attr sub_reorder0_attr;
        sub_reorder0_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        // per-head: reorder src1 to dense, for first matmul
        memory::dims sub_src1_dims = {1, 1, seq_len, size_per_head};
        sub_src1_md = memory::desc(
                sub_src1_dims, dt_in_user, {1, 1, num_head * size_per_head, 1});
        auto sub_src1_d_md = memory::desc(sub_src1_dims, dt_in_user, tag::abcd);
        auto sub_reorder0_pd = reorder::primitive_desc(p_engine, sub_src1_md,
                p_engine, sub_src1_d_md, sub_reorder0_attr);
        sub_reorder0_prim = reorder(sub_reorder0_pd);

        auto &mgr = sg->fusion_info_mgr_;

        // per-head: reorder u8->s8 wei for first matmul
        // create reorder1 primitive attr
        dnnl::primitive_attr sub_reorder1_attr
                = make_primitive_attr(sdp_op[0], mgr);
        memory::dims sub_wei1_dims = {1, 1, size_per_head, seq_len};
        sub_wei1_user_md = memory::desc(sub_wei1_dims, dt_in_user,
                {1, size_per_head, 1, num_head * size_per_head});
        // Flip the format to have `ba` weights MBI item in per thread loop.
        sub_wei1_md = memory::desc(sub_wei1_dims, dt_wei, tag::abdc);
        auto sub_reorder1_pd = reorder::primitive_desc(p_engine,
                sub_wei1_user_md, p_engine, sub_wei1_md, sub_reorder1_attr);
        sub_reorder1_prim = reorder(sub_reorder1_pd);
        sub_wei1_zp_md = memory::desc({1}, dt_zp, tag::x);

        // first matmul
        // create first matmul primitive attr
        dnnl::primitive_attr sub_matmul1_attr
                = make_primitive_attr(sdp_op[1], mgr);
        memory::dims sub_mm1_src_dims = {1, 1, seq_len, size_per_head};
        memory::dims sub_mm1_wei_dims = {1, 1, size_per_head, seq_len};
        memory::dims sub_mm1_dst_dims = {1, 1, seq_len, seq_len};

        sub_mm1_src_md = memory::desc(sub_mm1_src_dims, dt_in_user, tag::abcd);
        sub_mm1_wei_md = memory::desc(sub_mm1_wei_dims, dt_wei, tag::abdc);
        sub_mm1_dst_md = memory::desc(sub_mm1_dst_dims, dt_inter, tag::abcd);
        dnnl::post_ops dnnl_pops;
        auto div_dt = static_cast<dnnl::memory::data_type>(
                ltw(inputs[input_id[3]]).data_type());
        // TODO: It is presupposed that the dims of div and add's src are certain,
        // which may not always be true.
        sub_mm1_post_div_md = memory::desc({1, 1, 1, 1}, div_dt, tag::abcd);
        sub_mm1_post_add_md
                = memory::desc({1, 1, 1, seq_len}, dt_inter, tag::abcd);
        dnnl_pops.append_binary(algorithm::binary_div, sub_mm1_post_div_md);
        dnnl_pops.append_binary(algorithm::binary_add, sub_mm1_post_add_md);
        sub_matmul1_attr.set_post_ops(std::move(dnnl_pops));
        auto sub_mm1_pd = matmul::primitive_desc(p_engine, sub_mm1_src_md,
                sub_mm1_wei_md, sub_mm1_dst_md, sub_matmul1_attr);
        sub_mm1_prim = matmul(sub_mm1_pd);

        // softmax
        // create softmax primitive attr
        dnnl::primitive_attr sub_softmax_attr
                = make_primitive_attr(sdp_op[2], mgr);
        sub_softmax_dst_md
                = memory::desc(sub_mm1_dst_dims, dt_in_user, tag::abcd);
        auto sub_softmax_pd = softmax_forward::primitive_desc(p_engine,
                prop_kind::forward_inference, algorithm::softmax_accurate,
                sub_mm1_dst_md, sub_softmax_dst_md,
                sub_mm1_dst_md.get_ndims() - 1, sub_softmax_attr);
        sub_softmax_prim = softmax_forward(sub_softmax_pd);

        // reorder u8->s8 wei for second matmul
        // create reorder2 primitive attr
        dnnl::primitive_attr sub_reorder2_attr
                = make_primitive_attr(sdp_op[3], mgr);
        memory::dims sub_wei2_dims = {1, 1, seq_len, size_per_head};
        sub_wei2_user_md = memory::desc(
                sub_wei2_dims, dt_in_user, {1, 1, num_head * size_per_head, 1});
        // The format is `abcd` due to performance of reorder to `abdc` is low.
        auto sub_wei2_md = memory::desc(sub_wei2_dims, dt_wei, tag::abcd);
        auto sub_reorder2_pd = reorder::primitive_desc(p_engine,
                sub_wei2_user_md, p_engine, sub_wei2_md, sub_reorder2_attr);
        sub_reorder2_prim = reorder(sub_reorder2_pd);
        sub_wei2_zp_md = memory::desc({1}, dt_zp, tag::x);

        // second matmul
        // create second matmul primitive attr
        dnnl::primitive_attr sub_matmul2_attr
                = make_primitive_attr(sdp_op[4], mgr);
        memory::dims sub_mm2_src_dims = {1, 1, seq_len, seq_len};
        memory::dims sub_mm2_wei_dims = {1, 1, seq_len, size_per_head};
        memory::dims sub_mm2_dst_dims = {1, 1, seq_len, size_per_head};
        auto sub_mm2_src_md
                = memory::desc(sub_mm2_src_dims, dt_in_user, tag::abcd);
        sub_mm2_wei_md = memory::desc(sub_mm2_wei_dims, dt_wei, tag::abcd);
        sub_mm2_dst_md = memory::desc(sub_mm2_dst_dims, dt_in_user, tag::abcd);
        auto sub_mm2_pd = matmul::primitive_desc(p_engine, sub_mm2_src_md,
                sub_mm2_wei_md, sub_mm2_dst_md, sub_matmul2_attr);
        sub_mm2_prim = matmul(sub_mm2_pd);

        // per-head: reorder dst2 from dense to strided
        primitive_attr sub_reorder3_attr;
        sub_reorder3_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        memory::dims sub_dst_dims = {1, 1, seq_len, size_per_head};
        sub_dst_md = memory::desc(sub_dst_dims, dt_in_user, tag::abcd);
        sub_dst_user_md = memory::desc(
                sub_dst_dims, dt_in_user, {1, 1, num_head * size_per_head, 1});
        auto sub_reorder3_pd = reorder::primitive_desc(p_engine, sub_dst_md,
                p_engine, sub_dst_user_md, sub_reorder3_attr);
        sub_reorder3_prim = reorder(sub_reorder3_pd);
        ////////////////////////////////////////////////////////////////////////
        /////////////// End Creating primitives ////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        /////////////// Start Constructing exec args ///////////////////////////
        ////////////////////////////////////////////////////////////////////////
        memory::desc max_scratchpad_md, sub_max_src1_src2_md,
                sub_max_dst1_wei2_md;
        size_t max_scratchpad_size = 0;
        // all the scratchpads required by the primitives.
        const std::vector<memory::desc> scratchpads {
                sub_reorder0_pd.scratchpad_desc(),
                sub_reorder1_pd.scratchpad_desc(), sub_mm1_pd.scratchpad_desc(),
                sub_softmax_pd.scratchpad_desc(),
                sub_reorder2_pd.scratchpad_desc(), sub_mm2_pd.scratchpad_desc(),
                sub_reorder3_pd.scratchpad_desc()};

        for (auto &sp : scratchpads) {
            const size_t size = sp.get_size();
            if (size > max_scratchpad_size) {
                max_scratchpad_size = size;
                max_scratchpad_md = sp;
            }
        }

        auto sub_src1_size = sub_src1_d_md.get_size();
        auto sub_src2_size = sub_softmax_dst_md.get_size();
        sub_max_src1_src2_md = sub_src1_size > sub_src2_size
                ? sub_src1_d_md
                : sub_softmax_dst_md;

        auto sub_dst1_size = sub_mm1_dst_md.get_size();
        auto sub_wei2_size = sub_mm2_wei_md.get_size();
        sub_max_dst1_wei2_md = sub_dst1_size > sub_wei2_size ? sub_mm1_dst_md
                                                             : sub_mm2_wei_md;

        // Initialize memory object with empty buffer
        sub_max_src1_src2 = memory(sub_max_src1_src2_md, p_engine, nullptr);
        sub_max_dst1_wei2 = memory(sub_max_dst1_wei2_md, p_engine, nullptr);
        // reorder0: 2d strided -> 2d ab
        sub_src1 = memory(sub_src1_md, p_engine, nullptr);
        // reorder1: 2d strided u8 -> 2d ba s8
        sub_wei1_user = memory(sub_wei1_user_md, p_engine, nullptr);
        sub_wei1_zp = memory(sub_wei1_zp_md, p_engine, &reorder1_zp);
        // mm1
        sub_mm1_src = memory(sub_mm1_src_md, p_engine, nullptr);
        sub_mm1_wei = memory(sub_mm1_wei_md, p_engine, nullptr);
        sub_mm1_dst = memory(sub_mm1_dst_md, p_engine, nullptr);
        sub_mm1_post_div = memory(sub_mm1_post_div_md, p_engine, nullptr);
        sub_mm1_post_add = memory(sub_mm1_post_add_md, p_engine, nullptr);
        // softmax
        sub_softmax_dst = memory(sub_softmax_dst_md, p_engine, nullptr);
        // reorder2
        sub_wei2_user = memory(sub_wei2_user_md, p_engine, nullptr);
        sub_wei2_zp = memory(sub_wei2_zp_md, p_engine, &reorder2_zp);
        // mm2
        sub_mm2_wei = memory(sub_mm2_wei_md, p_engine, nullptr);
        sub_mm2_dst = memory(sub_mm2_dst_md, p_engine, nullptr);
        //reorder3
        sub_dst_user = memory(sub_dst_user_md, p_engine, nullptr);

        // scratchpad, each thread will have a largest scratchpad.
        sub_scratchpad = memory(max_scratchpad_md, p_engine, nullptr);

        sub_reorder0_args
                = {{DNNL_ARG_SRC, sub_src1}, {DNNL_ARG_DST, sub_mm1_src},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_reorder1_args
                = {{DNNL_ARG_SRC, sub_wei1_user}, {DNNL_ARG_DST, sub_mm1_wei},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, sub_wei1_zp},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_mm1_args = {{DNNL_ARG_SRC, sub_mm1_src},
                {DNNL_ARG_WEIGHTS, sub_mm1_wei}, {DNNL_ARG_DST, sub_mm1_dst},
                {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                        sub_mm1_post_div},
                {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                        sub_mm1_post_add},
                {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_softmax_args
                = {{DNNL_ARG_SRC, sub_mm1_dst}, {DNNL_ARG_DST, sub_softmax_dst},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_reorder2_args
                = {{DNNL_ARG_SRC, sub_wei2_user}, {DNNL_ARG_DST, sub_mm2_wei},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, sub_wei2_zp},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_mm2_args = {{DNNL_ARG_SRC, sub_softmax_dst},
                {DNNL_ARG_WEIGHTS, sub_mm2_wei}, {DNNL_ARG_DST, sub_mm2_dst},
                {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        sub_reorder3_args
                = {{DNNL_ARG_SRC, sub_mm2_dst}, {DNNL_ARG_DST, sub_dst_user},
                        {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

        // add scales and zps for mm1, softmax, mm2
        prepare_sdp_scales_zps(mgr, sdp_op[1], 2, sub_mm1_args, p_engine);
        prepare_sdp_scales_zps(mgr, sdp_op[2], 1, sub_softmax_args, p_engine);
        prepare_sdp_scales_zps(mgr, sdp_op[4], 2, sub_mm2_args, p_engine);
        ////////////////////////////////////////////////////////////////////////
        /////////////// End Constructing exec args /////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        // memory planing for buffer sharing
        memory_planning(sdp_registry, p_engine);
        return status::success;
    }

private:
    op_ptr get_post_op(const op_ptr &op) const {
        const auto out_val = op->get_output_value(0);
        const auto &consumers = out_val->get_consumers();
        if (consumers.size() != 1) return nullptr;
        return consumers[0].get_op().shared_from_this();
    }

    impl::status_t record_input_offset(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs) {
        auto find_input_id = [&](std::shared_ptr<value_t> val) {
            while (val->has_producer()) {
                val = val->get_producer().get_input_value(0);
            }
            for (int i = 0; i < (int)inputs.size(); i++) {
                if (val->get_logical_tensor().id == inputs[i].id) { return i; }
            }
            return 0;
        };
        op_ptr mm1, mm2, div, add;
        for (const auto &cur_op : sg->get_ops()) {
            if (mm1 != nullptr && mm2 != nullptr) break;
            if (cur_op->get_kind() != graph::op_kind::MatMul) continue;
            auto post_op = get_post_op(cur_op);
            if (post_op->get_kind() == graph::op_kind::Divide
                    || post_op->get_kind() == graph::op_kind::Multiply) {
                mm1 = cur_op;
                div = post_op;
                add = get_post_op(post_op);
            } else
                mm2 = cur_op;
        }
        int src1_id = find_input_id(mm1->get_input_value(0));
        input_id.emplace_back(src1_id);
        int wei1_id = find_input_id(mm1->get_input_value(1));
        input_id.emplace_back(wei1_id);
        int div_id = find_input_id(div->get_input_value(1));
        input_id.emplace_back(div_id);
        int add_id = find_input_id(add->get_input_value(1));
        input_id.emplace_back(add_id);
        int wei2_id = find_input_id(mm2->get_input_value(1));
        input_id.emplace_back(wei2_id);
        return status::success;
    }

    impl::status_t record_sdp_ops(
            std::shared_ptr<subgraph_t> &sg, bool is_quantize) {
        const auto get_wei_pre_op = [](const op_ptr &op) -> op_ptr {
            const auto out_val = op->get_input_value(1);
            auto &producer = out_val->get_producer();
            return producer.shared_from_this();
        };

        subgraph_rewriter_t rewriter(sg);

        for (const auto &cur_op : sg->get_ops()) {
            if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;
            auto post_op = get_post_op(cur_op);
            if (post_op == nullptr
                    || post_op->get_kind() != op_kind::dnnl_softmax)
                continue;
            auto ppost_op = get_post_op(post_op);
            op_ptr reorder1;
            op_ptr reorder2;
            if (is_quantize) {
                reorder1 = get_wei_pre_op(cur_op);
                reorder2 = get_wei_pre_op(ppost_op);
            }

            this->sdp_op = {reorder1, cur_op, post_op, reorder2, ppost_op};
            break;
        }
        return status::success;
    }

    void memory_planning(registry_t &sdp_registry, dnnl::engine p_engine) {
        // Registry is used to do the memory planning for sdp decompostion
        // algorithm. We reused some internal memory to reduce the memory
        // footprint for better cache hit. And here the key in registar of each
        // memory is planned in a specific order.
        registrar_t temporary_registrar = sdp_registry.registrar();

        // Here we initialize the map based on certain memory reuse logic. Those
        // memories(mds) who share the same buffer have the same registar key in
        // this map. So if we want to change the memory reuse logic, we need to
        // change the value of map here.
        mem_key_map = {{sub_max_src1_src2.get(), 0}, {sub_mm1_wei.get(), 1},
                {sub_max_dst1_wei2.get(), 2}, {sub_softmax_dst.get(), 0},
                {sub_mm2_dst.get(), 3}, {sub_scratchpad.get(), 4}};

        temporary_registrar.book(mem_key_map[sub_max_src1_src2.get()],
                sub_max_src1_src2.get_desc().get_size() * nthr);
        temporary_registrar.book(mem_key_map[sub_mm1_wei.get()],
                sub_mm1_wei.get_desc().get_size() * nthr);
        temporary_registrar.book(mem_key_map[sub_max_dst1_wei2.get()],
                sub_max_dst1_wei2.get_desc().get_size() * nthr);
        temporary_registrar.book(mem_key_map[sub_mm2_dst.get()],
                sub_mm2_dst.get_desc().get_size() * nthr);
        temporary_registrar.book(mem_key_map[sub_scratchpad.get()],
                sub_scratchpad.get_desc().get_size() * nthr);
    }

    impl::status_t prepare_sdp_scales_zps(const fusion_info_mgr_t &mgr,
            std::shared_ptr<op_t> &op, int index,
            std::unordered_map<int, memory> &args,
            const dnnl::engine &p_engine) {
        const auto dt_scale = memory::data_type::f32,
                   dt_zp = memory::data_type::s32;
        // scale zp order:
        // 1. src scale, wei scale
        // 2. src zp, wei zp
        // 3. dst scale, dst zp
        if (op->has_attr(op_attr::fusion_info_key)
                && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
            fusion_info_t fusion_info = mgr.get_info(key);
            if (fusion_info.with_runtime_scales(true, 0)) {
                memory::desc sub_src_scale_md
                        = memory::desc({1}, dt_scale, tag::x);
                memory sub_src_scale = memory(sub_src_scale_md, p_engine);
                float *src_scale_val_ptr = reinterpret_cast<float *>(
                        sub_src_scale.get_data_handle());
                src_scale_val_ptr[0] = get_attr_value<float, float>(
                        op, index++, op_attr::scales);

                args.insert(
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sub_src_scale});
            }
            if (fusion_info.with_runtime_scales(true, 1)) {
                memory::desc sub_wei_scale_md
                        = memory::desc({1}, dt_scale, tag::x);
                memory sub_wei_scale = memory(sub_wei_scale_md, p_engine);
                float *wei_scale_val_ptr = reinterpret_cast<float *>(
                        sub_wei_scale.get_data_handle());
                wei_scale_val_ptr[0] = get_attr_value<float, float>(
                        op, index++, op_attr::scales);
                args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                        sub_wei_scale});
            }

            // src_zp and wei_zp
            if (fusion_info.with_runtime_zero_points(true, 0)) {
                memory::desc sub_src_zp_md = memory::desc({1}, dt_zp, tag::x);
                memory sub_src_zp = memory(sub_src_zp_md, p_engine);
                int *src_zp_val_ptr
                        = reinterpret_cast<int *>(sub_src_zp.get_data_handle());
                src_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                        op, index++, op_attr::zps);
                args.insert(
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, sub_src_zp});
            }
            if (fusion_info.with_runtime_zero_points(true, 1)) {
                memory::desc sub_wei_zp_md = memory::desc({1}, dt_zp, tag::x);
                memory sub_wei_zp = memory(sub_wei_zp_md, p_engine);
                int *wei_zp_val_ptr
                        = reinterpret_cast<int *>(sub_wei_zp.get_data_handle());
                wei_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                        op, index++, op_attr::zps);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                        sub_wei_zp});
            }

            // dst scale, dst zp
            if (fusion_info.with_runtime_scales(false, 0)) {
                memory::desc sub_dst_scale_md
                        = memory::desc({1}, dt_scale, tag::x);
                memory sub_dst_scale = memory(sub_dst_scale_md, p_engine);
                float *dst_scale_val_ptr = reinterpret_cast<float *>(
                        sub_dst_scale.get_data_handle());
                dst_scale_val_ptr[0] = get_attr_value<float, float>(
                        op, index++, op_attr::scales);
                args.insert(
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, sub_dst_scale});
            }
            if (fusion_info.with_runtime_zero_points(false, 0)) {
                memory::desc sub_dst_zp_md = memory::desc({1}, dt_zp, tag::x);
                memory sub_dst_zp = memory(sub_dst_zp_md, p_engine);
                int *dst_zp_val_ptr
                        = reinterpret_cast<int *>(sub_dst_zp.get_data_handle());
                dst_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                        op, index++, op_attr::zps);
                args.insert(
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, sub_dst_zp});
            }
        }
        return status::success;
    }

    template <typename attr_dt, typename target_dt>
    target_dt get_attr_value(
            std::shared_ptr<op_t> &op, int i, op_attr_t attr_name) {
        const auto in_val = op->get_input_value(i);
        auto &producer = in_val->get_producer();
        return static_cast<target_dt>(
                producer.get_attr<std::vector<attr_dt>>(attr_name)[0]);
    }

    dnnl::primitive_attr make_primitive_attr(
            std::shared_ptr<op_t> &op, fusion_info_mgr_t &mgr) {
        fusion_info_t fusion_info;
        dnnl::primitive_attr attr;
        if (op != nullptr && op->has_attr(op_attr::fusion_info_key)
                && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
            fusion_info = mgr.get_info(key);
            attr = make_dnnl_primitive_attr(op, fusion_info);
        }
        if (op != nullptr && op->get_kind() == op_kind::dnnl_reorder) {
            // generate mask
            int mask = 0;
            if (op->has_attr(op_attr::axis) && op->has_attr(op_attr::qtype)) {
                int64_t axis = op->get_attr<int64_t>(op_attr::axis);
                std::string qtype = op->get_attr<std::string>(op_attr::qtype);
                mask = qtype == "per_tensor" ? 0 : 1 << axis;
            }

            if (op->has_attr(op_attr::with_runtime_dst_zps)
                    && op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
                // runtime dst zps
                attr.set_zero_points_mask(DNNL_ARG_TO, mask);
            } else if (op->has_attr(op_attr::dst_zps)) {
                assertm(false, "only support runtime dst zero points.\n");
            }
        }
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        return attr;
    }
};

// The second template param dt is used to indicate the internal data type of
// int8 sdp pattern. It doesn't take any effect if quantized param is false.
template <bool quantized = false, memory::data_type dt = memory::data_type::f32>
class sdp_decomp_kernel_t : public kernel_base_t {
private:
    allocator_t *g_alloc_ = nullptr;
    // used for sdp internal memory planning
    registry_t sdp_registry_;
    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;
    subgraph_visualizer_t vis_;

    // SDP-related params
    sdp_decomp_config_t sdp_cfg_;

public:
    sdp_decomp_kernel_t() {
        thread_local_cache_t<sdp_args_set_t> res_cache;
        res_cache.retain();
    }

    ~sdp_decomp_kernel_t() override {
        thread_local_cache_t<sdp_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
        res_cache.release();
    }

    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override {
        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = reinterpret_cast<graph::allocator_t *>(
                g_engine->get_allocator());

        // get subgraph from the deep copied partition
        subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_,
                part->get_fpmath_mode(), part->get_use_blocked_layout(), true);
        BACKEND_DNNL_CHECK(
                set_given_inputs_outputs(subgraph_, inputs, outputs));

        // Check if it's supported by decompostion kernel
        if (!sdp_cfg_.initial_check(subgraph_, inputs))
            return status::unimplemented;

        subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
            return this->memory_planner_.get_memory_info(val);
        });
        pass_pipeline_t pipeline = pass_pipeline_t(vis);

        BACKEND_DNNL_ADD_PASS(pipeline, lower_down);
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
        if (quantized) {
            BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_scales);
            BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_scales);
            BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_zero_points);
            BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_zero_points);
            BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);
        }

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

        return status::success;
    }

    void prepare_sub_args(const grantor_t &var_grantor, const int id,
            std::unordered_map<dnnl_memory_t, std::vector<memory>> &mem_map) {
        mem_map[sdp_cfg_.sub_mm1_wei.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_mm1_wei.get()])
                + id * sdp_cfg_.sub_mm1_wei.get_desc().get_size());
        // mm1
        mem_map[sdp_cfg_.sub_mm1_src.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_src1_src2.get()])
                + id * sdp_cfg_.sub_max_src1_src2.get_desc().get_size());
        mem_map[sdp_cfg_.sub_mm1_dst.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_dst1_wei2.get()])
                + id * sdp_cfg_.sub_max_dst1_wei2.get_desc().get_size());
        // softmax
        mem_map[sdp_cfg_.sub_softmax_dst.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_src1_src2.get()])
                + id * sdp_cfg_.sub_max_src1_src2.get_desc().get_size());
        // mm2
        mem_map[sdp_cfg_.sub_mm2_wei.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_dst1_wei2.get()])
                + id * sdp_cfg_.sub_max_dst1_wei2.get_desc().get_size());
        mem_map[sdp_cfg_.sub_mm2_dst.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_mm2_dst.get()])
                + id * sdp_cfg_.sub_mm2_dst.get_desc().get_size());
        // scratchpad, each thread will have a largest scratchpad.
        mem_map[sdp_cfg_.sub_scratchpad.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_scratchpad.get()])
                + id * sdp_cfg_.sub_scratchpad.get_desc().get_size());
    }

    status_t execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override {
        dnnl::stream strm = make_dnnl_stream(p_engine_, *g_stream);
        // each thread's own local resource
        thread_local_cache_t<sdp_args_set_t> res_cache;
        sdp_args_set_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        int MBO = sdp_cfg_.batch_size, MBI = sdp_cfg_.num_head,
            M1 = sdp_cfg_.seq_len, K1 = sdp_cfg_.size_per_head,
            N1 = sdp_cfg_.seq_len, M2 = sdp_cfg_.seq_len, K2 = sdp_cfg_.seq_len,
            N2 = sdp_cfg_.size_per_head;

        char *src1_user_pointer = static_cast<char *>(
                inputs[sdp_cfg_.input_id[0]].get_data_handle());
        char *wei1_user_pointer = static_cast<char *>(
                inputs[sdp_cfg_.input_id[1]].get_data_handle());
        char *wei2_user_pointer = static_cast<char *>(
                inputs[sdp_cfg_.input_id[4]].get_data_handle());
        char *dst2_user_pointer
                = static_cast<char *>(outputs[0].get_data_handle());

        // allocate the internal memory
        temporary_scratchpad_t scratchpad(
                sdp_registry_.size(), p_engine_, *g_alloc_);
        assertm(scratchpad.size() >= sdp_registry_.size(),
                "no enough scratchpad memory");
        grantor_t var_grantor = sdp_registry_.grantor(scratchpad.get_buffer());

        const auto get_mem_dt_size = [](const memory &m) -> size_t {
            return memory::data_type_size(m.get_desc().get_data_type());
        };

        const auto loop = [&](int tid, int nthr, dim_t bo, dim_t bi) {
            // prepare execution args and allocate real memory
            prepare_sub_args(var_grantor, tid, res->mem_map);

            // reorder0
            auto &sub_src1_tid = res->mem_map[sdp_cfg_.sub_src1.get()][tid];
            // reorder1:
            auto &sub_wei1_user_tid
                    = res->mem_map[sdp_cfg_.sub_wei1_user.get()][tid];

            // matmul1
            auto &sub_mm1_post_div_tid
                    = res->mem_map[sdp_cfg_.sub_mm1_post_div.get()][tid];
            auto &sub_mm1_post_add_tid
                    = res->mem_map[sdp_cfg_.sub_mm1_post_add.get()][tid];
            sub_mm1_post_div_tid.set_data_handle(
                    inputs[sdp_cfg_.input_id[2]].get_data_handle());
            sub_mm1_post_add_tid.set_data_handle(
                    static_cast<char *>(
                            inputs[sdp_cfg_.input_id[3]].get_data_handle())
                    + bo * sdp_cfg_.seq_len
                            * get_mem_dt_size(sdp_cfg_.sub_mm1_post_add));

            // reorder2:
            auto &sub_wei2_user_tid
                    = res->mem_map[sdp_cfg_.sub_wei2_user.get()][tid];

            //reorder3
            auto &sub_dst_user_tid
                    = res->mem_map[sdp_cfg_.sub_dst_user.get()][tid];

            const size_t sub_src1_offset = (bo * MBI * M1 * K1 + bi * K1)
                    * get_mem_dt_size(sub_src1_tid);
            const size_t sub_wei1_offset = (bo * MBI * K1 * N1 + bi * K1)
                    * get_mem_dt_size(sub_wei1_user_tid);
            const size_t sub_wei2_offset = (bo * MBI * K2 * N2 + bi * N2)
                    * get_mem_dt_size(sub_wei2_user_tid);
            const size_t sub_dst_user_offset = (bo * MBI * M2 * N2 + bi * N2)
                    * get_mem_dt_size(sub_dst_user_tid);

            sub_wei1_user_tid.set_data_handle(
                    wei1_user_pointer + sub_wei1_offset);
            sub_src1_tid.set_data_handle(src1_user_pointer + sub_src1_offset);
            sub_wei2_user_tid.set_data_handle(
                    wei2_user_pointer + sub_wei2_offset);
            sub_dst_user_tid.set_data_handle(
                    dst2_user_pointer + sub_dst_user_offset);

            // in parallel region - these primitives should use single thread.
            sdp_cfg_.sub_reorder0_prim.execute(
                    strm, res->sub_reorder0_args[tid]);
            sdp_cfg_.sub_reorder1_prim.execute(
                    strm, res->sub_reorder1_args[tid]);
            sdp_cfg_.sub_mm1_prim.execute(strm, res->sub_mm1_args[tid]);

            sdp_cfg_.sub_softmax_prim.execute(strm, res->sub_softmax_args[tid]);

            sdp_cfg_.sub_reorder2_prim.execute(
                    strm, res->sub_reorder2_args[tid]);

            sdp_cfg_.sub_mm2_prim.execute(strm, res->sub_mm2_args[tid]);
            sdp_cfg_.sub_reorder3_prim.execute(
                    strm, res->sub_reorder3_args[tid]);
        };
        // TODO: remove this when primitive new API ready
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
        omp_set_num_threads(sdp_cfg_.nthr);
#endif
        parallel_nd_ext(sdp_cfg_.nthr, MBO, MBI, loop);
        return status::success;
    }

    class sdp_args_set_t {
    public:
        sdp_args_set_t(sdp_decomp_kernel_t<quantized, dt> *sdp_kernel) {
            int nthr = sdp_kernel->sdp_cfg_.nthr;
            //consrtuct new args
            auto args_ctor = [this, nthr](const std::unordered_map<int, memory>
                                                  &ori_args,
                                     std::vector<std::unordered_map<int,
                                             memory>> &args) {
                args.resize(nthr);
                for (auto iter : ori_args) {
                    memory ori_mem = iter.second;
                    if (mem_map.count(ori_mem.get()) == 0) {
                        //consrtuct new memorys
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
                    sdp_kernel->sdp_cfg_.sub_reorder0_args, sub_reorder0_args);
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder1_args, sub_reorder1_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_mm1_args, sub_mm1_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_softmax_args, sub_softmax_args);
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder2_args, sub_reorder2_args);
            args_ctor(sdp_kernel->sdp_cfg_.sub_mm2_args, sub_mm2_args);
            args_ctor(
                    sdp_kernel->sdp_cfg_.sub_reorder3_args, sub_reorder3_args);
        }
        std::unordered_map<dnnl_memory_t, std::vector<memory>> mem_map;
        // execution args for each op in the subgraph
        std::vector<std::unordered_map<int, memory>> sub_reorder0_args,
                sub_reorder1_args, sub_mm1_args, sub_softmax_args,
                sub_reorder2_args, sub_mm2_args, sub_reorder3_args;
    };

    std::function<std::shared_ptr<sdp_args_set_t>()> resource_ctor_;

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
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
