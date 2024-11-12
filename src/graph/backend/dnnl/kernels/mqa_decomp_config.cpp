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

#include "graph/backend/dnnl/kernels/mqa_decomp_config.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

bool mqa_decomp_config_t::initial_check(const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    // The order of input logical tensors in inputs is not certain, we need
    // to record the input offset in a certain order of ops.
    record_input_offset(sg, inputs);

    // Key(3-dims): batch_size * seq_len * size_per_head
    memory::dims src1_user_dims = ltw(inputs[graph_inport[0]]).vdims();
    // Query(3-dims): batch_size * size_per_head * (num_head * seq_len)
    memory::dims wei1_user_dims = ltw(inputs[graph_inport[1]]).vdims();
    if (src1_user_dims.size() != 3 || wei1_user_dims.size() != 3) return false;

    // Initialize MQA input dimension according to the src of mm1
    batch_size = src1_user_dims[0];
    seq_len = src1_user_dims[1];
    size_per_head = src1_user_dims[2];
    num_head = wei1_user_dims[2] / seq_len;

    //  Check batch size compatibility.
    dims wei2_user_dims = ltw(inputs[graph_inport[3]]).vdims();
    if (batch_size != wei1_user_dims[0] || batch_size != wei2_user_dims[0]) {
        return false;
    }

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
// RATIO is an empirical value used to determine the numerical relationship
// between batch_size, num_head and thread number to determine whether to use
// decompose kernel. The key to the decompose kernel is that we do parallel in
// the batch_size and num_head dimensions. Therefore, if the batch_size or
// num_head is too small, it will cause many idle threads and affect efficiency
// which may even worse than the original sequential kernel. Here we set this
// ratio based on the experimental value to ensure that users do not have any
// regression when using the decompose kernel.
// TODO: Refine the inequation based on the relationship of cache size and mqa
// memory footprint requirements.
#define RATIO 2
    // Initialize nthr with current threads num
    nthr = dnnl_get_current_num_threads();
    return batch_size * num_head > RATIO * nthr;
#else
    return true;
#endif
}

template <bool quantized, memory::data_type dt>
status_t mqa_decomp_config_t::construct_params(std::shared_ptr<subgraph_t> &sg,
        registry_t &mqa_registry, const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs) {

    // Record the ops inside of MQA pattern in a specific order.
    record_mqa_ops(sg);

    // Acquire the data type from input param for later primitive creation.
    // The src and wei dt of both quantized mqa and float mqa are the same.
    memory::data_type dt_src_user = static_cast<memory::data_type>(
            ltw(inputs[graph_inport[0]]).data_type());
    memory::data_type dt_wei_user = static_cast<memory::data_type>(
            ltw(inputs[graph_inport[1]]).data_type());
    memory::data_type dt_wei = quantized ? memory::data_type::s8 : dt_src_user;
    memory::data_type dt_inter = quantized ? dt : dt_src_user;

    ////////////////////////////////////////////////////////////////////////
    ////////////// Start Creating primitives ///////////////////////////////
    ////////////////////////////////////////////////////////////////////////
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    // TODO: Here we create primitive with single thread, no exact reason,
    // pending on primitive investigation and fix
    omp_set_num_threads(1);
#endif
    // intermediate md used to create primitives
    memory::desc sub_src1_md, sub_wei1_user_md, sub_wei1_md, sub_mm1_src_md,
            sub_mm1_wei_md, sub_mm1_dst_md, sub_mm1_post_add_md,
            sub_softmax_dst_md, sub_src2_user_md, sub_mm2_src_md,
            sub_mm2_dst_md, sub_dst_md, sub_dst_user_md;

    // must use user mode to support concurrent execution
    primitive_attr sub_reorder0_attr;
    sub_reorder0_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // per-head: reorder src1 to dense, for first matmul
    memory::dims sub_src1_dims = {1, seq_len, size_per_head};
    sub_src1_md
            = memory::desc(sub_src1_dims, dt_src_user, {1, size_per_head, 1});
    auto sub_src1_d_md = memory::desc(sub_src1_dims, dt_src_user, tag::abc);
    auto sub_reorder0_pd = reorder::primitive_desc(
            p_engine, sub_src1_md, p_engine, sub_src1_d_md, sub_reorder0_attr);
    sub_reorder0.init(sub_reorder0_pd);

    auto &mgr = sg->fusion_info_mgr_;

    // per-head: reorder wei1 to dense, first matmul
    // create reorder1 primitive attr
    auto original_reorder1 = mqa_op[0];
    dnnl::primitive_attr sub_reorder1_attr
            = make_primitive_attr(original_reorder1, mgr);
    memory::dims sub_wei1_dims = {1, size_per_head, seq_len};

    auto original_matmul1 = mqa_op[1];
    auto wei_md = make_dnnl_memory_desc(
            original_matmul1->get_input_value(1)->get_logical_tensor());
    sub_wei1_user_md = memory::desc(
            sub_wei1_dims, dt_wei_user, {1, seq_len * num_head, 1});
    // Flip the format to have `ba` weights MBI item in per thread loop.
    sub_wei1_md = memory::desc(sub_wei1_dims, dt_wei, tag::abc);
    auto sub_reorder1_pd = reorder::primitive_desc(p_engine, sub_wei1_user_md,
            p_engine, sub_wei1_md, sub_reorder1_attr);
    sub_reorder1.init(sub_reorder1_pd);

    // first matmul
    // create first matmul primitive attr
    dnnl::primitive_attr sub_matmul1_attr
            = make_primitive_attr(original_matmul1, mgr);
    memory::dims sub_mm1_src_dims = {1, seq_len, size_per_head};
    memory::dims sub_mm1_wei_dims = {1, size_per_head, seq_len};
    memory::dims sub_mm1_dst_dims = {1, seq_len, seq_len};

    sub_mm1_src_md = memory::desc(sub_mm1_src_dims, dt_src_user, tag::abc);
    sub_mm1_wei_md = memory::desc(sub_mm1_wei_dims, dt_wei, tag::abc);
    sub_mm1_dst_md = memory::desc(sub_mm1_dst_dims, dt_inter, tag::abc);
    dnnl::post_ops dnnl_pops;
    auto mask_dt = static_cast<dnnl::memory::data_type>(
            ltw(inputs[graph_inport[2]]).data_type());
    sub_mm1_post_add_md
            = memory::desc({1, seq_len, seq_len}, mask_dt, tag::abc);
    auto ori_dnnl_pops = sub_matmul1_attr.get_post_ops();
    auto alg
            = static_cast<algorithm>(ori_dnnl_pops.get()->entry_[0].binary.alg);
    dnnl_pops.append_binary(alg, sub_mm1_post_add_md);
    sub_matmul1_attr.set_post_ops(std::move(dnnl_pops));
    auto sub_mm1_pd = matmul::primitive_desc(p_engine, sub_mm1_src_md,
            sub_mm1_wei_md, sub_mm1_dst_md, sub_matmul1_attr);
    sub_mm1_prim = matmul(sub_mm1_pd);

    // Here in the original graph, we have reshape and transpose op to
    // change the dimension and layout of matmul's output. But with the
    // decompose kernel, no need to reshape or transpose the internal buffer.

    // softmax
    // create softmax primitive attr
    auto original_softmax = mqa_op[2];
    dnnl::primitive_attr sub_softmax_attr
            = make_primitive_attr(original_softmax, mgr);
    sub_softmax_dst_md = memory::desc(sub_mm1_dst_dims, dt_src_user, tag::abc);
    auto sub_softmax_pd = softmax_forward::primitive_desc(p_engine,
            prop_kind::forward_inference, algorithm::softmax_accurate,
            sub_mm1_dst_md, sub_softmax_dst_md, sub_mm1_dst_md.get_ndims() - 1,
            sub_softmax_attr);
    sub_softmax_prim = softmax_forward(sub_softmax_pd);

    // reorder src of second matmul (Value)
    // create reorder2 primitive attr
    auto original_reorder2 = mqa_op[3];
    dnnl::primitive_attr sub_reorder2_attr
            = make_primitive_attr(original_reorder2, mgr);
    memory::dims sub_src2_dims = {1, size_per_head, seq_len};
    sub_src2_user_md
            = memory::desc(sub_src2_dims, dt_src_user, {1, seq_len, 1});
    // The format is `abc` due to performance of reorder to `acb` is low.
    auto sub_src2_md = memory::desc(sub_src2_dims, dt_src_user, tag::abc);
    auto sub_reorder2_pd = reorder::primitive_desc(p_engine, sub_src2_user_md,
            p_engine, sub_src2_md, sub_reorder2_attr);
    sub_reorder2.init(sub_reorder2_pd);

    // second matmul
    // create second matmul primitive attr
    auto original_matmul2 = mqa_op[4];
    dnnl::primitive_attr sub_matmul2_attr
            = make_primitive_attr(original_matmul2, mgr);
    memory::dims sub_mm2_src_dims = {1, size_per_head, seq_len};
    memory::dims sub_mm2_wei_dims = {1, seq_len, seq_len};
    memory::dims sub_mm2_dst_dims = {1, size_per_head, seq_len};
    sub_mm2_src_md = memory::desc(sub_mm2_src_dims, dt_src_user, tag::abc);
    auto sub_mm2_wei_md = memory::desc(sub_mm2_wei_dims, dt_src_user, tag::abc);
    sub_mm2_dst_md = memory::desc(sub_mm2_dst_dims, dt_src_user, tag::abc);
    auto sub_mm2_pd = matmul::primitive_desc(p_engine, sub_mm2_src_md,
            sub_mm2_wei_md, sub_mm2_dst_md, sub_matmul2_attr);
    sub_mm2_prim = matmul(sub_mm2_pd);

    // per-head: reorder dst2 from dense to strided
    primitive_attr sub_reorder3_attr;
    sub_reorder3_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    memory::dims sub_dst_dims = {1, size_per_head, seq_len};
    sub_dst_md = memory::desc(sub_dst_dims, dt_src_user, tag::abc);
    sub_dst_user_md = memory::desc(
            sub_dst_dims, dt_src_user, {1, seq_len * num_head, 1});
    auto sub_reorder3_pd = reorder::primitive_desc(
            p_engine, sub_dst_md, p_engine, sub_dst_user_md, sub_reorder3_attr);
    sub_reorder3.init(sub_reorder3_pd);
    ////////////////////////////////////////////////////////////////////////
    /////////////// End Creating primitives ////////////////////////////////
    ////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    /////////////// Start Constructing exec args ///////////////////////////
    ////////////////////////////////////////////////////////////////////////
    memory::desc max_scratchpad_md, sub_max_src1_src2_md, sub_max_dst1_dst2_md;
    size_t max_scratchpad_size = 0;
    // all the scratchpads required by the primitives.
    const std::vector<memory::desc> scratchpads {
            sub_reorder0_pd.scratchpad_desc(),
            sub_reorder1_pd.scratchpad_desc(), sub_mm1_pd.scratchpad_desc(),
            sub_softmax_pd.scratchpad_desc(), sub_reorder2_pd.scratchpad_desc(),
            sub_mm2_pd.scratchpad_desc(), sub_reorder3_pd.scratchpad_desc()};

    for (auto &sp : scratchpads) {
        const size_t size = sp.get_size();
        if (size > max_scratchpad_size) {
            max_scratchpad_size = size;
            max_scratchpad_md = sp;
        }
    }

    auto sub_src1_size = sub_src1_d_md.get_size();
    auto sub_src2_size = sub_mm2_src_md.get_size();
    sub_max_src1_src2_md
            = sub_src1_size > sub_src2_size ? sub_src1_d_md : sub_mm2_src_md;

    auto sub_dst1_size = sub_mm1_dst_md.get_size();
    auto sub_dst2_size = sub_mm2_dst_md.get_size();
    sub_max_dst1_dst2_md
            = sub_dst1_size > sub_dst2_size ? sub_mm1_dst_md : sub_mm2_dst_md;

    // Initialize memory object with empty buffer
    sub_max_src1_src2 = memory(sub_max_src1_src2_md, p_engine, nullptr);
    sub_max_dst1_dst2 = memory(sub_max_dst1_dst2_md, p_engine, nullptr);
    // reorder0: 2d strided -> 2d ab
    sub_src1 = memory(sub_src1_md, p_engine, nullptr);
    // reorder1: 2d strided u8 -> 2d ba s8
    sub_wei1_user = memory(sub_wei1_user_md, p_engine, nullptr);
    // mm1
    sub_mm1_src = memory(sub_mm1_src_md, p_engine, nullptr);
    sub_mm1_wei = memory(sub_mm1_wei_md, p_engine, nullptr);
    sub_mm1_dst = memory(sub_mm1_dst_md, p_engine, nullptr);
    // sub_mm1_post_scale = memory(sub_mm1_post_scale_md, p_engine, nullptr);
    sub_mm1_post_add = memory(sub_mm1_post_add_md, p_engine, nullptr);
    // softmax
    sub_softmax_dst = memory(sub_softmax_dst_md, p_engine, nullptr);
    // reorder2
    sub_src2_user = memory(sub_src2_user_md, p_engine, nullptr);
    // mm2
    sub_mm2_src = memory(sub_mm2_src_md, p_engine, nullptr);
    sub_mm2_dst = memory(sub_mm2_dst_md, p_engine, nullptr);
    //reorder3
    sub_dst_user = memory(sub_dst_user_md, p_engine, nullptr);

    // scratchpad, each thread will have a largest scratchpad.
    sub_scratchpad = memory(max_scratchpad_md, p_engine, nullptr);

    sub_reorder0_args = {{DNNL_ARG_SRC, sub_src1}, {DNNL_ARG_DST, sub_mm1_src},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_reorder1_args = {{DNNL_ARG_SRC, sub_wei1_user},
            {DNNL_ARG_DST, sub_mm1_wei}, {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_mm1_args = {{DNNL_ARG_SRC, sub_mm1_src},
            {DNNL_ARG_WEIGHTS, sub_mm1_wei}, {DNNL_ARG_DST, sub_mm1_dst},
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                    sub_mm1_post_add},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_softmax_args
            = {{DNNL_ARG_SRC, sub_mm1_dst}, {DNNL_ARG_DST, sub_softmax_dst},
                    {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_reorder2_args = {{DNNL_ARG_SRC, sub_src2_user},
            {DNNL_ARG_DST, sub_mm2_src}, {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_mm2_args = {{DNNL_ARG_SRC, sub_mm2_src},
            {DNNL_ARG_WEIGHTS, sub_softmax_dst}, {DNNL_ARG_DST, sub_mm2_dst},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_reorder3_args
            = {{DNNL_ARG_SRC, sub_mm2_dst}, {DNNL_ARG_DST, sub_dst_user},
                    {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    ////////////////////////////////////////////////////////////////////////
    /////////////// End Constructing exec args /////////////////////////////
    ////////////////////////////////////////////////////////////////////////

    // memory planing for buffer sharing
    memory_planning(mqa_registry);
    return status::success;
}

op_ptr mqa_decomp_config_t::get_post_op(const op_ptr &op) const {
    const auto out_val = op->get_output_value(0);
    const auto &consumers = out_val->get_consumers();
    if (consumers.size() != 1) return nullptr;
    return consumers[0].get_op().shared_from_this();
}

status_t mqa_decomp_config_t::record_input_offset(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    auto find_graph_inport = [&](std::shared_ptr<value_t> val) {
        // for quantized matmul, it has producer such as add_zp,sub_zp,mul_scale.
        if (val->get_consumers()[0].get_op().get_kind()
                == graph::op_kind::MatMul) {
            while (val->has_producer()) {
                val = val->get_producer().get_input_value(0);
            }
        }
        for (int i = 0; i < (int)inputs.size(); i++) {
            if (val->get_logical_tensor().id == inputs[i].id) { return i; }
        }
        // If the corresponding input is not found, return an invalid value
        return -1;
    };
    op_ptr mm1, mm2, add;
    for (const auto &cur_op : sg->get_ops()) {
        if (mm1 != nullptr && mm2 != nullptr) break;
        if (cur_op->get_kind() != graph::op_kind::MatMul) continue;
        auto post_op = get_post_op(cur_op);
        if (post_op != nullptr
                && post_op->get_kind() == graph::op_kind::StaticReshape) {
            mm1 = cur_op;
            auto transpose = get_post_op(post_op);
            if (transpose != nullptr
                    && transpose->get_kind()
                            == graph::op_kind::StaticTranspose) {
                add = get_post_op(transpose);
            }
        } else
            mm2 = cur_op;
    }
    if (impl::utils::one_of(nullptr, mm1, mm2, add))
        return status::invalid_graph;

    int src1_id = find_graph_inport(mm1->get_input_value(0));
    graph_inport.emplace_back(src1_id);
    int wei1_id = find_graph_inport(mm1->get_input_value(1));
    graph_inport.emplace_back(wei1_id);
    // for scale and add op. The input order is uncertain.
    int add_id = find_graph_inport(add->get_input_value(0));
    if (add_id == -1) add_id = find_graph_inport(add->get_input_value(1));
    graph_inport.emplace_back(add_id);

    int src2_id = find_graph_inport(mm2->get_input_value(0));
    graph_inport.emplace_back(src2_id);
    return status::success;
}

status_t mqa_decomp_config_t::record_mqa_ops(std::shared_ptr<subgraph_t> &sg) {
    op_ptr reorder1, reorder2, matmul1, softmax, matmul2;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;
        if (get_post_op(cur_op) != nullptr) {
            matmul1 = cur_op;
            auto reshape = get_post_op(cur_op);
            auto transpose = get_post_op(reshape);
            softmax = get_post_op(transpose);
        } else {
            matmul2 = cur_op;
        }
    }
    this->mqa_op = {reorder1, matmul1, softmax, reorder2, matmul2};
    return status::success;
}

void mqa_decomp_config_t::memory_planning(registry_t &mqa_registry) {
    // Registry is used to do the memory planning for mqa decomposition
    // algorithm. We reused some internal memory to reduce the memory
    // footprint for better cache hit. And here the key in registar of each
    // memory is planned in a specific order.
    registrar_t temporary_registrar = mqa_registry.registrar();

    // Here we initialize the map based on certain memory reuse logic. Those
    // memories(mds) who share the same buffer have the same registar key in
    // this map. So if we want to change the memory reuse logic, we need to
    // change the value of map here.
    mem_key_map = {{sub_max_src1_src2.get(), 0}, {sub_mm1_wei.get(), 1},
            {sub_max_dst1_dst2.get(), 2}, {sub_softmax_dst.get(), 3},
            {sub_scratchpad.get(), 4}};

    temporary_registrar.book(mem_key_map[sub_max_src1_src2.get()],
            sub_max_src1_src2.get_desc().get_size());
    temporary_registrar.book(
            mem_key_map[sub_mm1_wei.get()], sub_mm1_wei.get_desc().get_size());
    temporary_registrar.book(mem_key_map[sub_max_dst1_dst2.get()],
            sub_max_dst1_dst2.get_desc().get_size());
    temporary_registrar.book(mem_key_map[sub_softmax_dst.get()],
            sub_softmax_dst.get_desc().get_size());
    temporary_registrar.book(mem_key_map[sub_scratchpad.get()],
            sub_scratchpad.get_desc().get_size());
}

dnnl::primitive_attr mqa_decomp_config_t::make_primitive_attr(
        std::shared_ptr<op_t> &op, fusion_info_mgr_t &mgr) {
    dnnl::primitive_attr attr;
    if (op && op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        const fusion_info_t &fusion_info = mgr.get_info(key);
        attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    if (op && op->get_kind() == op_kind::dnnl_reorder) {
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

template status_t
mqa_decomp_config_t::construct_params<false, dnnl::memory::data_type::f32>(
        std::shared_ptr<subgraph_t> &sg, registry_t &mqa_registry,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs);

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
