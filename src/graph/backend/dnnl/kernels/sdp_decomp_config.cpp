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

#include "graph/backend/dnnl/kernels/sdp_decomp_config.hpp"
#include "graph/interface/shape_infer.hpp"

#define VCHECK_SDP_DECOMP(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_decomp_kernel_t, (cond), status, msg, \
            ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

bool sdp_decomp_config_t::initial_check(const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    // The order of input logical tensors in inputs is not certain, we need
    // to record the input offset in a certain order of ops.
    CHECK_BOOL(record_input_offset(sg, inputs));
    dims src1_user_dims = ltw(inputs[graph_inport[0]]).vdims();
    VCHECK_SDP_DECOMP(src1_user_dims.size() == 4, false,
            "Input dims should be 4, but got %zu", src1_user_dims.size());

    // Initialize SDP input dimension according to the src of mm1
    batch_size = src1_user_dims[0];
    num_head_q = src1_user_dims[1];
    seq_len_q = src1_user_dims[2];
    size_per_head = src1_user_dims[3];

    dims wei1_user_dims = ltw(inputs[graph_inport[1]]).vdims();
    num_head_kv = wei1_user_dims[1];

    // Check batch size compatibility.
    dims wei2_user_dims = ltw(inputs[graph_inport[4]]).vdims();
    VCHECK_SDP_DECOMP(
            batch_size == wei1_user_dims[0] && batch_size == wei2_user_dims[0],
            false,
            "Batch size mismatch, batch_size: %lld, wei1: %lld, wei2: %lld",
            batch_size, wei1_user_dims[0], wei2_user_dims[0]);

    // Check scale size
    if (graph_inport[2] != -1) {
        auto scale_sz = ltw(inputs[graph_inport[2]]).nelems();
        VCHECK_SDP_DECOMP(scale_sz == 1, false,
                "Only supports single scale value, but got %lld", scale_sz);
    }

    // Check select cond and src0 shape
    if (graph_inport[5] != -1 && graph_inport[6] != -1) {
        const auto select_cond_dims = ltw(inputs[graph_inport[5]]).vdims();
        const auto select_src0_dims = ltw(inputs[graph_inport[6]]).vdims();
        VCHECK_SDP_DECOMP(select_cond_dims != select_src0_dims, false,
                "Only supports select for case requiring broadcast cond input, "
                "but got cond dims %s and src0 dims %s",
                dims2str(select_cond_dims).c_str(),
                dims2str(select_src0_dims).c_str());
    }

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
// RATIO is an empirical value used to determine the numerical relationship
// between batch_size, num_head_q and thread number to determine whether to use
// decompose kernel. The key to the decompose kernel is that we do parallel in
// the batch_size and num_head_q dimensions. Therefore, if the batch_size or
// num_head_q is too small, it will cause many idle threads and affect
// efficiency which may even worse than the original sequential kernel. Here we
// set this ratio based on the experimental value to ensure that users do not
// have any regression when using the decompose kernel.
// TODO: Refine the inequation based on the relationship of cache size and sdp
// memory footprint requirements.
#define RATIO 2
    // Initialize nthr with current threads num
    nthr = dnnl_get_current_num_threads();
    VCHECK_SDP_DECOMP(batch_size * num_head_q > RATIO * nthr, false,
            "Doesn't meet condition for decompose: Batch size * num_head_q "
            "should be larger than ratio * nthr, but got batch_size %lld, "
            "num_head_q %lld, ration %d , nthr %d",
            batch_size, num_head_q, RATIO, nthr);
#endif
    return true;
}

template <bool quantized, memory::data_type dt>
impl::status_t sdp_decomp_config_t::construct_params(
        std::shared_ptr<subgraph_t> &sg, registry_t &sdp_registry,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs) {

    // Record the ops inside of SDP pattern for later usage
    CHECK(record_sdp_ops(sg, quantized));

    // Update SDPA input params. Sequence length for query and key/value are
    // NOT always same.
    const auto &lt_wei = sdp_op[1]->get_input_value(1)->get_logical_tensor();
    const ltw ltw_wei(lt_wei);
    seq_len_kv = ltw_wei.vdims()[3];

    // Acquire the data type from input param for later primitive creation.
    // The src and wei dt of both quantized sdp and float sdp are the same.
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
            sub_mm1_wei_md, sub_mm1_dst_md, sub_softmax_dst_md,
            sub_wei2_user_md, sub_mm2_wei_md, sub_mm2_dst_md, sub_dst_md,
            sub_dst_user_md;
    std::vector<memory::desc> sub_mm1_post_md;

    // must use user mode to support concurrent execution
    primitive_attr sub_reorder0_attr;
    sub_reorder0_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // per-head: reorder src1 to dense, for first matmul
    dims sub_src1_dims = {1, 1, seq_len_q, size_per_head};
    src1_strides = ltw(inputs[graph_inport[0]]).vstrides();
    sub_src1_md = memory::desc(sub_src1_dims, dt_src_user,
            {1, 1, src1_strides[2], src1_strides[3]});
    auto sub_src1_d_md = memory::desc(sub_src1_dims, dt_src_user, tag::abcd);
    auto sub_reorder0_pd = reorder::primitive_desc(
            p_engine, sub_src1_md, p_engine, sub_src1_d_md, sub_reorder0_attr);
    sub_reorder0.init(sub_reorder0_pd);

    auto &mgr = sg->fusion_info_mgr_;

    // per-head: reorder u8->s8 wei for first matmul
    // create reorder1 primitive attr
    dnnl::primitive_attr sub_reorder1_attr
            = make_primitive_attr(sdp_op[0], mgr);
    dims sub_wei1_dims = {1, 1, size_per_head, seq_len_kv};
    auto wei_md = make_dnnl_memory_desc(
            sdp_op[1]->get_input_value(1)->get_logical_tensor());
    wei1_strides = wei_md.get_strides();
    sub_wei1_user_md = memory::desc(sub_wei1_dims, dt_wei_user,
            {1, 1, wei1_strides[2], wei1_strides[3]});
    // Flip the format to have `ba` weights MBI item in per thread loop.
    sub_wei1_md = memory::desc(sub_wei1_dims, dt_wei, tag::abdc);
    auto sub_reorder1_pd = reorder::primitive_desc(p_engine, sub_wei1_user_md,
            p_engine, sub_wei1_md, sub_reorder1_attr);
    sub_reorder1.init(sub_reorder1_pd);

    // first matmul
    // create first matmul primitive attr
    dnnl::primitive_attr sub_matmul1_attr = make_primitive_attr(sdp_op[1], mgr);
    dims sub_mm1_src_dims = {1, 1, seq_len_q, size_per_head};
    dims sub_mm1_wei_dims = {1, 1, size_per_head, seq_len_kv};
    dims sub_mm1_dst_dims = {1, 1, seq_len_q, seq_len_kv};

    sub_mm1_src_md = memory::desc(sub_mm1_src_dims, dt_src_user, tag::abcd);
    sub_mm1_wei_md = memory::desc(sub_mm1_wei_dims, dt_wei, tag::abdc);
    sub_mm1_dst_md = memory::desc(sub_mm1_dst_dims, dt_inter, tag::abcd);
    dnnl::post_ops dnnl_pops;
    auto ori_dnnl_pops = sub_matmul1_attr.get_post_ops();
    for (int i = 0; i < ori_dnnl_pops.get()->len(); i++) {
        auto alg = static_cast<algorithm>(
                ori_dnnl_pops.get()->entry_[i].binary.alg);
        const dnnl::impl::memory_desc_t &ori_desc
                = ori_dnnl_pops.get()->entry_[i].binary.user_src1_desc;
        auto post_shape = ori_desc.dims;
        auto post_stride = ori_desc.format_desc.blocking.strides;
        auto post_dt = static_cast<memory::data_type>(ori_desc.data_type);
        dims post_stride_dims = dims(post_stride, post_stride + ori_desc.ndims);
        auto new_sub_md = memory::desc({1, 1, post_shape[2], post_shape[3]},
                post_dt, post_stride_dims);
        sub_mm1_post_md.emplace_back(new_sub_md);
        dnnl_pops.append_binary(alg, new_sub_md);
    }
    sub_matmul1_attr.set_post_ops(std::move(dnnl_pops));
    auto sub_mm1_pd = matmul::primitive_desc(p_engine, sub_mm1_src_md,
            sub_mm1_wei_md, sub_mm1_dst_md, sub_matmul1_attr);
    sub_mm1_prim = matmul(sub_mm1_pd);

    // softmax
    // create softmax primitive attr
    dnnl::primitive_attr sub_softmax_attr = make_primitive_attr(sdp_op[2], mgr);
    sub_softmax_dst_md = memory::desc(sub_mm1_dst_dims, dt_src_user, tag::abcd);
    auto sub_softmax_pd = softmax_forward::primitive_desc(p_engine,
            prop_kind::forward_inference, algorithm::softmax_accurate,
            sub_mm1_dst_md, sub_softmax_dst_md, sub_mm1_dst_md.get_ndims() - 1,
            sub_softmax_attr);
    sub_softmax_prim = softmax_forward(sub_softmax_pd);

    // reorder u8->s8 wei for second matmul
    // create reorder2 primitive attr
    dnnl::primitive_attr sub_reorder2_attr
            = make_primitive_attr(sdp_op[3], mgr);
    dims sub_wei2_dims = {1, 1, seq_len_kv, size_per_head};
    wei2_strides = ltw(inputs[graph_inport[4]]).vstrides();
    sub_wei2_user_md = memory::desc(sub_wei2_dims, dt_wei_user,
            {1, 1, wei2_strides[2], wei2_strides[3]});
    // The format is `abcd` due to performance of reorder to `abdc` is low.
    auto sub_wei2_md = memory::desc(sub_wei2_dims, dt_wei, tag::abcd);
    auto sub_reorder2_pd = reorder::primitive_desc(p_engine, sub_wei2_user_md,
            p_engine, sub_wei2_md, sub_reorder2_attr);
    sub_reorder2.init(sub_reorder2_pd);

    // second matmul
    // create second matmul primitive attr
    dnnl::primitive_attr sub_matmul2_attr = make_primitive_attr(sdp_op[4], mgr);
    dims sub_mm2_src_dims = {1, 1, seq_len_q, seq_len_kv};
    dims sub_mm2_wei_dims = {1, 1, seq_len_kv, size_per_head};
    dims sub_mm2_dst_dims = {1, 1, seq_len_q, size_per_head};
    auto sub_mm2_src_md
            = memory::desc(sub_mm2_src_dims, dt_src_user, tag::abcd);
    sub_mm2_wei_md = memory::desc(sub_mm2_wei_dims, dt_wei, tag::abcd);
    sub_mm2_dst_md = memory::desc(sub_mm2_dst_dims, dt_src_user, tag::abcd);
    auto sub_mm2_pd = matmul::primitive_desc(p_engine, sub_mm2_src_md,
            sub_mm2_wei_md, sub_mm2_dst_md, sub_matmul2_attr);
    sub_mm2_prim = matmul(sub_mm2_pd);

    // per-head: reorder dst2 from dense to strided
    primitive_attr sub_reorder3_attr;
    sub_reorder3_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    dims sub_dst_dims = {1, 1, seq_len_q, size_per_head};
    auto out_lt = sdp_op[4]->get_output_value(0)->get_logical_tensor();
    dst_strides = ltw(out_lt).vstrides();
    sub_dst_md = memory::desc(sub_dst_dims, dt_src_user, tag::abcd);
    sub_dst_user_md = memory::desc(
            sub_dst_dims, dt_src_user, {1, 1, dst_strides[2], dst_strides[3]});
    auto sub_reorder3_pd = reorder::primitive_desc(
            p_engine, sub_dst_md, p_engine, sub_dst_user_md, sub_reorder3_attr);
    sub_reorder3.init(sub_reorder3_pd);
    ////////////////////////////////////////////////////////////////////////
    /////////////// End Creating primitives ////////////////////////////////
    ////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    /////////////// Start Constructing exec args ///////////////////////////
    ////////////////////////////////////////////////////////////////////////
    memory::desc max_scratchpad_md, sub_max_src1_src2_md, sub_max_dst1_wei2_md;
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
    auto sub_src2_size = sub_softmax_dst_md.get_size();
    sub_max_src1_src2_md = sub_src1_size > sub_src2_size ? sub_src1_d_md
                                                         : sub_softmax_dst_md;

    auto sub_dst1_size = sub_mm1_dst_md.get_size();
    auto sub_wei2_size = sub_mm2_wei_md.get_size();
    sub_max_dst1_wei2_md
            = sub_dst1_size > sub_wei2_size ? sub_mm1_dst_md : sub_mm2_wei_md;

    // Initialize memory object with empty buffer
    sub_max_src1_src2 = memory(sub_max_src1_src2_md, p_engine, nullptr);
    sub_max_dst1_wei2 = memory(sub_max_dst1_wei2_md, p_engine, nullptr);
    // reorder0: 2d strided -> 2d ab
    sub_src1 = memory(sub_src1_md, p_engine, nullptr);
    // reorder1: 2d strided u8 -> 2d ba s8
    sub_wei1_user = memory(sub_wei1_user_md, p_engine, nullptr);
    // mm1
    sub_mm1_src = memory(sub_mm1_src_md, p_engine, nullptr);
    sub_mm1_wei = memory(sub_mm1_wei_md, p_engine, nullptr);
    sub_mm1_dst = memory(sub_mm1_dst_md, p_engine, nullptr);
    for (size_t i = 0; i < sub_mm1_post_md.size(); i++) {
        sub_mm1_post_mem.emplace_back(sub_mm1_post_md[i], p_engine, nullptr);
    }
    // softmax
    sub_softmax_dst = memory(sub_softmax_dst_md, p_engine, nullptr);
    // reorder2
    sub_wei2_user = memory(sub_wei2_user_md, p_engine, nullptr);
    // mm2
    sub_mm2_wei = memory(sub_mm2_wei_md, p_engine, nullptr);
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
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
    for (int i = 0; i < (int)sub_mm1_post_mem.size(); i++) {
        sub_mm1_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
                sub_mm1_post_mem[i]});
    }

    sub_softmax_args
            = {{DNNL_ARG_SRC, sub_mm1_dst}, {DNNL_ARG_DST, sub_softmax_dst},
                    {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_reorder2_args = {{DNNL_ARG_SRC, sub_wei2_user},
            {DNNL_ARG_DST, sub_mm2_wei}, {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_mm2_args = {{DNNL_ARG_SRC, sub_softmax_dst},
            {DNNL_ARG_WEIGHTS, sub_mm2_wei}, {DNNL_ARG_DST, sub_mm2_dst},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_reorder3_args
            = {{DNNL_ARG_SRC, sub_mm2_dst}, {DNNL_ARG_DST, sub_dst_user},
                    {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    // add scales and zps for mm1, softmax, mm2
    prepare_sdp_scales_zps(mgr, sdp_op[0], 1, sub_reorder1_args, p_engine);
    prepare_sdp_scales_zps(mgr, sdp_op[1], 2, sub_mm1_args, p_engine);
    prepare_sdp_scales_zps(mgr, sdp_op[2], 1, sub_softmax_args, p_engine);
    prepare_sdp_scales_zps(mgr, sdp_op[3], 1, sub_reorder2_args, p_engine);
    prepare_sdp_scales_zps(mgr, sdp_op[4], 2, sub_mm2_args, p_engine);
    ////////////////////////////////////////////////////////////////////////
    /////////////// End Constructing exec args /////////////////////////////
    ////////////////////////////////////////////////////////////////////////

    // memory planing for buffer sharing
    memory_planning(sdp_registry);
    // TODO: remove this when primitive new API ready
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    omp_set_num_threads(nthr);
#endif
    return status::success;
}

impl::status_t sdp_decomp_config_t::record_select_ops(
        std::shared_ptr<subgraph_t> &sg, std::vector<op_ptr> &select_out_ops) {

    //post scale isn't from select.
    //so the post binary number from select is post_op's size - 1
    const auto select_out_ops_size = sub_mm1_post_mem.size() - 1;
    select_out_ops.resize(select_out_ops_size);
    //sdp_op[1] is mm1.
    size_t input_size = sdp_op[1]->num_inputs();
    /*
            src wei   post_scale attn_mask* post_binary...(from select)
              \   \       /       /         /
               \   \     /     /         /
                 \  \   /   /        /
                   \ \ / /       /
                     mm1
        */
    // input_size - select_out_ops_size is the starting index of post ops
    // from select.
    for (size_t i = 0; i < select_out_ops_size; i++) {
        select_out_ops[i] = sdp_op[1]
                                    ->get_input_value(input_size
                                            - select_out_ops_size + i)
                                    ->get_producer()
                                    .shared_from_this();
    }

    const std::unordered_set<op_kind_t> select_kind
            = {op_kind::dnnl_eltwise, op_kind::dnnl_binary};
    return topo_order_visit(
            sg->get_output_ops(), [&select_kind, this](op_t *op) {
                bool is_select = false;
                if (select_kind.count(op->get_kind())) is_select = true;
                if (op->get_kind() == op_kind::dnnl_reorder
                        || op->get_kind() == op_kind::dnnl_unsqueeze) {
                    auto post_op = get_post_op(op->shared_from_this());
                    if (post_op != nullptr
                            && select_kind.count(post_op->get_kind()))
                        is_select = true;
                }
                if (is_select)
                    this->select_op.emplace_back(op->shared_from_this());
                return status::success;
            });
}

impl::status_t sdp_decomp_config_t::record_select_out_index(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<op_ptr> &select_out_ops) {
    // select_outop_index is used to record the topo order index of output
    // ops from the new select subgraph. -1 means this array isn't
    // initialized.
    select_outop_index.resize(select_out_ops.size(), -1);
    int temp = 0;
    return topo_order_visit(
            sg->get_output_ops(), [&temp, this, &select_out_ops](op_t *op) {
                for (size_t i = 0; i < select_out_ops.size(); i++) {
                    if (select_out_ops[i].get() == op) {
                        select_outop_index[i] = temp;
                        break;
                    }
                }
                temp++;
                return status::success;
            });
}

op_ptr sdp_decomp_config_t::get_post_op(const op_ptr &op) const {
    const auto out_val = op->get_output_value(0);
    const auto &consumers = out_val->get_consumers();
    if (consumers.size() != 1) return nullptr;
    return consumers[0].get_op().shared_from_this();
}

impl::status_t sdp_decomp_config_t::record_input_offset(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    auto find_graph_inport = [&](std::shared_ptr<value_t> val) {
        // for quantized matmul, it has producer such as add_zp,sub_zp,mul_scale.
        if (val->get_consumers()[0].get_op().get_kind()
                        == graph::op_kind::MatMul
                || (val->has_producer()
                        && val->get_producer().get_kind()
                                == graph::op_kind::StaticReshape)) {
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
    op_ptr mm1 = nullptr, mm2 = nullptr, scale = nullptr, add = nullptr,
           select = nullptr;
    const std::unordered_set<graph::op_kind_t> post_op_kind
            = {graph::op_kind::Divide, graph::op_kind::Multiply,
                    graph::op_kind::Add, graph::op_kind::Select,
                    graph::op_kind::SoftMax};
    for (const auto &cur_op : sg->get_ops()) {
        const auto &op_kind = cur_op->get_kind();
        VCHECK_SDP_DECOMP(op_kind != graph::op_kind::GenIndex,
                status::unimplemented, "Not support implicit causal mask");
        VCHECK_SDP_DECOMP(op_kind != graph::op_kind::DynamicDequantize,
                status::unimplemented,
                "Decomposed kernel does not support dynamic quantization");
        // both mm1 and mm2 are found.
        if (mm1 && mm2) break;
        if (op_kind != graph::op_kind::MatMul) continue;

        auto post_op = get_post_op(cur_op);
        if (post_op && post_op_kind.count(post_op->get_kind())) {
            // find mm1
            mm1 = cur_op;
            // TODO(xxx): Currently, p2 is not supported by decomp kernel.
            // p1: [matmul] --> [scale] --> [select] --> [mask] --> ...
            // p2: [matmul] --> [select] --> [scale] --> [mask] --> ...
            VCHECK_SDP_DECOMP(post_op->get_kind() != graph::op_kind::Select,
                    status::unimplemented,
                    "Not support select between matmul1 and scale");
            // find scale
            if (post_op->get_kind() == graph::op_kind::Divide
                    || post_op->get_kind() == graph::op_kind::Multiply) {
                has_scale = true;
                scale = post_op;
                post_op = get_post_op(post_op);
            }

            if (post_op) {
                // find mask
                if (post_op->get_kind() == graph::op_kind::Add) {
                    add = std::move(post_op);
                    has_attention_mask = true;
                } else if (post_op->get_kind() == graph::op_kind::Select) {
                    // mm1 -> scale -> select -> ...
                    select = std::move(post_op);
                    has_select = true;
                }
            }
        } else {
            // find mm2
            mm2 = cur_op;
        }
    }
    VCHECK_SDP_DECOMP(mm1 != nullptr && mm2 != nullptr, status::invalid_graph,
            "Failed to find matmul1 or matmul2");
    int src1_id = find_graph_inport(mm1->get_input_value(0));
    graph_inport.emplace_back(src1_id);
    int wei1_id = find_graph_inport(mm1->get_input_value(1));
    graph_inport.emplace_back(wei1_id);
    // for scale and add op. The input order is uncertain.
    if (has_scale) {
        int scale_id = find_graph_inport(scale->get_input_value(1));
        if (scale_id == -1)
            scale_id = find_graph_inport(scale->get_input_value(0));
        graph_inport.emplace_back(scale_id);
    } else {
        //placeholder
        graph_inport.emplace_back(-1);
    }
    if (has_attention_mask) {
        int add_id = find_graph_inport(add->get_input_value(1));
        if (add_id == -1) add_id = find_graph_inport(add->get_input_value(0));
        graph_inport.emplace_back(add_id);
    } else {
        //placeholder
        graph_inport.emplace_back(-1);
    }
    int wei2_id = find_graph_inport(mm2->get_input_value(1));
    graph_inport.emplace_back(wei2_id);
    if (has_select) {
        int cond_id = find_graph_inport(select->get_input_value(0));
        int src0_id = find_graph_inport(select->get_input_value(1));
        graph_inport.emplace_back(cond_id);
        graph_inport.emplace_back(src0_id);
    } else {
        //placeholder
        graph_inport.emplace_back(-1);
        graph_inport.emplace_back(-1);
    }
    return status::success;
}

impl::status_t sdp_decomp_config_t::record_sdp_ops(
        std::shared_ptr<subgraph_t> &sg, bool is_quantize) {
    const auto get_wei_pre_op = [](const op_ptr &op) -> op_ptr {
        const auto out_val = op->get_input_value(1);
        if (out_val->has_producer()) {
            auto &producer = out_val->get_producer();
            if (producer.get_kind() != op_kind::dnnl_reorder) return nullptr;
            return producer.shared_from_this();
        } else
            return nullptr;
    };

    for (const auto &cur_op : sg->get_ops()) {
        if (!cur_op || cur_op->get_kind() != op_kind::dnnl_matmul) continue;
        auto post_op = get_post_op(cur_op);
        if (!post_op || post_op->get_kind() != op_kind::dnnl_softmax) continue;
        auto ppost_op = get_post_op(post_op);
        VCHECK_SDP_DECOMP(ppost_op != nullptr, status::invalid_graph,
                "Failed to find post post op for matmul");

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

void sdp_decomp_config_t::memory_planning(registry_t &sdp_registry) {
    // Registry is used to do the memory planning for sdp decomposition
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
            sub_max_src1_src2.get_desc().get_size());
    temporary_registrar.book(
            mem_key_map[sub_mm1_wei.get()], sub_mm1_wei.get_desc().get_size());
    temporary_registrar.book(mem_key_map[sub_max_dst1_wei2.get()],
            sub_max_dst1_wei2.get_desc().get_size());
    temporary_registrar.book(
            mem_key_map[sub_mm2_dst.get()], sub_mm2_dst.get_desc().get_size());
    temporary_registrar.book(mem_key_map[sub_scratchpad.get()],
            sub_scratchpad.get_desc().get_size());
}

impl::status_t sdp_decomp_config_t::prepare_sdp_scales_zps(
        const fusion_info_mgr_t &mgr, std::shared_ptr<op_t> &op, int index,
        std::unordered_map<int, memory> &args, const dnnl::engine &p_engine) {
    const auto dt_scale = memory::data_type::f32,
               dt_zp = memory::data_type::s32;
    // scale zp order:
    // 1. src scale, wei scale
    // 2. src zp, wei zp
    // 3. dst scale, dst zp
    if (op && op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        const fusion_info_t &fusion_info = mgr.get_info(key);
        if (fusion_info.with_runtime_scales(true, 0)) {
            memory::desc sub_src_scale_md = memory::desc({1}, dt_scale, tag::x);
            memory sub_src_scale = memory(sub_src_scale_md, p_engine);
            float *src_scale_val_ptr = reinterpret_cast<float *>(
                    sub_src_scale.get_data_handle());
            src_scale_val_ptr[0] = get_attr_value<float, float>(
                    op, index++, op_attr::scales);

            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sub_src_scale});
        }
        if (fusion_info.with_runtime_scales(true, 1)) {
            memory::desc sub_wei_scale_md = memory::desc({1}, dt_scale, tag::x);
            memory sub_wei_scale = memory(sub_wei_scale_md, p_engine);
            float *wei_scale_val_ptr = reinterpret_cast<float *>(
                    sub_wei_scale.get_data_handle());
            wei_scale_val_ptr[0] = get_attr_value<float, float>(
                    op, index++, op_attr::scales);
            args.insert(
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sub_wei_scale});
        }

        // src_zp and wei_zp
        if (fusion_info.with_runtime_zero_points(true, 0)) {
            memory::desc sub_src_zp_md = memory::desc({1}, dt_zp, tag::x);
            memory sub_src_zp = memory(sub_src_zp_md, p_engine);
            int *src_zp_val_ptr
                    = reinterpret_cast<int *>(sub_src_zp.get_data_handle());
            src_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                    op, index++, op_attr::zps);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, sub_src_zp});
        }
        if (fusion_info.with_runtime_zero_points(true, 1)) {
            memory::desc sub_wei_zp_md = memory::desc({1}, dt_zp, tag::x);
            memory sub_wei_zp = memory(sub_wei_zp_md, p_engine);
            int *wei_zp_val_ptr
                    = reinterpret_cast<int *>(sub_wei_zp.get_data_handle());
            wei_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                    op, index++, op_attr::zps);
            args.insert(
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, sub_wei_zp});
        }

        // dst scale, dst zp
        if (fusion_info.with_runtime_scales(false, 0)) {
            memory::desc sub_dst_scale_md = memory::desc({1}, dt_scale, tag::x);
            memory sub_dst_scale = memory(sub_dst_scale_md, p_engine);
            float *dst_scale_val_ptr = reinterpret_cast<float *>(
                    sub_dst_scale.get_data_handle());
            dst_scale_val_ptr[0] = get_attr_value<float, float>(
                    op, index++, op_attr::scales);
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, sub_dst_scale});
        }
        if (fusion_info.with_runtime_zero_points(false, 0)) {
            memory::desc sub_dst_zp_md = memory::desc({1}, dt_zp, tag::x);
            memory sub_dst_zp = memory(sub_dst_zp_md, p_engine);
            int *dst_zp_val_ptr
                    = reinterpret_cast<int *>(sub_dst_zp.get_data_handle());
            dst_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                    op, index++, op_attr::zps);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, sub_dst_zp});
        }
    }
    if (op && op->get_kind() == op_kind::dnnl_reorder) {
        if (op->has_attr(op_attr::with_runtime_dst_zps)
                && op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
            memory::desc sub_dst_zp_md = memory::desc({1}, dt_zp, tag::x);
            memory sub_dst_zp = memory(sub_dst_zp_md, p_engine);
            int *dst_zp_val_ptr
                    = reinterpret_cast<int *>(sub_dst_zp.get_data_handle());
            dst_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                    op, index++, op_attr::zps);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, sub_dst_zp});
        }
    }
    return status::success;
}

dnnl::primitive_attr sdp_decomp_config_t::make_primitive_attr(
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
sdp_decomp_config_t::construct_params<false, dnnl::memory::data_type::f32>(
        std::shared_ptr<subgraph_t> &sg, registry_t &mqa_registry,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs);
template status_t
sdp_decomp_config_t::construct_params<true, dnnl::memory::data_type::f32>(
        std::shared_ptr<subgraph_t> &sg, registry_t &mqa_registry,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs);
template status_t
sdp_decomp_config_t::construct_params<true, dnnl::memory::data_type::bf16>(
        std::shared_ptr<subgraph_t> &sg, registry_t &mqa_registry,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs);

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
