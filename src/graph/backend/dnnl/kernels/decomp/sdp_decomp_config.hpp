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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_DECOMP_CONFIG_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_DECOMP_CONFIG_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "common/dnnl_thread.hpp"

#include "graph/interface/c_types_map.hpp"

#include "graph/backend/dnnl/scratchpad.hpp"
#include "graph/backend/dnnl/subgraph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using ltw = logical_tensor_wrapper_t;
using op_ptr = std::shared_ptr<op_t>;
using registry_key = size_t;

// TODO: merge with mqa_reorder_t
struct sdp_reorder_t {
public:
    status_t init(const dnnl::reorder::primitive_desc &pd) {
        auto src_desc = pd.src_desc();
        auto dst_desc = pd.dst_desc();
        if (src_desc == dst_desc) is_inplace_ = true;
        reorder_prim_ = reorder(pd);
        return status::success;
    }

    bool get_inplace() const { return is_inplace_; }

    status_t execute(const dnnl::stream &astream,
            const std::unordered_map<int, dnnl::memory> &args) const {
        if (is_inplace_) {
            void *handle = args.at(DNNL_ARG_SRC).get_data_handle();
            args.at(DNNL_ARG_DST).set_data_handle(handle);
        } else
            reorder_prim_.execute(astream, args);
        return status::success;
    }

private:
    dnnl::primitive reorder_prim_;
    bool is_inplace_ = false;
};

struct sdp_decomp_config_t {
public:
    sdp_decomp_config_t() = default;

    // SDP input dimension
    dim_t batch_size, num_head_q, num_head_kv, seq_len_q, seq_len_kv,
            size_per_head;

    // SDP input and output strides
    dims src1_strides, wei1_strides, wei2_strides, dst_strides,
            post_add_strides;

    // Thread nums during the workflow
    int nthr;

    // Used to record the exact input offset in subgraph
    // [mm1_src,mm1_wei,mm1_scale,mm1_add,mm2_wei,select_condition,select_other_input]
    std::vector<int> graph_inport;

    // Primitives that actually perform calculations
    primitive sub_mm1_prim, sub_softmax_prim, sub_mm2_prim;
    sdp_reorder_t sub_reorder0, sub_reorder1, sub_reorder2, sub_reorder3;

    // Args used in the execution of primitives
    std::unordered_map<int, memory> sub_reorder0_args, sub_reorder1_args,
            sub_mm1_args, sub_softmax_args, sub_reorder2_args, sub_mm2_args,
            sub_reorder3_args;

    // A map from memory to registry key, used to record the internal memories
    // location inside of the whole buffer.
    std::unordered_map<dnnl_memory_t, registry_key> mem_key_map;

    // Internal memory objects for each primitive in each threads.
    // reorder0
    memory sub_src1;
    // reorder1
    memory sub_wei1_user, sub_wei1_zp;
    //mm1
    memory sub_mm1_src, sub_mm1_wei, sub_mm1_dst;
    // sub_mm1_post_mem contains [post_scale, attn_mask(optional), post_binary(from select)...]
    std::vector<memory> sub_mm1_post_mem;
    //softmax
    memory sub_softmax_dst;
    //reorder2
    memory sub_wei2_user, sub_wei2_zp;
    //mm2
    memory sub_mm2_wei, sub_mm2_dst;
    //reorder3
    memory sub_dst_user;
    //scratchpad
    memory sub_scratchpad;
    // shared memory
    memory sub_max_src1_src2, sub_max_dst1_wei2;

    bool has_scale = false, has_attention_mask = false, has_select = false;
    // Used to record the ops from select
    std::vector<op_ptr> select_op;
    std::vector<int> select_outop_index;

private:
    // Used to record the ops contained in SDP
    // sdp_op = [reorder1, mm1, softmax, reorder2, mm2]
    // reorder1 is using mm1 weight u8->s8
    // reorder2 is using mm2 weight u8->s8
    std::vector<op_ptr> sdp_op;

public:
    // The function is used to check if the configuration of SDP is supported by
    // current implementation of decomp kernel. Currently, this implementation
    // can handle 4-dims tensor and limits the numerical relationship between
    // batch_size, num_head and thread num.
    // If the check passes, initialize few members according to inputs
    // If no, return unimplemented status directly and fallback to large kernel
    bool initial_check(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs);

    // Used to construct all params that SDP need
    template <bool quantized = false,
            memory::data_type dt = memory::data_type::f32>
    impl::status_t construct_params(std::shared_ptr<subgraph_t> &sg,
            registry_t &sdp_registry, const dnnl::engine &p_engine,
            const std::vector<logical_tensor_t> &inputs);

    impl::status_t record_select_ops(std::shared_ptr<subgraph_t> &sg,
            std::vector<op_ptr> &select_out_ops);
    impl::status_t record_select_out_index(
            const std::shared_ptr<subgraph_t> &sg,
            const std::vector<op_ptr> &select_out_ops);

private:
    op_ptr get_post_op(const op_ptr &op) const;

    impl::status_t record_input_offset(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs);

    impl::status_t record_sdp_ops(
            std::shared_ptr<subgraph_t> &sg, bool is_quantize);

    void memory_planning(registry_t &sdp_registry);

    impl::status_t prepare_sdp_scales_zps(const fusion_info_mgr_t &mgr,
            std::shared_ptr<op_t> &op, int index,
            std::unordered_map<int, memory> &args,
            const dnnl::engine &p_engine);

    template <typename attr_dt, typename target_dt>
    target_dt get_attr_value(
            std::shared_ptr<op_t> &op, int i, op_attr_t attr_name) {
        const auto in_val = op->get_input_value(i);
        auto &producer = in_val->get_producer();
        return static_cast<target_dt>(
                producer.get_attr<std::vector<attr_dt>>(attr_name)[0]);
    }

    dnnl::primitive_attr make_primitive_attr(
            std::shared_ptr<op_t> &op, fusion_info_mgr_t &mgr);
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
