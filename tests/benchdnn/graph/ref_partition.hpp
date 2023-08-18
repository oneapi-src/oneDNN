/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_REF_PARTITION_HPP
#define BENCHDNN_GRAPH_REF_PARTITION_HPP

#include <list>
#include "deserialize.hpp"
#include "dnnl_common.hpp"
#include "graph_bridge.hpp"
#include "input_displacer.hpp"
#include <unordered_set>

namespace graph {

class ref_partition_t {
public:
    ref_partition_t() = default;
    // to get a Topo ordered partition OPs reference and save the map
    // of input/output logical tensors ids to partition OPs reference
    ref_partition_t(const deserialized_graph &dg,
            const dnnl::graph::partition &par,
            const std::vector<dnnl::graph::logical_tensor> &ins,
            const std::vector<dnnl::graph::logical_tensor> &outs);

    // prepare memorise in both paths, one by one ref primitive
    void init_ref(const bench_mode_t mode,
            const std::vector<size_t> &graph_ports,
            partition_mem_map_t &partition_mem_map, res_t *res);
    // run partition in ref path, one by one ref primitive
    void exec_ops(res_t *res);

    // ref execution and cmp
    void check_partition_correctness(
            partition_mem_map_t &partition_mem_map, res_t *res);

    // link previous primitive's args to current primitive
    void link_args(const deserialized_op &cur_op, res_t *res);

    // get the reference of ops of the partition
    const op_ref_list_t &get_partition_ops() const {
        return partition_ops_ref_;
    }

protected:
    template <typename setting_t, typename prb_t, typename init_pd_func_t,
            typename supported_exec_args_func_t, typename setup_cmp_func_t>
    void init_op(const deserialized_op &cur_op, const init_pd_func_t &init_pd,
            const supported_exec_args_func_t &supported_exec_args,
            const setup_cmp_func_t &setup_cmp,
            partition_mem_map_t &graph_mem_map, const engine_t &ref_eng,
            res_t *res) {
        setting_t op_setting = get_setting<setting_t>(
                cur_op, bf16_to_f32_rewrite_lt_id_, res);
        if (res->state == INVALID_ARGUMENTS) return;

        auto pprb = std::make_shared<prb_t>(op_setting);

        // Memory Preparation Flow:
        // 1. Generate memory and data from primitive
        // 2. Replace the data filling if needed
        // 3. Replace memory from other OP in link_args
        //      a. If the memory input is an output of other OP
        //      b. If the memory input is a shared partition input with other OP
        // 4. The data for execution is finalized and generate graph memory args.

        init_prim<prb_t>(ref_prims_, cur_op, init_pd, supported_exec_args,
                setup_cmp, pprb, ref_eng, res);
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return;

        // prepare graph memory
        // 1. init tensors based on primitive memories
        // 2. maintain input and output tensors of the partition
        int op_id = static_cast<int>(cur_op.id_);
        auto &mems = std::get<1>(ref_prims_[op_id]);

        // displace the input tensor
        for (size_t i = 0; i < cur_op.in_lts_.size(); i++) {
            const auto &in = cur_op.in_lts_[i];
            int arg = get_prim_arg_name_from_graph_op_input_offset(
                    opstr2kind(cur_op.kind_), static_cast<int>(i));
            auto &mem = const_cast<dnn_mem_t &>(
                    ::std::get<3>(ref_prims_[op_id]).find(arg));
            data_displacer.displace_input_data(in.id_, mem, res);
            if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return;
        }

        link_args(cur_op, res);
        init_graph_memory_args(mems, graph_mem_map, partition_in_ids_,
                partition_out_ids_, cur_op, res);
    }

    template <typename prb_t>
    void exec_op(const deserialized_op &cur_op, const prb_t *prb, res_t *res) {
        execute_prim(ref_prims_, cur_op, prb, res);
    }

private:
    // check whether partition input ops support bf16-in-f32-out rewrite
    bool check_valid_bf16_in() const;

    // check whether partition output ops support bf16-in-f32-out rewrite
    bool check_valid_bf16_out() const;

    bool is_bf16_partition_support_f32_intermediate_result() const;

    // bf16 cases:use f32 as intermediate tensor dt to improve accuracy
    void handle_special_case_bf16(res_t *res);

    // Objects below are constructed.
    // OPs in the partition, which is Topo ordered
    op_ref_list_t partition_ops_ref_;
    // map of input logical tensor id to its consumer ops
    std::unordered_map<size_t, op_ref_list_t> in_lt_2_ops_;
    // map of output logical tensor id to its producer op
    std::unordered_map<size_t, std::reference_wrapper<const deserialized_op>>
            out_lt_2_op_;
    ::graph::partition_data_displacer_t data_displacer;
    // partition in logical tensors' ids
    std::vector<size_t> partition_in_ids_;
    // partition out logical tensors' ids
    std::vector<size_t> partition_out_ids_;
    // Objects below are modified at special bf16 and int8 cases.
    // IDs of logical tensors to replace bf16 data type with fp32.
    std::unordered_set<size_t> bf16_to_f32_rewrite_lt_id_;

    // Objects below are modified during run()
    // reference primitives for a single partition
    ref_prims_t ref_prims_;
};

} // namespace graph
#endif
