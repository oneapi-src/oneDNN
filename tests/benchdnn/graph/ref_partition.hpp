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

    // run partition in ref path, one by one ref primitive
    void run(partition_mem_map_t &partition_mem_map, res_t *res);

    // ref execution and cmp
    void check_partition_correctness(
            partition_mem_map_t &partition_mem_map, res_t *res);

    // link previous primitive's args to current primitive
    void link_args(const deserialized_op &cur_op, res_t *res);

    // copy current primitive's args to previous primitive
    void reverse_link_args(const deserialized_op &cur_op,
            partition_mem_map_t &graph_mem_map, res_t *res);

    // get the reference of ops of the partition
    const op_ref_list_t &get_partition_ops() const {
        return partition_ops_ref_;
    }

protected:
    template <typename setting_t, typename prb_t, typename init_pd_func_t,
            typename supported_exec_args_func_t, typename setup_cmp_func_t>
    void handle_leading_op(const deserialized_op &cur_op,
            const init_pd_func_t &init_pd,
            const supported_exec_args_func_t &supported_exec_args,
            const setup_cmp_func_t &setup_cmp,
            const std::unordered_map<size_t, const std::string> &map_off_to_dt,
            partition_mem_map_t &graph_mem_map, const engine_t &ref_eng,
            res_t *res) {
        setting_t op_setting = get_setting<setting_t>(
                cur_op, bf16_to_f32_rewrite_lt_id_, res);
        if (res->state == INVALID_ARGUMENTS) return;

        auto pprb = std::make_shared<prb_t>(op_setting);
        prb_t *prb = pprb.get();

        set_prb_cfg<prb_t>(prb, map_off_to_dt, res);

        std::vector<size_t> par_in_and_leading_ids(
                partition_in_ids_.begin(), partition_in_ids_.end());
        for (const auto &lt : cur_op.in_lts_)
            par_in_and_leading_ids.emplace_back(lt.id_);

        init_prim<prb_t>(ref_prims_, cur_op, init_pd, supported_exec_args,
                setup_cmp, pprb, ref_eng, res);
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return;

        int op_id = static_cast<int>(cur_op.id_);
        auto &mems = std::get<1>(ref_prims_[op_id]);
        init_graph_memory_args(mems, graph_mem_map, par_in_and_leading_ids,
                partition_out_ids_, cur_op, true, res);
        if (res->state == FAILED) return;
    }

    template <typename setting_t, typename prb_t, typename init_pd_func_t,
            typename supported_exec_args_func_t, typename setup_cmp_func_t>
    void handle_op(const deserialized_op &cur_op, const init_pd_func_t &init_pd,
            const supported_exec_args_func_t &supported_exec_args,
            const setup_cmp_func_t &setup_cmp,
            partition_mem_map_t &graph_mem_map, const engine_t &ref_eng,
            res_t *res) {
        setting_t op_setting = get_setting<setting_t>(
                cur_op, bf16_to_f32_rewrite_lt_id_, res);
        if (res->state == INVALID_ARGUMENTS) return;

        auto pprb = std::make_shared<prb_t>(op_setting);
        prb_t *prb = pprb.get();

        init_prim<prb_t>(ref_prims_, cur_op, init_pd, supported_exec_args,
                setup_cmp, pprb, ref_eng, res);
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return;

        // prepare graph memory
        // 1. init tensors based on primitive memories
        // 2. maintain input and output tensors of the partition
        int op_id = static_cast<int>(cur_op.id_);
        auto &mems = std::get<1>(ref_prims_[op_id]);
        init_graph_memory_args(mems, graph_mem_map, partition_in_ids_,
                partition_out_ids_, cur_op, false, res);
        if (res->state == FAILED) return;

        if (cur_op.kind_ == "Dequantize" && is_quantized_) {
            // move leading driver primitive's input memory to the
            // current primitive input

            size_t cur_op_in_lt_id = cur_op.in_lts_.front().id_;
            bool is_partition_input
                    = std::find(partition_in_ids_.begin(),
                              partition_in_ids_.end(), cur_op_in_lt_id)
                    != partition_in_ids_.end();
            if (is_partition_input)
                // for patterns like conv -> dq -> q -> conv, no need to
                // implement link args reversing for dequantize that are
                // not inputs of the partition
                reverse_link_args(cur_op, graph_mem_map, res);
        }

        link_args(cur_op, res);
        execute_prim(ref_prims_, cur_op, prb, res);
    }

private:
    // check whether partition input ops support bf16-in-f32-out rewrite
    bool check_valid_bf16_in() const;

    // check whether partition output ops support bf16-in-f32-out rewrite
    bool check_valid_bf16_out() const;

    // find the partition leading op based on input lts
    bool get_leading_op_group(op_ref_list_t &leading_ops_group);

    void get_leading_op_input_offset_to_dt_map(
            const deserialized_op &leading_op,
            std::unordered_map<size_t, const std::string> &map_off_to_dt);

    bool is_bf16_partition_support_f32_intermediate_result() const;

    // bf16 cases:use f32 as intermediate tensor dt to improve accuracy
    void handle_special_case_bf16(res_t *res);

    // int8 cases:special processing of data filling to prevent accumulate
    // overflow
    void handle_special_case_int8(
            partition_mem_map_t &partition_mem_map, res_t *res);

    // Engine used to run correctness ref path for testing.
    const engine_t &get_ref_engine() const;

    // Objects below are constructed.
    // OPs in the partition, which is Topo ordered
    op_ref_list_t partition_ops_ref_;
    // map of input logical tensor id to its consumer ops
    std::unordered_map<size_t, op_ref_list_t> in_lt_2_ops_;
    // map of output logical tensor id to its producer op
    std::unordered_map<size_t, std::reference_wrapper<const deserialized_op>>
            out_lt_2_op_;
    // partition in logical tensors' ids
    std::vector<size_t> partition_in_ids_;
    // partition out logical tensors' ids
    std::vector<size_t> partition_out_ids_;

    // Objects below are modified at special bf16 and int8 cases.
    // IDs of logical tensors to replace bf16 data type with fp32.
    std::unordered_set<size_t> bf16_to_f32_rewrite_lt_id_;
    // is quantized partition
    bool is_quantized_ {false};

    // Objects below are modified during run()
    // reference primitives for a single partition
    ref_prims_t ref_prims_;
};

} // namespace graph
#endif
