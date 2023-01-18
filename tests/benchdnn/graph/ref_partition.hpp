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
    void run(std::vector<dnnl::graph::tensor> &input_ts,
            std::vector<dnnl::graph::tensor> &output_ts, res_t *res);

    // fill primitive memory data to partition input/output tensors,
    // so that the execution results of ref_partition and partition can be
    // comparable
    void init_graph_mem(std::vector<dnnl::graph::tensor> &input_ts,
            std::vector<dnnl::graph::tensor> &output_ts,
            const deserialized_op &cur_op, res_t *res);

    // ref execution and cmp
    void check_partition_correctness(res_t *res);

    // link previous primitive's args to current primitive
    void link_args(const deserialized_op &cur_op, res_t *res);

    // copy current primitive's args to previous primitive
    void reverse_link_args(const deserialized_op &cur_op, res_t *res);

protected:
    template <typename setting_t, typename prb_t, typename init_pd_func_t,
            typename supported_exec_args_func_t, typename setup_cmp_func_t>
    void handle_leading_op(const deserialized_op &cur_op,
            const init_pd_func_t &init_pd,
            const supported_exec_args_func_t &supported_exec_args,
            const setup_cmp_func_t &setup_cmp,
            const std::unordered_map<size_t, const std::string> &map_off_to_dt,
            const engine_t &ref_eng, res_t *res) {
        setting_t op_setting = get_setting<setting_t>(
                cur_op, bf16_to_f32_rewrite_lt_id_, res);
        if (res->state == INVALID_ARGUMENTS) return;

        prb_t prb_(op_setting), *prb = &prb_;

        set_prb_cfg<prb_t>(prb, map_off_to_dt, res);
        init_prim<prb_t>(ref_prims_, cur_op, init_pd, supported_exec_args,
                setup_cmp, prb, ref_eng, res);
    }

    template <typename setting_t, typename prb_t, typename init_pd_func_t,
            typename supported_exec_args_func_t, typename setup_cmp_func_t>
    void handle_op(const deserialized_op &cur_op, const init_pd_func_t &init_pd,
            const supported_exec_args_func_t &supported_exec_args,
            const setup_cmp_func_t &setup_cmp,
            std::vector<dnnl::graph::tensor> &input_ts,
            std::vector<dnnl::graph::tensor> &output_ts,
            const engine_t &ref_eng, res_t *res) {
        setting_t op_setting = get_setting<setting_t>(
                cur_op, bf16_to_f32_rewrite_lt_id_, res);
        if (res->state == INVALID_ARGUMENTS) return;

        prb_t prb_(op_setting), *prb = &prb_;

        init_prim<prb_t>(ref_prims_, cur_op, init_pd, supported_exec_args,
                setup_cmp, prb, ref_eng, res);
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return;

        if (cur_op.kind_ == "Dequantize" && is_quantized_) {
            // move leading driver primitive's input memory to the current
            // primitive input
            reverse_link_args(cur_op, res);
        }

        link_args(cur_op, res);
        init_graph_mem(input_ts, output_ts, cur_op, res);
        execute_prim(ref_prims_, cur_op, prb, res);
    }

private:
    // check whether partition input ops support bf16-in-f32-out rewrite
    bool check_valid_bf16_in() const;

    // check whether partition output ops support bf16-in-f32-out rewrite
    bool check_valid_bf16_out() const;

    // find the partition leading op based on input lts
    bool get_leading_op_group(
            std::list<std::reference_wrapper<const deserialized_op>>
                    &leading_ops_group);

    void get_leading_op_input_offset_to_dt_map(
            const deserialized_op &leading_op,
            std::unordered_map<size_t, const std::string> &map_off_to_dt);

    bool is_bf16_partition_support_f32_intermediate_result() const;

    // bf16 cases:use f32 as intermediate tensor dt to improve accuracy
    void handle_special_case_bf16(res_t *res);

    // int8 cases:special processing of data filling to prevent accumulate
    // overflow
    void handle_special_case_int8(res_t *res);

    // Engine used to run correctness ref path for testing.
    const engine_t &get_ref_engine() const;

    // Objects below are constructed.
    // OPs in the partition, which is Topo ordered
    std::list<std::reference_wrapper<const deserialized_op>> partition_ops_ref_;
    // map of input logical tensor id to its consumer ops
    std::unordered_map<size_t,
            std::list<std::reference_wrapper<const deserialized_op>>>
            in_lt_2_ops_;
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
    std::unordered_map<int, graph_link_t> ref_prims_;
    // check wether has eltwise op to use relaxed comparison
    bool has_eltwise_ {false};
    // check whether allow nan/inf of output
    bool output_has_nans_ {false};

    // partition output tensors wrapped in args_t
    // used for later correctness check
    args_t partition_output_args_;

    // save the copied primitive mems here to avoid early free
    std::unordered_map<size_t, std::shared_ptr<dnn_mem_t>>
            partition_in_out_mems_;
};

} // namespace graph
#endif
