/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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
#include "graph_memory.hpp"
#include "input_displacer.hpp"
#include "ref_primitive.hpp"
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

    // get the reference of ops of the partition
    const op_ref_list_t &get_partition_ops() const {
        return partition_ops_ref_;
    }

private:
    // check whether partition input ops support bf16-in-f32-out rewrite
    bool check_valid_bf16_in() const;

    // check whether partition output ops support bf16-in-f32-out rewrite
    bool check_valid_bf16_out() const;

    bool is_bf16_partition_support_f32_intermediate_result() const;

    // bf16 cases:use f32 as intermediate tensor dt to improve accuracy
    void handle_special_case_bf16(res_t *res);

    // rewrite x16->f32 from typecast (f32->x16) to typecast (x16->f32)
    void handle_typecast_x16();

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

    // reference primitives for a single partition
    std::unordered_map<size_t, ::std::shared_ptr<ref_primitive_t>> ref_prims_;

    // keep the memory for each logical tensor
    // before the execution of each reference primitive,
    // replace the args with the memory from this map.
    std::unordered_map<size_t, const dnn_mem_t &> lt_id_2_mems_;
    // keep the lt id for fake output which is not supported by primitive
    std::unordered_set<size_t> fake_lt_ids_;

    std::unordered_map<size_t, const deserialized_lt &> lt_id_2_lt_;
};

} // namespace graph
#endif
