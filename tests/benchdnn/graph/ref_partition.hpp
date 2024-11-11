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
#include <unordered_set>

#include "dnnl_common.hpp"

#include "deserialize.hpp"
#include "graph_memory.hpp"
#include "input_displacer.hpp"
#include "ref_primitive.hpp"

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

    // prepare memories in both paths, one by one ref primitive
    int init_ref(const std::vector<size_t> &graph_ports,
            partition_mem_map_t &partition_mem_map, res_t *res);
    // run partition in ref path, one by one ref primitive
    void exec_ops(res_t *res);

    // ref execution and cmp
    int check_partition_correctness(
            partition_mem_map_t &partition_mem_map, res_t *res);

    // check the partition memory footprint of graph path
    int check_partition_total_size(const deserialized_op &op, res_t *res);

    // check the partition memory footprint of reference path
    int check_partition_total_size(
            const check_mem_size_args_t &check_mem_size_args, bool is_output_op,
            res_t *res);

    // get the reference of ops of the partition
    const op_ref_list_t &get_partition_ops() const {
        return partition_ops_ref_;
    }

private:
    // Returns `true` if an `op` has a parent op in the partition for any of
    // its logical tensors.
    // When `check_all_in_lts` is set to true, returns `true` if only the op has
    // a parent for each of its logical tensors.
    bool has_parent_op(const deserialized_op &op, bool check_all_in_lts) const;

    // Returns `true` if an `op` has a child op in the partition.
    // If `child_op_ptr` is not empty, updates the pointer with a child op.
    //
    // Note: double pointer is needed to initialize a pointer. A pointer is
    // needed to avoid a copy of an `child_op` object.
    bool has_child_op(const deserialized_op &op,
            const deserialized_op **child_op_ptr) const;

    // Returns a pointer to parent op for a given input lt id. If the parent is
    // not found, an empty pointer is returned.
    const deserialized_op *get_parent_op(size_t in_lt_id) const;

    // Returns `true` if unfusable transcendental op should have cropped output.
    // `dt` is a target data type for following transform. Updated only when the
    // function returns `true`.
    bool need_unfusable_output_crop(
            const deserialized_op &op, dnnl_data_type_t &dt) const;

    bool is_input_op(const deserialized_op &op) const;
    bool is_output_op(const deserialized_op &op) const;
    std::vector<size_t> get_in_out_lt_ids(const deserialized_op &op) const;

    const deserialized_graph *dg_;
    // Objects below are constructed.
    // OPs in the partition, which is Topo ordered
    op_ref_list_t partition_ops_ref_;
    // map of input logical tensor id to its consumer ops
    std::unordered_map<size_t, op_ref_list_t> in_lt_2_ops_;
    // map of output logical tensor id to its producer op
    std::unordered_map<size_t, op_ref_t> out_lt_2_op_;
    ::graph::partition_data_displacer_t data_displacer;
    // partition in logical tensors' ids
    std::vector<size_t> partition_in_ids_;
    // partition out logical tensors' ids
    std::vector<size_t> partition_out_ids_;

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
