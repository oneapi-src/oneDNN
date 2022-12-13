/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <unordered_map>

#include "dnnl_common.hpp"
#include "execution_context.hpp"
#include "graph.hpp"

namespace graph {

using namespace dnnl::graph;

std::string case_to_str(const std::string &json_file,
        const std::map<size_t, std::string> &in_shapes,
        const std::map<size_t, std::string> &op_attrs, const int64_t mb) {
    std::stringstream s;
    dump_global_params(s);

    if (mb != 0) { s << "--mb=" << mb << " "; }

    if (!(in_shapes.size() == 1 && in_shapes.count(0)
                && in_shapes.at(0) == "default")) {
        s << "--in-shapes=";
        std::string tmp;
        for (const auto &in_shape : in_shapes) {
            tmp += (std::to_string(in_shape.first) + ":" + in_shape.second
                    + "+");
        }
        s << tmp.substr(0, tmp.length() - 1);
        s << " ";
    }

    if (!(op_attrs.size() == 1 && op_attrs.count(0)
                && op_attrs.at(0) == "default")) {
        s << "--op-attrs=";
        std::string tmp;
        for (const auto &op_attr : op_attrs) {
            tmp += (std::to_string(op_attr.first) + ":" + op_attr.second + "+");
        }
        s << tmp.substr(0, tmp.length() - 1);
        s << " ";
    }

    s << "--case=" << json_file;
    return s.str();
}

/// Set any layout according to the connection relationship of partitions
///
/// @param partitions a list of partitions
/// @param id_to_set_any_layout a set of ids of logical tensors with any layout
///     type
void set_any_layout(const std::vector<dnnl::graph::partition> &partitions,
        std::unordered_set<size_t> &id_to_set_any_layout) {
    // mapping from output tensor id to the all supported flags of
    // supported partitions, we may only need outputs' supported flags
    std::unordered_map<size_t, std::vector<bool>> output_to_flag_map;
    for (const auto &p : partitions) {
        for (const auto &out : p.get_output_ports()) {
            size_t id = out.get_id();
            if (p.is_supported()
                    && output_to_flag_map.find(id)
                            == output_to_flag_map.end()) {
                output_to_flag_map[id] = {};
            }
        }

        for (const auto &in : p.get_input_ports()) {
            size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            if (iter != output_to_flag_map.end()) {
                // collect all of supported flags of this tensor's uses
                // Considering we have such a graph:
                //
                //   partition_A  partition_B
                //        \           |
                //      tensor1    tensor2
                //           \     /     |
                //         partition_C  unsuppported partition
                //              |
                //           tensor3
                //
                // so the mapping of partition_A's output will be { true }
                // the mapping of partition_B's output will be { true, false }
                // The mapping of partition_C's output will be { false }
                // Only when all supported flags are true, users can set any
                // layout.
                iter->second.push_back(p.is_supported());
            }
        }
    }

    for (const auto &p : partitions) {
        // no need to set `any` layout if this partition is not supported
        if (!p.is_supported()) continue;
        for (const auto &in : p.get_input_ports()) {
            size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            // if this input tensor is not an output of another supported
            // partition, just skip
            if (iter == output_to_flag_map.end()) continue;
            const auto &flag_vec = iter->second;
            // check if all of uses of this tensor are supported partitions,
            // if not, no need to set ANY layout.
            bool need_set_any = std::all_of(
                    flag_vec.begin(), flag_vec.end(), [](bool a) { return a; });
            if (!need_set_any) continue;

            // record the id of logical tensor that will be set to ANY layout
            id_to_set_any_layout.insert(id);
        }
    }
}

/// Update tensors with ANY layout
///
/// @param lts a list of logical tensors
/// @param id_to_set_any_layout a set of ids of logical tensors with any layout
///     type
void update_tensors_with_any_layout(
        std::vector<dnnl::graph::logical_tensor> &lts,
        const std::unordered_set<size_t> &id_to_set_any_layout) {
    for (auto &lt : lts) {
        auto id = lt.get_id();
        if (id_to_set_any_layout.find(id) == id_to_set_any_layout.end())
            continue;

        const auto &ori_dims = lt.get_dims();
        const auto ori_dtype = lt.get_data_type();
        // update old logical tensor with ANY layout
        lt = dnnl::graph::logical_tensor(id, ori_dtype, ori_dims,
                dnnl::graph::logical_tensor::layout_type::any);
    }
}

/// Replace original logical tensors with queried logical tensors
///
/// @param lts a list of logical tensors to be updated
/// @param id_to_queried_logical_tensors a mapping from (logical tensor) id to
///     the corresponding logical tensor queried from a compiled partition
void replace_with_queried_logical_tensors(
        std::vector<dnnl::graph::logical_tensor> &lts,
        const std::unordered_map<size_t, dnnl::graph::logical_tensor>
                &id_to_queried_logical_tensors) {
    for (auto &lt : lts) {
        auto id = lt.get_id();
        auto iter = id_to_queried_logical_tensors.find(id);
        if (iter != id_to_queried_logical_tensors.end()) lt = iter->second;
    }
}

/// Record queried logical tensor in a map
///
/// @param lts a list of logical tensors used to provide ids
/// @param c_partition target compiled partition
/// @param id_to_queried_logical_tensors a map to store the mapping from
///     (logical tensor) id to the corresponding logical tensor queried from
///     target compiled partition
void record_queried_logical_tensors(
        const std::vector<dnnl::graph::logical_tensor> &lts,
        const dnnl::graph::compiled_partition &c_partition,
        std::unordered_map<size_t, dnnl::graph::logical_tensor>
                &id_to_queried_logical_tensors) {
    for (const auto &lt : lts) {
        auto id = lt.get_id();
        id_to_queried_logical_tensors[id]
                = c_partition.query_logical_tensor(id);
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;
    const auto &dg = prb->dg;
    auto ograph = dg.to_graph(prb->fpmath_mode);
    DNN_GRAPH_SAFE(ograph.finalize(), WARN);
    const auto partitions = ograph.get_partitions();
    if (partitions.empty()) {
        BENCHDNN_PRINT(0, "FAIL: partition empty %d.\n", 0);
        return res->state = FAILED, FAIL;
    }
    BENCHDNN_PRINT(1, "Partition size %zd.\n", partitions.size());

    for (size_t i = 0; i < partitions.size(); ++i) {
        if (!partitions[i].is_supported()) {
            BENCHDNN_PRINT(1, "Partition %zd is unsupported!\n", i);
            res->state = UNIMPLEMENTED;
            return OK;
        }

        auto in_out_lts = partitions[i].get_input_ports();
        const auto &outputs = partitions[i].get_output_ports();
        in_out_lts.insert(in_out_lts.end(), outputs.begin(), outputs.end());
        std::vector<dnnl_data_type_t> in_out_dt;
        for (const auto &lt : in_out_lts) {
            switch (lt.get_data_type()) {
                case logical_tensor::data_type::bf16:
                    in_out_dt.emplace_back(dnnl_bf16);
                    break;
                case logical_tensor::data_type::f16:
                    in_out_dt.emplace_back(dnnl_f16);
                    break;
                default: break;
            }
        }
        // Get partition direction from op's kind which used for skipping
        // unsupported cases.
        dir_t dir = FLAG_FWD;
        const auto &op_ids = partitions[i].get_ops();
        for (const auto &aop : dg.ops_) {
            if (std::count(op_ids.begin(), op_ids.end(), aop.id_)) {
                if (aop.kind_.find("Backward") != std::string::npos) {
                    dir = FLAG_BWD;
                    break;
                }
            }
        }
        skip_unimplemented_data_type(in_out_dt, dir, res);
    }

    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const auto &eng = get_test_engine();

    // mark the output logical tensors of partition as ANY layout enabled
    std::unordered_set<size_t> id_to_set_any_layout;
    std::vector<compiled_partition> c_partitions;
    std::vector<std::vector<tensor>> tensors_in, tensors_out;

    // mapping from id to tensors
    tensor_map tm;

    // mapping from id to queried logical tensor from compiled partition used to
    // record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    // mark partition outputs id to set as ANY layout
    set_any_layout(partitions, id_to_set_any_layout);

    for (size_t i = 0; i < partitions.size(); ++i) {
        auto inputs = partitions[i].get_input_ports();
        auto outputs = partitions[i].get_output_ports();

        // replace input logical tensor with the queried one
        replace_with_queried_logical_tensors(
                inputs, id_to_queried_logical_tensors);

        // update output logical tensors with ANY layout
        update_tensors_with_any_layout(outputs, id_to_set_any_layout);

        // compile to generate compiled partition
        DNN_GRAPH_SAFE(c_partitions.emplace_back(
                               partitions[i].compile(inputs, outputs, eng)),
                CRIT);

        record_queried_logical_tensors(
                outputs, c_partitions.back(), id_to_queried_logical_tensors);

        // Creating tensors and allocating memory buffer
        auto input_ts = tm.construct_and_initialize_tensors(
                inputs, c_partitions.back(), eng, 128);
        auto output_ts = tm.construct_and_initialize_tensors(
                outputs, c_partitions.back(), eng, 0);
        tensors_in.emplace_back(input_ts);
        tensors_out.emplace_back(output_ts);
    }

    SAFE(execute_and_wait(c_partitions, tensors_in, tensors_out, res), WARN);

    if (is_bench_mode(PERF)) {
        SAFE(measure_perf(res->timer_map.perf_timer(), c_partitions, tensors_in,
                     tensors_out, res),
                WARN);
    }
    return OK;
}
} // namespace graph
