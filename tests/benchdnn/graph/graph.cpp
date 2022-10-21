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

#include "deserialize.hpp"
#include "dnnl_graph_common.hpp"
#include "execution_context.hpp"
#include "graph.hpp"
#include "helpers_any_layout.hpp"

namespace graph {

using namespace benchdnnext;

std::string case_to_str(const std::string json_file,
        const std::map<size_t, std::string> in_shapes,
        const std::map<size_t, std::string> op_attrs, const int64_t mb) {
    std::stringstream s;
    dump_global_params(s);
    std::string temp_s = "";

    if (mb != 0) {
        s << "--mb=" << mb;
        s << " ";
    }

    if (!(in_shapes.size() == 1 && in_shapes.count(0)
                && in_shapes.at(0) == "default")) {
        s << "--in-shapes=";
        temp_s.clear();
        for (auto &in_shape : in_shapes) {
            temp_s += (std::to_string(in_shape.first) + ":" + in_shape.second
                    + "+");
        }
        s << temp_s.substr(0, temp_s.length() - 1);
        s << " ";
    }

    if (!(op_attrs.size() == 1 && op_attrs.count(0)
                && op_attrs.at(0) == "default")) {
        s << "--op-attrs=";
        temp_s.clear();
        for (auto &op_attr : op_attrs) {
            temp_s += (std::to_string(op_attr.first) + ":" + op_attr.second
                    + "+");
        }
        s << temp_s.substr(0, temp_s.length() - 1);
        s << " ";
    }

    s << "--case=" << json_file;
    return s.str();
}

int doit(const prb_t *prb, res_t *res) {
    deserialized_graph dg = prb->dg;
    auto ograph = dg.to_graph(prb->fpmath_mode);
    const auto partitions = ograph.get_partitions();
    if (partitions.empty()) {
        BENCHDNN_PRINT(0, "FAIL: partition empty %d.\n", 0);
        return res->state = FAILED, FAIL;
    }
    BENCHDNN_PRINT(1, "Partition size %zd.\n", partitions.size());

    for (size_t i = 0; i < partitions.size(); ++i) {
        if (partitions[i].is_supported()) {
            std::vector<logical_tensor> in_out_lts
                    = partitions[i].get_in_ports();
            std::vector<logical_tensor> outputs = partitions[i].get_out_ports();
            in_out_lts.insert(in_out_lts.end(), outputs.begin(), outputs.end());
            // Get partition direction from op's kind which used for skipping unsupported cases.
            // For now, partition contains only one bwd op, so the direction can be inferred directly,
            // need to update the logic if library support fwd+bwd pattern later.
            dir_t dir = FLAG_FWD;
            const auto &op_ids = partitions[i].get_ops();
            for (const auto &aop : dg.ops_) {
                if (std::count(op_ids.begin(), op_ids.end(), aop.id_) > 0) {
                    if (aop.kind_.rfind("Backprop") != std::string::npos) {
                        dir = FLAG_BWD;
                        break;
                    }
                }
            }
            skip_unimplemented_data_type(in_out_lts, dir, res);
        }
    }

    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    engine eng = benchdnnext::get_test_engine();

    /// mark the output logical tensors of partition as ANY layout enabled
    std::unordered_set<size_t> id_to_set_any_layout;
    std::vector<compiled_partition> c_partitions;
    std::vector<std::vector<dnnl::graph::tensor>> tensors_in, tensors_out;

    // mapping from id to tensors
    tensor_map tm;

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    for (size_t i = 0; i < partitions.size(); ++i) {
        if (partitions[i].is_supported()) {
            std::vector<logical_tensor> inputs = partitions[i].get_in_ports();
            std::vector<logical_tensor> outputs = partitions[i].get_out_ports();

            /// replace input logical tensor with the queried one
            replace_with_queried_logical_tensors(
                    inputs, id_to_queried_logical_tensors);

            /// update output logical tensors with ANY layout
            update_tensors_with_any_layout(outputs, id_to_set_any_layout);

            /// compile to generate compiled partition
            c_partitions.emplace_back(
                    partitions[i].compile(inputs, outputs, eng));

            record_queried_logical_tensors(partitions[i].get_out_ports(),
                    c_partitions.back(), id_to_queried_logical_tensors);

            // Creating tensors and allocating memory buffer
            std::vector<tensor> input_ts = tm.construct_and_initialize_tensors(
                    inputs, c_partitions.back(), eng, 128);
            std::vector<tensor> output_ts = tm.construct_and_initialize_tensors(
                    outputs, c_partitions.back(), eng, 0);
            tensors_in.emplace_back(input_ts);
            tensors_out.emplace_back(output_ts);
        } else {
            BENCHDNN_PRINT(1, "Partition %zd is unsupported!\n", i);
            res->state = UNIMPLEMENTED;
            return OK;
        }
    }

    if (is_bench_mode(PERF)) {
        SAFE(measure_perf(res->timer_map.perf_timer(), c_partitions, tensors_in,
                     tensors_out, res),
                WARN);
    }
    return OK;
}
} // namespace graph
