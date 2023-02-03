/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

void get_combination(const std::vector<std::vector<int64_t>> &candidate,
        std::vector<std::vector<int64_t>> &combination) {
    if (candidate.size() == 1) {
        for (auto i : candidate[0]) {
            combination.push_back({i});
        }
    } else if (candidate.size() == 2) {
        for (auto i : candidate[0]) {
            for (auto j : candidate[1]) {
                combination.push_back({i, j});
            }
        }
    } else {
        BENCHDNN_PRINT(
                1, "Size %zd is out of limitations!\n", candidate.size());
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;
    deserialized_graph dg = prb->dg;
    bool has_dynamic_dim = dg.has_dynamic_dim;
    std::map<size_t, std::vector<std::vector<int64_t>>> real_shape_candidates
            = dg.dynamic_dims;

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
    // mapping from logical tensor id to output tensors
    // used to the connection relationship between partitions (e.g partition 0's
    // output tensor is fed into partition 1)
    std::unordered_map<size_t, tensor> global_outputs_ts_map;
    // manage the lifetime of memory buffers binded to those input/output tensors
    std::vector<std::shared_ptr<void>> data_buffers;
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

            if (has_dynamic_dim
                    && eng.get_kind() == dnnl::graph::engine::kind::cpu) {
                // when no candidate is given (e.g. no '--in-shapes' is given),
                // set dynamic dim value as 1
                if (real_shape_candidates.size() == 0) {
                    // for mlp --in-shapes:0:[1]x13
                    // for mha --in-shapes:0:[1]x16x[1]x64
                    for (const auto &in : inputs) {
                        std::vector<std::vector<int64_t>> default_one;
                        auto exec_dims = in.get_dims();
                        for (auto dim : exec_dims) {
                            if (dim == -2) { default_one.push_back({1}); }
                        }
                        real_shape_candidates.emplace(in.get_id(), default_one);
                        default_one.clear();
                    }
                }

                // NOTE: limitation#1: every id have the same numbers of dynamic dims:
                // 0:[2]x[64-128]x1024, 3:[2]x[64-128]x1024, 13:[2]x1x1x[64-128], 6:[2]x[64-128]x1024
                // all have two dynamic dims: {{2}, {64,128}}
                // get real case dynamic dims combination {{2,64}, {2,128}}
                std::vector<std::vector<int64_t>> combination;
                get_combination(real_shape_candidates[0], combination);

                // group all id with dynamic dims
                std::vector<size_t> dynamic_ids;
                for (auto itr = real_shape_candidates.begin();
                        itr != real_shape_candidates.end(); ++itr) {
                    dynamic_ids.push_back(itr->first);
                }

                for (const auto &combi : combination) {
                    // create tensors_in & tensors_out
                    std::vector<tensor> input_ts, output_ts;

                    input_ts.reserve(inputs.size());
                    output_ts.reserve(outputs.size());
                    for (const auto &in : inputs) {
                        size_t id = in.get_id();
                        auto exec_dims = in.get_dims();
                        int tag = -1;
                        for (auto &dim : exec_dims) {
                            if (dim == -2) { dim = combi[++tag]; }
                        }
                        logical_tensor exec_in {id, in.get_data_type(),
                                exec_dims, in.get_layout_type(),
                                in.get_property_type()};
                        if (exec_in.get_layout_type()
                                == logical_tensor::layout_type::any) {
                            BENCHDNN_PRINT(0,
                                    "Layout %d is unsupported for compiler "
                                    "backend!\n",
                                    1);
                            res->state = UNIMPLEMENTED;
                            return OK;
                        }
                        size_t mem_size = exec_in.get_mem_size();
                        // check if the input is an output of another partition
                        auto pos = global_outputs_ts_map.find(id);
                        if (pos != global_outputs_ts_map.end()) {
                            input_ts.push_back(pos->second);
                            continue;
                        }
                        // memory allocation
                        data_buffers.push_back({});
                        data_buffers.back().reset(
                                malloc(mem_size), cpu_deletor {});
                        input_ts.push_back(tensor {
                                exec_in, eng, data_buffers.back().get()});
                    }
                    tensors_in.emplace_back(input_ts);
                    // NOTE: imitation#2: how about the dynamic info for output,
                    // which can not be infered from '--in-shapes'
                    for (const auto &out : outputs) {
                        auto exec_dims = out.get_dims();
                        int tag = -1;
                        for (auto &dim : exec_dims) {
                            if (dim == -2) { dim = combi[++tag]; }
                        }
                        logical_tensor exec_out {out.get_id(),
                                out.get_data_type(), exec_dims,
                                out.get_layout_type(), out.get_property_type()};
                        size_t mem_size = exec_out.get_mem_size();
                        // memory allocate
                        data_buffers.push_back({});
                        data_buffers.back().reset(
                                malloc(mem_size), cpu_deletor {});
                        output_ts.push_back(tensor {
                                exec_out, eng, data_buffers.back().get()});
                        global_outputs_ts_map[exec_out.get_id()]
                                = output_ts.back();
                    }
                    tensors_out.emplace_back(output_ts);
                }
            } else {
                // static shape
                // Creating tensors and allocating memory buffer
                std::vector<tensor> input_ts
                        = tm.construct_and_initialize_tensors(
                                inputs, c_partitions.back(), eng, 128);
                std::vector<tensor> output_ts
                        = tm.construct_and_initialize_tensors(
                                outputs, c_partitions.back(), eng, 0);
                tensors_in.emplace_back(input_ts);
                tensors_out.emplace_back(output_ts);
            }
        } else {
            BENCHDNN_PRINT(1, "Partition %zd is unsupported!\n", i);
            res->state = UNIMPLEMENTED;
            return OK;
        }
    }

    // for dynamic shape, compiled_partition should make copies for all real shape
    // candidates
    size_t real_inputs = tensors_in.size();
    size_t cp_size = c_partitions.size();

    assert(real_inputs == tensors_out.size());

    if (has_dynamic_dim && eng.get_kind() == dnnl::graph::engine::kind::cpu) {
        if (cp_size < real_inputs) {
            for (size_t i = 0; i < real_inputs - cp_size; ++i) {
                c_partitions.push_back(c_partitions.back());
            }
        }
    }

    if (is_bench_mode(INIT)) return res->state = INITIALIZED, OK;

    if (is_bench_mode(PERF)) {
        SAFE(measure_perf(res->timer_map.perf_timer(), c_partitions, tensors_in,
                     tensors_out, res),
                WARN);
    }
    return OK;
}
} // namespace graph
