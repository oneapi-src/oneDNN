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

#include "dnnl_common.hpp"
#include "graph.hpp"
#include "ref_partition.hpp"

namespace {

/// Set any layout according to the connection relationship of partitions
/// @param dg a deserialized graph
/// @param partitions a list of partitions
/// @param id_to_set_any_layout a set of ids of logical tensors with any layout
///     type
void set_any_layout(const graph::deserialized_graph &dg,
        const std::vector<dnnl::graph::partition> &partitions,
        std::unordered_set<size_t> &id_to_set_any_layout) {
    // mapping from output tensor id to the all supported flags of
    // supported partitions, we may only need outputs' supported flags
    std::unordered_map<size_t, std::vector<bool>> output_to_flag_map;
    // record in & out of all Reoder ops in the current graph
    std::unordered_set<size_t> reorder_in_out_ids;

    for (const auto &aop : dg.ops_) {
        if (aop.kind_ == "Reorder") {
            // reorder only has one input and one output
            reorder_in_out_ids.emplace(aop.out_lts_.front().id_);
            reorder_in_out_ids.emplace(aop.in_lts_.front().id_);
        }
    }

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

            // if current id is not a input of Reorder or a output of Reorder
            // record the id of logical tensor that will be set to ANY layout
            auto iter_find = reorder_in_out_ids.find(id);
            if (iter_find == reorder_in_out_ids.end()) {
                id_to_set_any_layout.insert(id);
            }
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

/// Find the logical tensor and op with given logical tensor id and op list
///
/// @param lt_id an id of the logical tensor to be found
/// @param ops a list of ops of the partition
/// @param aop a deserialized op to be updated
/// @param alt a deserialized logical tensor to be updated
/// @param is_input a boolean flag to indicate to search input or output lts
int find_logical_tensor(size_t lt_id, const graph::op_ref_list_t &ops,
        graph::deserialized_op &aop, graph::deserialized_lt &alt,
        const bool is_input) {

    for (const auto &op : ops) {
        const auto &lts = is_input ? op.get().in_lts_ : op.get().out_lts_;
        for (const auto &op_lt : lts) {
            if (op_lt.id_ == lt_id) {
                aop = op;
                alt = op_lt;
                return OK;
            }
        }
    }
    return FAIL;
}

/// map graph memories to device before primitive execution or unmap graph
/// memories back to host after primitive execution
///
/// @param partition_mem_map a mapping from logical tensor id to graph memory
/// @param lts a vector of logical tensors
/// @param map_flag a flag to indicate whether to do mapping or unmapping
/// @param res a res_t struct that records the result
int map_unmap_partition_mem(graph::partition_mem_map_t &partition_mem_map,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        const int &map_flag, res_t *res) {
    // In case one logical tensor is used for multiple inputs, record the
    // processed logical tensor ids to avoid duplicate processing
    std::unordered_set<size_t> processed_ids;
    for (const auto &lt : lts) {
        const auto &lt_id = lt.get_id();
        if (processed_ids.find(lt_id) != processed_ids.end()) continue;

        const auto iter = partition_mem_map.find(lt_id);
        if (iter == partition_mem_map.end()) {
            BENCHDNN_PRINT(0,
                    "FAIL: Cannot find graph memory with lt id %zu! \n", lt_id);
            return res->state = FAILED, FAIL;
        }
        auto &graph_mem = iter->second;
        if (map_flag == MAP)
            graph_mem.map_mem(); // Map graph memory to host
        else if (map_flag == UNMAP)
            graph_mem.unmap_mem(); // Unmap graph memory from host
        else
            return res->state = UNIMPLEMENTED, FAIL;

        processed_ids.insert(lt_id);
    }

    return OK;
}

/// Get input tensors for the partition
///
/// @param input_ts a vector of input tensors
/// @param partition_mem_map a mapping from logical tensor id to graph memory
/// of the partition
/// @param ops a list of op references of the partition
/// @param ins a vector of logical tensors of partition inputs
int make_input_tensors(std::vector<dnnl::graph::tensor> &input_ts,
        const graph::partition_mem_map_t &partition_mem_map,
        const graph::op_ref_list_t &ops,
        const std::vector<dnnl::graph::logical_tensor> &ins) {
    for (size_t idx = 0; idx < ins.size(); ++idx) {
        // find the op id of the input logical tensor
        const auto &in = ins[idx];
        const auto &lt_id = in.get_id();
        graph::deserialized_lt lt;
        graph::deserialized_op op;
        if (find_logical_tensor(lt_id, ops, op, lt, true) != OK) {
            BENCHDNN_PRINT(0,
                    "FAIL: Cannot find logical tensor with id %zu! \n", lt_id);
            return FAIL;
        }

        // generate tensor for graph path

        const auto iter = partition_mem_map.find(lt_id);
        if (iter != partition_mem_map.end()) {
            const auto &graph_mem = iter->second;
            input_ts[idx] = graph_mem.make_graph_tensor(lt);
        } else {
            BENCHDNN_PRINT(0,
                    "FAIL: Cannot find graph memory with lt id %zu! \n", lt_id);
            return FAIL;
        }
    }
    return OK;
}

/// Get output tensors for the partition
///
/// @param output_ts a vector of output tensors
/// @param partition_mem_map a mapping from logical tensor id to graph memory
/// of the partition
/// @param ops a list of op references of the partition
/// @param outs a vector of logical tensors of partition outputs
int make_output_tensors(std::vector<dnnl::graph::tensor> &output_ts,
        const graph::partition_mem_map_t &partition_mem_map,
        const graph::op_ref_list_t &ops,
        const std::vector<dnnl::graph::logical_tensor> &outs,
        const std::vector<std::pair<size_t, size_t>> &inplace_ports) {

    for (size_t idx = 0; idx < outs.size(); ++idx) {
        // find the op id of the output logical tensor
        const auto &out = outs[idx];
        const auto &lt_id = out.get_id();
        graph::deserialized_op op;
        graph::deserialized_lt lt;
        if (find_logical_tensor(lt_id, ops, op, lt, false) != OK) {
            BENCHDNN_PRINT(0,
                    "FAIL: Cannot find logical tensor with id %zu! \n", lt_id);
            return FAIL;
        }

        // generate tensor for graph path
        const auto iter = partition_mem_map.find(lt_id);
        if (iter == partition_mem_map.end()) {
            BENCHDNN_PRINT(0,
                    "FAIL: Cannot find graph memory with lt id %zu! \n", lt_id);
            return FAIL;
        }
        const auto &graph_mem = iter->second;
        if (has_bench_mode_bit(mode_bit_t::corr)) {
            output_ts[idx] = graph_mem.make_graph_tensor(lt);
        } else {
            // For performance mode, we need special handling for graph
            // with in-place ports by using the graph memory of input
            // logical tensor to construct tensor. Meanwhile, for
            // correctness mode it's not needed as we only care about
            // the result correctness.
            auto pos = std::find_if(inplace_ports.begin(), inplace_ports.end(),
                    [lt_id](const std::pair<size_t, size_t> &p) {
                        return lt_id == p.second;
                    });
            if (pos != inplace_ports.end()) {
                const auto &inplace_lt_id = pos->first;
                const auto inplace_iter = partition_mem_map.find(inplace_lt_id);
                if (inplace_iter != partition_mem_map.end()) {
                    const auto &inplace_graph_mem = inplace_iter->second;
                    output_ts[idx] = inplace_graph_mem.make_graph_tensor(lt);
                } else {
                    BENCHDNN_PRINT(0,
                            "FAIL: Cannot find logical tensor with id %zu! "
                            "\n",
                            inplace_lt_id);
                    return FAIL;
                }

            } else {
                output_ts[idx] = graph_mem.make_graph_tensor(lt);
            }
        }
    }
    return OK;
}

} // namespace

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

/// @brief check if the current partition is actually an End op
/// @param parti the current partition
/// @param end_op_ids a collection of End op's ids
/// @return return true, when current partition is an End op
bool is_single_end_op_partition(const dnnl::graph::partition &parti,
        const std::vector<size_t> &end_op_ids) {
    const auto parti_op_ids = parti.get_ops();
    if (!end_op_ids.empty() && parti_op_ids.size() == 1
            && std::count(end_op_ids.begin(), end_op_ids.end(),
                    parti_op_ids.front())) {
        return true;
    }
    return false;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == bench_mode_t::list) return res->state = LISTED, OK;

    skip_start(res);
    if (res->state == SKIPPED) return OK;

    const auto &dg = prb->dg;
    const auto graph_in_ports = dg.get_input_ports();
    auto ograph = dg.to_graph(prb->fpmath_mode);
    DNN_GRAPH_SAFE(ograph.finalize(), WARN);
    const auto partitions = ograph.get_partitions();
    // a collection of End op's id in this graph
    std::vector<size_t> end_opid_v {};
    for (const auto &aop : dg.ops_) {
        if (aop.kind_ == "End") { end_opid_v.emplace_back(aop.id_); }
    }

    if (partitions.empty()) {
        BENCHDNN_PRINT(0, "FAIL: partition empty %d.\n", 0);
        return res->state = FAILED, FAIL;
    }
    BENCHDNN_PRINT(1, "Partition size %zd.\n", partitions.size());

    for (size_t i = 0; i < partitions.size(); ++i) {
        // Single end op partition is an unsupported partition in the library
        if (!partitions[i].is_supported()) {
            BENCHDNN_PRINT(1, "Partition %zd is unsupported!\n", i);
            if (is_single_end_op_partition(partitions[i], end_opid_v)) {
                BENCHDNN_PRINT(1, "Partition %zd is End op!\n", i);
                continue;
            }
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

    const auto &eng = get_graph_engine();
    cpp_stream_t strm {eng};

    // mark the output logical tensors of partition as ANY layout enabled
    std::unordered_set<size_t> id_to_set_any_layout;
    std::vector<compiled_partition> c_partitions;
    std::vector<std::vector<tensor>> input_ts_all, output_ts_all;
    // Extend the partition_mem_map_t's lifecycle as input_ts/output_ts hold the
    // same addresses as in partition_mem_map_t for perf mode
    // TODO: Once the API allocating memory when creating tensors is provided by
    // the Graph library, use a single partition_mem_map_t object, and move it
    // inside of the loop, perform tensor copy to input_ts/output_ts when
    // make_graph_tensor
    std::vector<partition_mem_map_t> partition_mem_map_v(partitions.size());

    // mapping from id to queried logical tensor from compiled partition used to
    // record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    // Mark partition outputs id to set as ANY layout. Used in perf mode only
    // to connect partitions in most optimized way avoiding extra reorder.
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        set_any_layout(dg, partitions, id_to_set_any_layout);
    }

    // the index offset for current partition compared with the previous partition index
    size_t idx_offset = 0;
    for (size_t i = 0; i < partitions.size(); ++i) {
        if (is_single_end_op_partition(partitions[i], end_opid_v)) {
            idx_offset += 1;
            continue;
        }

        auto inputs = partitions[i].get_input_ports();
        auto outputs = partitions[i].get_output_ports();

        // replace input logical tensor with the queried one
        replace_with_queried_logical_tensors(
                inputs, id_to_queried_logical_tensors);

        // Update output logical tensors with ANY layout. See `set_any_layout`
        // comment above.
        if (has_bench_mode_bit(mode_bit_t::perf)) {
            update_tensors_with_any_layout(outputs, id_to_set_any_layout);
        }

        DNN_GRAPH_SAFE(c_partitions.emplace_back(
                               partitions[i].compile(inputs, outputs, eng)),
                WARN);

        record_queried_logical_tensors(outputs, c_partitions[i - idx_offset],
                id_to_queried_logical_tensors);
    }
    if (bench_mode == bench_mode_t::init) return res->state = INITIALIZED, OK;

    idx_offset = 0;
    for (size_t i = 0; i < partitions.size(); ++i) {
        if (is_single_end_op_partition(partitions[i], end_opid_v)) {
            idx_offset += 1;
            continue;
        }

        auto inputs = partitions[i].get_input_ports();
        auto outputs = partitions[i].get_output_ports();
        // replace input logical tensor with the queried one
        replace_with_queried_logical_tensors(
                inputs, id_to_queried_logical_tensors);

        std::vector<dnnl::graph::tensor> input_ts(inputs.size());
        std::vector<dnnl::graph::tensor> output_ts(outputs.size());

        ref_partition_t ref_partition(dg, partitions[i], inputs, outputs);
        // Construct memory for both perf & corr modes
        ref_partition.init_ref(
                bench_mode, graph_in_ports, partition_mem_map_v[i], res);

        if (has_bench_mode_bit(mode_bit_t::corr)) {
            // correctness mode, run ref partition
            if (res->state == UNTESTED || res->state == EXECUTED) {
                ref_partition.exec_ops(res);
                if (res->state == FAILED) return FAIL;
                if (res->state == SKIPPED || res->state == UNIMPLEMENTED)
                    return OK;
            } else {
                // once a partition failed on init_ref, terminate whole graph execution
                return FAIL;
            }
        }

        // unmap memory from host to device
        map_unmap_partition_mem(partition_mem_map_v[i], inputs, UNMAP, res);
        map_unmap_partition_mem(partition_mem_map_v[i], outputs, UNMAP, res);
        if (res->state == FAIL) {
            BENCHDNN_PRINT(0,
                    "FAIL: Fail to unmap memories to host for partition "
                    "%zu.\n",
                    i);
            return FAIL;
        }

        const op_ref_list_t &op_list = ref_partition.get_partition_ops();
        const auto &inplace_ports
                = c_partitions[i - idx_offset].get_inplace_ports();
        if (make_input_tensors(
                    input_ts, partition_mem_map_v[i], op_list, inputs)
                != OK) {
            BENCHDNN_PRINT(0,
                    "FAIL: Fail to construct input tesnors for partition "
                    "%zu.\n",
                    i);
            return res->state = FAILED, FAIL;
        }
        if (make_output_tensors(output_ts, partition_mem_map_v[i], op_list,
                    outputs, inplace_ports)
                != OK) {
            BENCHDNN_PRINT(0,
                    "FAIL: Fail to construct output tesnors for partition "
                    "%zu.\n",
                    i);
            return res->state = FAILED, FAIL;
        }

        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

        input_ts_all.emplace_back(input_ts);
        output_ts_all.emplace_back(output_ts);

        c_partitions[i - idx_offset].execute(strm, input_ts, output_ts);
        strm.wait();

        // map memory from device back to host
        map_unmap_partition_mem(partition_mem_map_v[i], inputs, MAP, res);
        map_unmap_partition_mem(partition_mem_map_v[i], outputs, MAP, res);
        if (res->state == FAIL) {
            BENCHDNN_PRINT(0,
                    "FAIL: Fail to map memories back to host for partition "
                    "%zu.\n",
                    i);
            return FAIL;
        }

        res->state = EXECUTED;

        if (has_bench_mode_bit(mode_bit_t::corr)) {
            // args for correctness check of the last op
            ref_partition.check_partition_correctness(
                    partition_mem_map_v[i], res);
        }
    }

    if (has_bench_mode_bit(mode_bit_t::perf)) {
        SAFE(measure_perf(res->timer_map.perf_timer(), c_partitions,
                     input_ts_all, output_ts_all, res),
                WARN);
    }
    return OK;
}
} // namespace graph
