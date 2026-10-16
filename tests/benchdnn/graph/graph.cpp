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

#include "allocator.hpp"
#include "dnnl_common.hpp"
#include "graph.hpp"
#include "ref_partition.hpp"
#include "utils/execution_mode.hpp"
#include "utils/parser.hpp"
#include "utils/stream_kind.hpp"
#include "utils/stringstream.hpp"

namespace {

/// Set any layout according to the connection relationship of partitions
/// @param dg a deserialized graph
/// @param partitions a list of partitions
/// @param id_to_set_any_layout a set of ids of logical tensors with any layout
///     type
void set_any_layout(const graph::deserialized_graph_t &dg,
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
        graph::deserialized_op_t &aop, graph::deserialized_lt_t &alt,
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

    // Not map or unmap the reference primitive memories for `no_ref_memory`
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

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
        const std::vector<dnnl::graph::logical_tensor> &ins, res_t *res) {
    for (size_t idx = 0; idx < ins.size(); ++idx) {
        // find the op id of the input logical tensor
        const auto &in = ins[idx];
        const auto &lt_id = in.get_id();
        graph::deserialized_lt_t lt;
        graph::deserialized_op_t op;
        if (find_logical_tensor(lt_id, ops, op, lt, true) != OK) {
            BENCHDNN_PRINT(0,
                    "FAIL: Cannot find logical tensor with id %zu! \n", lt_id);
            return FAIL;
        }

        // generate tensor for graph path
        const auto iter = partition_mem_map.find(lt_id);
        if (iter != partition_mem_map.end()) {
            const auto &graph_mem = iter->second;
            input_ts[idx] = graph_mem.make_graph_tensor(lt, res);
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
        const std::vector<std::pair<size_t, size_t>> &inplace_ports,
        res_t *res) {

    for (size_t idx = 0; idx < outs.size(); ++idx) {
        // find the op id of the output logical tensor
        const auto &out = outs[idx];
        const auto &lt_id = out.get_id();
        graph::deserialized_op_t op;
        graph::deserialized_lt_t lt;
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
            output_ts[idx] = graph_mem.make_graph_tensor(lt, res);
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
                    output_ts[idx]
                            = inplace_graph_mem.make_graph_tensor(lt, res);
                } else {
                    BENCHDNN_PRINT(0,
                            "FAIL: Cannot find logical tensor with id %zu! "
                            "\n",
                            inplace_lt_id);
                    return FAIL;
                }

            } else {
                output_ts[idx] = graph_mem.make_graph_tensor(lt, res);
            }
        }
    }
    return OK;
}

} // namespace

namespace graph {

using namespace dnnl::graph;

prb_t::prb_t(const deserialized_graph_t &dg, const size_t &expected_n_partition)
    : dg(dg), expected_n_partition(expected_n_partition) {

    const auto &fpmath = dg.get_fpmath_mode();
    fpmath_mode.mode_ = fpmath.first;
    if (!fpmath.second.empty()) {
        fpmath_mode.apply_to_int_ = parser::parsers::str2bool(fpmath.second);
    }
}

std::string case_to_str(const std::string &json_file,
        const std::map<size_t, std::string> &in_shapes,
        const std::map<size_t, std::string> &op_attrs,
        const graph_fpmath_mode_t &fpmath_mode,
        const size_t expected_n_partitions, const int64_t mb,
        const dnnl_data_type_t dt,
        const std::map<size_t, dnnl_data_type_t> &dt_map,
        const std::map<size_t, std::string> &op_kind_map,
        const std::map<size_t, std::string> &tensor_property) {
    stringstream_t s;

    if (mb != 0) { s << "--mb=" << mb << " "; }

    if (dt != dnnl_data_type_undef) { s << "--dt=" << dt << " "; }

    const bool skip_dts = dt_map.empty()
            || (dt_map.size() == 1 && dt_map.count(SIZE_MAX) == 1);
    if (!skip_dts) {
        s << "--dt=";
        std::string tmp;
        for (const auto &v : dt_map) {
            tmp += (std::to_string(v.first) + ":" + dt2str(v.second) + "+");
        }
        s << tmp.substr(0, tmp.length() - 1) << " ";
    }

    if (!(op_kind_map.size() == 1 && op_kind_map.count(SIZE_MAX) == 1
                && op_kind_map.at(SIZE_MAX) == "default")) {
        s << "--op-kind=";
        std::string tmp;
        for (const auto &v : op_kind_map) {
            tmp += (std::to_string(v.first) + ":" + v.second + "+");
        }
        // Remove dangling '+'.
        s << tmp.substr(0, tmp.size() - 1) << " ";
    }

    if (!(tensor_property.size() == 1 && tensor_property.count(SIZE_MAX) == 1
                && tensor_property.at(SIZE_MAX) == "default")) {
        s << "--tensor-property=";
        std::string tmp;
        for (const auto &v : tensor_property) {
            tmp += (std::to_string(v.first) + ":" + v.second + "+");
        }
        // Remove dangling '+'.
        s << tmp.substr(0, tmp.size() - 1) << " ";
    }

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

    if (fpmath_mode.override_json_value_) {
        s << "--attr-fpmath=" << fpmath_mode.mode_.c_str();
        if (fpmath_mode.apply_to_int_) { s << ":true"; }
        s << " ";
    }

    if (expected_n_partitions != 1) {
        s << "--expected-n-partitions=" << std::to_string(expected_n_partitions)
          << " ";
    }

    s << "--case=" << json_file;
    return s.str();
}

int skip_unimplemented_ops(const dnnl::graph::partition &partition,
        const deserialized_graph_t &dg, res_t *res) {
    // A list of ops that don't have DNNL backend support so far.
    static const std::vector<std::string> unimplemented_ops {"Pow"};
    // A list of ops that don't have DNNL backend support so far on GPU.
    static const std::vector<std::string> unimplemented_ops_gpu {};
    const bool is_gpu = get_graph_engine().is_gpu();
    // For an unsupported partition, retrieve all operation IDs, find a
    // correspondent operation kind in a deserialized_graph_t and match it against
    // a list of known unsupported ops.
    const std::vector<size_t> &partition_op_ids = partition.get_ops();
    for (const size_t op_id : partition_op_ids) {
        const std::string &dg_op_kind = dg.get_op(op_id).kind_;
        const bool has_unimplemented_op = std::any_of(unimplemented_ops.begin(),
                unimplemented_ops.end(),
                [&dg_op_kind](const std::string &kind) {
            return dg_op_kind == kind;
        });
        if (has_unimplemented_op) {
            BENCHDNN_PRINT(
                    2, "[INFO]: Unimplemented op: %s.\n", dg_op_kind.c_str());
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return OK;
        }

        if (is_gpu) {
            const bool has_unimplemented_op_gpu = std::any_of(
                    unimplemented_ops_gpu.begin(), unimplemented_ops_gpu.end(),
                    [&dg_op_kind](const std::string &kind) {
                return dg_op_kind == kind;
            });
            if (has_unimplemented_op_gpu) {
                BENCHDNN_PRINT(2, "[INFO]: Unimplemented op on GPU: %s.\n",
                        dg_op_kind.c_str());
                res->state = SKIPPED;
                res->reason = reason_t::skip_not_supported;
                return OK;
            }
        }
    }
    return OK;
}

// Currently, all primitives support "strict" and "relaxed" accumulation modes
// on CPU.
int skip_unimplemented_accumulation_mode(
        const dnnl::graph::partition &partition, const deserialized_graph_t &dg,
        res_t *res) {
    const bool is_cpu = get_graph_engine().is_cpu();
    if (!is_cpu) return OK;
    const std::vector<size_t> &partition_op_ids = partition.get_ops();
    for (const size_t op_id : partition_op_ids) {
        std::string acc_mode;
        if (dg.get_op(op_id).get_attr_string(acc_mode, "accumulation_mode")) {
            if (acc_mode != "strict" && acc_mode != "relaxed") {
                BENCHDNN_PRINT(2,
                        "[INFO]: Unimplemented accumulation mode: %s.\n",
                        acc_mode.c_str());
                res->state = SKIPPED;
                res->reason = reason_t::skip_not_supported;
                return OK;
            }
        }
    }
    return OK;
}

int skip_unimplemented_partitions(const std::vector<partition> &partitions,
        const deserialized_graph_t &dg, const prb_t *prb, res_t *res) {

    if (partitions.empty()) {
        BENCHDNN_PRINT(0, "%s\n", "Error: partitions are empty");
        SAFE(FAIL, WARN);
    }

    BENCHDNN_PRINT(3, "[INFO]: n_partitions:%zd; ops_in_partitions:%s\n",
            partitions.size(), verbose_partitions_n_ops(partitions).c_str());

    const bool partition_num_mismatch = (prb->expected_n_partition > 0
            && partitions.size() != prb->expected_n_partition);

    // If there are multiple partitions, check all of them for skip conditions
    // before reporting a failure, as an unsupported data type in any partition
    // may cause the mismatch.
    int first_unsupported = -1;
    for (size_t i = 0; i < partitions.size(); ++i) {
        // Skip the partition contains op with unimplemented accumulation_mode
        skip_unimplemented_accumulation_mode(partitions[i], dg, res);
        if (res->state == SKIPPED) return OK;

        // If the partition number mismatches the requirement, check whether
        // there are unsupported data types.
        const bool is_supported = partitions[i].is_supported();
        if (is_supported && !partition_num_mismatch) continue;
        if (!is_supported && first_unsupported < 0)
            first_unsupported = static_cast<int>(i);

        skip_unimplemented_ops(partitions[i], dg, res);
        if (res->state == SKIPPED) return OK;

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
                case logical_tensor::data_type::f8_e5m2:
                    in_out_dt.emplace_back(dnnl_f8_e5m2);
                    break;
                case logical_tensor::data_type::f8_e4m3:
                    in_out_dt.emplace_back(dnnl_f8_e4m3);
                    break;
                case logical_tensor::data_type::s4:
                    in_out_dt.emplace_back(dnnl_s4);
                    break;
                case logical_tensor::data_type::u4:
                    in_out_dt.emplace_back(dnnl_u4);
                default: break;
            }
        }
        // Get partition direction from op's kind which used for skipping
        // unsupported cases.
        dir_t dir = FWD_I;
        const auto &op_ids = partitions[i].get_ops();
        for (const auto &aop : dg.ops_) {
            if (std::count(op_ids.begin(), op_ids.end(), aop.id_)) {
                if (aop.kind_.find("Backward") != std::string::npos) {
                    dir = BWD_DW;
                    break;
                }
                // set the flag back for this specific op.
                if (aop.kind_ == "BatchNormForwardTraining") {
                    dir = FLAG_FWD;
                    break;
                }
            }
        }
        if (in_out_dt.empty()) continue;
        skip_unimplemented_data_type(in_out_dt, dir, res);
        if (res->state == SKIPPED) return OK;
    }

    if (first_unsupported >= 0) {
        BENCHDNN_PRINT(3, "[INFO]: partition #%d is unsupported!\n",
                first_unsupported);
        return res->state = UNIMPLEMENTED, FAIL;
    }

    if (partition_num_mismatch) {
        BENCHDNN_PRINT(0,
                "Error: the expected number of partitions (%zu) doesn't "
                "coincide with the actual number of partitions returned "
                "(%zu).\n ",
                prb->expected_n_partition, partitions.size());
        SAFE(FAIL, WARN);
    }
    return OK;
}

int skip_unimplemented_memory_kind(res_t *res) {
    // currently, only USM memory is supported by graph driver, skip other
    // memory kinds (eg., buffer).
    if (memory_kind != memory_kind_ext_t::usm) {
        BENCHDNN_PRINT(2, "%s\n",
                "[INFO]: only USM memory is supported by graph driver");
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
    }

    return OK;
}

int skip_unimplemented_execution_mode(
        const deserialized_graph_t &dg, const engine_t &eng, res_t *res) {
    if (use_sycl_graph_exec(eng)
            && (dg.has_backward_op()
                    || dg.get_recognized_pattern()
                            == graph_recognized_pattern_t::sdpa_bwd)) {
        BENCHDNN_PRINT(2, "%s\n",
                "[INFO]: Skip backward ops and patterns for SYCL graph "
                "execution mode.");
        res->state = SKIPPED;
        res->reason = reason_t::skip_execution_mode;
        return OK;
    }
    return OK;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == bench_mode_t::list) return res->state = LISTED, OK;

    skip_start(res);
    if (res->state == SKIPPED) return OK;

    skip_unimplemented_memory_kind(res);
    if (res->state == SKIPPED) return OK;

    const auto &dg = prb->dg;
    const auto &graph_in_ports = dg.get_input_ports();
    auto ograph = dg.to_graph(prb->fpmath_mode);
    DNN_GRAPH_SAFE(ograph.finalize(), WARN, res);

    const auto &partitions = ograph.get_partitions();
    SAFE(skip_unimplemented_partitions(partitions, dg, prb, res), WARN);
    if (res->state == SKIPPED) return OK;

    const dnnl::engine &eng = get_graph_engine();
    stream_t strm(eng);

    skip_unimplemented_execution_mode(dg, eng, res);
    if (res->state == SKIPPED) return OK;

    // mark the output logical tensors of partition as ANY layout enabled
    std::unordered_set<size_t> id_to_set_any_layout;
    std::vector<compiled_partition> c_partitions;
    std::vector<std::vector<tensor>> input_ts_all, output_ts_all;
    std::vector<tensor> scratchpad_ts_all;
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

    for (size_t i = 0; i < partitions.size(); ++i) {
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
                WARN, res);

        record_queried_logical_tensors(
                outputs, c_partitions.back(), id_to_queried_logical_tensors);
    }

    res->impl_name.clear();
    for (size_t i = 0; i < c_partitions.size(); ++i) {
        const char *impl_info = nullptr;
        DNN_GRAPH_SAFE(impl_info = c_partitions[i].impl_info_str(), WARN, res);
        if (!impl_info || impl_info[0] == '\0') {
            BENCHDNN_PRINT(0,
                    "Error: empty implementation info for compiled partition "
                    "#%zu.\n",
                    i);
            return res->state = FAILED, FAIL;
        }
        // For case containing multiple compiled partitions, concatenate their
        // implementation info with '+'.
        if (!res->impl_name.empty()) res->impl_name += "+";
        res->impl_name += impl_info;
    }
    BENCHDNN_PRINT(
            5, "oneDNN graph implementation: %s\n", res->impl_name.c_str());

    scratchpad_ts_all.reserve(c_partitions.size());
    auto &graph_mem_mgr = graph_mem_manager_t::get_instance();

    if (bench_mode == bench_mode_t::init) return res->state = INITIALIZED, OK;

    // `idx_offset` points to the correspondent `compiled_partition`, if any
    // of `partitions` were skipped expectedly and not compiled.
    size_t idx_offset = 0;
    for (size_t i = 0; i < partitions.size(); ++i) {
        auto inputs = partitions[i].get_input_ports();
        auto outputs = partitions[i].get_output_ports();
        // replace input logical tensor with the queried one
        replace_with_queried_logical_tensors(
                inputs, id_to_queried_logical_tensors);

        std::vector<dnnl::graph::tensor> input_ts(inputs.size());
        std::vector<dnnl::graph::tensor> output_ts(outputs.size());

        ref_partition_t ref_partition(dg, partitions[i], inputs, outputs);

        // Reference execution must use direct execution mode. Use RAII guard to
        // ensure the original execution mode is restored.
        // TODO: introduce exec_args_t for `execute_and_wait` to explicitly
        // specify `execution_mode` value instead of updating the global value.
        {
            execution_mode_guard_t exec_mode_guard(execution_mode_t::direct);

            // Construct memory for both perf & corr modes
            SAFE(ref_partition.init_ref(graph_in_ports, res), WARN);
            if (res->state == SKIPPED) return OK;

            if (has_bench_mode_bit(mode_bit_t::corr)) {
                // correctness mode, run ref partition
                if (res->state == UNTESTED || res->state == EXECUTED
                        || res->state == DEFERRED) {
                    ref_partition.exec_ops(res);
                    if (res->state == FAILED) return FAIL;
                    if (res->state == SKIPPED || res->state == UNIMPLEMENTED)
                        return OK;
                } else {
                    // once a partition failed on init_ref, terminate whole graph execution
                    return FAIL;
                }
            }

            // To ensure data consistency between graph and reference paths,
            // memory initialization and data displacement for reference
            // primitives (performed in ref_partition.init_ref and
            // ref_partition.exec_ops) must be completed before initializing
            // graph memory.
            SAFE(ref_partition.init_graph_mem(partition_mem_map_v[i], res),
                    WARN);
            if (res->state == SKIPPED) return OK;

            // unmap memory from host to device
            SAFE(map_unmap_partition_mem(
                         partition_mem_map_v[i], inputs, UNMAP, res),
                    WARN);
            SAFE(map_unmap_partition_mem(
                         partition_mem_map_v[i], outputs, UNMAP, res),
                    WARN);
        }

        const op_ref_list_t &op_list = ref_partition.get_partition_ops();
        const auto &inplace_ports
                = c_partitions[i - idx_offset].get_inplace_ports();
        if (make_input_tensors(
                    input_ts, partition_mem_map_v[i], op_list, inputs, res)
                != OK) {
            BENCHDNN_PRINT(0,
                    "FAIL: Fail to construct input tesnors for partition "
                    "%zu.\n",
                    i);
            return res->state = FAILED, FAIL;
        }
        if (make_output_tensors(output_ts, partition_mem_map_v[i], op_list,
                    outputs, inplace_ports, res)
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

        {
            graph_mem_check_guard_t graph_mem_check_guard(graph_mem_mgr);
            auto scratchpad_lt = c_partitions[i - idx_offset]
                                         .get_scratchpad_logical_tensor();
            if (scratchpad_lt.get_mem_size() > 0) {
                DNN_GRAPH_SAFE(
                        scratchpad_ts_all.emplace_back(scratchpad_lt, eng),
                        (WARN | NEED_CLEANUP), res);
            } else {
                scratchpad_ts_all.emplace_back();
            }
            if (res->state == SKIPPED) return OK;

            BENCHDNN_PRINT(
                    3, "[INFO]: Start execution of partition #%zd.\n", i);

            if (use_sycl_graph_exec(eng)) {
                std::function<void()> record_fn
                        = std::bind(compiled_partition_executor,
                                c_partitions[i - idx_offset], std::ref(strm),
                                input_ts, output_ts, scratchpad_ts_all.back());
                SAFE(execute_in_graph_mode(strm, record_fn, res), WARN);
            } else {
                stream_staller_t staller(strm);
                // Need following clean-up steps as the memories have been mappped
                // to device. Otherwise the deconstruction will fail.
                DNN_GRAPH_SAFE(
                        c_partitions[i - idx_offset].execute(strm, input_ts,
                                output_ts, scratchpad_ts_all.back()),
                        (WARN | NEED_CLEANUP), res);
                staller.release();

                DNN_GRAPH_SAFE(strm.wait(), WARN, res);
            }
        }
        // map memory from device back to host
        SAFE(map_unmap_partition_mem(partition_mem_map_v[i], inputs, MAP, res),
                WARN);
        SAFE(map_unmap_partition_mem(partition_mem_map_v[i], outputs, MAP, res),
                WARN);

        // If the device is out-of-memory due to graph path execution, skip the
        // case.
        if (res->state == SKIPPED) return OK;
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
            SAFE(ref_partition.check_partition_correctness(
                         partition_mem_map_v[i], res),
                    WARN);
        }

        auto &graph_mem_req = graph_memory_req_args_t::get_instance();
        // release the memory assigned for the reference path of the partition,
        // while the memory for the graph path needs to be kept for the
        // performance mode if needed.
        graph_mem_req.reset_path(REF);
        if (!has_bench_mode_bit(mode_bit_t::perf)) {
            graph_mem_req.reset_path(GRAPH_USER);
        }
    }

    if (has_bench_mode_bit(mode_bit_t::perf)) {
        SAFE(measure_perf(res->timer_map.perf_timer(), c_partitions,
                     input_ts_all, output_ts_all, scratchpad_ts_all, res),
                WARN);
    }

    return OK;
}
} // namespace graph
