/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_UNIT_BACKEND_DNNL_DNNL_TEST_COMMON_HPP
#define GRAPH_UNIT_BACKEND_DNNL_DNNL_TEST_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/passes/utils.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

static inline dnnl::impl::graph::pass::pass_base_ptr get_pass(
        const std::string &pass_name) {
    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());
    auto &passes = pm.get_passes();
    auto find = std::find_if(passes.begin(), passes.end(),
            [&pass_name](const dnnl::impl::graph::pass::pass_base_ptr &p)
                    -> bool { return p->get_pass_name() == pass_name; });

    return *find;
}

static inline void run_all_passes(dnnl::impl::graph::graph_t &agraph) {
    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());
    pm.run_passes(agraph, "", dnnl::impl::graph::partition_policy::fusion);
}

static inline void run_all_single_passes(dnnl::impl::graph::graph_t &agraph) {
    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());
    auto &passes = pm.get_passes();
    for (auto &p : passes) {
        if (p->get_priority() == 8.f) {
            //single op pass
            p->run(agraph);
        }
    }
}

// This function run the unfused graph op by op. The output tensor of the whole
// graph should also be strided so that we can read and check the results. The
// given graph will be deep copied first so that all the changes inside the
// function are not visible outside.
static inline dnnl::impl::graph::status_t run_graph(
        dnnl::impl::graph::graph_t &agraph,
        const std::vector<test_tensor> &g_in_ts,
        const std::vector<test_tensor> &g_out_ts,
        dnnl::impl::graph::engine_t &eng, dnnl::impl::graph::stream_t &strm) {
    namespace graph = dnnl::impl::graph;
    namespace dnnl_impl = graph::dnnl_impl;
    graph::status_t ret;
    graph::graph_t copied(agraph);
    auto ops = copied.get_ops();

    // force each tensor to be strided
    for (auto &op : ops) {
        for (auto val : op->get_input_values()) {
            val->set_layout_type(graph::layout_type::strided);
        }

        for (auto val : op->get_output_values()) {
            val->set_layout_type(graph::layout_type::strided);
        }
    }

    // set the given in/outputs to the graph
    std::vector<graph::logical_tensor_t> g_in_lts, g_out_lts;
    for (auto &in_t : g_in_ts) {
        g_in_lts.emplace_back(in_t.get().get_logical_tensor());
    }
    for (auto &out_t : g_out_ts) {
        g_out_lts.emplace_back(out_t.get().get_logical_tensor());
    }
    ret = dnnl_impl::set_given_inputs_outputs(ops, g_in_lts, g_out_lts);
    if (ret != graph::status::success) return ret;

    // do shape inference
    ret = copied.infer_shape();
    if (ret != graph::status::success) return ret;

    // used to hold the temporary buffers
    std::unordered_map<size_t, test_tensor> temp_data;

    // compile and execute each op in topo order
    return graph::topo_order_visit(
            copied.get_output_ops(), [&](graph::op_t *op) {
                // construct a single op partition
                graph::graph_t g(eng.kind());
                graph::op_t single_op(op->get_kind());
                single_op.merge_attributes(op->get_attributes());
                single_op.remove_attr(graph::op_attr::matched);
                single_op.remove_attr(graph::op_attr::op_depth);
                single_op.set_partition(nullptr);
                for (size_t i = 0; i < op->num_inputs(); i++) {
                    single_op.add_input(
                            op->get_input_value(i)->get_logical_tensor());
                }
                for (size_t i = 0; i < op->num_outputs(); i++) {
                    single_op.add_output(
                            op->get_output_value(i)->get_logical_tensor());
                }
                g.add_op(&single_op);
                g.finalize();
                run_all_single_passes(g);

                auto part = g.get_partitions()[0];
                // compile
                graph::partition_t p;
                p.init(part);

                // prepare logical tensors
                std::vector<graph::logical_tensor_t> in_lts, out_lts;
                std::vector<const graph::logical_tensor_t *> in_lt_ptrs,
                        out_lt_ptrs;
                in_lts.reserve(op->num_inputs());
                in_lt_ptrs.reserve(op->num_inputs());
                out_lts.reserve(op->num_outputs());
                out_lt_ptrs.reserve(op->num_outputs());

                for (auto &in : op->get_input_values()) {
                    in_lts.emplace_back(in->get_logical_tensor());
                    in_lt_ptrs.emplace_back(&in_lts.back());
                }

                for (auto &out : op->get_output_values()) {
                    out_lts.emplace_back(out->get_logical_tensor());
                    out_lt_ptrs.emplace_back(&out_lts.back());
                }

                // compile
                graph::compiled_partition_t cp(p);
                ret = p.compile(&cp, in_lt_ptrs, out_lt_ptrs, &eng);
                if (ret != graph::status::success) return ret;

                // update the layout info in output values
                for (auto &out_val : op->get_output_values()) {
                    graph::logical_tensor_t compiled_lt;
                    cp.query_logical_tensor(
                            out_val->get_logical_tensor().id, &compiled_lt);
                    out_val->set_logical_tensor(compiled_lt);
                }

                // prepare tensors
                std::vector<graph::tensor_t> in_ts, out_ts;
                for (auto &in_val : op->get_input_values()) {
                    auto in_lt = in_val->get_logical_tensor();
                    auto pos = std::find_if(g_in_ts.begin(), g_in_ts.end(),
                            [&](const test_tensor &t) {
                                return in_lt.id
                                        == t.get().get_logical_tensor().id;
                            });
                    if (pos != g_in_ts.end()) {
                        in_ts.emplace_back(pos->get());
                        continue;
                    }
                    if (temp_data.find(in_lt.id) == temp_data.end()) {
                        temp_data.insert({in_lt.id, test_tensor(in_lt, &eng)});
                    }
                    in_ts.emplace_back(temp_data.at(in_lt.id).get());
                }
                for (auto &out_val : op->get_output_values()) {
                    auto out_lt = out_val->get_logical_tensor();
                    auto pos = std::find_if(g_out_ts.begin(), g_out_ts.end(),
                            [&](const test_tensor &t) {
                                return out_lt.id
                                        == t.get().get_logical_tensor().id;
                            });
                    if (pos != g_out_ts.end()) {
                        out_ts.emplace_back(pos->get());
                        continue;
                    }
                    if (temp_data.find(out_lt.id) == temp_data.end()) {
                        temp_data.insert(
                                {out_lt.id, test_tensor(out_lt, &eng)});
                    }
                    out_ts.emplace_back(temp_data.at(out_lt.id).get());
                }

                // execute
                ret = cp.execute(&strm, in_ts, out_ts);
                if (ret != graph::status::success) return ret;

                strm.wait();
                return graph::status::success;
            });
}

template <typename T>
static inline bool allclose(const std::vector<T> &a, const std::vector<T> &b,
        float rtol, float atol) {
    if (a.size() != b.size()) return false;
    bool flag = true;
    for (size_t i = 0; i < a.size(); i++) {
        float diff
                = std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        if (diff > (atol + rtol * std::abs(static_cast<float>(b[i])))) {
            flag = false;
            std::cout << "index = " << i << ", a = " << static_cast<float>(a[i])
                      << ", b = " << static_cast<float>(b[i])
                      << ", diff = " << diff << ", atol = " << atol
                      << ", rtol = " << rtol << ". Failed.\n";
            break;
        }
    }
    return flag;
}

template <typename T>
static inline bool allclose(
        const test_tensor &a, const test_tensor &b, float rtol, float atol) {
    auto av = a.as_vec_type<T>();
    auto bv = b.as_vec_type<T>();
    return allclose(av, bv, rtol, atol);
}

static inline size_t product(std::vector<int64_t> &in) {
    if (in.empty()) return 0;
    int64_t prod = std::accumulate(in.begin(), in.end(),
            static_cast<int64_t>(1), std::multiplies<int64_t>());
    return static_cast<size_t>(prod);
}

#define for_ for
#define SET_Q_DQ_DATA_ATTR(q_dq_data) \
    q_dq_data.set_attr<std::string>( \
            dnnl::impl::graph::op_attr::qtype, "per_tensor"); \
    q_dq_data.set_attr<std::vector<int64_t>>( \
            dnnl::impl::graph::op_attr::zps, {zp_src}); \
    q_dq_data.set_attr<std::vector<float>>( \
            dnnl::impl::graph::op_attr::scales, {scale_src}); \
    q_dq_data.set_attr<int64_t>(dnnl::impl::graph::op_attr::axis, 0);

#define SET_Q_DQ_WEIGHT_ATTR(q_dq_weight, pc_axis) \
    q_dq_weight.set_attr<std::string>( \
            dnnl::impl::graph::op_attr::qtype, wei_qtype); \
    q_dq_weight.set_attr<std::vector<int64_t>>( \
            dnnl::impl::graph::op_attr::zps, zp_wei); \
    q_dq_weight.set_attr<std::vector<float>>( \
            dnnl::impl::graph::op_attr::scales, scale_wei); \
    q_dq_weight.set_attr<int64_t>(dnnl::impl::graph::op_attr::axis, pc_axis);

#define SET_CONV_ATTR(conv, nd) \
    conv.set_attr<dims>(dnnl::impl::graph::op_attr::strides, dims(nd, 1)); \
    conv.set_attr<dims>(dnnl::impl::graph::op_attr::dilations, dims(nd, 1)); \
    conv.set_attr<dims>(dnnl::impl::graph::op_attr::pads_begin, dims(nd, 0)); \
    conv.set_attr<dims>(dnnl::impl::graph::op_attr::pads_end, dims(nd, 0)); \
    conv.set_attr<int64_t>(dnnl::impl::graph::op_attr::groups, g); \
    conv.set_attr<std::string>( \
            dnnl::impl::graph::op_attr::data_format, "NCX"); \
    conv.set_attr<std::string>( \
            dnnl::impl::graph::op_attr::weights_format, "OIX");

#define SET_CONVTRANSPOSE_ATTR(convtranspose, nd) \
    convtranspose.set_attr<dims>( \
            dnnl::impl::graph::op_attr::strides, dims(nd, 1)); \
    convtranspose.set_attr<dims>( \
            dnnl::impl::graph::op_attr::dilations, dims(nd, 1)); \
    convtranspose.set_attr<dims>( \
            dnnl::impl::graph::op_attr::pads_begin, dims(nd, 0)); \
    convtranspose.set_attr<dims>( \
            dnnl::impl::graph::op_attr::pads_end, dims(nd, 0)); \
    convtranspose.set_attr<int64_t>(dnnl::impl::graph::op_attr::groups, g); \
    convtranspose.set_attr<std::string>( \
            dnnl::impl::graph::op_attr::data_format, "NCX"); \
    convtranspose.set_attr<std::string>( \
            dnnl::impl::graph::op_attr::weights_format, "IOX");

#define SET_Q_DQ_OUT_ATTR(q_dq_out) \
    q_dq_out.set_attr<std::string>( \
            dnnl::impl::graph::op_attr::qtype, "per_tensor"); \
    q_dq_out.set_attr<std::vector<int64_t>>( \
            dnnl::impl::graph::op_attr::zps, {zp_out}); \
    q_dq_out.set_attr<std::vector<float>>( \
            dnnl::impl::graph::op_attr::scales, {scale_out}); \
    q_dq_out.set_attr<int64_t>(dnnl::impl::graph::op_attr::axis, 0);

#endif
