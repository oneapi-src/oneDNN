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

#include <vector>
#include "../transform/transform.hpp"
#include "../visitor.hpp"
#include "pass.hpp"
#include <compiler/ir/graph/quantization/quantize_op.hpp>
#include <ops/convolution.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/fusible/ternary_elemwise.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <ops/managed_matmul_core.hpp>
#include <ops/matmul_core.hpp>
#include <ops/reduce_mean.hpp>
#include <ops/reshape.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <runtime/dynamic_dispatch/op_func_decl.hpp>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
using namespace ops;
SC_MODULE(graph.pass.dynamic_infer_shape)
static void print_shapes(
        const std::string &name, runtime::dynamic_tensor_t *in) {
    std::stringstream ss;
    ss << "Shape after op " << name << " is :[";
    for (int i = 0; i < in->ndims_; i++) {
        if (i) { ss << ","; }
        ss << std::to_string(in->dims_[i]);
    }
    ss << "]\n";
    SC_MODULE_INFO << ss.str();
}
SC_API void dynamic_infer_shape_by_graph(sc_graph_t &graph,
        runtime::dynamic_tensor_t **ins, runtime::dynamic_tensor_t **outs,
        size_t num_ins, size_t num_outs) {
    if (graph.empty() || !graph.is_dynamic()) { return; }
    COMPILE_ASSERT(num_ins == graph.get_input_ops().size()
                    && num_outs == graph.get_output_ops().size(),
            "Given input/output number does not match the graph.");
    std::unordered_map<graph_tensor_ptr, runtime::dynamic_tensor_t *> ltsr_dtsr;
    std::vector<runtime::dynamic_tensor_t> inner_dyn_tsr;
    std::vector<std::vector<sc_dim>> inner_shape_tsr;

    auto inp_ops = graph.get_input_ops();
    auto out_ops = graph.get_output_ops();
    for (size_t i = 0; i < num_ins; i++) {
        ltsr_dtsr[inp_ops[i]->get_outputs()[0]] = ins[i];
    }
    for (size_t i = 0; i < num_outs; i++) {
        ltsr_dtsr[out_ops[i]->get_inputs()[0]] = outs[i];
    }
    // avoid realloc
    size_t ltsr_num = 0;
    for (auto &op : graph.ops_) {
        ltsr_num += op->get_inputs().size() + op->get_outputs().size();
    }
    ltsr_num -= inp_ops.size() + out_ops.size();
    inner_dyn_tsr.reserve(ltsr_num);
    inner_shape_tsr.reserve(ltsr_num);
    auto get_or_create_dyn_tsr = [&ltsr_dtsr, &inner_dyn_tsr, &inner_shape_tsr](
                                         const graph_tensor_ptr &in) {
        auto it = ltsr_dtsr.find(in);
        if (it != ltsr_dtsr.end()) { return it->second; }
        auto plain_dims = in->details_.get_plain_dims();
        inner_shape_tsr.emplace_back(plain_dims);
        inner_dyn_tsr.emplace_back(nullptr, inner_shape_tsr.back().data(),
                static_cast<int>(plain_dims.size()), 0, 0);
        ltsr_dtsr.insert(std::make_pair(in, &inner_dyn_tsr.back()));
        return &inner_dyn_tsr.back();
    };
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(graph.ops_.size());
    vis.visit_graph(graph,
            [&get_or_create_dyn_tsr](op_visitor_t *vis, const sc_op_ptr &node) {
                if (node->isa<input_op>() || node->isa<output_op>()) {
                    return;
                } else if (node->isa<constant_op_t>()) {
                    // don't need to infer.
                    get_or_create_dyn_tsr(node->get_outputs()[0]);
                } else if (node->isa<matmul_core_op_t>()
                        || node->isa<managed_matmul_core_op_t>()) {
                    auto *data = get_or_create_dyn_tsr(node->get_inputs()[0]);
                    auto *weight = get_or_create_dyn_tsr(node->get_inputs()[1]);
                    auto *out = get_or_create_dyn_tsr(node->get_outputs()[0]);
                    infer_shape_matmul_op(out, data, weight);
                    print_shapes(node->op_name_, out);
                } else if (node->isa<conv_fwd_core_op_t>()) {
                    auto *data = get_or_create_dyn_tsr(node->get_inputs()[0]);
                    auto *weight = get_or_create_dyn_tsr(node->get_inputs()[1]);
                    auto *out = get_or_create_dyn_tsr(node->get_outputs()[0]);
                    auto &stride = node->attrs_.get<sc_dims>("strides");
                    auto dilations = get_dilations(node->attrs_);
                    auto &pads_begin = node->attrs_.has_key("pads_begin")
                            ? node->attrs_.get<sc_dims>("pads_begin")
                            : node->attrs_.get<sc_dims>("paddings");
                    auto &pads_end = node->attrs_.has_key("pads_end")
                            ? node->attrs_.get<sc_dims>("pads_end")
                            : node->attrs_.get<sc_dims>("paddings");
                    auto dyn_conv_info = data->ndims_ == 4
                            ? dyn_conv_fwd_runtime_info_t(stride[0], stride[1],
                                    pads_begin[0], pads_begin[1], pads_end[0],
                                    pads_end[1])
                            : dyn_conv_fwd_runtime_info_t(stride[0], stride[1],
                                    stride[2], pads_begin[0], pads_begin[1],
                                    pads_begin[2], pads_end[0], pads_end[1],
                                    pads_end[2]);
                    infer_shape_conv_fwd_op(out, data, weight, dyn_conv_info);
                    print_shapes(node->op_name_, out);
                } else if (node->isa<unary_elementwise_op_t>()
                        || node->isa<reorder_op_t>()
                        || node->isa<quantize::quantize_op_t>()
                        || node->isa<quantize::dequantize_op_t>()) {
                    auto *in = get_or_create_dyn_tsr(node->get_inputs()[0]);
                    auto *out = get_or_create_dyn_tsr(node->get_outputs()[0]);
                    infer_shape_unary_fusible_op(out, in);
                    print_shapes(node->op_name_, out);
                } else if (node->isa<binary_elementwise_op_t>()) {
                    auto *in0 = get_or_create_dyn_tsr(node->get_inputs()[0]);
                    auto *in1 = get_or_create_dyn_tsr(node->get_inputs()[1]);
                    auto *out = get_or_create_dyn_tsr(node->get_outputs()[0]);
                    infer_shape_binary_fusible_op(out, in0, in1);
                    print_shapes(node->op_name_, out);
                } else if (node->isa<reduce_op_t>()
                        || node->isa<reduce_mean_op_t>()) {
                    auto *in = get_or_create_dyn_tsr(node->get_inputs()[0]);
                    auto *out = get_or_create_dyn_tsr(node->get_outputs()[0]);
                    auto rd_axis
                            = node->attrs_.get<std::vector<int>>("rd_axis");
                    infer_shape_reduce_op(out, in, rd_axis.data(),
                            static_cast<int>(rd_axis.size()));
                    print_shapes(node->op_name_, out);
                } else if (node->isa<transpose_op_t>()) {
                    auto *in = get_or_create_dyn_tsr(node->get_inputs()[0]);
                    auto *out = get_or_create_dyn_tsr(node->get_outputs()[0]);
                    auto order = node->attrs_.get<std::vector<int>>("order");
                    infer_shape_transpose_op(out, in, order.data(),
                            static_cast<int>(order.size()));
                    print_shapes(node->op_name_, out);
                } else if (node->isa<tensor_view_op_t>()) {
                    auto *in = get_or_create_dyn_tsr(node->get_inputs()[0]);
                    auto *out = get_or_create_dyn_tsr(node->get_outputs()[0]);
                    auto new_shape
                            = node->attrs_.get<std::vector<sc_dim>>("shape");
                    auto input_plain_dims
                            = node->get_inputs()[0]->details_.get_plain_dims();
                    infer_shape_tensor_view_op(out, in, input_plain_dims.data(),
                            static_cast<int>(input_plain_dims.size()),
                            new_shape.data(),
                            static_cast<int>(new_shape.size()));
                    print_shapes(node->op_name_, out);
                } else if (node->isa<select_op_t>()) {
                    auto *in0 = get_or_create_dyn_tsr(node->get_inputs()[0]);
                    auto *in1 = get_or_create_dyn_tsr(node->get_inputs()[1]);
                    auto *in2 = get_or_create_dyn_tsr(node->get_inputs()[2]);
                    auto *out = get_or_create_dyn_tsr(node->get_outputs()[0]);
                    infer_shape_select_op(out, in0, in1, in2);
                    print_shapes(node->op_name_, out);
                } else {
                    COMPILE_ASSERT(false,
                            "Unsupported op for shape inference: "
                                    << node->op_name_);
                }
            });
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
