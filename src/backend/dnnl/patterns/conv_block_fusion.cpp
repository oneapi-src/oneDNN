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

#include "backend/dnnl/kernels/large_partition.hpp"
#include "backend/dnnl/patterns/fusions.hpp"
#include "backend/dnnl/patterns/transformation_pattern.hpp"
#include "backend/dnnl/patterns/utils.hpp"

#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = impl::pass::FCreatePattern;

namespace {
template <bool GROUPED>
bool check_grouped(op_t *op) {
    if (GROUPED) {
        return op->has_attr(op_attr::groups)
                && op->get_attr<int64_t>(op_attr::groups) > 1;
    } else {
        return !op->has_attr(op_attr::groups)
                || op->get_attr<int64_t>(op_attr::groups) <= 1;
    }
}

// Block creators used to construct large patterns
pm::pb_op_t *conv_bias(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool grouped = false, bool use_biasadd = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op_t *conv = pgraph->append_op(impl::op_kind::Convolution, in_edges);
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    return conv_bias_dst;
};

pm::pb_op_t *conv_bias_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool grouped = false, bool use_biasadd = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }

    pm::pb_op_t *conv = pgraph->append_op(impl::op_kind::Convolution, in_edges);
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op_t *relu = pgraph->append_op(
            impl::op_kind::ReLU, in_edges_t {in_edge(0, conv_bias_dst, 0)});
    return relu;
};

pm::pb_op_t *conv_bias_add_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, pm::pb_op_t *post_src, bool grouped = false,
        bool use_biasadd = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op_t *conv = pgraph->append_op(impl::op_kind::Convolution, in_edges);
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);

    in_edges_t add_in_edges = in_edges_t {in_edge(0, conv_bias_dst, 0)};
    if (post_src) { add_in_edges.emplace_back(in_edge(1, post_src, 0)); }
    pm::pb_op_t *add = pgraph->append_op(impl::op_kind::Add, add_in_edges);

    pm::pb_op_t *relu = pgraph->append_op(
            impl::op_kind::ReLU, in_edges_t {in_edge(0, add, 0)});
    return relu;
};

pm::pb_op_t *int8_conv_bias(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op_t *dequant_src
            = pgraph->append_op(impl::op_kind::Dequantize, in_edges);
    pm::pb_op_t *dequant_wei = nullptr;
    if (use_quant_wei) {
        pm::pb_op_t *quant_wei = pgraph->append_op(impl::op_kind::Quantize);
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize,
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize);
    }
    pm::pb_op_t *conv = pgraph->append_op(impl::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op_t *quant_dst = pgraph->append_op(
            impl::op_kind::Quantize, in_edges_t {in_edge(0, conv_bias_dst, 0)});
    return quant_dst;
};

pm::pb_op_t *int8_conv_bias_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op_t *dequant_src
            = pgraph->append_op(impl::op_kind::Dequantize, in_edges);
    pm::pb_op_t *dequant_wei = nullptr;
    if (use_quant_wei) {
        pm::pb_op_t *quant_wei = pgraph->append_op(impl::op_kind::Quantize);
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize,
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize);
    }
    pm::pb_op_t *conv = pgraph->append_op(impl::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op_t *relu = pgraph->append_op(
            impl::op_kind::ReLU, in_edges_t {in_edge(0, conv_bias_dst, 0)});
    pm::pb_op_t *quant_dst = pgraph->append_op(
            impl::op_kind::Quantize, in_edges_t {in_edge(0, relu, 0)});
    return quant_dst;
};

pm::pb_op_t *int8_conv_bias_add_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, pm::pb_op_t *post_src, bool grouped = false,
        bool use_biasadd = false, bool use_quant_wei = false,
        bool f32_output = false) {
    in_edges_t in_edges, post_src_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    if (post_src) { post_src_edges = in_edges_t {in_edge(0, post_src, 0)}; }
    pm::pb_op_t *dequant_src
            = pgraph->append_op(impl::op_kind::Dequantize, in_edges);
    pm::pb_op_t *dequant_wei = nullptr;
    if (use_quant_wei) {
        pm::pb_op_t *quant_wei = pgraph->append_op(impl::op_kind::Quantize);
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize,
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize);
    }
    pm::pb_op_t *dequant_other
            = pgraph->append_op(impl::op_kind::Dequantize, post_src_edges);
    pm::pb_op_t *conv = pgraph->append_op(impl::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    pm::pb_op_t *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op_t *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op_t *add = pgraph->append_op(impl::op_kind::Add,
            in_edges_t {in_edge(0, conv_bias_dst, 0),
                    in_edge(1, dequant_other, 0)});
    pm::pb_op_t *relu = pgraph->append_op(
            impl::op_kind::ReLU, in_edges_t {in_edge(0, add, 0)});
    if (f32_output) {
        return relu;
    } else {
        pm::pb_op_t *quant_dst = pgraph->append_op(
                impl::op_kind::Quantize, in_edges_t {in_edge(0, relu, 0)});
        return quant_dst;
    }
};

// The F(x)+x basic residual block
pm::pb_op_t *int8_identical_basic_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op_t *quant_dst0
            = int8_conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *quant_dst1 = int8_conv_bias_add_relu(
            pgraph, quant_dst0, input, grouped, use_biasadd);
    return quant_dst1;
};

// The F(x)+G(x) basic residual block
pm::pb_op_t *int8_convolutional_basic_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op_t *quant_dst0
            = int8_conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *quant_dst1
            = int8_conv_bias(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *quant_dst2 = int8_conv_bias_add_relu(
            pgraph, quant_dst0, quant_dst1, grouped, use_biasadd);
    return quant_dst2;
};

// The F(x)+x bottleneck residual block
pm::pb_op_t *int8_identical_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false, bool f32_output = false) {
    pm::pb_op_t *quant_dst0 = int8_conv_bias_relu(
            pgraph, input, false, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst1 = int8_conv_bias_relu(
            pgraph, quant_dst0, grouped, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst2 = int8_conv_bias_add_relu(pgraph, quant_dst1, input,
            false, use_biasadd, use_quant_wei, f32_output);
    return quant_dst2;
};

// The F(x)+G(x) bottleneck residual block
pm::pb_op_t *int8_convolutional_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    pm::pb_op_t *quant_dst0 = int8_conv_bias_relu(
            pgraph, input, false, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst1 = int8_conv_bias_relu(
            pgraph, quant_dst0, grouped, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst2
            = int8_conv_bias(pgraph, input, false, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst3 = int8_conv_bias_add_relu(
            pgraph, quant_dst1, quant_dst2, false, use_biasadd, use_quant_wei);
    return quant_dst3;
};

pm::pb_op_t *int8_convolutional_bottleneck_resblock_v2(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    pm::pb_op_t *quant_dst0 = int8_conv_bias_relu(
            pgraph, input, grouped, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst1 = int8_conv_bias_relu(
            pgraph, quant_dst0, grouped, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst2 = int8_conv_bias(
            pgraph, quant_dst1, grouped, use_biasadd, use_quant_wei);
    pm::pb_op_t *quant_dst3 = int8_conv_bias_add_relu(
            pgraph, input, quant_dst2, grouped, use_biasadd, use_quant_wei);
    return quant_dst3;
};

pm::pb_op_t *convolutional_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op_t *dst0 = conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *dst1 = conv_bias_relu(pgraph, dst0, grouped, use_biasadd);
    pm::pb_op_t *dst2 = conv_bias(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *dst3
            = conv_bias_add_relu(pgraph, dst1, dst2, grouped, use_biasadd);
    return dst3;
};

pm::pb_op_t *identical_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op_t *dst0 = conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op_t *dst1 = conv_bias_relu(pgraph, dst0, grouped, use_biasadd);
    pm::pb_op_t *dst2
            = conv_bias_add_relu(pgraph, dst1, input, grouped, use_biasadd);
    return dst2;
};

} // namespace

/*!
 * \brief This provides ResNet block-related fusion, i.e.
 *        resblock fusion, bottleneck fusion, etc.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(conv_block_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, conv_simple_resblock_fusion)
        .set_priority(5.f) // low priority to avoid current functionality
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *conv0
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv0->append_decision_function(check_input_num<2>);
                    pm::pb_op_t *relu0 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv0, 0)});

                    pm::pb_op_t *conv1
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, relu0, 0)});
                    conv1->append_decision_function(check_input_num<2>);
                    pm::pb_op_t *relu1 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv1, 0)});

                    pm::pb_op_t *conv2
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, relu1, 0)});
                    conv2->append_decision_function(check_input_num<2>);

                    pm::pb_op_t *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv2, 0)});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// Two conv fusion for f32 resnet
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, conv_bias_relu_conv_bias_relu_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *relu0 = conv_bias_relu(pgraph, nullptr);
                    conv_bias_relu(pgraph, relu0);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, conv_bias_relu_conv_bias_add_relu_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *relu0 = conv_bias_relu(pgraph, nullptr);
                    conv_bias_add_relu(pgraph, relu0, nullptr);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// Two conv fusion for int8 resnet
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_conv_bias_relu_conv_bias_relu_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *quant_dst0
                            = int8_conv_bias_relu(pgraph, nullptr);
                    int8_conv_bias_relu(pgraph, quant_dst0);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, identical_bottleneck_resblock_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_resblock(pgraph, nullptr);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_resblock(pgraph, nullptr, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, convolutional_bottleneck_resblock_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_resblock(pgraph, nullptr);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_identical_bottleneck_resblock_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    int8_identical_bottleneck_resblock(pgraph, nullptr);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    int8_identical_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl,
        int8_convolutional_bottleneck_resblock_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    int8_convolutional_bottleneck_resblock(pgraph, nullptr);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_resnet50_stage_1_4_fusion)
        .set_priority(22.f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    output = int8_identical_bottleneck_resblock(pgraph, output);
                    output = int8_identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_resnet50_stage_2_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_resnet50_stage_3_fusion)
        .set_priority(22.2f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_resnet34_stage_1_4_fusion)
        .set_priority(22.f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_identical_basic_resblock(pgraph, nullptr);
                    output = int8_identical_basic_resblock(pgraph, output);
                    output = int8_identical_basic_resblock(pgraph, output);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_resnet34_stage_2_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_basic_resblock(pgraph, nullptr);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_basic_resblock(pgraph, output);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_resnet34_stage_3_fusion)
        .set_priority(22.2f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_basic_resblock(pgraph, nullptr);
                    // 5 F(x)+x blocks
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_basic_resblock(pgraph, output);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, f32_resnet50_stage_1_4_fusion)
        .set_priority(22.f) // high priority to support itex
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    output = identical_bottleneck_resblock(pgraph, output);
                    output = identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    output = identical_bottleneck_resblock(
                            pgraph, output, false, true);
                    output = identical_bottleneck_resblock(
                            pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, f32_resnet50_stage_2_fusion)
        .set_priority(22.1f) // high priority to support itex
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, f32_resnet50_stage_3_fusion)
        .set_priority(22.2f) // high priority to support itex
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// For itex int8 rn50 only (include the weight quantize into pattern)
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, itex_int8_resnet50_stage_1_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// For itex int8 rn50 only (include the weight quantize into pattern)
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, itex_int8_resnet50_stage_2_fusion)
        .set_priority(22.2f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// For itex int8 rn50 only (include the weight quantize into pattern)
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, itex_int8_resnet50_stage_3_fusion)
        .set_priority(22.3f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, itex_int8_resnet50_stage_4_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true, true);
                    output = int8_identical_bottleneck_resblock(pgraph, output,
                            false, true, true, /* f32 output */ true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// ResNeXt101 stage 1 and stage 4 have 1 convolutional bottleneck residual block
// and followed by 2 identical bottleneck blocks. The convolution's bias can be
// connected to conv op directly as an optional input, or it also can be
// performed by using a separated biasadd op
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_resnext101_stage_1_4_fusion)
        .set_enable(true)
        .set_priority(22.f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, true, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// ResNeXt101 stage 2 has 1 convolutional bottleneck residual block and followed
// by 3 identical bottleneck blocks. The convolution's bias can be connected to
// conv op directly as an optional input, or it also can be performed by using a
// separated biasadd op
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_resnext101_stage_2_fusion)
        .set_enable(true)
        .set_priority(22.1f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, true);
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, true, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// ResNeXt101 stage 3 has 1 convolutional bottleneck residual block and followed
// by 22 identical bottleneck blocks. The convolution's bias can be connected to
// conv op directly as an optional input, or it also can be performed by using a
// separated biasadd op
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_resnext101_stage_3_fusion)
        .set_enable(true)
        .set_priority(22.2f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, true);
                    const size_t identical_residual_block_num = 22;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, true, true);
                    const size_t identical_residual_block_num = 22;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

// ResNeXt101 backbone is the composition of 4 stages, which has 102 conv inside
// it. The convolution's bias can be connected to conv op directly as an
// optional input, or it also can be performed by using a separated biasadd op
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(
        dnnl, int8_resnext101_backbone_fusion)
        .set_enable(true)
        .set_priority(23.f) // high priority to support lz models
        .set_kind(impl::partition_kind::quantized_residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    // stage 1
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, true);
                    for (size_t i = 0; i < 2; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true);
                    // stage 2
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true);
                    for (size_t i = 0; i < 3; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true);
                    // stage 3
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true);
                    for (size_t i = 0; i < 22; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true);
                    // stage 4
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true);
                    for (size_t i = 0; i < 2; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *output = nullptr;
                    // stage 1
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, true, true);
                    for (size_t i = 0; i < 2; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                    // stage 2
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true, true);
                    for (size_t i = 0; i < 3; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                    // stage 3
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true, true);
                    for (size_t i = 0; i < 22; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                    // stage 4
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, output, true, true);
                    for (size_t i = 0; i < 2; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, true, true);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
