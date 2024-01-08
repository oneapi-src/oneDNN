/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_TEST_UTILS_HPP
#define BACKEND_GRAPH_COMPILER_TEST_UTILS_HPP

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "compiler/config/context.hpp"
#include "graph/unit/utils.hpp"
#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"
#include "runtime/config.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace tests {
namespace unit {
namespace compiler {
namespace utils {

#define REQUIRE_AVX512() \
    if (!::dnnl::impl::graph::gc::get_default_context() \
                    ->machine_.cpu_flags_.fAVX512F) { \
        GTEST_SKIP(); \
        return; \
    }

#define REQUIRE_VNNI_AMXINT8() \
    REQUIRE_AVX512() \
    if (!::dnnl::impl::graph::gc::get_default_context() \
                    ->machine_.cpu_flags_.fAVX512VNNI \
            && !::dnnl::impl::graph::gc::get_default_context() \
                        ->machine_.cpu_flags_.fAVX512AMXINT8) { \
        GTEST_SKIP(); \
        return; \
    }

#define REQUIRE_BF16_AMXBF16() \
    REQUIRE_AVX512() \
    if (!::dnnl::impl::graph::gc::get_default_context() \
                    ->machine_.cpu_flags_.fAVX512BF16 \
            && !::dnnl::impl::graph::gc::get_default_context() \
                        ->machine_.cpu_flags_.fAVX512AMXBF16) { \
        GTEST_SKIP(); \
        return; \
    }

#define REQUIRE_AMX() \
    if (!::dnnl::impl::graph::gc::get_default_context() \
                    ->machine_.cpu_flags_.fAVX512AMXTILE) { \
        GTEST_SKIP(); \
        return; \
    }

#define REQUIRE_AMXBF16() \
    if (!::dnnl::impl::graph::gc::get_default_context() \
                    ->machine_.cpu_flags_.fAVX512AMXBF16) { \
        GTEST_SKIP(); \
        return; \
    }

#define REQUIRE_SINGLE_OP_PATTERN() \
    if (!graph::utils::getenv_int_internal( \
                "ENABLE_GRAPH_COMPILER_SINGLE_OP_PATTERN", 0)) { \
        GTEST_SKIP(); \
        return; \
    }

#define SKIP_WHEN_SINGLE_OP_PATTERN_ON() \
    if (graph::utils::getenv_int_internal( \
                "ENABLE_GRAPH_COMPILER_SINGLE_OP_PATTERN", 0)) { \
        GTEST_SKIP(); \
        return; \
    }

#define REQUIRE_CPU_ENGINE() \
    impl::engine_t *engine = get_engine(); \
    SKIP_IF(engine->kind() == graph::engine_kind::gpu, "skip on gpu");

#define DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(name) \
    name.set_attr(op_attr::qtype, std::string("per_tensor")); \
    name.set_attr(op_attr::axis, (int64_t)0);

#define DEFINE_DEFAULT_PER_CHANNEL_DYN_QUANT_ATTR(name, shape, ax) \
    name.set_attr(op_attr::qtype, std::string("per_channel")); \
    name.set_attr(op_attr::axis, (int64_t)ax);

#define DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(name) \
    name.set_attr(graph::op_attr::scales, std::vector<float>({0.12f})); \
    name.set_attr(graph::op_attr::zps, std::vector<int64_t>({2})); \
    name.set_attr(graph::op_attr::qtype, std::string("per_tensor")); \
    name.set_attr(graph::op_attr::axis, (int64_t)0);

#define DEFINE_DEFAULT_PER_CHANNEL_QUANT_ATTR(name, shape, ax) \
    name.set_attr(graph::op_attr::scales, std::vector<float>(shape, 0.12f)); \
    name.set_attr(graph::op_attr::zps, std::vector<int64_t>(shape, 0)); \
    name.set_attr(graph::op_attr::qtype, std::string("per_channel")); \
    name.set_attr(graph::op_attr::axis, (int64_t)ax);

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

typedef enum {
    RESHAPE_INCLUDED = 0,
    RESHAPE_EXCLUDED = 1,
} quantize_position_t;

// this function can add fp32/bf16/int8-fp32/int8-bf16 MHA graph
// if is_itex is false, it is the default setting
// if itex is true, it will put K side's second transpose to matmul_qk
inline void add_MHA_subgraph(graph::graph_t *agraph, bool use_bf16 = false,
        bool use_int8 = false, bool is_itex = false,
        graph::dim_t batch_size = 128, graph::dim_t seq_len = 384,
        graph::dim_t num_head = 16, graph::dim_t head_dim = 1024,
        bool dyn_quant = false) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    graph::dim_t size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> EXTENDED_ATTENTION_MASK_SHAPE = is_itex
            ? std::vector<graph::dim_t> {batch_size, 1, seq_len, seq_len}
            : std::vector<graph::dim_t> {batch_size, 1, 1, seq_len};
    std::vector<graph::dim_t> QKV_INPUT_SHAPE = is_itex
            ? std::vector<graph::dim_t> {batch_size * seq_len, head_dim}
            : std::vector<graph::dim_t> {batch_size, seq_len, head_dim};
    std::vector<graph::dim_t> QKV_RESHAPED_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<graph::dim_t> QKV_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<graph::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<graph::dim_t> CONST_SHAPE {1};
    auto dtype = use_bf16 ? graph::data_type::bf16 : graph::data_type::f32;

    graph::logical_tensor_t query_dequantize_input, key_dequantize_input,
            value_dequantize_input;
    query_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, graph::data_type::u8);
    key_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, graph::data_type::u8);
    value_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, graph::data_type::u8);
    // dyn quant scale and zp tensor.
    graph::logical_tensor_t qkv_scale_desc, qkv_zp_desc;
    if (dyn_quant) {
        qkv_scale_desc = utils::logical_tensor_init(
                logical_tensor_idx++, {1}, graph::data_type::f32);
        qkv_zp_desc = utils::logical_tensor_init(
                logical_tensor_idx++, {1}, graph::data_type::u8);
    }

    graph::logical_tensor_t query_typecast_input, key_typecast_input,
            value_typecast_input;
    query_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, graph::data_type::f32);
    key_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, graph::data_type::f32);
    value_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t query_reshape_input, key_reshape_input,
            value_reshape_input;
    query_reshape_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, dtype);
    key_reshape_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, dtype);
    value_reshape_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, dtype);

    graph::logical_tensor_t query_transpose_input, key_transpose_input,
            value_transpose_input;
    query_transpose_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    key_transpose_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    value_transpose_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);

    graph::logical_tensor_t key_transpose2_input;
    key_transpose2_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t query_matmul_input, key_matmul_input,
            value_matmul_input;
    query_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);
    key_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, dtype);
    value_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(
            logical_tensor_idx++, EXTENDED_ATTENTION_MASK_SHAPE, dtype);

    graph::logical_tensor_t fscore_scale, fscore_scale_out;
    fscore_scale = utils::logical_tensor_init(logical_tensor_idx++, CONST_SHAPE,
            is_itex || (use_bf16 && !use_int8) ? dtype : graph::data_type::f32);
    fscore_scale_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t fscore_add_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t softmax_cast_out;
    softmax_cast_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t softmax_quantize_out;
    softmax_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, graph::data_type::u8);
    graph::logical_tensor_t softmax_dequantize_out;
    softmax_dequantize_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t softmax_dequantize_out_cast;
    softmax_dequantize_out_cast
            = utils::logical_tensor_init(logical_tensor_idx++,
                    MATMUL_QK_OUTPUT_SHAPE, graph::data_type::bf16);
    graph::logical_tensor_t out_scale_desc, out_zp_desc;
    if (dyn_quant) {
        out_scale_desc = utils::logical_tensor_init(
                logical_tensor_idx++, {1}, graph::data_type::f32);
        out_zp_desc = utils::logical_tensor_init(
                logical_tensor_idx++, {1}, graph::data_type::s8);
    }

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    context_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, dtype);

    graph::logical_tensor_t context_cast_out;
    context_cast_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t context_quantize_out;
    context_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, graph::data_type::u8);

    graph::op_t dequantize_query {op_idx++,
            dyn_quant ? graph::op_kind::DynamicDequantize
                      : graph::op_kind::Dequantize,
            "dequantize_query"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequantize_query);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_query);
    }
    graph::op_t dequantize_key {op_idx++,
            dyn_quant ? graph::op_kind::DynamicDequantize
                      : graph::op_kind::Dequantize,
            "dequantize_key"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequantize_key);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_key);
    }
    graph::op_t dequantize_value {op_idx++,
            dyn_quant ? graph::op_kind::DynamicDequantize
                      : graph::op_kind::Dequantize,
            "dequantize_value"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequantize_value);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_value);
    }
    graph::op_t typecast_query {
            op_idx++, graph::op_kind::TypeCast, "typecast_query"};
    graph::op_t typecast_key {
            op_idx++, graph::op_kind::TypeCast, "typecast_key"};
    graph::op_t typecast_value {
            op_idx++, graph::op_kind::TypeCast, "typecast_value"};

    graph::op_t query_reshape {
            op_idx++, graph::op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr(op_attr::shape, QKV_RESHAPED_SHAPE);
    query_reshape.set_attr(op_attr::special_zero, false);
    graph::op_t query_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr(op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t key_reshape {
            op_idx++, graph::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(op_attr::shape, QKV_RESHAPED_SHAPE);
    key_reshape.set_attr(op_attr::special_zero, false);
    graph::op_t key_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr(op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t key_transpose2 {
            op_idx++, graph::op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr(op_attr::order, std::vector<int64_t> {0, 1, 3, 2});

    graph::op_t matmul_qk {op_idx++, graph::op_kind::MatMul, "matmul_qk"};
    if (is_itex) { matmul_qk.set_attr(op_attr::transpose_b, true); }

    graph::op_t fscore_rescale {op_idx++,
            is_itex ? graph::op_kind::Multiply : graph::op_kind::Divide,
            "fscore_rescale"};
    fscore_rescale.set_attr(op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t fscore_add {op_idx++, graph::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t softmax {op_idx++, graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(op_attr::axis, (int64_t)3);

    graph::op_t softmax_cast {
            op_idx++, graph::op_kind::TypeCast, "softmax_cast"};
    graph::op_t quantize_softmax {op_idx++,
            dyn_quant ? graph::op_kind::DynamicQuantize
                      : graph::op_kind::Quantize,
            "quantize_softmax"};
    graph::op_t dequantize_softmax {op_idx++,
            dyn_quant ? graph::op_kind::DynamicDequantize
                      : graph::op_kind::Dequantize,
            "dequantize_softmax"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(quantize_softmax);
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequantize_softmax);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_softmax);
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_softmax);
    }

    graph::op_t dequantize_softmax_cast {
            op_idx++, graph::op_kind::TypeCast, "dequantize_softmax_cast"};

    graph::op_t value_reshape {
            op_idx++, graph::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(graph::op_attr::shape, QKV_RESHAPED_SHAPE);
    value_reshape.set_attr(graph::op_attr::special_zero, false);
    graph::op_t value_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});

    graph::op_t matmul_v {op_idx++, graph::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    graph::op_t transpose_output {
            op_idx++, graph::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t reshape_output {
            op_idx++, graph::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(op_attr::special_zero, false);
    reshape_output.set_attr(op_attr::shape, QKV_INPUT_SHAPE);

    graph::op_t typecast_output {
            op_idx++, graph::op_kind::TypeCast, "typecast_output"};
    graph::op_t quantize_output {op_idx++,
            dyn_quant ? graph::op_kind::DynamicQuantize
                      : graph::op_kind::Quantize,
            "quantize_output"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(quantize_output);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_output);
    }

    if (use_int8) {
        dequantize_query.add_input(query_dequantize_input);
        dequantize_key.add_input(key_dequantize_input);
        dequantize_value.add_input(value_dequantize_input);
        if (!use_bf16) {
            dequantize_query.add_output(query_reshape_input);
            dequantize_key.add_output(key_reshape_input);
            dequantize_value.add_output(value_reshape_input);
        } else {
            dequantize_query.add_output(query_typecast_input);
            dequantize_key.add_output(key_typecast_input);
            dequantize_value.add_output(value_typecast_input);
            typecast_query.add_input(query_typecast_input);
            typecast_key.add_input(key_typecast_input);
            typecast_value.add_input(value_typecast_input);
            typecast_query.add_output(query_reshape_input);
            typecast_key.add_output(key_reshape_input);
            typecast_value.add_output(value_reshape_input);
        }
        if (dyn_quant) {
            dequantize_query.add_input(qkv_scale_desc);
            dequantize_query.add_input(qkv_zp_desc);
            dequantize_key.add_input(qkv_scale_desc);
            dequantize_key.add_input(qkv_zp_desc);
            dequantize_value.add_input(qkv_scale_desc);
            dequantize_value.add_input(qkv_zp_desc);
        }
    }

    query_reshape.add_input(query_reshape_input);
    query_reshape.add_output(query_transpose_input);
    query_transpose.add_input(query_transpose_input);
    query_transpose.add_output(query_matmul_input);

    key_reshape.add_input(key_reshape_input);
    key_reshape.add_output(key_transpose_input);
    key_transpose.add_input(key_transpose_input);
    key_transpose.add_output(key_transpose2_input);

    matmul_qk.add_input(query_matmul_input);

    if (!is_itex) {
        key_transpose2.add_input(key_transpose2_input);
        key_transpose2.add_output(key_matmul_input);
        matmul_qk.add_input(key_matmul_input);
    } else {
        matmul_qk.add_input(key_transpose2_input);
    }

    matmul_qk.add_output(matmul_qk_out);

    fscore_rescale.add_input(matmul_qk_out);
    fscore_rescale.add_input(fscore_scale);
    fscore_rescale.add_output(fscore_scale_out);
    fscore_add.add_input(fscore_scale_out);
    fscore_add.add_input(attention_mask_flt);
    fscore_add.add_output(fscore_add_out);
    softmax.add_input(fscore_add_out);
    softmax.add_output(softmax_out);

    if (use_int8) {
        quantize_softmax.add_output(softmax_quantize_out);
        dequantize_softmax.add_input(softmax_quantize_out);
        dequantize_softmax.add_output(softmax_dequantize_out);
        if (!use_bf16) {
            quantize_softmax.add_input(softmax_out);
            matmul_v.add_input(softmax_dequantize_out);
        } else {
            softmax_cast.add_input(softmax_out);
            softmax_cast.add_output(softmax_cast_out);
            quantize_softmax.add_input(softmax_cast_out);
            dequantize_softmax_cast.add_input(softmax_dequantize_out);
            dequantize_softmax_cast.add_output(softmax_dequantize_out_cast);
            matmul_v.add_input(softmax_dequantize_out_cast);
        }
        if (dyn_quant) {
            quantize_softmax.add_input(qkv_scale_desc);
            quantize_softmax.add_input(qkv_zp_desc);
            dequantize_softmax.add_input(qkv_scale_desc);
            dequantize_softmax.add_input(qkv_zp_desc);
        }
    } else {
        matmul_v.add_input(softmax_out);
    }

    value_reshape.add_input(value_reshape_input);
    value_reshape.add_output(value_transpose_input);
    value_transpose.add_input(value_transpose_input);
    value_transpose.add_output(value_matmul_input);

    matmul_v.add_input(value_matmul_input);
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);
    reshape_output.add_input(context_transpose_out);
    reshape_output.add_output(context_reshape_out);

    if (use_int8) {
        quantize_output.add_output(context_quantize_out);
        if (!use_bf16) {
            quantize_output.add_input(context_reshape_out);
        } else {
            typecast_output.add_input(context_reshape_out);
            typecast_output.add_output(context_cast_out);
            quantize_output.add_input(context_cast_out);
        }
        if (dyn_quant) {
            quantize_output.add_input(out_scale_desc);
            quantize_output.add_input(out_zp_desc);
        }
    }

    if (use_int8) {
        agraph->add_op(&dequantize_query);
        agraph->add_op(&dequantize_key);
        agraph->add_op(&dequantize_value);
        if (use_bf16) {
            agraph->add_op(&typecast_query);
            agraph->add_op(&typecast_key);
            agraph->add_op(&typecast_value);
        }
    }

    agraph->add_op(&query_reshape);
    agraph->add_op(&query_transpose);
    agraph->add_op(&key_reshape);
    agraph->add_op(&key_transpose);

    if (!is_itex) { agraph->add_op(&key_transpose2); }

    agraph->add_op(&value_reshape);
    agraph->add_op(&value_transpose);

    agraph->add_op(&matmul_qk);
    agraph->add_op(&fscore_rescale);
    agraph->add_op(&fscore_add);
    agraph->add_op(&softmax);

    if (use_int8) {
        agraph->add_op(&quantize_softmax);
        agraph->add_op(&dequantize_softmax);
        if (use_bf16) {
            agraph->add_op(&softmax_cast);
            agraph->add_op(&dequantize_softmax_cast);
        }
    }

    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
    agraph->add_op(&reshape_output);

    if (use_int8) {
        agraph->add_op(&quantize_output);
        if (use_bf16) { agraph->add_op(&typecast_output); }
    }
}

// add MHA fp32/bf16/int8 graph that has starting dequantize
// after reshape/transpose && ends with reorder
inline void add_MHA_subgraph_alternative(graph::graph_t *agraph,
        bool use_bf16 = false, bool use_int8 = false,
        graph::op_kind_t output_op_kind = graph::op_kind::Reorder,
        graph::dim_t batch_size = 128, graph::dim_t seq_len = 384,
        graph::dim_t num_head = 16, graph::dim_t head_dim = 1024,
        bool dyn_quant = false) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    graph::dim_t size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<graph::dim_t> QKV_RESHAPED_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<graph::dim_t> QKV_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<graph::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<graph::dim_t> MATMUL_V_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> RESHAPE_OUTPUT_SHAPE {
            batch_size * seq_len, num_head * size_per_head};
    std::vector<graph::dim_t> CONST_SHAPE {1};
    auto dtype = use_bf16 ? graph::data_type::bf16 : graph::data_type::f32;

    graph::logical_tensor_t query_dequantize_input, key_dequantize_input,
            value_dequantize_input;
    query_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::u8);
    key_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, graph::data_type::u8);
    value_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::u8);

    // dyn quant scale and zp tensor.
    graph::logical_tensor_t qkv_scale_desc, qkv_zp_desc;
    if (dyn_quant) {
        qkv_scale_desc = utils::logical_tensor_init(
                logical_tensor_idx++, {1}, graph::data_type::f32);
        qkv_zp_desc = utils::logical_tensor_init(
                logical_tensor_idx++, {1}, graph::data_type::s32);
    }

    graph::logical_tensor_t query_typecast_input, key_typecast_input,
            value_typecast_input;
    query_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::f32);
    key_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, graph::data_type::f32);
    value_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t query_matmul_input, key_matmul_input,
            value_matmul_input;
    query_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);
    key_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, dtype);
    value_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(
            logical_tensor_idx++, EXTENDED_ATTENTION_MASK_SHAPE, dtype);

    graph::logical_tensor_t fscore_scale, fscore_div_out;
    fscore_scale = utils::logical_tensor_init(logical_tensor_idx++, CONST_SHAPE,
            graph::data_type::f32); // fscore_scale is always fp32
    fscore_div_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t fscore_add_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t softmax_cast_out;
    softmax_cast_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t softmax_quantize_out;
    softmax_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, graph::data_type::u8);
    graph::logical_tensor_t softmax_dequantize_out;
    softmax_dequantize_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t softmax_dequantize_out_cast;
    softmax_dequantize_out_cast
            = utils::logical_tensor_init(logical_tensor_idx++,
                    MATMUL_QK_OUTPUT_SHAPE, graph::data_type::bf16);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t context_transpose_out, context_final_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    context_final_out = utils::logical_tensor_init(logical_tensor_idx++, dtype);

    graph::logical_tensor_t context_cast_out;
    context_cast_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32);

    graph::logical_tensor_t context_quantize_out;
    context_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::u8);

    graph::op_t dequantize_query {op_idx++,
            dyn_quant ? graph::op_kind::DynamicDequantize
                      : graph::op_kind::Dequantize,
            "dequantize_query"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequantize_query);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_query);
    }
    graph::op_t dequantize_key {op_idx++,
            dyn_quant ? graph::op_kind::DynamicDequantize
                      : graph::op_kind::Dequantize,
            "dequantize_key"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequantize_key);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_key);
    }
    graph::op_t dequantize_value {op_idx++,
            dyn_quant ? graph::op_kind::DynamicDequantize
                      : graph::op_kind::Dequantize,
            "dequantize_value"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequantize_value);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_value);
    }
    graph::op_t typecast_query {
            op_idx++, graph::op_kind::TypeCast, "typecast_query"};
    graph::op_t typecast_key {
            op_idx++, graph::op_kind::TypeCast, "typecast_key"};
    graph::op_t typecast_value {
            op_idx++, graph::op_kind::TypeCast, "typecast_value"};

    graph::op_t matmul_qk {op_idx++, graph::op_kind::MatMul, "matmul_qk"};

    graph::op_t fscore_div {op_idx++, graph::op_kind::Divide, "fscore_div"};
    fscore_div.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t fscore_add {op_idx++, graph::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t softmax {op_idx++, graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)3);

    graph::op_t softmax_cast {
            op_idx++, graph::op_kind::TypeCast, "softmax_cast"};
    graph::op_t quantize_softmax {op_idx++,
            dyn_quant ? graph::op_kind::DynamicQuantize
                      : graph::op_kind::Quantize,
            "quantize_softmax"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(quantize_softmax);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_softmax);
    }
    graph::op_t dequantize_softmax {op_idx++,
            dyn_quant ? graph::op_kind::DynamicDequantize
                      : graph::op_kind::Dequantize,
            "dequantize_softmax"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequantize_softmax);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_softmax);
    }

    graph::op_t dequantize_softmax_cast {
            op_idx++, graph::op_kind::TypeCast, "dequantize_softmax_cast"};

    graph::op_t matmul_v {op_idx++, graph::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    graph::op_t transpose_output {
            op_idx++, graph::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t reshape_reorder_output {
            op_idx++, output_op_kind, "reshape_reorder_output"};
    if (output_op_kind == graph::op_kind::StaticReshape) {
        // if static reshape
        reshape_reorder_output.set_attr(
                graph::op_attr::shape, RESHAPE_OUTPUT_SHAPE);
        reshape_reorder_output.set_attr(graph::op_attr::special_zero, false);
    }

    graph::op_t typecast_output {
            op_idx++, graph::op_kind::TypeCast, "typecast_output"};
    graph::op_t quantize_output {op_idx++,
            dyn_quant ? graph::op_kind::DynamicQuantize
                      : graph::op_kind::Quantize,
            "quantize_output"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(quantize_output);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_output);
    }

    if (use_int8) {
        dequantize_query.add_input(query_dequantize_input);
        dequantize_key.add_input(key_dequantize_input);
        dequantize_value.add_input(value_dequantize_input);
        if (!use_bf16) {
            dequantize_query.add_output(query_matmul_input);
            dequantize_key.add_output(key_matmul_input);
            dequantize_value.add_output(value_matmul_input);
        } else {
            dequantize_query.add_output(query_typecast_input);
            dequantize_key.add_output(key_typecast_input);
            dequantize_value.add_output(value_typecast_input);
            typecast_query.add_input(query_typecast_input);
            typecast_key.add_input(key_typecast_input);
            typecast_value.add_input(value_typecast_input);
            typecast_query.add_output(query_matmul_input);
            typecast_key.add_output(key_matmul_input);
            typecast_value.add_output(value_matmul_input);
        }
        if (dyn_quant) {
            dequantize_query.add_input(qkv_scale_desc);
            dequantize_query.add_input(qkv_zp_desc);
            dequantize_key.add_input(qkv_scale_desc);
            dequantize_key.add_input(qkv_zp_desc);
            dequantize_value.add_input(qkv_scale_desc);
            dequantize_value.add_input(qkv_zp_desc);
        }
    }

    matmul_qk.add_input(query_matmul_input);
    matmul_qk.add_input(key_matmul_input);
    matmul_qk.add_output(matmul_qk_out);

    fscore_div.add_input(matmul_qk_out);
    fscore_div.add_input(fscore_scale);
    fscore_div.add_output(fscore_div_out);
    fscore_add.add_input(fscore_div_out);
    fscore_add.add_input(attention_mask_flt);
    fscore_add.add_output(fscore_add_out);
    softmax.add_input(fscore_add_out);
    softmax.add_output(softmax_out);

    if (use_int8) {
        quantize_softmax.add_output(softmax_quantize_out);
        dequantize_softmax.add_input(softmax_quantize_out);
        dequantize_softmax.add_output(softmax_dequantize_out);
        if (!use_bf16) {
            quantize_softmax.add_input(softmax_out);
            matmul_v.add_input(softmax_dequantize_out);
        } else {
            softmax_cast.add_input(softmax_out);
            softmax_cast.add_output(softmax_cast_out);
            quantize_softmax.add_input(softmax_cast_out);
            dequantize_softmax_cast.add_input(softmax_dequantize_out);
            dequantize_softmax_cast.add_output(softmax_dequantize_out_cast);
            matmul_v.add_input(softmax_dequantize_out_cast);
        }
        if (dyn_quant) {
            quantize_softmax.add_input(qkv_scale_desc);
            quantize_softmax.add_input(qkv_zp_desc);
            dequantize_softmax.add_input(qkv_scale_desc);
            dequantize_softmax.add_input(qkv_zp_desc);
        }
    } else {
        matmul_v.add_input(softmax_out);
    }

    matmul_v.add_input(value_matmul_input);
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);
    reshape_reorder_output.add_input(context_transpose_out);
    reshape_reorder_output.add_output(context_final_out);

    if (use_int8) {
        quantize_output.add_output(context_quantize_out);
        if (!use_bf16) {
            quantize_output.add_input(context_final_out);
        } else {
            typecast_output.add_input(context_final_out);
            typecast_output.add_output(context_cast_out);
            quantize_output.add_input(context_cast_out);
        }
        if (dyn_quant) {
            quantize_output.add_input(qkv_scale_desc);
            quantize_output.add_input(qkv_zp_desc);
        }
    }

    if (use_int8) {
        agraph->add_op(&dequantize_query);
        agraph->add_op(&dequantize_key);
        agraph->add_op(&dequantize_value);
        if (use_bf16) {
            agraph->add_op(&typecast_query);
            agraph->add_op(&typecast_key);
            agraph->add_op(&typecast_value);
        }
    }

    agraph->add_op(&matmul_qk);
    agraph->add_op(&fscore_div);
    agraph->add_op(&fscore_add);
    agraph->add_op(&softmax);

    if (use_int8) {
        agraph->add_op(&quantize_softmax);
        agraph->add_op(&dequantize_softmax);
        if (use_bf16) {
            agraph->add_op(&softmax_cast);
            agraph->add_op(&dequantize_softmax_cast);
        }
    }

    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
    agraph->add_op(&reshape_reorder_output);

    if (use_int8) {
        agraph->add_op(&quantize_output);
        if (use_bf16) { agraph->add_op(&typecast_output); }
    }
}

// this function can add a simple MHA without intermediate tensor shape
// aiming for testing infer shape
inline void add_MHA_infer_shape(graph::graph_t *agraph,
        graph::dim_t batch_size = 128, graph::dim_t seq_len = 384,
        graph::dim_t num_head = 16, graph::dim_t head_dim = 1024) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    graph::dim_t size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> MIXED_LAYER_INPUT_SHAPE {
            batch_size, seq_len, head_dim};
    std::vector<graph::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<graph::dim_t> CONST_SHAPE {1};
    std::vector<graph::dim_t> OUTPUT_SHAPE {batch_size, seq_len, head_dim};

    graph::logical_tensor_t query_gemm_out_flt, qk_bmm_wei_flt,
            value_bmm_wei_flt;
    query_gemm_out_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, graph::data_type::f32);
    qk_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, graph::data_type::f32);
    value_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(logical_tensor_idx++,
            EXTENDED_ATTENTION_MASK_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t query_reshape_out, query_transpose_out;
    query_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);
    query_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);

    graph::logical_tensor_t key_reshape_out, key_transpose_out;
    key_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);
    key_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);

    graph::logical_tensor_t key_transpose_out2;
    key_transpose_out2 = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);

    graph::logical_tensor_t fscore_scale, fscore_div_out;
    fscore_scale = utils::logical_tensor_init(
            logical_tensor_idx++, CONST_SHAPE, graph::data_type::f32);
    fscore_div_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);

    graph::logical_tensor_t fscore_add_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);

    graph::logical_tensor_t value_reshape_out, value_transpose_out;
    value_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);
    value_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);

    graph::logical_tensor_t softmax_out_q, softmax_out_deq;
    softmax_out_q = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);
    softmax_out_deq = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);

    graph::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32, layout_type::strided);
    context_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, graph::data_type::f32);

    // reshape + transpose for query + key
    graph::op_t query_reshape {
            op_idx++, graph::op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr(graph::op_attr::shape,
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    query_reshape.set_attr(graph::op_attr::special_zero, false);
    graph::op_t query_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t key_reshape {
            op_idx++, graph::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(graph::op_attr::shape,
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    key_reshape.set_attr(graph::op_attr::special_zero, false);
    graph::op_t key_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t key_transpose2 {
            op_idx++, graph::op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 1, 3, 2});
    graph::op_t matmul_qk {op_idx++, graph::op_kind::MatMul, "matmul_qk"};

    graph::op_t fscore_div {op_idx++, graph::op_kind::Divide, "fscore_div"};
    fscore_div.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t fscore_add {op_idx++, graph::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t softmax {op_idx++, graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)3);

    // reshape + transpose for value
    graph::op_t value_reshape {
            op_idx++, graph::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(graph::op_attr::shape,
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    value_reshape.set_attr(graph::op_attr::special_zero, false);
    graph::op_t value_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t matmul_v {op_idx++, graph::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    graph::op_t transpose_output {
            op_idx++, graph::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t reshape_output {
            op_idx++, graph::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(graph::op_attr::special_zero, false);
    reshape_output.set_attr(graph::op_attr::shape,
            std::vector<int64_t> {batch_size, seq_len, head_dim});

    query_reshape.add_input(query_gemm_out_flt);
    key_reshape.add_input(qk_bmm_wei_flt);
    value_reshape.add_input(value_bmm_wei_flt);

    query_reshape.add_output(query_reshape_out);
    query_transpose.add_input(query_reshape_out);
    query_transpose.add_output(query_transpose_out);

    key_reshape.add_output(key_reshape_out);
    key_transpose.add_input(key_reshape_out);
    key_transpose.add_output(key_transpose_out);
    key_transpose2.add_input(key_transpose_out);
    key_transpose2.add_output(key_transpose_out2);

    matmul_qk.add_input(query_transpose_out);
    matmul_qk.add_input(key_transpose_out2);
    matmul_qk.add_output(matmul_qk_out);

    fscore_div.add_input(matmul_qk_out);
    fscore_div.add_input(fscore_scale);
    fscore_div.add_output(fscore_div_out);
    fscore_add.add_input(fscore_div_out);
    fscore_add.add_input(attention_mask_flt);
    fscore_add.add_output(fscore_add_out);
    softmax.add_input(fscore_add_out);
    softmax.add_output(softmax_out);

    value_reshape.add_output(value_reshape_out);
    value_transpose.add_input(value_reshape_out);
    value_transpose.add_output(value_transpose_out);
    matmul_v.add_input(softmax_out);
    matmul_v.add_input(value_transpose_out);
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);
    reshape_output.add_input(context_transpose_out);
    reshape_output.add_output(context_reshape_out);

    agraph->add_op(&query_reshape);
    agraph->add_op(&query_transpose);
    agraph->add_op(&key_reshape);
    agraph->add_op(&key_transpose);
    agraph->add_op(&key_transpose2);
    agraph->add_op(&matmul_qk);
    agraph->add_op(&fscore_div);
    agraph->add_op(&fscore_add);
    agraph->add_op(&softmax);

    agraph->add_op(&value_reshape);
    agraph->add_op(&value_transpose);
    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
    agraph->add_op(&reshape_output);
}

inline void get_int8_MHA_subgraph_varients(graph::graph_t *agraph,
        bool use_div = true,
        const std::vector<quantize_position_t> &quantize_positions
        = std::vector<quantize_position_t>(4, RESHAPE_INCLUDED),
        graph::dim_t add_inport = 0, graph::dim_t batch_size = 128,
        graph::dim_t seq_len = 384, graph::dim_t num_head = 16,
        graph::dim_t head_dim = 1024) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    graph::dim_t size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> MIXED_LAYER_INPUT_SHAPE {
            batch_size, seq_len, head_dim};
    std::vector<graph::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<graph::dim_t> QKV_RESHAPED_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<graph::dim_t> QKV_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<graph::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<graph::dim_t> MATMUL_V_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> CONST_SHAPE {1};
    std::vector<graph::dim_t> OUTPUT_SHAPE {batch_size, seq_len, head_dim};

    graph::logical_tensor_t query_gemm_out_flt, qk_bmm_wei_flt,
            value_bmm_wei_flt, attention_mask_flt;
    query_gemm_out_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, graph::data_type::f32);
    qk_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, graph::data_type::f32);
    value_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, graph::data_type::f32);
    attention_mask_flt = utils::logical_tensor_init(logical_tensor_idx++,
            EXTENDED_ATTENTION_MASK_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t query_gemm_out_q, qk_bmm_wei_q, value_bmm_wei_q;
    query_gemm_out_q = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[0] ? QKV_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            graph::data_type::u8);
    qk_bmm_wei_q = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[1] ? KEY_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            graph::data_type::u8);
    value_bmm_wei_q = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[2] ? QKV_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            graph::data_type::u8);

    graph::logical_tensor_t query_gemm_out_deq, qk_bmm_wei_deq,
            value_bmm_wei_deq;
    query_gemm_out_deq = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[0] ? QKV_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            graph::data_type::f32);
    qk_bmm_wei_deq = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[1] ? KEY_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            graph::data_type::f32);
    value_bmm_wei_deq = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[2] ? QKV_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            graph::data_type::f32);

    graph::logical_tensor_t query_reshape_out, query_transpose_out;
    query_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, graph::data_type::f32);
    query_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t key_reshape_out, key_transpose_out;
    key_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, graph::data_type::f32);
    key_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t key_transpose_out2;
    key_transpose_out2 = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t fscore_scale, fscore_div_out;
    fscore_scale = utils::logical_tensor_init(
            logical_tensor_idx++, CONST_SHAPE, graph::data_type::f32);
    fscore_div_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t fscore_add_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);
    softmax_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t value_reshape_out, value_transpose_out;
    value_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, graph::data_type::f32);
    value_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t softmax_out_q, softmax_out_deq;
    softmax_out_q = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, graph::data_type::u8);
    softmax_out_deq = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, graph::data_type::f32);
    context_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t context_out_q, context_out_deq;
    context_out_q = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[3] ? MATMUL_V_OUTPUT_SHAPE : OUTPUT_SHAPE,
            graph::data_type::u8);
    context_out_deq = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[3] ? MATMUL_V_OUTPUT_SHAPE : OUTPUT_SHAPE,
            graph::data_type::f32);

    // add quantize-dequantize
    graph::op_t quantize_query_gemm {
            op_idx++, graph::op_kind::Quantize, "quantize_query_gemm"};
    graph::op_t quantize_key_gemm {
            op_idx++, graph::op_kind::Quantize, "quantize_key_gemm"};
    graph::op_t quantize_value_gemm {
            op_idx++, graph::op_kind::Quantize, "quantize_value_gemm"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_query_gemm);
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_key_gemm);
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_value_gemm);

    graph::op_t dequantize_query_gemm {
            op_idx++, graph::op_kind::Dequantize, "dequantize_query_gemm"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_query_gemm);
    graph::op_t dequantize_key_gemm {
            op_idx++, graph::op_kind::Dequantize, "dequantize_key_gemm"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_key_gemm);
    graph::op_t dequantize_value_gemm {
            op_idx++, graph::op_kind::Dequantize, "dequantize_value_gemm"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_value_gemm);

    // reshape + transpose for query + key
    graph::op_t query_reshape {
            op_idx++, graph::op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr(graph::op_attr::special_zero, false);
    query_reshape.set_attr(graph::op_attr::shape, QKV_RESHAPED_SHAPE);
    graph::op_t query_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});

    graph::op_t key_reshape {
            op_idx++, graph::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(graph::op_attr::special_zero, false);
    key_reshape.set_attr(graph::op_attr::shape, QKV_RESHAPED_SHAPE);
    graph::op_t key_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t key_transpose2 {
            op_idx++, graph::op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 1, 3, 2});

    graph::op_t matmul_qk {op_idx++, graph::op_kind::MatMul, "matmul_qk"};

    graph::op_t fscore_rescale {op_idx++,
            use_div ? graph::op_kind::Divide : graph::op_kind::Multiply,
            "fscore_rescale"};
    fscore_rescale.set_attr(
            graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t fscore_add {op_idx++, graph::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t softmax {op_idx++, graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)3);
    // quantize-dequantize softmax's output
    graph::op_t quantize_softmax {
            op_idx++, graph::op_kind::Quantize, "quantize_softmax"};
    graph::op_t dequantize_softmax {
            op_idx++, graph::op_kind::Dequantize, "dequantize_softmax"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_softmax);
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_softmax);

    // reshape + transpose for value
    graph::op_t value_reshape {
            op_idx++, graph::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(graph::op_attr::special_zero, false);
    value_reshape.set_attr(graph::op_attr::shape, QKV_RESHAPED_SHAPE);
    graph::op_t value_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});

    graph::op_t matmul_v {op_idx++, graph::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    graph::op_t transpose_output {
            op_idx++, graph::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t reshape_output {
            op_idx++, graph::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(graph::op_attr::special_zero, false);
    reshape_output.set_attr(graph::op_attr::shape, OUTPUT_SHAPE);

    // quantize dequantize output
    graph::op_t quantize_output {
            op_idx++, graph::op_kind::Quantize, "quantize_output"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_output);
    graph::op_t dequantize_output {
            op_idx++, graph::op_kind::Dequantize, "dequantize_output"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_output);

    // query part: quantize's input; reshape's input;
    quantize_query_gemm.add_output(query_gemm_out_q);
    dequantize_query_gemm.add_input(query_gemm_out_q);
    dequantize_query_gemm.add_output(query_gemm_out_deq);
    query_reshape.add_output(query_reshape_out);
    query_transpose.add_input(query_reshape_out);
    query_transpose.add_output(query_transpose_out);
    if (quantize_positions[0] == RESHAPE_INCLUDED) {
        quantize_query_gemm.add_input(query_gemm_out_flt);
        query_reshape.add_input(query_gemm_out_deq);
    } else {
        quantize_query_gemm.add_input(query_transpose_out);
        query_reshape.add_input(query_gemm_out_flt);
    }

    // key part
    quantize_key_gemm.add_output(qk_bmm_wei_q);
    dequantize_key_gemm.add_input(qk_bmm_wei_q);
    dequantize_key_gemm.add_output(qk_bmm_wei_deq);
    key_reshape.add_output(key_reshape_out);
    key_transpose.add_input(key_reshape_out);
    key_transpose.add_output(key_transpose_out);
    key_transpose2.add_input(key_transpose_out);
    key_transpose2.add_output(key_transpose_out2);
    if (quantize_positions[1] == RESHAPE_INCLUDED) {
        quantize_key_gemm.add_input(qk_bmm_wei_flt);
        key_reshape.add_input(qk_bmm_wei_deq);
    } else {
        quantize_key_gemm.add_input(key_transpose_out2);
        key_reshape.add_input(qk_bmm_wei_flt);
    }

    // value part
    quantize_value_gemm.add_output(value_bmm_wei_q);
    dequantize_value_gemm.add_input(value_bmm_wei_q);
    dequantize_value_gemm.add_output(value_bmm_wei_deq);
    value_reshape.add_output(value_reshape_out);
    value_transpose.add_input(value_reshape_out);
    value_transpose.add_output(value_transpose_out);
    if (quantize_positions[2] == RESHAPE_INCLUDED) {
        quantize_value_gemm.add_input(value_bmm_wei_flt);
        value_reshape.add_input(value_bmm_wei_deq);
    } else {
        quantize_value_gemm.add_input(value_transpose_out);
        value_reshape.add_input(value_bmm_wei_flt);
    }

    // matmul qk
    if (quantize_positions[0] == RESHAPE_INCLUDED) {
        matmul_qk.add_input(query_transpose_out);
    } else {
        matmul_qk.add_input(query_gemm_out_deq);
    }
    if (quantize_positions[1] == RESHAPE_INCLUDED) {
        matmul_qk.add_input(key_transpose_out2);
    } else {
        matmul_qk.add_input(qk_bmm_wei_deq);
    }
    matmul_qk.add_output(matmul_qk_out);

    fscore_rescale.add_input(matmul_qk_out);
    fscore_rescale.add_input(fscore_scale);
    fscore_rescale.add_output(fscore_div_out);

    // add commutativity
    if (add_inport == 0) {
        fscore_add.add_input(fscore_div_out);
        fscore_add.add_input(attention_mask_flt);
    } else {
        fscore_add.add_input(attention_mask_flt);
        fscore_add.add_input(fscore_div_out);
    }
    fscore_add.add_output(fscore_add_out);
    softmax.add_input(fscore_add_out);
    softmax.add_output(softmax_out);
    quantize_softmax.add_input(softmax_out);
    quantize_softmax.add_output(softmax_out_q);
    dequantize_softmax.add_input(softmax_out_q);
    dequantize_softmax.add_output(softmax_out_deq);

    // matmul v
    matmul_v.add_input(softmax_out_deq);
    if (quantize_positions[2] == RESHAPE_INCLUDED) {
        matmul_v.add_input(value_transpose_out);
    } else {
        matmul_v.add_input(value_bmm_wei_deq);
    }
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_output(context_transpose_out);
    reshape_output.add_input(context_transpose_out);
    reshape_output.add_output(context_reshape_out);
    quantize_output.add_output(context_out_q);
    dequantize_output.add_input(context_out_q);
    dequantize_output.add_output(context_out_deq);
    if (quantize_positions[3] == RESHAPE_INCLUDED) {
        transpose_output.add_input(matmul_v_out);
        quantize_output.add_input(context_reshape_out);
    } else {
        quantize_output.add_input(matmul_v_out);
        transpose_output.add_input(context_out_deq);
    }

    agraph->add_op(&quantize_query_gemm);
    agraph->add_op(&quantize_key_gemm);
    agraph->add_op(&quantize_value_gemm);
    agraph->add_op(&dequantize_query_gemm);
    agraph->add_op(&dequantize_key_gemm);
    agraph->add_op(&dequantize_value_gemm);
    agraph->add_op(&query_reshape);
    agraph->add_op(&query_transpose);
    agraph->add_op(&key_reshape);
    agraph->add_op(&key_transpose);
    agraph->add_op(&key_transpose2);
    agraph->add_op(&matmul_qk);
    agraph->add_op(&fscore_rescale);
    agraph->add_op(&fscore_add);
    agraph->add_op(&softmax);
    agraph->add_op(&quantize_softmax);
    agraph->add_op(&dequantize_softmax);
    agraph->add_op(&value_reshape);
    agraph->add_op(&value_transpose);
    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
    agraph->add_op(&quantize_output);
    agraph->add_op(&dequantize_output);
    agraph->add_op(&reshape_output);
}

inline void add_MHA_training_subgraph(graph::graph_t *agraph,
        bool use_bf16 = false, bool has_mul_from_select = false,
        graph::dim_t batch_size = 128, graph::dim_t seq_len = 384,
        graph::dim_t num_head = 16, graph::dim_t head_dim = 1024) {
    auto dtype = use_bf16 ? graph::data_type::bf16 : graph::data_type::f32;
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    graph::dim_t size_per_head = head_dim / num_head;

    std::vector<graph::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<graph::dim_t> QKV_RESHAPED_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<graph::dim_t> QKV_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<graph::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<graph::dim_t> MATMUL_V_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> SOFTMAX_SUM_SHAPE {
            batch_size, num_head, seq_len, 1};
    std::vector<graph::dim_t> CONST_SHAPE {1};
    std::vector<graph::dim_t> OUTPUT_SHAPE {batch_size, seq_len, head_dim};

    // start constructing forward graph
    graph::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(
            logical_tensor_idx++, EXTENDED_ATTENTION_MASK_SHAPE, dtype);

    graph::logical_tensor_t query_transpose_out;
    query_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t key_transpose_out;
    key_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t fscore_scale, fscore_div_out;
    fscore_scale = utils::logical_tensor_init(
            logical_tensor_idx++, CONST_SHAPE, graph::data_type::f32);
    fscore_div_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t fscore_add_out, softmax_out, dropout, dropout_out,
            select, select_out;
    fscore_add_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);
    dropout = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);
    dropout_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);
    select = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);
    select_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t value_transpose_out;
    value_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    context_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, dtype);

    graph::op_t matmul_qk {op_idx++, graph::op_kind::MatMul, "matmul_qk"};
    matmul_qk.set_attr(graph::op_attr::transpose_b, true);
    matmul_qk.add_input(query_transpose_out);
    matmul_qk.add_input(key_transpose_out);
    matmul_qk.add_output(matmul_qk_out);

    graph::op_t fscore_div {op_idx++, graph::op_kind::Divide, "fscore_div"};
    fscore_div.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    fscore_div.add_input(matmul_qk_out);
    fscore_div.add_input(fscore_scale);
    fscore_div.add_output(fscore_div_out);

    graph::op_t fscore_add {op_idx++, graph::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    fscore_add.add_input(fscore_div_out);
    fscore_add.add_input(attention_mask_flt);
    fscore_add.add_output(fscore_add_out);

    graph::op_t softmax {op_idx++, graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)3);
    softmax.add_input(fscore_add_out);
    softmax.add_output(softmax_out);

    graph::op_t mul_dropout {op_idx++, graph::op_kind::Multiply, "mul_dropout"};
    mul_dropout.add_input(softmax_out);
    mul_dropout.add_input(dropout);
    mul_dropout.add_output(dropout_out);

    graph::op_t mul_select {op_idx++, graph::op_kind::Multiply, "mul_select"};
    mul_select.add_input(dropout_out);
    mul_select.add_input(select);
    mul_select.add_output(select_out);

    graph::op_t matmul_v {op_idx++, graph::op_kind::MatMul, "matmul_v"};
    matmul_v.add_input(has_mul_from_select ? select_out : dropout_out);
    matmul_v.add_input(value_transpose_out);
    matmul_v.add_output(matmul_v_out);

    // transpose + reshape before output
    graph::op_t transpose_output {
            op_idx++, graph::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);

    graph::op_t reshape_output {
            op_idx++, graph::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(graph::op_attr::special_zero, false);
    reshape_output.set_attr(graph::op_attr::shape, OUTPUT_SHAPE);
    reshape_output.add_input(context_transpose_out);
    reshape_output.add_output(context_reshape_out);

    // adding ops
    agraph->add_op(&matmul_qk);
    agraph->add_op(&fscore_div);
    agraph->add_op(&fscore_add);
    agraph->add_op(&softmax);
    agraph->add_op(&mul_dropout);
    if (has_mul_from_select) { agraph->add_op(&mul_select); }
    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
    agraph->add_op(&reshape_output);

    // start constructing backward graph
    graph::logical_tensor_t backward_in;
    backward_in = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, dtype);
    graph::logical_tensor_t in_reshape;
    in_reshape = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    graph::logical_tensor_t in_transpose;
    in_transpose = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t bmm_v_grad_weight;
    bmm_v_grad_weight = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t value_transpose;
    value_transpose = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t bmm_v_grad_data;
    bmm_v_grad_data = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t select_grad;
    select_grad = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t dropout_grad;
    dropout_grad = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t softmax_mul;
    softmax_mul = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t softmax_sum;
    softmax_sum = utils::logical_tensor_init(
            logical_tensor_idx++, SOFTMAX_SUM_SHAPE, dtype);

    graph::logical_tensor_t softmax_sub;
    softmax_sub = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t fscore_grad;
    fscore_grad = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t bmm_q_grad_weight;
    bmm_q_grad_weight = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t bmm_k_grad_weight;
    bmm_k_grad_weight = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    graph::op_t reshape_bwd {
            op_idx++, graph::op_kind::StaticReshape, "reshape_bwd"};
    reshape_bwd.set_attr(graph::op_attr::shape, QKV_RESHAPED_SHAPE);
    reshape_bwd.set_attr(graph::op_attr::special_zero, false);
    reshape_bwd.add_input(backward_in);
    reshape_bwd.add_output(in_reshape);

    graph::op_t transpose_bwd {
            op_idx++, graph::op_kind::StaticTranspose, "transpose_bwd"};
    transpose_bwd.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    transpose_bwd.add_input(in_reshape);
    transpose_bwd.add_output(in_transpose);

    graph::op_t grad_v {op_idx++, graph::op_kind::MatMul, "grad_v"};
    grad_v.set_attr(graph::op_attr::transpose_a, true);
    grad_v.add_input(dropout_out);
    grad_v.add_input(in_transpose);
    grad_v.add_output(bmm_v_grad_weight);

    graph::op_t grad_dropout {op_idx++, graph::op_kind::MatMul, "grad_dropout"};
    grad_dropout.set_attr(graph::op_attr::transpose_b, true);
    grad_dropout.add_input(in_transpose);
    grad_dropout.add_input(value_transpose);
    grad_dropout.add_output(bmm_v_grad_data);

    graph::op_t grad_select {op_idx++, graph::op_kind::Multiply, "grad_select"};
    grad_select.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    grad_select.add_input(bmm_v_grad_data);
    grad_select.add_input(select);
    grad_select.add_output(select_grad);

    graph::op_t mul {op_idx++, graph::op_kind::Multiply, "mul"};
    mul.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    mul.add_input(has_mul_from_select ? select_grad : bmm_v_grad_data);
    mul.add_input(dropout);
    mul.add_output(dropout_grad);

    graph::op_t mul_2 {op_idx++, graph::op_kind::Multiply, "mul_2"};
    mul_2.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    mul_2.add_input(dropout_grad);
    mul_2.add_input(softmax_out);
    mul_2.add_output(softmax_mul);

    graph::op_t reduce_sum {op_idx++, graph::op_kind::ReduceSum, "reduce_sum"};
    reduce_sum.set_attr(graph::op_attr::keep_dims, true);
    reduce_sum.set_attr(graph::op_attr::axes, std::vector<int64_t> {-1});
    reduce_sum.add_input(softmax_mul);
    reduce_sum.add_output(softmax_sum);

    graph::op_t sub {op_idx++, graph::op_kind::Subtract, "sub"};
    sub.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    sub.add_input(dropout_grad);
    sub.add_input(softmax_sum);
    sub.add_output(softmax_sub);

    graph::op_t div {op_idx++, graph::op_kind::Divide, "div"};
    div.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    div.add_input(softmax_sub);
    div.add_input(fscore_scale);
    div.add_output(fscore_grad);

    graph::op_t grad_q {op_idx++, graph::op_kind::MatMul, "grad_q"};
    grad_q.add_input(fscore_grad);
    grad_q.add_input(key_transpose_out);
    grad_q.add_output(bmm_q_grad_weight);

    graph::op_t grad_k {op_idx++, graph::op_kind::MatMul, "grad_k"};
    grad_k.set_attr(graph::op_attr::transpose_a, true);
    grad_k.add_input(fscore_grad);
    grad_k.add_input(query_transpose_out);
    grad_k.add_output(bmm_k_grad_weight);

    // adding ops
    agraph->add_op(&reshape_bwd);
    agraph->add_op(&transpose_bwd);
    agraph->add_op(&grad_v);
    agraph->add_op(&grad_dropout);
    if (has_mul_from_select) { agraph->add_op(&grad_select); };
    agraph->add_op(&mul);
    agraph->add_op(&mul_2);
    agraph->add_op(&reduce_sum);
    agraph->add_op(&sub);
    agraph->add_op(&div);
    agraph->add_op(&grad_q);
    agraph->add_op(&grad_k);
}

inline void add_distill_bert_MHA(graph::graph_t *agraph, bool use_bf16 = false,
        bool use_int8 = false,
        graph::op_kind_t output_op_kind = graph::op_kind::Reorder,
        graph::dim_t batch_size = 224, graph::dim_t seq_len = 128,
        graph::dim_t num_head = 12, graph::dim_t head_dim = 768) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    graph::dim_t size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<graph::dim_t> QKV_RESHAPED_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<graph::dim_t> QKV_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<graph::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<graph::dim_t> MATMUL_V_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> RESHAPE_OUTPUT_SHAPE {
            batch_size * seq_len, num_head * size_per_head};
    std::vector<graph::dim_t> CONST_SHAPE {1};
    auto dtype = use_bf16 ? graph::data_type::bf16 : graph::data_type::f32;

    graph::logical_tensor_t query_dequantize_input, key_dequantize_input,
            value_dequantize_input;
    query_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::u8);
    key_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, graph::data_type::u8);
    value_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::u8);

    graph::logical_tensor_t query_typecast_input, key_typecast_input,
            value_typecast_input;
    query_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::f32);
    key_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, graph::data_type::f32);
    value_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t query_matmul_input, key_matmul_input,
            value_matmul_input;
    query_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);
    key_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, dtype);
    value_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(logical_tensor_idx++,
            EXTENDED_ATTENTION_MASK_SHAPE, graph::data_type::boolean);

    graph::logical_tensor_t fscore_scale, select_out;
    fscore_scale = utils::logical_tensor_init(
            logical_tensor_idx++, CONST_SHAPE, dtype);
    select_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t softmax_out;
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t softmax_cast_out;
    softmax_cast_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t softmax_quantize_out;
    softmax_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, graph::data_type::u8);
    graph::logical_tensor_t softmax_dequantize_out;
    softmax_dequantize_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t softmax_dequantize_out_cast;
    softmax_dequantize_out_cast
            = utils::logical_tensor_init(logical_tensor_idx++,
                    MATMUL_QK_OUTPUT_SHAPE, graph::data_type::bf16);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t context_transpose_out, context_final_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    context_final_out = utils::logical_tensor_init(logical_tensor_idx++, dtype);

    graph::logical_tensor_t context_cast_out;
    context_cast_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32);

    graph::logical_tensor_t context_quantize_out;
    context_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::u8);

    graph::op_t dequantize_query {
            op_idx++, graph::op_kind::Dequantize, "dequantize_query"};
    dequantize_query.set_attr(
            graph::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_query.set_attr(graph::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_query.set_attr(graph::op_attr::qtype, std::string("per_tensor"));
    dequantize_query.set_attr(graph::op_attr::axis, (int64_t)0);
    graph::op_t dequantize_key {
            op_idx++, graph::op_kind::Dequantize, "dequantize_key"};
    dequantize_key.set_attr(
            graph::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_key.set_attr(graph::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_key.set_attr(graph::op_attr::qtype, std::string("per_tensor"));
    dequantize_key.set_attr(graph::op_attr::axis, (int64_t)0);
    graph::op_t dequantize_value {
            op_idx++, graph::op_kind::Dequantize, "dequantize_value"};
    dequantize_value.set_attr(
            graph::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_value.set_attr(graph::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_value.set_attr(graph::op_attr::qtype, std::string("per_tensor"));
    dequantize_value.set_attr(graph::op_attr::axis, (int64_t)0);
    graph::op_t typecast_query {
            op_idx++, graph::op_kind::TypeCast, "typecast_query"};
    graph::op_t typecast_key {
            op_idx++, graph::op_kind::TypeCast, "typecast_key"};
    graph::op_t typecast_value {
            op_idx++, graph::op_kind::TypeCast, "typecast_value"};

    graph::op_t matmul_qk {op_idx++, graph::op_kind::MatMul, "matmul_qk"};

    graph::op_t select {op_idx++, graph::op_kind::Select, "select"};
    select.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t softmax {op_idx++, graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)3);

    graph::op_t softmax_cast {
            op_idx++, graph::op_kind::TypeCast, "softmax_cast"};
    graph::op_t quantize_softmax {
            op_idx++, graph::op_kind::Quantize, "quantize_softmax"};
    quantize_softmax.set_attr(
            graph::op_attr::scales, std::vector<float>({0.12f}));
    quantize_softmax.set_attr(graph::op_attr::zps, std::vector<int64_t>({2}));
    quantize_softmax.set_attr(graph::op_attr::qtype, std::string("per_tensor"));
    quantize_softmax.set_attr(graph::op_attr::axis, (int64_t)0);
    graph::op_t dequantize_softmax {
            op_idx++, graph::op_kind::Dequantize, "dequantize_softmax"};
    dequantize_softmax.set_attr(
            graph::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_softmax.set_attr(graph::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_softmax.set_attr(
            graph::op_attr::qtype, std::string("per_tensor"));
    dequantize_softmax.set_attr(graph::op_attr::axis, (int64_t)0);

    graph::op_t dequantize_softmax_cast {
            op_idx++, graph::op_kind::TypeCast, "dequantize_softmax_cast"};

    graph::op_t matmul_v {op_idx++, graph::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    graph::op_t transpose_output {
            op_idx++, graph::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t reshape_reorder_output {
            op_idx++, output_op_kind, "reshape_reorder_output"};
    if (output_op_kind == graph::op_kind::StaticReshape) {
        // if static reshape
        reshape_reorder_output.set_attr(
                graph::op_attr::shape, RESHAPE_OUTPUT_SHAPE);
        reshape_reorder_output.set_attr(graph::op_attr::special_zero, false);
    }

    graph::op_t typecast_output {
            op_idx++, graph::op_kind::TypeCast, "typecast_output"};
    graph::op_t quantize_output {
            op_idx++, graph::op_kind::Quantize, "quantize_output"};
    quantize_output.set_attr(
            graph::op_attr::scales, std::vector<float>({0.12f}));
    quantize_output.set_attr(graph::op_attr::zps, std::vector<int64_t>({2}));
    quantize_output.set_attr(graph::op_attr::qtype, std::string("per_tensor"));
    quantize_output.set_attr(graph::op_attr::axis, (int64_t)0);

    if (use_int8) {
        dequantize_query.add_input(query_dequantize_input);
        dequantize_key.add_input(key_dequantize_input);
        dequantize_value.add_input(value_dequantize_input);
        if (!use_bf16) {
            dequantize_query.add_output(query_matmul_input);
            dequantize_key.add_output(key_matmul_input);
            dequantize_value.add_output(value_matmul_input);
        } else {
            dequantize_query.add_output(query_typecast_input);
            dequantize_key.add_output(key_typecast_input);
            dequantize_value.add_output(value_typecast_input);
            typecast_query.add_input(query_typecast_input);
            typecast_key.add_input(key_typecast_input);
            typecast_value.add_input(value_typecast_input);
            typecast_query.add_output(query_matmul_input);
            typecast_key.add_output(key_matmul_input);
            typecast_value.add_output(value_matmul_input);
        }
    }

    matmul_qk.add_input(query_matmul_input);
    matmul_qk.add_input(key_matmul_input);
    matmul_qk.add_output(matmul_qk_out);

    select.add_input(attention_mask_flt);
    select.add_input(fscore_scale);
    select.add_input(matmul_qk_out);
    select.add_output(select_out);
    softmax.add_input(select_out);
    softmax.add_output(softmax_out);

    if (use_int8) {
        quantize_softmax.add_output(softmax_quantize_out);
        dequantize_softmax.add_input(softmax_quantize_out);
        dequantize_softmax.add_output(softmax_dequantize_out);
        if (!use_bf16) {
            quantize_softmax.add_input(softmax_out);
            matmul_v.add_input(softmax_dequantize_out);
        } else {
            softmax_cast.add_input(softmax_out);
            softmax_cast.add_output(softmax_cast_out);
            quantize_softmax.add_input(softmax_cast_out);
            dequantize_softmax_cast.add_input(softmax_dequantize_out);
            dequantize_softmax_cast.add_output(softmax_dequantize_out_cast);
            matmul_v.add_input(softmax_dequantize_out_cast);
        }
    } else {
        matmul_v.add_input(softmax_out);
    }

    matmul_v.add_input(value_matmul_input);
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);
    reshape_reorder_output.add_input(context_transpose_out);
    reshape_reorder_output.add_output(context_final_out);

    if (use_int8) {
        quantize_output.add_output(context_quantize_out);
        if (!use_bf16) {
            quantize_output.add_input(context_final_out);
        } else {
            typecast_output.add_input(context_final_out);
            typecast_output.add_output(context_cast_out);
            quantize_output.add_input(context_cast_out);
        }
    }

    if (use_int8) {
        agraph->add_op(&dequantize_query);
        agraph->add_op(&dequantize_key);
        agraph->add_op(&dequantize_value);
        if (use_bf16) {
            agraph->add_op(&typecast_query);
            agraph->add_op(&typecast_key);
            agraph->add_op(&typecast_value);
        }
    }

    agraph->add_op(&matmul_qk);
    agraph->add_op(&select);
    agraph->add_op(&softmax);

    if (use_int8) {
        agraph->add_op(&quantize_softmax);
        agraph->add_op(&dequantize_softmax);
        if (use_bf16) {
            agraph->add_op(&softmax_cast);
            agraph->add_op(&dequantize_softmax_cast);
        }
    }

    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
    agraph->add_op(&reshape_reorder_output);

    if (use_int8) {
        agraph->add_op(&quantize_output);
        if (use_bf16) { agraph->add_op(&typecast_output); }
    }
}

inline void add_mlp_subgraph(graph::graph_t *agraph, bool use_bf16 = false,
        graph::dim_t batch_size = 1, graph::dim_t layer = 1,
        std::vector<graph::dim_t> hidden_size = {13, 512},
        std::vector<graph::op_kind_t> act_type = {graph::op_kind::ReLU},
        bool separate_bias_add = false) {
    size_t lt_idx = 0;
    size_t op_idx = 0;

    auto dtype = use_bf16 ? graph::data_type::bf16 : graph::data_type::f32;

    std::vector<graph::dim_t> layer_input_size {batch_size, hidden_size[0]};
    graph::logical_tensor_t input
            = utils::logical_tensor_init(lt_idx++, layer_input_size, dtype);

    for (graph::dim_t i = 0; i < layer; ++i) {
        std::vector<graph::dim_t> layer_weight_size {
                hidden_size[i], hidden_size[i + 1]};
        std::vector<graph::dim_t> layer_bias_size {hidden_size[i + 1]};
        std::vector<graph::dim_t> layer_dst_size {
                batch_size, hidden_size[i + 1]};

        graph::logical_tensor_t weight, bias, matmul_dst, add_dst, act_dst;

        weight = utils::logical_tensor_init(lt_idx++, layer_weight_size, dtype);
        weight.property = property_type::constant;
        bias = utils::logical_tensor_init(lt_idx++, layer_bias_size, dtype);
        bias.property = property_type::constant;
        matmul_dst
                = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
        add_dst = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
        act_dst = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);

        std::string layer_suffix = "_layer" + std::to_string(i);
        graph::op_t matmul {
                op_idx++, graph::op_kind::MatMul, "matmul" + layer_suffix};
        graph::op_t add {op_idx++, graph::op_kind::Add, "add" + layer_suffix};
        graph::op_t activation {
                op_idx++, act_type[i], "activation" + layer_suffix};

        matmul.add_input(input);
        matmul.add_input(weight);
        matmul.add_output(matmul_dst);
        if (separate_bias_add) {
            add.add_input(matmul_dst);
            add.add_input(bias);
            add.add_output(add_dst);
        } else {
            matmul.add_input(bias);
        }

        if (act_type[i] == graph::op_kind::Wildcard) {
            input = separate_bias_add ? add_dst : matmul_dst;
        } else {
            activation.add_input(separate_bias_add ? add_dst : matmul_dst);
            activation.add_output(act_dst);
            input = act_dst;
        }

        agraph->add_op(&matmul);
        if (separate_bias_add) { agraph->add_op(&add); }
        if (act_type[i] != graph::op_kind::Wildcard) {
            agraph->add_op(&activation);
        }
    }
}

inline void add_int8_mlp_subgraph(graph_t *agraph,
        std::vector<graph::dim_t> batch_dims = {1, 384}, graph::dim_t layer = 1,
        std::vector<graph::dim_t> hidden_size = {1024, 1024},
        std::vector<op_kind_t> act_type = {op_kind::ReLU},
        bool mixed_dtype = false, bool dyn_quant = false) {
    size_t lt_idx = 0;
    size_t op_idx = 0;

    auto dtype = mixed_dtype ? graph::data_type::bf16 : graph::data_type::f32;
    std::vector<graph::dim_t> layer_input_size(batch_dims);
    layer_input_size.push_back(hidden_size[0]);
    graph::logical_tensor_t input_desc
            = utils::logical_tensor_init(lt_idx++, layer_input_size, dtype);

    for (graph::dim_t i = 0; i < layer; ++i) {
        std::vector<graph::dim_t> layer_weight_size {
                hidden_size[i], hidden_size[i + 1]};
        std::vector<graph::dim_t> layer_bias_size {hidden_size[i + 1]};
        std::vector<graph::dim_t> layer_dst_size(batch_dims);
        layer_dst_size.push_back(hidden_size[i + 1]);
        // creating logical tensors of each layer
        graph::logical_tensor_t casted_input_desc, quant_input_desc,
                dequant_input_desc, casted_dequant_input_desc,
                quant_weight_desc, dequant_weight_desc,
                casted_dequant_weight_desc, bias_desc, matmul_dst_desc,
                act_dst_desc;
        // logical tensors for dynamic quantization.
        graph::logical_tensor_t input_scale_desc, input_zp_desc,
                weight_scale_desc;
        casted_input_desc = utils::logical_tensor_init(
                lt_idx++, layer_input_size, graph::data_type::f32);
        quant_input_desc = utils::logical_tensor_init(
                lt_idx++, layer_input_size, graph::data_type::u8);
        dequant_input_desc = utils::logical_tensor_init(
                lt_idx++, layer_input_size, graph::data_type::f32);
        casted_dequant_input_desc
                = utils::logical_tensor_init(lt_idx++, layer_input_size, dtype);
        quant_weight_desc = utils::logical_tensor_init(
                lt_idx++, layer_weight_size, graph::data_type::s8);
        quant_weight_desc.property = property_type::constant;
        dequant_weight_desc = utils::logical_tensor_init(
                lt_idx++, layer_weight_size, graph::data_type::f32);
        casted_dequant_weight_desc = utils::logical_tensor_init(
                lt_idx++, layer_weight_size, dtype);
        bias_desc
                = utils::logical_tensor_init(lt_idx++, layer_bias_size, dtype);
        bias_desc.property = property_type::constant;
        matmul_dst_desc
                = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
        act_dst_desc
                = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
        if (dyn_quant) {
            input_scale_desc = utils::logical_tensor_init(
                    lt_idx++, {1}, graph::data_type::f32);
            input_zp_desc = utils::logical_tensor_init(
                    lt_idx++, {1}, graph::data_type::s32);
            weight_scale_desc = utils::logical_tensor_init(
                    lt_idx++, {hidden_size[i + 1]}, graph::data_type::f32);
        }
        // defining ops of each layer
        std::string layer_suffix = "_layer" + std::to_string(i);
        graph::op_t typecast_input_f32 {op_idx++, graph::op_kind::TypeCast,
                "typecast_input_f32" + layer_suffix};
        graph::op_t quant_input {op_idx++,
                dyn_quant ? graph::op_kind::DynamicQuantize
                          : graph::op_kind::Quantize,
                "quantize_input" + layer_suffix};
        if (dyn_quant) {
            DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(quant_input);
        } else {
            DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quant_input);
        }
        graph::op_t dequant_input {op_idx++,
                dyn_quant ? graph::op_kind::DynamicDequantize
                          : graph::op_kind::Dequantize,
                "dequantize_input" + layer_suffix};
        if (dyn_quant) {
            DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequant_input);
        } else {
            DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequant_input);
        }
        graph::op_t typecast_input_bf16 {op_idx++, graph::op_kind::TypeCast,
                "typecast_input_bf16" + layer_suffix};
        graph::op_t dequant_weight {op_idx++,
                dyn_quant ? graph::op_kind::DynamicDequantize
                          : graph::op_kind::Dequantize,
                "dequantize_weight" + layer_suffix};
        if (dyn_quant) {
            DEFINE_DEFAULT_PER_CHANNEL_DYN_QUANT_ATTR(
                    dequant_weight, hidden_size[i + 1], 1);
        } else {
            DEFINE_DEFAULT_PER_CHANNEL_QUANT_ATTR(
                    dequant_weight, hidden_size[i + 1], 1);
        }
        graph::op_t typecast_weight_bf16 {op_idx++, graph::op_kind::TypeCast,
                "typecast_input_bf16" + layer_suffix};
        graph::op_t matmul {
                op_idx++, graph::op_kind::MatMul, "matmul" + layer_suffix};
        graph::op_t activation {
                op_idx++, act_type[i], "activation" + layer_suffix};
        // defining op connection of each layer
        quant_input.add_output(quant_input_desc);
        dequant_input.add_input(quant_input_desc);
        dequant_input.add_output(dequant_input_desc);
        dequant_weight.add_input(quant_weight_desc);
        dequant_weight.add_output(dequant_weight_desc);
        if (mixed_dtype) {
            typecast_input_f32.add_input(input_desc);
            typecast_input_f32.add_output(casted_input_desc);
            quant_input.add_input(casted_input_desc);
            typecast_input_bf16.add_input(dequant_input_desc);
            typecast_input_bf16.add_output(casted_dequant_input_desc);
            typecast_weight_bf16.add_input(dequant_weight_desc);
            typecast_weight_bf16.add_output(casted_dequant_weight_desc);
            matmul.add_input(casted_dequant_input_desc);
            matmul.add_input(casted_dequant_weight_desc);
        } else {
            quant_input.add_input(input_desc);
            matmul.add_input(dequant_input_desc);
            matmul.add_input(dequant_weight_desc);
        }
        if (dyn_quant) {
            quant_input.add_input(input_scale_desc);
            quant_input.add_input(input_zp_desc);
            dequant_input.add_input(input_scale_desc);
            dequant_input.add_input(input_zp_desc);
            dequant_weight.add_input(weight_scale_desc);
        }

        matmul.add_input(bias_desc);
        matmul.add_output(matmul_dst_desc);

        if (act_type[i] == graph::op_kind::Wildcard) {
            input_desc = matmul_dst_desc;
        } else {
            activation.add_input(matmul_dst_desc);
            activation.add_output(act_dst_desc);
            input_desc = act_dst_desc;
        }
        // adding ops of each layer
        if (mixed_dtype) {
            agraph->add_op(&typecast_input_f32);
            agraph->add_op(&typecast_input_bf16);
            agraph->add_op(&typecast_weight_bf16);
        }
        agraph->add_op(&quant_input);
        agraph->add_op(&dequant_input);
        agraph->add_op(&dequant_weight);
        agraph->add_op(&matmul);
        if (act_type[i] != graph::op_kind::Wildcard) {
            agraph->add_op(&activation);
        }

        layer_input_size = layer_dst_size;
    }

    // defining output layer logical tensors
    graph::logical_tensor_t casted_output_desc = utils::logical_tensor_init(
            lt_idx++, layer_input_size, graph::data_type::f32);
    graph::logical_tensor_t quant_output_desc = utils::logical_tensor_init(
            lt_idx++, layer_input_size, graph::data_type::u8);
    graph::logical_tensor_t dequant_output_desc = utils::logical_tensor_init(
            lt_idx++, layer_input_size, graph::data_type::f32);
    graph::logical_tensor_t casted_dequant_output_desc
            = utils::logical_tensor_init(lt_idx++, layer_input_size, dtype);
    // defining logical tensors for dynamic quantization.
    graph::logical_tensor_t output_scale_desc
            = utils::logical_tensor_init(lt_idx++, {1}, graph::data_type::f32);
    graph::logical_tensor_t output_zp_desc
            = utils::logical_tensor_init(lt_idx++, {1}, graph::data_type::s32);
    // defining output layer ops
    graph::op_t typecast_output_f32 {
            op_idx++, graph::op_kind::TypeCast, "typecast_output_f32"};
    graph::op_t quant_output {op_idx++,
            dyn_quant ? graph::op_kind::DynamicQuantize
                      : graph::op_kind::Quantize,
            "quantize_output"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(quant_output);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quant_output);
    }
    graph::op_t dequant_output {op_idx++,
            dyn_quant ? graph::op_kind::DynamicDequantize
                      : graph::op_kind::Dequantize,
            "dequantize_output"};
    if (dyn_quant) {
        DEFINE_DEFAULT_PER_TENSOR_DYN_QUANT_ATTR(dequant_output);
    } else {
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequant_output);
    }
    graph::op_t typecast_output_bf16 {
            op_idx++, graph::op_kind::TypeCast, "typecast_output_bf16"};
    // defining connection between output ops
    quant_output.add_output(quant_output_desc);
    dequant_output.add_input(quant_output_desc);
    dequant_output.add_output(dequant_output_desc);
    if (mixed_dtype) {
        typecast_output_f32.add_input(input_desc);
        typecast_output_f32.add_output(casted_output_desc);
        quant_output.add_input(casted_output_desc);
        typecast_output_bf16.add_input(dequant_output_desc);
        typecast_output_bf16.add_output(casted_dequant_output_desc);
    } else {
        quant_output.add_input(input_desc);
    }
    if (dyn_quant) {
        quant_output.add_input(output_scale_desc);
        quant_output.add_input(output_zp_desc);
        dequant_output.add_input(output_scale_desc);
        dequant_output.add_input(output_zp_desc);
    }
    // adding ops
    agraph->add_op(&quant_output);
    agraph->add_op(&dequant_output);
    if (mixed_dtype) {
        agraph->add_op(&typecast_output_f32);
        agraph->add_op(&typecast_output_bf16);
    }
}

inline void add_int8_mlp_subgraph(graph_t *agraph, graph::dim_t batch_size = 1,
        graph::dim_t layer = 1,
        std::vector<graph::dim_t> hidden_size = {13, 512},
        std::vector<op_kind_t> act_type = {op_kind::ReLU},
        bool mixed_dtype = false, bool dyn_quant = false) {
    add_int8_mlp_subgraph(agraph, std::vector<graph::dim_t> {batch_size}, layer,
            hidden_size, act_type, mixed_dtype, dyn_quant);
}

static bool apply_use_dst(bool use_dst, graph::op_kind_t act_op_kind) {
    if (!use_dst
            && (act_op_kind == graph::op_kind::ReLUBackward
                    || act_op_kind == graph::op_kind::SigmoidBackward)) {
        return false;
    }
    return true;
}

/*
act_type: Activation function op after each layer of matmul (forward order)
act_backprop_type: Activation backprop function op before each matmul
                   (reverse order)
e.g. {ReLU, Sigmoid} ==> {SigmoidBackprop, ReLUBackprop}
formula: X_{i+1} = Activation(Z{i}); Z_{i} = X_{i}W_{i} + bias_{i}
*/
inline void add_mlp_training_graph(graph::graph_t *agraph,
        graph::dim_t batch_size, graph::dim_t layer,
        std::vector<graph::dim_t> hidden_size,
        std::vector<graph::op_kind_t> act_type,
        std::vector<graph::op_kind_t> act_backprop_type, bool use_bf16 = false,
        bool use_dst = true, bool weight_transposed = false) {
    graph::data_type_t dtype
            = use_bf16 ? graph::data_type::bf16 : graph::data_type::f32;
    size_t lt_idx = 0;
    size_t op_idx = 0;
    std::vector<graph::logical_tensor_t> z_lts;
    std::vector<graph::logical_tensor_t> weight_lts;
    std::vector<graph::logical_tensor_t> x_lts;

    std::vector<graph::dim_t> layer_input_size {batch_size, hidden_size[0]};
    graph::logical_tensor_t input
            = utils::logical_tensor_init(lt_idx++, layer_input_size, dtype);
    x_lts.push_back(input);

    // constructing the forward calculations
    for (graph::dim_t i = 0; i < layer; ++i) {
        std::vector<graph::dim_t> layer_weight_size {
                hidden_size[i], hidden_size[i + 1]};
        std::vector<graph::dim_t> layer_bias_size {hidden_size[i + 1]};
        std::vector<graph::dim_t> layer_dst_size {
                batch_size, hidden_size[i + 1]};

        graph::logical_tensor_t weight, bias, matmul_dst, act_dst;

        weight = utils::logical_tensor_init(lt_idx++, layer_weight_size, dtype);
        weight_lts.push_back(weight);
        bias = utils::logical_tensor_init(lt_idx++, layer_bias_size, dtype);
        matmul_dst
                = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
        z_lts.push_back(matmul_dst);
        act_dst = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
        x_lts.push_back(act_dst);

        std::string layer_suffix = "_layer" + std::to_string(i);
        graph::op_t matmul {
                op_idx++, graph::op_kind::MatMul, "matmul" + layer_suffix};
        graph::op_t activation {
                op_idx++, act_type[i], "activation" + layer_suffix};

        matmul.add_input(input);
        matmul.add_input(weight);
        matmul.add_input(bias);
        matmul.add_output(matmul_dst);

        activation.add_input(matmul_dst);
        activation.add_output(act_dst);

        agraph->add_op(&matmul);
        agraph->add_op(&activation);

        input = act_dst;
    }

    graph::logical_tensor_t grad_y = utils::logical_tensor_init(
            lt_idx++, {batch_size, hidden_size[layer]}, dtype);

    // constructing backward calculations
    for (graph::dim_t i = 0; i < layer; ++i) {
        std::vector<graph::dim_t> grad_z_size {
                batch_size, hidden_size[layer - i]};
        std::vector<graph::dim_t> transposed_grad_z_size {
                hidden_size[layer - i], batch_size};
        std::vector<graph::dim_t> grad_x_size {
                batch_size, hidden_size[layer - i - 1]};
        std::vector<graph::dim_t> grad_w_size = weight_transposed
                ? std::vector<graph::dim_t> {hidden_size[layer - i],
                        hidden_size[layer - i - 1]}
                : std::vector<graph::dim_t> {
                        hidden_size[layer - i - 1], hidden_size[layer - i]};
        std::vector<graph::dim_t> transposed_x_size {
                hidden_size[layer - i - 1], batch_size};
        std::vector<graph::dim_t> transposed_weight_size {
                hidden_size[layer - i], hidden_size[layer - i - 1]};
        std::vector<graph::dim_t> grad_bias_size {1, hidden_size[layer - i]};

        graph::logical_tensor_t activation_backprop_out,
                transposed_activation_backprop_out, grad_x, grad_weight,
                grad_bias, transposed_weight_out, transposed_x_out;
        activation_backprop_out
                = utils::logical_tensor_init(lt_idx++, grad_z_size, dtype);
        transposed_activation_backprop_out = utils::logical_tensor_init(
                lt_idx++, transposed_grad_z_size, dtype);
        grad_x = utils::logical_tensor_init(lt_idx++, grad_x_size, dtype);
        grad_weight = utils::logical_tensor_init(lt_idx++, grad_w_size, dtype);
        grad_bias = utils::logical_tensor_init(lt_idx++, grad_bias_size, dtype);
        transposed_weight_out = utils::logical_tensor_init(
                lt_idx++, transposed_weight_size, dtype);
        transposed_x_out = utils::logical_tensor_init(
                lt_idx++, transposed_x_size, dtype);

        std::string layer_suffix = "_layer" + std::to_string(layer - i - 1);
        graph::op_t activation_backward {op_idx++, act_backprop_type[i],
                "activation_backprop" + layer_suffix};
        if (!apply_use_dst(use_dst, act_backprop_type[i])) {
            activation_backward.set_attr(graph::op_attr::use_dst, false);
        }
        graph::op_t transpose_activation_backward {op_idx++,
                graph::op_kind::StaticTranspose,
                "transpose_activation_backward" + layer_suffix};
        transpose_activation_backward.set_attr(
                graph::op_attr::order, std::vector<int64_t> {1, 0});
        graph::op_t transpose_weight {op_idx++, graph::op_kind::StaticTranspose,
                "transpose_weight" + layer_suffix};
        transpose_weight.set_attr(
                graph::op_attr::order, std::vector<int64_t> {1, 0});
        graph::op_t transpose_x {op_idx++, graph::op_kind::StaticTranspose,
                "transpose_x" + layer_suffix};
        transpose_x.set_attr(
                graph::op_attr::order, std::vector<int64_t> {1, 0});
        graph::op_t matmul_weight {
                op_idx++, graph::op_kind::MatMul, "grad_weight" + layer_suffix};
        graph::op_t matmul_x {
                op_idx++, graph::op_kind::MatMul, "grad_x" + layer_suffix};
        graph::op_t reduce_bias {op_idx++, graph::op_kind::ReduceSum,
                "grad_bias" + layer_suffix};
        reduce_bias.set_attr(graph::op_attr::axes, std::vector<int64_t> {0});
        reduce_bias.set_attr(graph::op_attr::keep_dims, true);

        // X_{i+1} = act(Z_{i})
        if (!apply_use_dst(use_dst, act_backprop_type[i])) {
            activation_backward.add_input(z_lts[layer - i - 1]);
        } else {
            activation_backward.add_input(x_lts[layer - i]);
        }
        activation_backward.add_input(grad_y);
        activation_backward.add_output(activation_backprop_out);
        transpose_activation_backward.add_input(activation_backprop_out);
        transpose_activation_backward.add_output(
                transposed_activation_backprop_out);
        transpose_weight.add_input(weight_lts[layer - i - 1]);
        transpose_weight.add_output(transposed_weight_out);
        matmul_x.add_input(activation_backprop_out);
        matmul_x.add_input(transposed_weight_out);
        matmul_x.add_output(grad_x);
        transpose_x.add_input(x_lts[layer - i - 1]);
        transpose_x.add_output(transposed_x_out);
        if (weight_transposed) {
            matmul_weight.add_input(transposed_activation_backprop_out);
            matmul_weight.add_input(x_lts[layer - i - 1]);
            matmul_weight.add_output(grad_weight);
        } else {
            matmul_weight.add_input(transposed_x_out);
            matmul_weight.add_input(activation_backprop_out);
            matmul_weight.add_output(grad_weight);
        }
        reduce_bias.add_input(activation_backprop_out);
        reduce_bias.add_output(grad_bias);

        agraph->add_op(&activation_backward);
        if (weight_transposed) {
            agraph->add_op(&transpose_activation_backward);
        } else {
            agraph->add_op(&transpose_x);
        }
        agraph->add_op(&matmul_weight);
        if (i != layer - 1) {
            // no need to compute grad_input for the last backprop layer
            agraph->add_op(&matmul_x);
            agraph->add_op(&transpose_weight);
        }
        if (batch_size > 1) { agraph->add_op(&reduce_bias); }

        grad_y = grad_x;
    }
}

static inline void add_MHA_subgraph_alternative2(graph::graph_t *agraph,
        bool has_quantize = false, graph::dim_t batch_size = 64,
        graph::dim_t seq_len = 384, graph::dim_t num_head = 16,
        graph::dim_t head_dim = 1024) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    graph::dim_t size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> MIXED_LAYER_INPUT_SHAPE {
            batch_size, seq_len, head_dim};
    std::vector<graph::dim_t> QKV_RESHAPE_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<graph::dim_t> QV_TRANSPOSE_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> K_TRANSPOSE_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<graph::dim_t> MATMUL_QK_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<graph::dim_t> SOFTMAX_SHAPE {
            batch_size * num_head * seq_len, seq_len};
    std::vector<graph::dim_t> MATMUL_V_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<graph::dim_t> OUTPUT_SHAPE {batch_size, seq_len, head_dim};
    std::vector<graph::dim_t> CONST_SHAPE {1};

    graph::logical_tensor_t query_gemm_out_flt, qk_bmm_wei_flt,
            value_bmm_wei_flt;
    query_gemm_out_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, graph::data_type::f32);
    qk_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, graph::data_type::f32);
    value_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(logical_tensor_idx++,
            EXTENDED_ATTENTION_MASK_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t scale;
    scale = utils::logical_tensor_init(
            logical_tensor_idx++, CONST_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t query_reshape_out, query_transpose_out;
    query_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPE_SHAPE, graph::data_type::f32);
    query_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QV_TRANSPOSE_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t key_reshape_out, key_transpose_out;
    key_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPE_SHAPE, graph::data_type::f32);
    key_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, K_TRANSPOSE_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t fake_quantize_out, fake_dequantize_out;
    fake_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, graph::data_type::u8);
    fake_dequantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t add_out;
    add_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, graph::data_type::f32);
    graph::logical_tensor_t mul_out;
    mul_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t pre_softmax_reshape_out;
    pre_softmax_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, SOFTMAX_SHAPE, graph::data_type::f32);
    graph::logical_tensor_t softmax_out;
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, SOFTMAX_SHAPE, graph::data_type::f32);
    graph::logical_tensor_t softmax_reshape_out;
    softmax_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t value_reshape_out, value_transpose_out;
    value_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPE_SHAPE, graph::data_type::f32);
    value_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QV_TRANSPOSE_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPE_SHAPE, graph::data_type::f32);
    context_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, graph::data_type::f32);

    // reshape + transpose for query + key
    graph::op_t query_reshape {
            op_idx++, graph::op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr(graph::op_attr::shape, QKV_RESHAPE_SHAPE);
    query_reshape.set_attr(graph::op_attr::special_zero, false);
    graph::op_t query_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t key_reshape {
            op_idx++, graph::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(graph::op_attr::shape, QKV_RESHAPE_SHAPE);
    key_reshape.set_attr(graph::op_attr::special_zero, false);
    graph::op_t key_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 3, 1});

    graph::op_t matmul_qk {op_idx++, graph::op_kind::MatMul, "matmul_qk"};

    graph::op_t fake_quantize {
            op_idx++, graph::op_kind::Quantize, "fake_quantize"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(fake_quantize);
    graph::op_t fake_dequantize {
            op_idx++, graph::op_kind::Dequantize, "fake_dequantize"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(fake_dequantize);

    graph::op_t add {op_idx++, graph::op_kind::Add, "add"};
    add.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t mul {op_idx++, graph::op_kind::Multiply, "mul"};
    mul.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));

    graph::op_t pre_softmax_reshape {
            op_idx++, graph::op_kind::StaticReshape, "pre_softmax_reshape"};
    pre_softmax_reshape.set_attr(graph::op_attr::shape, SOFTMAX_SHAPE);
    pre_softmax_reshape.set_attr(graph::op_attr::special_zero, false);
    graph::op_t softmax {op_idx++, graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)1);
    graph::op_t softmax_reshape {
            op_idx++, graph::op_kind::StaticReshape, "softmax_reshape"};
    softmax_reshape.set_attr(graph::op_attr::shape, MATMUL_QK_SHAPE);
    softmax_reshape.set_attr(graph::op_attr::special_zero, false);

    // reshape + transpose for value
    graph::op_t value_reshape {
            op_idx++, graph::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(graph::op_attr::shape, QKV_RESHAPE_SHAPE);
    value_reshape.set_attr(graph::op_attr::special_zero, false);
    graph::op_t value_transpose {
            op_idx++, graph::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t matmul_v {op_idx++, graph::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    graph::op_t transpose_output {
            op_idx++, graph::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t reshape_output {
            op_idx++, graph::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(graph::op_attr::special_zero, false);
    reshape_output.set_attr(graph::op_attr::shape, OUTPUT_SHAPE);

    query_reshape.add_input(query_gemm_out_flt);
    key_reshape.add_input(qk_bmm_wei_flt);
    value_reshape.add_input(value_bmm_wei_flt);

    query_reshape.add_output(query_reshape_out);
    query_transpose.add_input(query_reshape_out);
    query_transpose.add_output(query_transpose_out);

    key_reshape.add_output(key_reshape_out);
    key_transpose.add_input(key_reshape_out);
    key_transpose.add_output(key_transpose_out);

    matmul_qk.add_input(query_transpose_out);
    matmul_qk.add_input(key_transpose_out);
    matmul_qk.add_output(matmul_qk_out);

    if (!has_quantize) {
        add.add_input(matmul_qk_out);
        add.add_input(attention_mask_flt);
        add.add_output(add_out);
        pre_softmax_reshape.add_input(add_out);
    } else {
        fake_quantize.add_input(matmul_qk_out);
        fake_quantize.add_output(fake_quantize_out);
        fake_dequantize.add_input(fake_quantize_out);
        fake_dequantize.add_output(fake_dequantize_out);
        add.add_input(fake_dequantize_out);
        add.add_input(attention_mask_flt);
        add.add_output(add_out);
        mul.add_input(add_out);
        mul.add_input(scale);
        mul.add_output(mul_out);
        pre_softmax_reshape.add_input(mul_out);
    }

    pre_softmax_reshape.add_output(pre_softmax_reshape_out);
    softmax.add_input(pre_softmax_reshape_out);
    softmax.add_output(softmax_out);
    softmax_reshape.add_input(softmax_out);
    softmax_reshape.add_output(softmax_reshape_out);

    value_reshape.add_output(value_reshape_out);
    value_transpose.add_input(value_reshape_out);
    value_transpose.add_output(value_transpose_out);
    matmul_v.add_input(softmax_reshape_out);
    matmul_v.add_input(value_transpose_out);
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);
    reshape_output.add_input(context_transpose_out);
    reshape_output.add_output(context_reshape_out);

    agraph->add_op(&query_reshape);
    agraph->add_op(&query_transpose);
    agraph->add_op(&key_reshape);
    agraph->add_op(&key_transpose);
    agraph->add_op(&matmul_qk);
    agraph->add_op(&add);
    if (has_quantize) {
        agraph->add_op(&fake_quantize);
        agraph->add_op(&fake_dequantize);
        agraph->add_op(&mul);
    }

    agraph->add_op(&pre_softmax_reshape);
    agraph->add_op(&softmax);
    agraph->add_op(&softmax_reshape);

    agraph->add_op(&value_reshape);
    agraph->add_op(&value_transpose);
    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
    agraph->add_op(&reshape_output);
}

static inline void add_MHA_subgraph_alternative4(graph::graph_t *agraph,
        bool use_int8 = false, graph::dim_t batch_size = 16,
        graph::dim_t seq_len = 2048, graph::dim_t num_head = 12,
        graph::dim_t head_dim = 768) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    graph::dim_t size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> QV_TRANSPOSE_SHAPE {
            batch_size * num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> K_TRANSPOSE_SHAPE {
            batch_size * num_head, size_per_head, seq_len};
    std::vector<graph::dim_t> MATMUL_QK_SHAPE {
            batch_size * num_head, seq_len, seq_len};
    std::vector<graph::dim_t> MATMUL_QK_VIEWED_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<graph::dim_t> MATMUL_V_SHAPE {
            batch_size * num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> MASK_SHAPE {batch_size, 1, 1, seq_len};

    auto dtype = graph::data_type::f32;

    graph::logical_tensor_t query_dequantize_input, key_dequantize_input,
            value_dequantize_input;
    query_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QV_TRANSPOSE_SHAPE, graph::data_type::u8);
    key_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, K_TRANSPOSE_SHAPE, graph::data_type::u8);
    value_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QV_TRANSPOSE_SHAPE, graph::data_type::u8);

    graph::logical_tensor_t query_matmul_input, key_matmul_input,
            value_matmul_input;
    query_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QV_TRANSPOSE_SHAPE, dtype);
    key_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, K_TRANSPOSE_SHAPE, dtype);
    value_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QV_TRANSPOSE_SHAPE, dtype);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, dtype);

    graph::logical_tensor_t attention_mask_flt, max_in_flt;
    attention_mask_flt = utils::logical_tensor_init(
            logical_tensor_idx++, MASK_SHAPE, dtype);
    max_in_flt = utils::logical_tensor_init(
            logical_tensor_idx++, MASK_SHAPE, dtype);

    graph::logical_tensor_t fscore_reshape1_out, fscore_add_out, fscore_max_out,
            fscore_reshape2_out, softmax_out;
    fscore_reshape1_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_VIEWED_SHAPE, dtype);
    fscore_add_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_VIEWED_SHAPE, dtype);
    fscore_max_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_VIEWED_SHAPE, dtype);
    fscore_reshape2_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, dtype);
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, dtype);

    graph::logical_tensor_t softmax_quantize_out, softmax_dequantize_out;
    softmax_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, graph::data_type::u8);
    softmax_dequantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_SHAPE, dtype);

    graph::logical_tensor_t context_quantize_out;
    context_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_SHAPE, graph::data_type::u8);

    graph::op_t dequantize_query {
            op_idx++, graph::op_kind::Dequantize, "dequantize_query"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_query);
    graph::op_t dequantize_key {
            op_idx++, graph::op_kind::Dequantize, "dequantize_key"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_key);
    graph::op_t dequantize_value {
            op_idx++, graph::op_kind::Dequantize, "dequantize_value"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_value);

    graph::op_t matmul_qk {op_idx++, graph::op_kind::MatMul, "matmul_qk"};

    graph::op_t fscore_view1 {
            op_idx++, graph::op_kind::StaticReshape, "fscore_view1"};
    fscore_view1.set_attr(graph::op_attr::shape, MATMUL_QK_VIEWED_SHAPE);
    fscore_view1.set_attr(graph::op_attr::special_zero, false);
    graph::op_t fscore_add {op_idx++, graph::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t fscore_max {op_idx++, graph::op_kind::Maximum, "fscore_max"};
    fscore_max.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t fscore_view2 {
            op_idx++, graph::op_kind::StaticReshape, "fscore_view2"};
    fscore_view2.set_attr(graph::op_attr::shape, MATMUL_QK_SHAPE);
    fscore_view2.set_attr(graph::op_attr::special_zero, false);
    graph::op_t softmax {op_idx++, graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)2);

    graph::op_t quantize_softmax {
            op_idx++, graph::op_kind::Quantize, "quantize_softmax"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_softmax);
    graph::op_t dequantize_softmax {
            op_idx++, graph::op_kind::Dequantize, "dequantize_softmax"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_softmax);

    graph::op_t matmul_v {op_idx++, graph::op_kind::MatMul, "matmul_v"};

    graph::op_t quantize_output {
            op_idx++, graph::op_kind::Quantize, "quantize_output"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_output);

    if (use_int8) {
        dequantize_query.add_input(query_dequantize_input);
        dequantize_key.add_input(key_dequantize_input);
        dequantize_value.add_input(value_dequantize_input);
        dequantize_query.add_output(query_matmul_input);
        dequantize_key.add_output(key_matmul_input);
        dequantize_value.add_output(value_matmul_input);
    }

    matmul_qk.add_input(query_matmul_input);
    matmul_qk.add_input(key_matmul_input);
    matmul_qk.add_output(matmul_qk_out);

    fscore_view1.add_input(matmul_qk_out);
    fscore_view1.add_output(fscore_reshape1_out);
    fscore_add.add_input(fscore_reshape1_out);
    fscore_add.add_input(attention_mask_flt);
    fscore_add.add_output(fscore_add_out);
    fscore_max.add_input(fscore_add_out);
    fscore_max.add_input(max_in_flt);
    fscore_max.add_output(fscore_max_out);
    fscore_view2.add_input(fscore_max_out);
    fscore_view2.add_output(fscore_reshape2_out);
    softmax.add_input(fscore_reshape2_out);
    softmax.add_output(softmax_out);

    if (use_int8) {
        quantize_softmax.add_input(softmax_out);
        quantize_softmax.add_output(softmax_quantize_out);
        dequantize_softmax.add_input(softmax_quantize_out);
        dequantize_softmax.add_output(softmax_dequantize_out);
        matmul_v.add_input(softmax_dequantize_out);
    } else {
        matmul_v.add_input(softmax_out);
    }

    matmul_v.add_input(value_matmul_input);
    matmul_v.add_output(matmul_v_out);

    if (use_int8) {
        quantize_output.add_input(matmul_v_out);
        quantize_output.add_output(context_quantize_out);
    }

    if (use_int8) {
        agraph->add_op(&dequantize_query);
        agraph->add_op(&dequantize_key);
        agraph->add_op(&dequantize_value);
    }

    agraph->add_op(&matmul_qk);
    agraph->add_op(&fscore_view1);
    agraph->add_op(&fscore_add);
    agraph->add_op(&fscore_max);
    agraph->add_op(&fscore_view2);
    agraph->add_op(&softmax);

    if (use_int8) {
        agraph->add_op(&quantize_softmax);
        agraph->add_op(&dequantize_softmax);
    }

    agraph->add_op(&matmul_v);

    if (use_int8) { agraph->add_op(&quantize_output); }
}

// returned vector with shape {ic, ks, oc}
static std::vector<graph::dim_t> extract_filter_info(
        const dims &shape, const std::string &filter_format) {
    size_t ndims = shape.size();
    return filter_format == "OIX"
            ? std::vector<graph::dim_t> {shape[1], shape[2], shape[0]}
            : std::vector<graph::dim_t> {
                    shape[ndims - 2], shape[0], shape[ndims - 1]};
}

inline graph::logical_tensor_t create_dyn_dequantize(
        utils::id_generator &id_gen, graph_t &agraph,
        const graph::logical_tensor_t &src, const std::string &qtype,
        int64_t axis, graph::dim_t channel_size = 1) {
    graph::op_t dq_op(
            id_gen.get_id(), graph::op_kind::DynamicDequantize, "dequantize");
    dq_op.set_attr<std::string>(op_attr::qtype, qtype);
    dq_op.set_attr<int64_t>(op_attr::axis, axis);

    auto dst = utils::logical_tensor_init(id_gen.get_id(), data_type::f32);
    graph::logical_tensor_t scale_desc = utils::logical_tensor_init(
            id_gen.get_id(),
            qtype == "per_channel" ? std::vector<graph::dim_t> {channel_size}
                                   : std::vector<graph::dim_t> {1},
            data_type::f32);
    dq_op.add_input(src);
    dq_op.add_input(scale_desc);
    dq_op.add_output(dst);
    agraph.add_op(&dq_op);
    return dst;
}

inline graph::logical_tensor_t create_dyn_quantize(utils::id_generator &id_gen,
        graph_t &agraph, const graph::logical_tensor_t &src,
        data_type_t dst_dtype, const std::string &qtype, int64_t axis) {
    graph::op_t q_op(
            id_gen.get_id(), graph::op_kind::DynamicQuantize, "quantize");
    q_op.set_attr<std::string>(op_attr::qtype, qtype);
    q_op.set_attr<int64_t>(op_attr::axis, axis);

    auto dst = utils::logical_tensor_init(id_gen.get_id(), dst_dtype);
    graph::logical_tensor_t scale_desc
            = utils::logical_tensor_init(id_gen.get_id(), {1}, data_type::f32);
    q_op.add_input(src);
    q_op.add_input(scale_desc);
    q_op.add_output(dst);
    agraph.add_op(&q_op);
    return dst;
}

inline graph::logical_tensor_t create_int8_convolution_dyn_quant(
        utils::id_generator &id_gen, graph_t &agraph,
        const graph::logical_tensor_t &src, int64_t ic, int64_t ks, int64_t oc,
        int64_t groups, const dims &strides, const dims &dilations,
        const dims &pads_begin, const dims &pads_end,
        const std::string &data_format, const std::string &filter_format,
        bool with_bias, bool with_bn, float epsilon, bool with_relu,
        data_type_t dst_dtype, bool is_quantize_dst = true,
        bool use_biasadd = false, bool is_quantize_wei = false) {
    assertm(!with_bn, "int8 conv not support bn now");

    auto dq_src = create_dyn_dequantize(id_gen, agraph, src, "per_tensor", 0);

    graph::op_t conv(id_gen.get_id(), graph::op_kind::Convolution, "conv");
    conv.set_attr<int64_t>(op_attr::groups, groups);
    conv.set_attr<dims>(op_attr::strides, strides);
    conv.set_attr<dims>(op_attr::dilations, dilations);
    conv.set_attr<dims>(op_attr::pads_begin, pads_begin);
    conv.set_attr<dims>(op_attr::pads_end, pads_end);
    conv.set_attr<std::string>(op_attr::data_format, data_format);
    conv.set_attr<std::string>(op_attr::weights_format, filter_format);

    dims wei_shape = (filter_format == "OIX") ? dims {oc, ic / groups, ks, ks}
                                              : dims {ks, ks, ic / groups, oc};

    graph::logical_tensor_t int8_wei;
    if (is_quantize_wei) {
        auto f32_wei = utils::logical_tensor_init(
                id_gen.get_id(), wei_shape, data_type::f32);
        f32_wei.property = property_type::constant;
        int8_wei = create_dyn_quantize(id_gen, agraph, f32_wei, data_type::s8,
                "per_channel", (filter_format == "OIX") ? 0 : 3);
    } else {
        int8_wei = utils::logical_tensor_init(
                id_gen.get_id(), wei_shape, data_type::s8);
        int8_wei.property = property_type::constant;
    }
    int64_t channel_axis = (filter_format == "OIX") ? 0 : 3;
    auto dq_wei = create_dyn_dequantize(id_gen, agraph, int8_wei, "per_channel",
            channel_axis, wei_shape[channel_axis]);

    auto dst = utils::logical_tensor_init(id_gen.get_id(), dq_src.data_type);

    conv.add_input(dq_src);
    conv.add_input(dq_wei);
    if (with_bias && !use_biasadd) {
        auto bias = utils::logical_tensor_init(
                id_gen.get_id(), dims {oc}, dq_src.data_type);
        bias.property = property_type::constant;
        conv.add_input(bias);
    }
    conv.add_output(dst);
    agraph.add_op(&conv);

    if (with_bias && use_biasadd) {
        graph::op_t biasadd_op(
                id_gen.get_id(), graph::op_kind::BiasAdd, "biasadd");
        biasadd_op.set_attr<std::string>(op_attr::data_format, data_format);

        auto biasadd_src = dst;
        auto bias = utils::logical_tensor_init(
                id_gen.get_id(), dims {oc}, biasadd_src.data_type);
        bias.property = property_type::constant;

        dst = utils::logical_tensor_init(
                id_gen.get_id(), biasadd_src.data_type);

        biasadd_op.add_input(biasadd_src);
        biasadd_op.add_input(bias);
        biasadd_op.add_output(dst);

        agraph.add_op(&biasadd_op);
    }

    if (with_relu) {
        graph::op_t relu_op(id_gen.get_id(), graph::op_kind::ReLU, "relu");
        auto relu_src = dst;
        dst = utils::logical_tensor_init(id_gen.get_id(), dq_src.data_type);
        relu_op.add_input(relu_src);
        relu_op.add_output(dst);
        agraph.add_op(&relu_op);
    }

    if (is_quantize_dst) {
        dst = create_dyn_quantize(
                id_gen, agraph, dst, dst_dtype, "per_tensor", 0);
    }

    return dst;
}

// filter_shape in the order of 1x1; 1x1 + 3x3 + 1x1
// strides and paddings also start with the single conv branch
inline void construct_convolutional_bottleneck_resblock(graph::graph_t *agraph,
        utils::id_generator &id_gen, const dims &input_shape,
        const std::vector<dims> &filter_shapes, bool is_bf16 = false,
        const std::vector<std::vector<int64_t>> &strides
        = {{2, 2}, {1, 1}, {2, 2}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NCX",
        const std::string &filter_format = "OIX") {
    auto dtype = is_bf16 ? graph::data_type::bf16 : graph::data_type::f32;
    auto input
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);

    std::vector<std::vector<graph::dim_t>> filter_infos;
    for (auto shape : filter_shapes) {
        filter_infos.push_back(extract_filter_info(shape, filter_format));
    }
    auto left = create_convolution(id_gen, *agraph, input, filter_infos[0][0],
            filter_infos[0][1], filter_infos[0][2], 1, strides[0], {1, 1},
            paddings[0], paddings[0], data_format, filter_format, true, false,
            1e-6f, false);

    auto right = input;
    for (size_t i = 1; i < filter_shapes.size(); ++i) {
        right = utils::create_convolution(id_gen, *agraph, right,
                filter_infos[i][0], filter_infos[i][1], filter_infos[i][2], 1,
                strides[i], {1, 1}, paddings[i], paddings[i], data_format,
                filter_format, true, false, 1e-6f,
                i < filter_shapes.size() - 1);
    }
    auto add = create_add(id_gen, *agraph, left, right);
    auto relu3 = create_relu(id_gen, *agraph, add);
    (void)(relu3);
}

inline void construct_identical_bottleneck_resblock(graph::graph_t *agraph,
        utils::id_generator &id_gen, const dims &input_shape,
        const std::vector<dims> &filter_shapes, bool is_bf16 = false,
        const std::vector<std::vector<int64_t>> &strides
        = {{1, 1}, {1, 1}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NCX",
        const std::string &filter_format = "OIX") {
    auto dtype = is_bf16 ? graph::data_type::bf16 : graph::data_type::f32;
    auto input
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    std::vector<std::vector<graph::dim_t>> filter_infos;
    for (auto shape : filter_shapes) {
        filter_infos.push_back(extract_filter_info(shape, filter_format));
    }
    auto temp_input = input;
    for (size_t i = 0; i < filter_shapes.size(); ++i) {
        temp_input = utils::create_convolution(id_gen, *agraph, temp_input,
                filter_infos[i][0], filter_infos[i][1], filter_infos[i][2], 1,
                strides[i], {1, 1}, paddings[i], paddings[i], data_format,
                filter_format, true, false, 1e-6f,
                i < filter_shapes.size() - 1);
    }
    auto add = utils::create_add(id_gen, *agraph, temp_input, input);
    auto relu3 = utils::create_relu(id_gen, *agraph, add);
    (void)(relu3);
}

inline void construct_int8_identical_bottleneck_resblock(graph::graph_t *agraph,
        utils::id_generator &id_gen, const dims &input_shape,
        const std::vector<dims> &filter_shapes,
        const std::vector<std::vector<int64_t>> &strides
        = {{1, 1}, {1, 1}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NCX",
        const std::string &filter_format = "OIX", bool dyn_quant = false) {
    float scale_src = 1 / 255.f, scale_out = 1;
    int64_t zp_src = 0, zp_out = 78;

    auto src = utils::logical_tensor_init(
            id_gen.get_id(), input_shape, graph::data_type::u8);

    std::vector<std::vector<graph::dim_t>> filter_infos;
    for (auto shape : filter_shapes) {
        filter_infos.push_back(extract_filter_info(shape, filter_format));
    }
    auto temp_src = src;
    for (size_t i = 0; i < filter_shapes.size(); ++i) {
        if (dyn_quant) {
            temp_src = create_int8_convolution_dyn_quant(id_gen, *agraph,
                    temp_src, filter_infos[i][0], filter_infos[i][1],
                    filter_infos[i][2], 1, strides[i], {1, 1}, paddings[i],
                    paddings[i], data_format, filter_format, true, false, 1e-6f,
                    /*has_relu? */ i < filter_shapes.size() - 1,
                    graph::data_type::u8,
                    /*is_quantize_dst? */ i < filter_shapes.size() - 1);
        } else {
            std::vector<float> scale_wei(filter_infos[i][2], 1 / 127.f);
            temp_src = utils::create_int8_convolution(id_gen, *agraph, temp_src,
                    filter_infos[i][0], filter_infos[i][1], filter_infos[i][2],
                    1, strides[i], {1, 1}, paddings[i], paddings[i],
                    data_format, filter_format, true, false, 1e-6f,
                    /*has_relu? */ i < filter_shapes.size() - 1, scale_src,
                    zp_src, scale_out, zp_out, scale_wei, graph::data_type::u8,
                    /*is_quantize_dst? */ i < filter_shapes.size() - 1);
        }
    }
    auto dq3 = dyn_quant
            ? create_dyn_dequantize(id_gen, *agraph, src, "per_tensor", 0)
            : create_dequantize(id_gen, *agraph, src, "per_tensor", {zp_src},
                    {scale_src}, 0);
    auto add0 = create_add(id_gen, *agraph, temp_src, dq3);
    auto relu0 = create_relu(id_gen, *agraph, add0);
    auto q2 = dyn_quant
            ? create_dyn_quantize(id_gen, *agraph, relu0, graph::data_type::u8,
                    "per_tensor", 0)
            : create_quantize(id_gen, *agraph, relu0, graph::data_type::u8,
                    "per_tensor", std::vector<int64_t> {zp_out},
                    std::vector<float> {scale_out}, 0);
    (void)(q2);
}

// filter_shape in the order of 1x1; 1x1+3x3+1x1
// strides and paddings also first consider the single conv branch
inline void construct_int8_convolutional_bottleneck_resblock(
        graph::graph_t *agraph, utils::id_generator &id_gen,
        const dims &input_shape, const std::vector<dims> &filter_shapes,
        const std::vector<std::vector<int64_t>> &strides
        = {{2, 2}, {1, 1}, {2, 2}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NCX",
        const std::string &filter_format = "OIX", bool dyn_quant = false) {
    float scale_src = 1 / 255.f, scale_out = 1;
    int64_t zp_src = 0, zp_out = 78;
    auto src = utils::logical_tensor_init(
            id_gen.get_id(), input_shape, graph::data_type::u8);

    std::vector<std::vector<graph::dim_t>> filter_infos;
    for (auto shape : filter_shapes) {
        filter_infos.push_back(extract_filter_info(shape, filter_format));
    }
    std::vector<float> scale_wei(filter_infos[0][2], 1 / 127.f);
    graph::logical_tensor_t left;
    if (dyn_quant) {
        left = create_int8_convolution_dyn_quant(id_gen, *agraph, src,
                filter_infos[0][0], filter_infos[0][1], filter_infos[0][2], 1,
                strides[0], {1, 1}, paddings[0], paddings[0], data_format,
                filter_format, true, false, 1e-6f,
                /*no relu*/ false, graph::data_type::u8);
    } else {
        left = utils::create_int8_convolution(id_gen, *agraph, src,
                filter_infos[0][0], filter_infos[0][1], filter_infos[0][2], 1,
                strides[0], {1, 1}, paddings[0], paddings[0], data_format,
                filter_format, true, false, 1e-6f,
                /*no relu*/ false, scale_src, zp_src, scale_out, zp_out,
                scale_wei, graph::data_type::u8);
    }

    auto right = src;
    for (size_t i = 1; i < filter_shapes.size(); ++i) {
        if (dyn_quant) {
            right = create_int8_convolution_dyn_quant(id_gen, *agraph, right,
                    filter_infos[i][0], filter_infos[i][1], filter_infos[i][2],
                    1, strides[i], {1, 1}, paddings[i], paddings[i],
                    data_format, filter_format, true, false, 1e-6f,
                    /*has relu*/ i < filter_shapes.size() - 1,
                    graph::data_type::u8,
                    /*is_quantize_dst? */ i < filter_shapes.size() - 1);
        } else {
            scale_wei = std::vector<float>(filter_infos[i][2], 1 / 127.f);
            right = utils::create_int8_convolution(id_gen, *agraph, right,
                    filter_infos[i][0], filter_infos[i][1], filter_infos[i][2],
                    1, strides[i], {1, 1}, paddings[i], paddings[i],
                    data_format, filter_format, true, false, 1e-6f,
                    /*has relu*/ i < filter_shapes.size() - 1, scale_src,
                    zp_src, scale_out, zp_out, scale_wei, graph::data_type::u8,
                    /*is_quantize_dst? */ i < filter_shapes.size() - 1);
        }
    }
    auto dq3 = dyn_quant
            ? create_dyn_dequantize(id_gen, *agraph, left, "per_tensor", 0)
            : create_dequantize(id_gen, *agraph, left, "per_tensor", {zp_src},
                    {scale_src}, 0);
    auto add0 = create_add(id_gen, *agraph, right, dq3);
    auto relu0 = create_relu(id_gen, *agraph, add0);
    auto q2 = dyn_quant
            ? create_dyn_quantize(id_gen, *agraph, relu0, graph::data_type::u8,
                    "per_tensor", 0)
            : create_quantize(id_gen, *agraph, relu0, graph::data_type::u8,
                    "per_tensor", std::vector<int64_t> {zp_out},
                    std::vector<float> {scale_out}, 0);
    (void)(q2);
}

inline graph::logical_tensor_t create_relu_bwd(utils::id_generator &id_gen,
        graph::graph_t &agraph, const graph::logical_tensor_t &dst,
        const graph::logical_tensor_t &delta_in, const dims &dims) {
    graph::op_t relu_bwd(
            id_gen.get_id(), graph::op_kind::ReLUBackward, "relu_bwd");
    auto delta
            = utils::logical_tensor_init(id_gen.get_id(), dims, dst.data_type);
    relu_bwd.add_input(dst);
    relu_bwd.add_input(delta);
    relu_bwd.add_output(delta_in);
    agraph.add_op(&relu_bwd);
    return delta;
}

inline graph::logical_tensor_t create_relu_bwd2(utils::id_generator &id_gen,
        graph::graph_t &agraph, const graph::logical_tensor_t &dst,
        const graph::logical_tensor_t &delta) {
    graph::op_t relu_bwd(
            id_gen.get_id(), graph::op_kind::ReLUBackward, "relu_bwd");
    auto output = utils::logical_tensor_init(id_gen.get_id(), dst.data_type);
    relu_bwd.add_input(dst);
    relu_bwd.add_input(delta);
    relu_bwd.add_output(output);
    agraph.add_op(&relu_bwd);
    return output;
}

// create conv training fwd and bwd together since the connection is complex
// src is the src of the first conv_fwd
// delta_in is the destination of the last conv_bwd
// (which is then connected to the previous subgraph)
/*
        (src)           (delta_in)
          |                 |
         conv          conv_data_bwd
          |                 |
          bn              bn_bwd
          |                 |
         relu            relu_bwd
          |                 |
        (dst)         (delta_in_next)
*/
inline std::vector<graph::logical_tensor_t> create_convolution_training(
        utils::id_generator &id_gen, graph::graph_t &agraph,
        const graph::logical_tensor_t &src, graph::logical_tensor_t &delta_in,
        const dims &filter_shape, const dims &strides, const dims &pads_begin,
        const dims &pads_end, const std::string &data_format,
        const std::string &filter_format, bool with_bn = false,
        float epsilon = 1e-6f, bool with_momentum = false,
        float momentum = 0.0f, bool with_relu = false,
        graph::logical_tensor_t *known_delta_in_next = nullptr,
        bool is_bn_bf16 = false) {
    auto bn_dtype = is_bn_bf16 ? graph::data_type::bf16 : graph::data_type::f32;
    graph::op_t conv_fwd(
            id_gen.get_id(), graph::op_kind::Convolution, "conv_fwd");
    conv_fwd.set_attr<dims>(graph::op_attr::strides, strides);
    conv_fwd.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    if (std::all_of(strides.begin(), strides.end(),
                [](size_t x) { return x == 1; })) {
        conv_fwd.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
        conv_fwd.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
        conv_fwd.set_attr<std::string>(graph::op_attr::auto_pad, "SAME_UPPER");
    } else {
        conv_fwd.set_attr<dims>(graph::op_attr::pads_begin, pads_begin);
        conv_fwd.set_attr<dims>(graph::op_attr::pads_end, pads_end);
    }
    conv_fwd.set_attr<std::string>(graph::op_attr::data_format, data_format);
    conv_fwd.set_attr<std::string>(
            graph::op_attr::weights_format, filter_format);

    graph::dim_t ih = (data_format == "NCX") ? src.dims[2] : src.dims[1];
    graph::dim_t iw = (data_format == "NCX") ? src.dims[3] : src.dims[2];
    graph::dim_t ks
            = (filter_format == "OIX") ? filter_shape[2] : filter_shape[0];
    graph::dim_t ic
            = (filter_format == "OIX") ? filter_shape[1] : filter_shape[2];
    graph::dim_t oc
            = (filter_format == "OIX") ? filter_shape[0] : filter_shape[3];
    graph::dim_t oh = (ih + pads_begin[0] + pads_end[0] - ks) / strides[0] + 1;
    graph::dim_t ow = (iw + pads_begin[1] + pads_end[1] - ks) / strides[1] + 1;

    dims src_shape = (data_format == "NCX") ? dims {src.dims[0], ic, ih, iw}
                                            : dims {src.dims[0], ih, iw, ic};
    dims dst_shape = (data_format == "NCX") ? dims {src.dims[0], oc, oh, ow}
                                            : dims {src.dims[0], oh, ow, oc};

    auto wei = utils::logical_tensor_init(
            id_gen.get_id(), filter_shape, src.data_type);
    auto dst = utils::logical_tensor_init(
            id_gen.get_id(), dst_shape, src.data_type);

    conv_fwd.add_input(src);
    conv_fwd.add_input(wei);
    conv_fwd.add_output(dst);
    agraph.add_op(&conv_fwd);

    // gamma, mean, var will be reused in bwd part
    graph::logical_tensor_t bn_src, gamma, mean, var;
    if (with_bn) {
        graph::op_t bn_fwd(id_gen.get_id(),
                graph::op_kind::BatchNormForwardTraining, "bn_fwd");
        bn_fwd.set_attr<std::string>(graph::op_attr::data_format, data_format);
        bn_fwd.set_attr<float>(graph::op_attr::epsilon, epsilon);
        if (with_momentum) {
            bn_fwd.set_attr<float>(graph::op_attr::momentum, momentum);
        }

        int64_t bn_ic = oc;

        bn_src = dst;
        gamma = utils::logical_tensor_init(id_gen.get_id(), {bn_ic}, bn_dtype);
        auto beta = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, bn_dtype);
        mean = utils::logical_tensor_init(id_gen.get_id(), {bn_ic}, bn_dtype);
        var = utils::logical_tensor_init(id_gen.get_id(), {bn_ic}, bn_dtype);

        dst = utils::logical_tensor_init(
                id_gen.get_id(), dst_shape, src.data_type);
        auto running_mean = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, bn_dtype);
        auto running_variance = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, bn_dtype);
        auto batch_mean = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, bn_dtype);
        auto batch_variance = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, bn_dtype);

        bn_fwd.add_input(bn_src);
        bn_fwd.add_input(mean);
        bn_fwd.add_input(var);
        bn_fwd.add_input(gamma);
        bn_fwd.add_input(beta);
        bn_fwd.add_output(dst);
        bn_fwd.add_output(running_mean);
        bn_fwd.add_output(running_variance);
        bn_fwd.add_output(batch_mean);
        bn_fwd.add_output(batch_variance);

        agraph.add_op(&bn_fwd);
    }
    if (with_relu) {
        // fwd relu
        graph::op_t relu_fwd(id_gen.get_id(), graph::op_kind::ReLU, "relu_fwd");
        auto relu_src = dst;
        dst = utils::logical_tensor_init(
                id_gen.get_id(), dst_shape, src.data_type);
        relu_fwd.add_input(relu_src);
        relu_fwd.add_output(dst);
        agraph.add_op(&relu_fwd);
    }

    // start constructing bwd part
    graph::logical_tensor_t output_delta = utils::logical_tensor_init(
            id_gen.get_id(), dst_shape, src.data_type);
    graph::logical_tensor_t delta_in_next;
    if (known_delta_in_next) {
        output_delta = *known_delta_in_next;
        delta_in_next = *known_delta_in_next;
    } else {
        delta_in_next = output_delta;
    }
    if (with_relu) {
        // bwd relu
        graph::op_t relu_bwd(
                id_gen.get_id(), graph::op_kind::ReLUBackward, "relu_bwd");
        relu_bwd.set_attr<bool>(graph::op_attr::use_dst, true);
        relu_bwd.add_input(dst); // dst is dst of fwd
        relu_bwd.add_input(output_delta);
        output_delta = utils::logical_tensor_init(
                id_gen.get_id(), dst_shape, src.data_type);
        relu_bwd.add_output(output_delta);
        agraph.add_op(&relu_bwd);
    }
    if (with_bn) {
        graph::op_t bn_bwd(id_gen.get_id(),
                graph::op_kind::BatchNormTrainingBackward, "bn_bwd");
        bn_bwd.set_attr<std::string>(graph::op_attr::data_format, data_format);
        bn_bwd.set_attr<float>(graph::op_attr::epsilon, epsilon);

        int64_t bn_ic = oc;
        auto gamma_delta = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, bn_dtype);
        auto beta_delta = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, bn_dtype);

        bn_bwd.add_input(bn_src);
        bn_bwd.add_input(output_delta);
        bn_bwd.add_input(mean);
        bn_bwd.add_input(var);
        bn_bwd.add_input(gamma);
        output_delta = utils::logical_tensor_init(
                id_gen.get_id(), dst_shape, src.data_type);
        bn_bwd.add_output(output_delta);
        bn_bwd.add_output(gamma_delta);
        bn_bwd.add_output(beta_delta);

        agraph.add_op(&bn_bwd);
    }

    graph::op_t conv_bwd_data(id_gen.get_id(),
            graph::op_kind::ConvolutionBackwardData, "conv_bwd_data");
    conv_bwd_data.set_attr<dims>(graph::op_attr::strides, strides);
    conv_bwd_data.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    if (std::all_of(strides.begin(), strides.end(),
                [](size_t x) { return x == 1; })) {
        conv_bwd_data.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
        conv_bwd_data.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
        conv_bwd_data.set_attr<std::string>(
                graph::op_attr::auto_pad, "SAME_UPPER");
    } else {
        conv_bwd_data.set_attr<dims>(graph::op_attr::pads_begin, pads_begin);
        conv_bwd_data.set_attr<dims>(graph::op_attr::pads_end, pads_end);
    }
    conv_bwd_data.set_attr<std::string>(
            graph::op_attr::data_format, data_format);
    conv_bwd_data.set_attr<std::string>(
            graph::op_attr::weights_format, filter_format);
    conv_bwd_data.set_attr<dims>(graph::op_attr::dst_shape, src_shape);
    conv_bwd_data.add_input(output_delta);
    conv_bwd_data.add_input(wei);
    conv_bwd_data.add_output(delta_in);
    agraph.add_op(&conv_bwd_data);

    graph::op_t conv_bwd_weight(id_gen.get_id(),
            graph::op_kind::ConvolutionBackwardWeights, "conv_bwd_weight");
    conv_bwd_weight.set_attr<dims>(graph::op_attr::strides, strides);
    conv_bwd_weight.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    if (std::all_of(strides.begin(), strides.end(),
                [](size_t x) { return x == 1; })) {
        conv_bwd_weight.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
        conv_bwd_weight.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
        conv_bwd_weight.set_attr<std::string>(
                graph::op_attr::auto_pad, "SAME_UPPER");
    } else {
        conv_bwd_weight.set_attr<dims>(graph::op_attr::pads_begin, pads_begin);
        conv_bwd_weight.set_attr<dims>(graph::op_attr::pads_end, pads_end);
    }
    conv_bwd_weight.set_attr<std::string>(
            graph::op_attr::data_format, data_format);
    conv_bwd_weight.set_attr<std::string>(
            graph::op_attr::weights_format, filter_format);
    conv_bwd_weight.set_attr<dims>(graph::op_attr::weights_shape, filter_shape);
    conv_bwd_weight.add_input(src);
    conv_bwd_weight.add_input(output_delta);
    auto delta_weight = utils::logical_tensor_init(
            id_gen.get_id(), filter_shape, src.data_type);
    conv_bwd_weight.add_output(delta_weight);
    agraph.add_op(&conv_bwd_weight);

    return {dst, delta_in_next};
}

inline void construct_identical_bottleneck_training_subgraph(
        graph::graph_t *agraph, utils::id_generator &id_gen,
        const dims &input_shape, const std::vector<dims> &filter_shapes,
        bool is_bf16 = false, bool is_bn_bf16 = false,
        const std::vector<std::vector<int64_t>> &strides
        = {{1, 1}, {1, 1}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NXC",
        const std::string &filter_format = "XIO") {
    auto dtype = is_bf16 ? graph::data_type::bf16 : graph::data_type::f32;
    auto src = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto delta_data
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto ret0 = create_convolution_training(id_gen, *agraph, src, delta_data,
            filter_shapes[0], strides[0], paddings[0], paddings[0], data_format,
            filter_format, true, 1e-5f, true, 0.1f, true, nullptr, is_bn_bf16);
    auto conv0 = ret0[0];
    auto delta_in0 = ret0[1];
    auto ret1 = create_convolution_training(id_gen, *agraph, conv0, delta_in0,
            filter_shapes[1], strides[1], paddings[1], paddings[1], data_format,
            filter_format, true, 1e-5f, true, 0.1f, true, nullptr, is_bn_bf16);
    auto conv1 = ret1[0];
    auto delta_in1 = ret1[1];
    auto ret2 = create_convolution_training(id_gen, *agraph, conv1, delta_in1,
            filter_shapes[2], strides[2], paddings[2], paddings[2], data_format,
            filter_format, true, 1e-5f, true, 0.1f, false, nullptr, is_bn_bf16);
    auto conv2 = ret2[0];
    auto delta_in2 = ret2[1];
    auto add = utils::create_add(id_gen, *agraph, src, conv2);
    auto relu = utils::create_relu(id_gen, *agraph, add);
    // relu_bwd is delta of forward_output
    auto relu_bwd
            = create_relu_bwd(id_gen, *agraph, relu, delta_in2, input_shape);
    auto add_bwd = utils::create_add(id_gen, *agraph, delta_in2, delta_data);
    auto relu_bwd_last_src
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto relu_bwd_last
            = create_relu_bwd2(id_gen, *agraph, relu_bwd_last_src, add_bwd);
    (void)(add_bwd);
    (void)(relu_bwd);
    (void)(relu_bwd_last);
}

// filter is in order {rhs, lhs0, lhs1, lhs2}
inline void construct_convolutional_bottleneck_training_subgraph(
        graph::graph_t *agraph, utils::id_generator &id_gen,
        const dims &input_shape, const std::vector<dims> &filter_shapes,
        bool is_bf16 = false, bool is_bn_bf16 = false,
        const std::vector<std::vector<int64_t>> &strides
        = {{2, 2}, {1, 1}, {2, 2}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NXC",
        const std::string &filter_format = "XIO") {
    auto infer_output_shape = [&](dims shape) {
        if (data_format == "NCX")
            return dims {
                    shape[0], filter_shapes[0][0], shape[2] / 2, shape[3] / 2};
        else
            return dims {
                    shape[0], shape[1] / 2, shape[2] / 2, filter_shapes[0][3]};
    };

    auto dtype = is_bf16 ? graph::data_type::bf16 : graph::data_type::f32;
    auto src = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto delta_data_lhs
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto delta_data_rhs
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto rhs = create_convolution_training(id_gen, *agraph, src, delta_data_rhs,
            filter_shapes[0], strides[0], paddings[0], paddings[0], data_format,
            filter_format, true, 1e-5f, true, 0.1f, false, nullptr, is_bn_bf16);
    auto conv_rhs0 = rhs[0];
    auto delta_in_rhs0 = rhs[1];
    auto lhs0 = create_convolution_training(id_gen, *agraph, src,
            delta_data_lhs, filter_shapes[1], strides[1], paddings[1],
            paddings[1], data_format, filter_format, true, 1e-5f, true, 0.1f,
            true, nullptr, is_bn_bf16);
    auto conv_lhs0 = lhs0[0];
    auto delta_in_lhs0 = lhs0[1];
    auto lhs1 = create_convolution_training(id_gen, *agraph, conv_lhs0,
            delta_in_lhs0, filter_shapes[2], strides[2], paddings[2],
            paddings[2], data_format, filter_format, true, 1e-5f, true, 0.1f,
            true, nullptr, is_bn_bf16);
    auto conv_lhs1 = lhs1[0];
    auto delta_in_lhs1 = lhs1[1];
    auto lhs2 = create_convolution_training(id_gen, *agraph, conv_lhs1,
            delta_in_lhs1, filter_shapes[3], strides[3], paddings[3],
            paddings[3], data_format, filter_format, true, 1e-5f, true, 0.1f,
            false, &delta_in_rhs0, is_bn_bf16);
    auto conv_lhs2 = lhs2[0];
    auto delta_in_lhs2 = lhs2[1];
    auto add = utils::create_add(id_gen, *agraph, conv_rhs0, conv_lhs2);
    auto relu = utils::create_relu(id_gen, *agraph, add);
    auto relu_bwd = create_relu_bwd(id_gen, *agraph, relu, delta_in_rhs0,
            infer_output_shape(input_shape));
    auto add_bwd = utils::create_add(
            id_gen, *agraph, delta_data_lhs, delta_data_rhs);
    auto relu_bwd_last_src
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto relu_bwd_last
            = create_relu_bwd2(id_gen, *agraph, relu_bwd_last_src, add_bwd);
    (void)(delta_in_lhs2);
    (void)(add_bwd);
    (void)(relu_bwd);
    (void)(relu_bwd_last);
}

inline void add_bart_MHA(graph::graph_t *agraph, bool use_bf16 = false,
        bool use_int8 = false, graph::dim_t batch_size = 1,
        graph::dim_t seq_len = 6, graph::dim_t num_head = 12,
        graph::dim_t head_dim = 768) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    graph::dim_t batch_dim = batch_size * num_head;
    graph::dim_t size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> QKV_TRANSPOSED_SHAPE {
            batch_dim, seq_len, size_per_head};
    std::vector<graph::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_dim, size_per_head, seq_len};
    std::vector<graph::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_dim, seq_len, seq_len};
    std::vector<graph::dim_t> MATMUL_V_OUTPUT_SHAPE {
            batch_dim, seq_len, size_per_head};
    auto dtype = use_bf16 ? graph::data_type::bf16 : graph::data_type::f32;

    graph::logical_tensor_t query_dequantize_input, key_dequantize_input,
            value_dequantize_input;
    query_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::u8);
    key_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, graph::data_type::u8);
    value_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::u8);

    graph::logical_tensor_t query_typecast_input, key_typecast_input,
            value_typecast_input;
    query_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::f32);
    key_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, graph::data_type::f32);
    value_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t query_matmul_input, key_matmul_input,
            value_matmul_input;
    query_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);
    key_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, dtype);
    value_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t softmax_out;
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t softmax_cast_out;
    softmax_cast_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t softmax_quantize_out;
    softmax_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, graph::data_type::u8);
    graph::logical_tensor_t softmax_dequantize_out;
    softmax_dequantize_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE, graph::data_type::f32);

    graph::logical_tensor_t softmax_dequantize_out_cast;
    softmax_dequantize_out_cast
            = utils::logical_tensor_init(logical_tensor_idx++,
                    MATMUL_QK_OUTPUT_SHAPE, graph::data_type::bf16);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    graph::logical_tensor_t context_cast_out;
    context_cast_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::f32);

    graph::logical_tensor_t context_quantize_out;
    context_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, graph::data_type::u8);

    graph::op_t dequantize_query {
            op_idx++, graph::op_kind::Dequantize, "dequantize_query"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_query);
    graph::op_t dequantize_key {
            op_idx++, graph::op_kind::Dequantize, "dequantize_key"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_key);
    graph::op_t dequantize_value {
            op_idx++, graph::op_kind::Dequantize, "dequantize_value"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_value);
    graph::op_t typecast_query {
            op_idx++, graph::op_kind::TypeCast, "typecast_query"};
    graph::op_t typecast_key {
            op_idx++, graph::op_kind::TypeCast, "typecast_key"};
    graph::op_t typecast_value {
            op_idx++, graph::op_kind::TypeCast, "typecast_value"};

    graph::op_t matmul_qk {op_idx++, graph::op_kind::MatMul, "matmul_qk"};
    graph::op_t softmax {op_idx++, graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)2);

    graph::op_t softmax_cast {
            op_idx++, graph::op_kind::TypeCast, "softmax_cast"};
    graph::op_t quantize_softmax {
            op_idx++, graph::op_kind::Quantize, "quantize_softmax"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_softmax);
    graph::op_t dequantize_softmax {
            op_idx++, graph::op_kind::Dequantize, "dequantize_softmax"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_softmax);

    graph::op_t dequantize_softmax_cast {
            op_idx++, graph::op_kind::TypeCast, "dequantize_softmax_cast"};

    graph::op_t matmul_v {op_idx++, graph::op_kind::MatMul, "matmul_v"};
    graph::op_t typecast_output {
            op_idx++, graph::op_kind::TypeCast, "typecast_output"};
    graph::op_t quantize_output {
            op_idx++, graph::op_kind::Quantize, "quantize_output"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_output);

    if (use_int8) {
        dequantize_query.add_input(query_dequantize_input);
        dequantize_key.add_input(key_dequantize_input);
        dequantize_value.add_input(value_dequantize_input);
        if (!use_bf16) {
            dequantize_query.add_output(query_matmul_input);
            dequantize_key.add_output(key_matmul_input);
            dequantize_value.add_output(value_matmul_input);
        } else {
            dequantize_query.add_output(query_typecast_input);
            dequantize_key.add_output(key_typecast_input);
            dequantize_value.add_output(value_typecast_input);
            typecast_query.add_input(query_typecast_input);
            typecast_key.add_input(key_typecast_input);
            typecast_value.add_input(value_typecast_input);
            typecast_query.add_output(query_matmul_input);
            typecast_key.add_output(key_matmul_input);
            typecast_value.add_output(value_matmul_input);
        }
    }
    matmul_qk.add_input(query_matmul_input);
    matmul_qk.add_input(key_matmul_input);
    matmul_qk.add_output(matmul_qk_out);
    softmax.add_input(matmul_qk_out);
    softmax.add_output(softmax_out);
    if (use_int8) {
        quantize_softmax.add_output(softmax_quantize_out);
        dequantize_softmax.add_input(softmax_quantize_out);
        dequantize_softmax.add_output(softmax_dequantize_out);
        if (!use_bf16) {
            quantize_softmax.add_input(softmax_out);
            matmul_v.add_input(softmax_dequantize_out);
        } else {
            softmax_cast.add_input(softmax_out);
            softmax_cast.add_output(softmax_cast_out);
            quantize_softmax.add_input(softmax_cast_out);
            dequantize_softmax_cast.add_input(softmax_dequantize_out);
            dequantize_softmax_cast.add_output(softmax_dequantize_out_cast);
            matmul_v.add_input(softmax_dequantize_out_cast);
        }
    } else {
        matmul_v.add_input(softmax_out);
    }
    matmul_v.add_input(value_matmul_input);
    matmul_v.add_output(matmul_v_out);
    if (use_int8) {
        quantize_output.add_output(context_quantize_out);
        if (!use_bf16) {
            quantize_output.add_input(matmul_v_out);
        } else {
            typecast_output.add_input(matmul_v_out);
            typecast_output.add_output(context_cast_out);
            quantize_output.add_input(context_cast_out);
        }
    }

    if (use_int8) {
        agraph->add_op(&dequantize_query);
        agraph->add_op(&dequantize_key);
        agraph->add_op(&dequantize_value);
        if (use_bf16) {
            agraph->add_op(&typecast_query);
            agraph->add_op(&typecast_key);
            agraph->add_op(&typecast_value);
        }
    }
    agraph->add_op(&matmul_qk);
    agraph->add_op(&softmax);
    if (use_int8) {
        agraph->add_op(&quantize_softmax);
        agraph->add_op(&dequantize_softmax);
        if (use_bf16) {
            agraph->add_op(&softmax_cast);
            agraph->add_op(&dequantize_softmax_cast);
        }
    }
    agraph->add_op(&matmul_v);
    if (use_int8) {
        agraph->add_op(&quantize_output);
        if (use_bf16) { agraph->add_op(&typecast_output); }
    }
}

inline void add_bart_mlp_residual_subgraph(graph::graph_t *agraph,
        bool use_bf16 = false, bool use_int8 = false,
        graph::dim_t batch_size = 1, graph::dim_t seq_len = 17) {
    size_t lt_idx = 0;
    size_t op_idx = 0;
    const graph::dim_t head_dim = 768;
    auto dtype = use_bf16 ? graph::data_type::bf16 : graph::data_type::f32;
    std::vector<graph::dim_t> input_size_1 {batch_size, seq_len, head_dim};
    std::vector<graph::dim_t> input_size_2 {batch_size, seq_len, head_dim};
    std::vector<graph::dim_t> input_size_3 {batch_size, seq_len, head_dim * 4};
    std::vector<graph::dim_t> output_size {batch_size, seq_len, head_dim};
    std::vector<graph::dim_t> weight_size_1 {head_dim, head_dim};
    std::vector<graph::dim_t> weight_size_2 {head_dim * 4, head_dim};
    std::vector<graph::dim_t> weight_size_3 {head_dim, head_dim * 4};
    std::vector<graph::dim_t> bias_size_1 {head_dim};
    std::vector<graph::dim_t> bias_size_2 {head_dim * 4};
    std::vector<graph::dim_t> bias_size_3 {head_dim};

    // layer1
    graph::logical_tensor_t input_desc_1 = utils::logical_tensor_init(
            lt_idx++, input_size_1, graph::data_type::u8);
    graph::logical_tensor_t dequant_input_desc_1 = utils::logical_tensor_init(
            lt_idx++, input_size_1, graph::data_type::f32);
    graph::logical_tensor_t typecast_input_desc_1
            = utils::logical_tensor_init(lt_idx++, input_size_1, dtype);

    graph::logical_tensor_t weight_desc_1 = utils::logical_tensor_init(
            lt_idx++, weight_size_1, graph::data_type::s8);
    weight_desc_1.property = property_type::constant;
    graph::logical_tensor_t dequant_weight_desc_1 = utils::logical_tensor_init(
            lt_idx++, weight_size_1, graph::data_type::f32);
    graph::logical_tensor_t typecast_weight_desc_1
            = utils::logical_tensor_init(lt_idx++, weight_size_1, dtype);
    graph::logical_tensor_t bias_desc_1
            = utils::logical_tensor_init(lt_idx++, bias_size_1, dtype);
    bias_desc_1.property = property_type::constant;

    graph::logical_tensor_t matmul_desc_1
            = utils::logical_tensor_init(lt_idx++, input_size_1, dtype);
    graph::logical_tensor_t add_other_input_desc
            = utils::logical_tensor_init(lt_idx++, input_size_1, dtype);
    graph::logical_tensor_t add_desc_1
            = utils::logical_tensor_init(lt_idx++, input_size_1, dtype);
    graph::logical_tensor_t layernorm_alpha_desc_1 = utils::logical_tensor_init(
            lt_idx++, bias_size_1, graph::data_type::f32);
    graph::logical_tensor_t layernorm_beta_desc_1 = utils::logical_tensor_init(
            lt_idx++, bias_size_1, graph::data_type::f32);
    graph::logical_tensor_t layernorm_desc_1
            = utils::logical_tensor_init(lt_idx++, input_size_1, dtype);
    graph::logical_tensor_t typecast_output_1 = utils::logical_tensor_init(
            lt_idx++, input_size_1, graph::data_type::f32);

    // layer2
    graph::logical_tensor_t quant_input_desc_2 = utils::logical_tensor_init(
            lt_idx++, input_size_2, graph::data_type::u8);
    graph::logical_tensor_t dequant_input_desc_2 = utils::logical_tensor_init(
            lt_idx++, input_size_2, graph::data_type::f32);
    graph::logical_tensor_t typecast_input_desc_2
            = utils::logical_tensor_init(lt_idx++, input_size_2, dtype);

    graph::logical_tensor_t weight_desc_2 = utils::logical_tensor_init(
            lt_idx++, weight_size_2, graph::data_type::s8);
    weight_desc_2.property = property_type::constant;
    graph::logical_tensor_t dequant_weight_desc_2 = utils::logical_tensor_init(
            lt_idx++, weight_size_2, graph::data_type::f32);
    graph::logical_tensor_t typecast_weight_desc_2
            = utils::logical_tensor_init(lt_idx++, weight_size_2, dtype);
    graph::logical_tensor_t bias_desc_2
            = utils::logical_tensor_init(lt_idx++, bias_size_2, dtype);
    bias_desc_2.property = property_type::constant;

    graph::logical_tensor_t matmul_desc_2
            = utils::logical_tensor_init(lt_idx++, input_size_3, dtype);
    graph::logical_tensor_t gelu_desc
            = utils::logical_tensor_init(lt_idx++, input_size_3, dtype);
    graph::logical_tensor_t typecast_output_2 = utils::logical_tensor_init(
            lt_idx++, input_size_3, graph::data_type::f32);

    // layer 3
    graph::logical_tensor_t quant_input_desc_3 = utils::logical_tensor_init(
            lt_idx++, input_size_3, graph::data_type::u8);
    graph::logical_tensor_t dequant_input_desc_3 = utils::logical_tensor_init(
            lt_idx++, input_size_3, graph::data_type::f32);
    graph::logical_tensor_t typecast_input_desc_3
            = utils::logical_tensor_init(lt_idx++, input_size_3, dtype);

    graph::logical_tensor_t weight_desc_3 = utils::logical_tensor_init(
            lt_idx++, weight_size_3, graph::data_type::s8);
    weight_desc_3.property = property_type::constant;
    graph::logical_tensor_t dequant_weight_desc_3 = utils::logical_tensor_init(
            lt_idx++, weight_size_3, graph::data_type::f32);
    graph::logical_tensor_t typecast_weight_desc_3
            = utils::logical_tensor_init(lt_idx++, weight_size_3, dtype);
    graph::logical_tensor_t bias_desc_3
            = utils::logical_tensor_init(lt_idx++, bias_size_3, dtype);
    bias_desc_3.property = property_type::constant;

    graph::logical_tensor_t matmul_desc_3
            = utils::logical_tensor_init(lt_idx++, output_size, dtype);
    graph::logical_tensor_t add_desc_2
            = utils::logical_tensor_init(lt_idx++, output_size, dtype);
    graph::logical_tensor_t layernorm_alpha_desc_2 = utils::logical_tensor_init(
            lt_idx++, bias_size_3, graph::data_type::f32);
    graph::logical_tensor_t layernorm_beta_desc_2 = utils::logical_tensor_init(
            lt_idx++, bias_size_3, graph::data_type::f32);
    graph::logical_tensor_t layernorm_desc_2
            = utils::logical_tensor_init(lt_idx++, output_size, dtype);

    // construct op
    graph::op_t dequant_input_1 {
            op_idx++, graph::op_kind::Dequantize, "dequantize_input_1"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequant_input_1);
    graph::op_t typecast_input_1 {
            op_idx++, graph::op_kind::TypeCast, "typecast_input_1"};

    graph::op_t dequant_weight_1 {
            op_idx++, graph::op_kind::Dequantize, "dequantize_weight_1"};
    DEFINE_DEFAULT_PER_CHANNEL_QUANT_ATTR(dequant_weight_1, bias_size_1[0], 0);
    graph::op_t typecast_weight_1 {
            op_idx++, graph::op_kind::TypeCast, "typecast_weight_1"};

    graph::op_t matmul_1 {op_idx++, graph::op_kind::MatMul, "matmul_1"};
    matmul_1.set_attr(graph::op_attr::transpose_b, true);
    graph::op_t add_1 {op_idx++, graph::op_kind::Add, "add_1"};
    graph::op_t layernorm_1 {
            op_idx++, graph::op_kind::LayerNorm, "layernorm_1"};
    layernorm_1.set_attr(graph::op_attr::keep_stats, false);
    graph::op_t cast_output_1 {
            op_idx++, graph::op_kind::TypeCast, "cast_output_1"};
    graph::op_t quant_input_2 {
            op_idx++, graph::op_kind::Quantize, "quantize_input_2"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quant_input_2);

    graph::op_t dequant_input_2 {
            op_idx++, graph::op_kind::Dequantize, "dequantize_input_2"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequant_input_2);
    graph::op_t typecast_input_2 {
            op_idx++, graph::op_kind::TypeCast, "typecast_input_2"};

    graph::op_t dequant_weight_2 {
            op_idx++, graph::op_kind::Dequantize, "dequantize_weight_2"};
    DEFINE_DEFAULT_PER_CHANNEL_QUANT_ATTR(dequant_weight_2, bias_size_2[0], 0);
    graph::op_t typecast_weight_2 {
            op_idx++, graph::op_kind::TypeCast, "typecast_weight_2"};

    graph::op_t matmul_2 {op_idx++, graph::op_kind::MatMul, "matmul_2"};
    matmul_2.set_attr(graph::op_attr::transpose_b, true);
    graph::op_t gelu_1 {op_idx++, graph::op_kind::GELU, "gelu_1"};
    graph::op_t cast_output_2 {
            op_idx++, graph::op_kind::TypeCast, "cast_output_2"};
    graph::op_t quant_input_3 {
            op_idx++, graph::op_kind::Quantize, "quantize_input_3"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quant_input_3);

    graph::op_t dequant_input_3 {
            op_idx++, graph::op_kind::Dequantize, "dequantize_input_3"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequant_input_3);
    graph::op_t typecast_input_3 {
            op_idx++, graph::op_kind::TypeCast, "typecast_input_3"};

    graph::op_t dequant_weight_3 {
            op_idx++, graph::op_kind::Dequantize, "dequantize_weight_3"};
    DEFINE_DEFAULT_PER_CHANNEL_QUANT_ATTR(dequant_weight_3, bias_size_3[0], 0);
    graph::op_t typecast_weight_3 {
            op_idx++, graph::op_kind::TypeCast, "typecast_weight_3"};

    graph::op_t matmul_3 {op_idx++, graph::op_kind::MatMul, "matmul_3"};
    matmul_3.set_attr(graph::op_attr::transpose_b, true);
    graph::op_t add_2 {op_idx++, graph::op_kind::Add, "add_2"};
    graph::op_t layernorm_2 {
            op_idx++, graph::op_kind::LayerNorm, "layernorm_2"};
    layernorm_2.set_attr(graph::op_attr::keep_stats, false);

    if (use_int8) {
        dequant_input_1.add_input(input_desc_1);
        dequant_input_1.add_output(dequant_input_desc_1);
        dequant_weight_1.add_input(weight_desc_1);
        dequant_weight_1.add_output(dequant_weight_desc_1);
        if (!use_bf16) {
            matmul_1.add_input(dequant_input_desc_1);
            matmul_1.add_input(dequant_weight_desc_1);
        } else {
            typecast_input_1.add_input(dequant_input_desc_1);
            typecast_input_1.add_output(typecast_input_desc_1);
            typecast_weight_1.add_input(dequant_weight_desc_1);
            typecast_weight_1.add_output(typecast_weight_desc_1);
            matmul_1.add_input(typecast_input_desc_1);
            matmul_1.add_input(typecast_weight_desc_1);
        }
    } else {
        matmul_1.add_input(typecast_input_desc_1);
        matmul_1.add_input(typecast_weight_desc_1);
    }
    matmul_1.add_input(bias_desc_1);
    matmul_1.add_output(matmul_desc_1);
    add_1.add_input(add_other_input_desc);
    add_1.add_input(matmul_desc_1);
    add_1.add_output(add_desc_1);
    layernorm_1.add_input(add_desc_1);
    layernorm_1.add_input(layernorm_alpha_desc_1);
    layernorm_1.add_input(layernorm_beta_desc_1);
    layernorm_1.add_output(layernorm_desc_1);
    if (use_int8) {
        quant_input_2.add_output(quant_input_desc_2);
        dequant_input_2.add_input(quant_input_desc_2);
        dequant_input_2.add_output(dequant_input_desc_2);
        dequant_weight_2.add_input(weight_desc_2);
        dequant_weight_2.add_output(dequant_weight_desc_2);
        if (!use_bf16) {
            quant_input_2.add_input(layernorm_desc_1);
            matmul_2.add_input(dequant_input_desc_2);
            matmul_2.add_input(dequant_weight_desc_2);
        } else {
            cast_output_1.add_input(layernorm_desc_1);
            cast_output_1.add_output(typecast_output_1);
            quant_input_2.add_input(typecast_output_1);
            typecast_input_2.add_input(dequant_input_desc_2);
            typecast_input_2.add_output(typecast_input_desc_2);
            typecast_weight_2.add_input(dequant_weight_desc_2);
            typecast_weight_2.add_output(typecast_weight_desc_2);
            matmul_2.add_input(typecast_input_desc_2);
            matmul_2.add_input(typecast_weight_desc_2);
        }
    } else {
        matmul_2.add_input(layernorm_desc_1);
        matmul_2.add_input(typecast_weight_desc_2);
    }

    matmul_2.add_input(bias_desc_2);
    matmul_2.add_output(matmul_desc_2);
    gelu_1.add_input(matmul_desc_2);
    gelu_1.add_output(gelu_desc);

    if (use_int8) {
        quant_input_3.add_output(quant_input_desc_3);
        dequant_input_3.add_input(quant_input_desc_3);
        dequant_input_3.add_output(dequant_input_desc_3);
        dequant_weight_3.add_input(weight_desc_3);
        dequant_weight_3.add_output(dequant_weight_desc_3);
        if (!use_bf16) {
            quant_input_3.add_input(gelu_desc);
            matmul_3.add_input(dequant_input_desc_3);
            matmul_3.add_input(dequant_weight_desc_3);
        } else {
            cast_output_2.add_input(gelu_desc);
            cast_output_2.add_output(typecast_output_2);
            quant_input_3.add_input(typecast_output_2);
            typecast_input_3.add_input(dequant_input_desc_3);
            typecast_input_3.add_output(typecast_input_desc_3);
            typecast_weight_3.add_input(dequant_weight_desc_3);
            typecast_weight_3.add_output(typecast_weight_desc_3);
            matmul_3.add_input(typecast_input_desc_3);
            matmul_3.add_input(typecast_weight_desc_3);
        }
    } else {
        matmul_3.add_input(gelu_desc);
        matmul_3.add_input(typecast_weight_desc_3);
    }

    matmul_3.add_input(bias_desc_3);
    matmul_3.add_output(matmul_desc_3);
    add_2.add_input(layernorm_desc_1);
    add_2.add_input(matmul_desc_3);
    add_2.add_output(add_desc_2);
    layernorm_2.add_input(add_desc_2);
    layernorm_2.add_input(layernorm_alpha_desc_2);
    layernorm_2.add_input(layernorm_beta_desc_2);
    layernorm_2.add_output(layernorm_desc_2);

    if (use_int8) {
        agraph->add_op(&dequant_input_1);
        agraph->add_op(&dequant_weight_1);
        agraph->add_op(&quant_input_2);
        agraph->add_op(&dequant_input_2);
        agraph->add_op(&dequant_weight_2);
        agraph->add_op(&quant_input_3);
        agraph->add_op(&dequant_input_3);
        agraph->add_op(&dequant_weight_3);
        if (use_bf16) {
            agraph->add_op(&typecast_input_1);
            agraph->add_op(&typecast_weight_1);
            agraph->add_op(&typecast_input_2);
            agraph->add_op(&typecast_weight_2);
            agraph->add_op(&typecast_input_3);
            agraph->add_op(&typecast_weight_3);
            agraph->add_op(&cast_output_1);
            agraph->add_op(&cast_output_2);
        }
    }

    agraph->add_op(&matmul_1);
    agraph->add_op(&add_1);
    agraph->add_op(&layernorm_1);
    agraph->add_op(&matmul_2);
    agraph->add_op(&gelu_1);
    agraph->add_op(&matmul_3);
    agraph->add_op(&add_2);
    agraph->add_op(&layernorm_2);
}

inline void construct_mul_quantize_subgraph(graph::graph_t *agraph,
        utils::id_generator &id_gen, const graph::dims &input_shape,
        bool is_mixed_precision = false) {
    graph::dims smooth_quant_scales_shape {input_shape[input_shape.size() - 1]};
    auto dtype = is_mixed_precision ? graph::data_type::bf16
                                    : graph::data_type::f32;
    auto mul_in
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto smooth_quant_scale = utils::logical_tensor_init(
            id_gen.get_id(), smooth_quant_scales_shape, graph::data_type::f32);
    auto mul_out = utils::logical_tensor_init(
            id_gen.get_id(), input_shape, graph::data_type::f32);
    auto quant_out = utils::logical_tensor_init(
            id_gen.get_id(), input_shape, graph::data_type::u8);

    graph::op_t mul {id_gen.get_id(), graph::op_kind::Multiply, "mul"};
    mul.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t quantize {
            id_gen.get_id(), graph::op_kind::Quantize, "quantize"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize);

    mul.add_input(mul_in);
    mul.add_input(smooth_quant_scale);
    mul.add_output(mul_out);
    quantize.add_input(mul_out);
    quantize.add_output(quant_out);

    agraph->add_op(&mul);
    agraph->add_op(&quantize);
}

inline void construct_gpt_mha_subgraph(graph::graph_t *agraph,
        utils::id_generator &id_gen, bool use_bf16 = false,
        bool use_int8 = false, int batch_size = 4, int seq_len = 34,
        int num_head = 16, int head_dim = 4096) {
    int size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> SELECT_BOOL_SHAPE {1, 1, 1, seq_len};
    std::vector<graph::dim_t> ADD_MASK_SHAPE {batch_size, 1, 1, seq_len};
    std::vector<graph::dim_t> QUERY_TRANSPOSED_SHAPE {
            batch_size, num_head, 1, size_per_head};
    std::vector<graph::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<graph::dim_t> VALUE_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_size, num_head, 1, seq_len};
    std::vector<graph::dim_t> CONST_SHAPE {1};
    auto dtype = use_bf16 ? graph::data_type::bf16 : graph::data_type::f32;

    graph::logical_tensor_t query_dequantize_input, key_dequantize_input,
            value_dequantize_input;
    query_dequantize_input = utils::logical_tensor_init(
            id_gen.get_id(), QUERY_TRANSPOSED_SHAPE, graph::data_type::u8);
    key_dequantize_input = utils::logical_tensor_init(
            id_gen.get_id(), KEY_TRANSPOSED_SHAPE, graph::data_type::u8);
    value_dequantize_input = utils::logical_tensor_init(
            id_gen.get_id(), VALUE_TRANSPOSED_SHAPE, graph::data_type::u8);

    graph::logical_tensor_t query_typecast_input, key_typecast_input,
            value_typecast_input;
    query_typecast_input = utils::logical_tensor_init(id_gen.get_id(),
            graph::data_type::f32, graph::layout_type::strided);
    key_typecast_input = utils::logical_tensor_init(id_gen.get_id(),
            graph::data_type::f32, graph::layout_type::strided);
    value_typecast_input = utils::logical_tensor_init(id_gen.get_id(),
            graph::data_type::f32, graph::layout_type::strided);

    graph::logical_tensor_t query_matmul_input, key_matmul_input,
            value_matmul_input;
    query_matmul_input = utils::logical_tensor_init(
            id_gen.get_id(), QUERY_TRANSPOSED_SHAPE, dtype);
    key_matmul_input = utils::logical_tensor_init(
            id_gen.get_id(), KEY_TRANSPOSED_SHAPE, dtype);
    value_matmul_input = utils::logical_tensor_init(
            id_gen.get_id(), VALUE_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t select_bool_lt, select_else_lt, select_out;
    select_bool_lt = utils::logical_tensor_init(
            id_gen.get_id(), SELECT_BOOL_SHAPE, graph::data_type::boolean);
    select_else_lt
            = utils::logical_tensor_init(id_gen.get_id(), CONST_SHAPE, dtype);
    select_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t fscore_scale, fscore_scale_out;
    fscore_scale
            = utils::logical_tensor_init(id_gen.get_id(), CONST_SHAPE, dtype);
    fscore_scale_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t fscore_add_out, fscore_add_mask, softmax_out;
    fscore_add_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);
    fscore_add_mask = utils::logical_tensor_init(
            id_gen.get_id(), ADD_MASK_SHAPE, dtype);
    softmax_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t softmax_cast_out;
    softmax_cast_out = utils::logical_tensor_init(id_gen.get_id(),
            graph::data_type::f32, graph::layout_type::strided);

    graph::logical_tensor_t softmax_quantize_out;
    softmax_quantize_out = utils::logical_tensor_init(
            id_gen.get_id(), graph::data_type::u8, graph::layout_type::strided);
    graph::logical_tensor_t softmax_dequantize_out;
    softmax_dequantize_out = utils::logical_tensor_init(id_gen.get_id(),
            graph::data_type::f32, graph::layout_type::strided);

    graph::logical_tensor_t softmax_dequantize_out_cast;
    softmax_dequantize_out_cast = utils::logical_tensor_init(id_gen.get_id(),
            graph::data_type::bf16, graph::layout_type::strided);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t context_transpose_out, context_reorder_out;
    context_transpose_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);
    context_reorder_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t context_cast_out;
    context_cast_out = utils::logical_tensor_init(id_gen.get_id(),
            graph::data_type::f32, graph::layout_type::strided);

    graph::logical_tensor_t context_quantize_out;
    context_quantize_out = utils::logical_tensor_init(
            id_gen.get_id(), graph::data_type::u8, graph::layout_type::strided);

    graph::op_t dequantize_query {
            id_gen.get_id(), graph::op_kind::Dequantize, "dequantize_query"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_query);
    graph::op_t dequantize_key {
            id_gen.get_id(), graph::op_kind::Dequantize, "dequantize_key"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_key);
    graph::op_t dequantize_value {
            id_gen.get_id(), graph::op_kind::Dequantize, "dequantize_value"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_value);
    graph::op_t typecast_query {
            id_gen.get_id(), graph::op_kind::TypeCast, "typecast_query"};
    graph::op_t typecast_key {
            id_gen.get_id(), graph::op_kind::TypeCast, "typecast_key"};
    graph::op_t typecast_value {
            id_gen.get_id(), graph::op_kind::TypeCast, "typecast_value"};

    graph::op_t matmul_qk {
            id_gen.get_id(), graph::op_kind::MatMul, "matmul_qk"};

    graph::op_t fscore_select {
            id_gen.get_id(), graph::op_kind::Select, "fscore_select"};
    graph::op_t fscore_rescale {
            id_gen.get_id(), graph::op_kind::Divide, "fscore_rescale"};
    fscore_rescale.set_attr(
            graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t fscore_add {id_gen.get_id(), graph::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t softmax {id_gen.get_id(), graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)3);

    graph::op_t softmax_cast1 {
            id_gen.get_id(), graph::op_kind::TypeCast, "softmax_cast1"};
    graph::op_t quantize_softmax {
            id_gen.get_id(), graph::op_kind::Quantize, "quantize_softmax"};
    graph::op_t dequantize_softmax {
            id_gen.get_id(), graph::op_kind::Dequantize, "dequantize_softmax"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_softmax);
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_softmax);

    graph::op_t dequantize_softmax_cast {id_gen.get_id(),
            graph::op_kind::TypeCast, "dequantize_softmax_cast"};

    graph::op_t matmul_v {id_gen.get_id(), graph::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    graph::op_t transpose_output {id_gen.get_id(),
            graph::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            graph::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    graph::op_t reorder_output {
            id_gen.get_id(), graph::op_kind::Reorder, "reorder_output"};

    graph::op_t typecast_output {
            id_gen.get_id(), graph::op_kind::TypeCast, "typecast_output"};
    graph::op_t quantize_output {
            id_gen.get_id(), graph::op_kind::Quantize, "quantize_output"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_output);

    if (use_int8) {
        dequantize_query.add_input(query_dequantize_input);
        dequantize_key.add_input(key_dequantize_input);
        dequantize_value.add_input(value_dequantize_input);
        if (!use_bf16) {
            dequantize_query.add_output(query_matmul_input);
            dequantize_key.add_output(key_matmul_input);
            dequantize_value.add_output(value_matmul_input);
        } else {
            dequantize_query.add_output(query_typecast_input);
            dequantize_key.add_output(key_typecast_input);
            dequantize_value.add_output(value_typecast_input);
            typecast_query.add_input(query_typecast_input);
            typecast_key.add_input(key_typecast_input);
            typecast_value.add_input(value_typecast_input);
            typecast_query.add_output(query_matmul_input);
            typecast_key.add_output(key_matmul_input);
            typecast_value.add_output(value_matmul_input);
        }
    }
    matmul_qk.add_input(query_matmul_input);
    matmul_qk.add_input(key_matmul_input);
    matmul_qk.add_output(matmul_qk_out);
    fscore_select.add_input(select_bool_lt);
    fscore_select.add_input(matmul_qk_out);
    fscore_select.add_input(select_else_lt);
    fscore_select.add_output(select_out);
    fscore_rescale.add_input(select_out);
    fscore_rescale.add_input(fscore_scale);
    fscore_rescale.add_output(fscore_scale_out);
    fscore_add.add_input(fscore_scale_out);
    fscore_add.add_input(fscore_add_mask);
    fscore_add.add_output(fscore_add_out);
    softmax.add_input(fscore_add_out);
    softmax.add_output(softmax_out);

    if (use_int8) {
        quantize_softmax.add_output(softmax_quantize_out);
        dequantize_softmax.add_input(softmax_quantize_out);
        dequantize_softmax.add_output(softmax_dequantize_out);
        if (!use_bf16) {
            quantize_softmax.add_input(softmax_out);
            matmul_v.add_input(softmax_dequantize_out);
        } else {
            softmax_cast1.add_input(softmax_out);
            softmax_cast1.add_output(softmax_cast_out);
            quantize_softmax.add_input(softmax_cast_out);
            dequantize_softmax_cast.add_input(softmax_dequantize_out);
            dequantize_softmax_cast.add_output(softmax_dequantize_out_cast);
            matmul_v.add_input(softmax_dequantize_out_cast);
        }
    } else {
        matmul_v.add_input(softmax_out);
    }
    matmul_v.add_input(value_matmul_input);
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);
    reorder_output.add_input(context_transpose_out);
    reorder_output.add_output(context_reorder_out);

    if (use_int8) {
        quantize_output.add_output(context_quantize_out);
        if (!use_bf16) {
            quantize_output.add_input(context_reorder_out);
        } else {
            typecast_output.add_input(context_reorder_out);
            typecast_output.add_output(context_cast_out);
            quantize_output.add_input(context_cast_out);
        }
    }
    if (use_int8) {
        agraph->add_op(&dequantize_query);
        agraph->add_op(&dequantize_key);
        agraph->add_op(&dequantize_value);
        if (use_bf16) {
            agraph->add_op(&typecast_query);
            agraph->add_op(&typecast_key);
            agraph->add_op(&typecast_value);
        }
    }
    agraph->add_op(&matmul_qk);
    agraph->add_op(&fscore_select);
    agraph->add_op(&fscore_rescale);
    agraph->add_op(&fscore_add);
    agraph->add_op(&softmax);

    if (use_int8) {
        agraph->add_op(&quantize_softmax);
        agraph->add_op(&dequantize_softmax);
        if (use_bf16) {
            agraph->add_op(&softmax_cast1);
            agraph->add_op(&dequantize_softmax_cast);
        }
    }

    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
    agraph->add_op(&reorder_output);

    if (use_int8) {
        agraph->add_op(&quantize_output);
        if (use_bf16) { agraph->add_op(&typecast_output); }
    }
}

static void construct_llama_mha_base(graph::graph_t *agraph,
        utils::id_generator &id_gen, graph::data_type_t dtype,
        int batch_size = 4, int seq_len = 34, int num_head = 32,
        int head_dim = 4096) {
    int size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> ADD_MASK_SHAPE {batch_size, 1, 1, seq_len};
    std::vector<graph::dim_t> QUERY_TRANSPOSED_SHAPE {
            batch_size, num_head, 1, size_per_head};
    std::vector<graph::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<graph::dim_t> VALUE_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<graph::dim_t> CONST_SHAPE {1};

    graph::logical_tensor_t query_matmul_input, key_matmul_input,
            value_matmul_input;
    query_matmul_input = utils::logical_tensor_init(
            id_gen.get_id(), QUERY_TRANSPOSED_SHAPE, dtype);
    key_matmul_input = utils::logical_tensor_init(
            id_gen.get_id(), KEY_TRANSPOSED_SHAPE, dtype);
    value_matmul_input = utils::logical_tensor_init(
            id_gen.get_id(), VALUE_TRANSPOSED_SHAPE, dtype);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t fscore_scale, fscore_scale_out;
    fscore_scale
            = utils::logical_tensor_init(id_gen.get_id(), CONST_SHAPE, dtype);
    fscore_scale_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t fscore_add_mask, fscore_add_out, fscore_max_in,
            fscore_max_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);
    fscore_add_mask = utils::logical_tensor_init(
            id_gen.get_id(), ADD_MASK_SHAPE, dtype);
    fscore_max_in
            = utils::logical_tensor_init(id_gen.get_id(), CONST_SHAPE, dtype);
    fscore_max_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);
    softmax_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::op_t dequantize_query {
            id_gen.get_id(), graph::op_kind::Dequantize, "dequantize_query"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_query);
    graph::op_t dequantize_key {
            id_gen.get_id(), graph::op_kind::Dequantize, "dequantize_key"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_key);
    graph::op_t dequantize_value {
            id_gen.get_id(), graph::op_kind::Dequantize, "dequantize_value"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_value);
    graph::op_t typecast_query {
            id_gen.get_id(), graph::op_kind::TypeCast, "typecast_query"};
    graph::op_t typecast_key {
            id_gen.get_id(), graph::op_kind::TypeCast, "typecast_key"};
    graph::op_t typecast_value {
            id_gen.get_id(), graph::op_kind::TypeCast, "typecast_value"};

    graph::op_t matmul_qk {
            id_gen.get_id(), graph::op_kind::MatMul, "matmul_qk"};

    graph::op_t fscore_rescale {
            id_gen.get_id(), graph::op_kind::Divide, "fscore_rescale"};
    fscore_rescale.set_attr(
            graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t fscore_add {id_gen.get_id(), graph::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t fscore_max {
            id_gen.get_id(), graph::op_kind::Maximum, "fscore_max"};
    fscore_max.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t softmax {id_gen.get_id(), graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)3);

    graph::op_t matmul_v {id_gen.get_id(), graph::op_kind::MatMul, "matmul_v"};

    matmul_qk.add_input(query_matmul_input);
    matmul_qk.add_input(key_matmul_input);
    matmul_qk.add_output(matmul_qk_out);
    fscore_rescale.add_input(matmul_qk_out);
    fscore_rescale.add_input(fscore_scale);
    fscore_rescale.add_output(fscore_scale_out);
    fscore_add.add_input(fscore_scale_out);
    fscore_add.add_input(fscore_add_mask);
    fscore_add.add_output(fscore_add_out);
    fscore_max.add_input(fscore_add_out);
    fscore_max.add_input(fscore_max_in);
    fscore_max.add_output(fscore_max_out);
    softmax.add_input(fscore_max_out);
    softmax.add_output(softmax_out);
    matmul_v.add_input(softmax_out);
    matmul_v.add_input(value_matmul_input);
    matmul_v.add_output(matmul_v_out);

    agraph->add_op(&matmul_qk);
    agraph->add_op(&fscore_rescale);
    agraph->add_op(&fscore_add);
    agraph->add_op(&fscore_max);
    agraph->add_op(&softmax);
    agraph->add_op(&matmul_v);
}

static void insert_quantization_before_op(graph::graph_t *agraph,
        utils::id_generator &id_gen, graph::op_t *op, int idx, bool is_bf16) {
    // init quant dequant
    graph::op_t quantize {
            id_gen.get_id(), graph::op_kind::Quantize, "quantize"};
    graph::op_t dequantize {
            id_gen.get_id(), graph::op_kind::Dequantize, "dequantize"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize);
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize);
    auto in_val = op->get_input_value(idx);
    in_val->remove_consumer(*op, idx);
    auto ref_lt = in_val->get_logical_tensor();
    auto quantize_out_lt = ref_lt;
    quantize_out_lt.id = id_gen.get_id();
    quantize_out_lt.data_type = graph::data_type::u8;
    quantize.add_output(quantize_out_lt);
    dequantize.add_input(quantize_out_lt);
    auto dequant_out_lt = ref_lt;
    dequant_out_lt.id = id_gen.get_id();
    dequant_out_lt.data_type = graph::data_type::f32;

    if (is_bf16) {
        graph::op_t typecast1 {
                id_gen.get_id(), graph::op_kind::TypeCast, "typecast1"};
        graph::op_t typecast2 {
                id_gen.get_id(), graph::op_kind::TypeCast, "typecast2"};

        auto typecast1_out_lt = ref_lt;
        typecast1_out_lt.id = id_gen.get_id();
        typecast1_out_lt.data_type = graph::data_type::f32;
        typecast1.add_input(in_val);
        typecast1.add_output(typecast1_out_lt);

        quantize.add_input(typecast1_out_lt);
        dequantize.add_output(dequant_out_lt);

        typecast2.add_input(dequant_out_lt);
        auto typecast2_out_lt = ref_lt;
        typecast2_out_lt.id = id_gen.get_id();
        typecast2_out_lt.data_type = graph::data_type::bf16;

        auto typecast2_out_val
                = std::make_shared<graph::value_t>(typecast2_out_lt, false);
        op->connect_input(idx, typecast2_out_val);
        typecast2.add_output(typecast2_out_val);

        agraph->add_op(&typecast1);
        agraph->add_op(&typecast2);
    } else {
        quantize.add_input(in_val);

        auto dequant_out_val
                = std::make_shared<graph::value_t>(dequant_out_lt, false);
        op->connect_input(idx, dequant_out_val);
        dequantize.add_output(dequant_out_val);
    }
    agraph->add_op(&quantize);
    agraph->add_op(&dequantize);
}

static void insert_quantization_after_output(graph::graph_t *agraph,
        utils::id_generator &id_gen, const graph::logical_tensor_t &output_lt,
        bool is_bf16) {
    // init quant dequant
    graph::op_t quantize {
            id_gen.get_id(), graph::op_kind::Quantize, "quantize"};
    graph::op_t dequantize {
            id_gen.get_id(), graph::op_kind::Dequantize, "dequantize"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize);
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize);
    auto quantize_out_lt = output_lt;
    quantize_out_lt.id = id_gen.get_id();
    quantize_out_lt.data_type = graph::data_type::u8;
    quantize.add_output(quantize_out_lt);
    dequantize.add_input(quantize_out_lt);
    auto dequant_out_lt = output_lt;
    dequant_out_lt.id = id_gen.get_id();
    dequant_out_lt.data_type = graph::data_type::f32;
    dequantize.add_output(dequant_out_lt);

    if (is_bf16) {
        graph::op_t typecast1 {
                id_gen.get_id(), graph::op_kind::TypeCast, "typecast1"};
        graph::op_t typecast2 {
                id_gen.get_id(), graph::op_kind::TypeCast, "typecast2"};

        auto typecast1_out_lt = output_lt;
        typecast1_out_lt.id = id_gen.get_id();
        typecast1_out_lt.data_type = graph::data_type::f32;
        typecast1.add_input(output_lt);
        typecast1.add_output(typecast1_out_lt);

        quantize.add_input(typecast1_out_lt);

        typecast2.add_input(dequant_out_lt);
        auto typecast2_out_lt = output_lt;
        typecast2_out_lt.id = id_gen.get_id();
        typecast2_out_lt.data_type = graph::data_type::bf16;

        auto typecast2_out_val
                = std::make_shared<graph::value_t>(typecast2_out_lt, false);
        typecast2.add_output(typecast2_out_val);

        agraph->add_op(&typecast1);
        agraph->add_op(&typecast2);
    } else {
        quantize.add_input(output_lt);
    }
    agraph->add_op(&quantize);
    agraph->add_op(&dequantize);
}

inline void construct_llama_mha_subgraph(graph::graph_t *agraph,
        utils::id_generator &id_gen, bool use_bf16 = false,
        bool use_int8 = false, int batch_size = 4, int seq_len = 34,
        int num_head = 32, int head_dim = 4096) {
    construct_llama_mha_base(agraph, id_gen,
            use_bf16 ? graph::data_type::bf16 : graph::data_type::f32,
            batch_size, seq_len, num_head, head_dim);
    auto ops = agraph->get_ops();
    // insert quantize / dequantize / typecast
    if (use_int8) {
        for (const auto &op : ops) {
            if (op->get_kind() == graph::op_kind::MatMul) {
                insert_quantization_before_op(
                        agraph, id_gen, op.get(), 0, use_bf16);
                insert_quantization_before_op(
                        agraph, id_gen, op.get(), 1, use_bf16);
            }
        }
    }
    // specially deal with bf16 case
    if (use_bf16 && !use_int8) {
        // change dtype of add/max/softmax
        for (const auto &op : ops) {
            if (op->get_kind() == graph::op_kind::Add
                    || op->get_kind() == graph::op_kind::Maximum
                    || op->get_kind() == graph::op_kind::SoftMax) {
                for (auto &val : op->get_input_values()) {
                    val->set_data_type(graph::data_type::f32);
                }
                for (auto &val : op->get_output_values()) {
                    val->set_data_type(graph::data_type::f32);
                }
            }
        }

        // insert typecast after div & softmax
        for (const auto &op : ops) {
            if (op->get_kind() == graph::op_kind::Divide
                    || op->get_kind() == graph::op_kind::SoftMax) {
                if (op->get_kind() == graph::op_kind::SoftMax) {
                    for (auto &val : op->get_output_values()) {
                        val->set_data_type(graph::data_type::bf16);
                    }
                } else if (op->get_kind() == graph::op_kind::Divide) {
                    for (auto &val : op->get_output_values()) {
                        val->set_data_type(graph::data_type::f32);
                    }
                }
                graph::op_t typecast {
                        id_gen.get_id(), graph::op_kind::TypeCast, "typecast"};
                auto out_val = op->get_output_value(0);
                typecast.add_output(out_val);
                auto typecast_in_lt = out_val->get_logical_tensor();
                typecast_in_lt.id = id_gen.get_id();
                typecast_in_lt.data_type
                        = typecast_in_lt.data_type == graph::data_type::bf16
                        ? graph::data_type::f32
                        : graph::data_type::bf16;
                auto typecast_in_val = std::make_shared<graph::value_t>(
                        *op, 0, typecast_in_lt, false);
                op->connect_output(0, typecast_in_val);
                typecast.add_input(typecast_in_val);

                agraph->add_op(&typecast);
            }
        }
    }
}

static inline graph::logical_tensor_t create_matmul(utils::id_generator &id_gen,
        graph::graph_t *agraph, const graph::logical_tensor_t *input,
        const graph::logical_tensor_t *weight,
        const graph::logical_tensor_t *bias = nullptr,
        bool transpose_b = false) {
    graph::op_t matmul(id_gen.get_id(), graph::op_kind::MatMul, "matmul");
    auto dst = utils::logical_tensor_init(id_gen.get_id(), input->data_type);
    if (transpose_b) { matmul.set_attr(graph::op_attr::transpose_b, true); }
    matmul.add_input(*input);
    matmul.add_input(*weight);
    if (bias != nullptr) { matmul.add_input(*bias); }
    matmul.add_output(dst);
    agraph->add_op(&matmul);
    return dst;
}

static graph::logical_tensor_t construct_gpt_mlp_base(graph::graph_t *agraph,
        utils::id_generator &id_gen, graph::data_type_t dtype,
        graph::dim_t batch_size = 4,
        std::vector<graph::dim_t> hidden_sizes = {4096, 16384, 4096}) {
    std::vector<graph::dim_t> input_shape {batch_size, 1, hidden_sizes[0]};
    std::vector<graph::dim_t> weight_size_1 {hidden_sizes[0], hidden_sizes[1]};
    std::vector<graph::dim_t> weight_size_2 {hidden_sizes[1], hidden_sizes[2]};
    std::vector<graph::dim_t> weight_size_3 {hidden_sizes[0], hidden_sizes[2]};
    std::vector<graph::dim_t> bias_size_1 {hidden_sizes[1]};
    std::vector<graph::dim_t> bias_size_2 {hidden_sizes[2]};

    auto input_desc_1
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto weight_desc_1
            = utils::logical_tensor_init(id_gen.get_id(), weight_size_1, dtype);
    auto bias_desc_1
            = utils::logical_tensor_init(id_gen.get_id(), bias_size_1, dtype);
    graph::logical_tensor_t matmul_desc_1 = create_matmul(
            id_gen, agraph, &input_desc_1, &weight_desc_1, &bias_desc_1);

    graph::op_t gelu(id_gen.get_id(), graph::op_kind::GELU, "gelu");
    auto gelu_dst = utils::logical_tensor_init(
            id_gen.get_id(), matmul_desc_1.data_type);
    gelu.add_input(matmul_desc_1);
    gelu.add_output(gelu_dst);
    agraph->add_op(&gelu);

    auto weight_desc_2
            = utils::logical_tensor_init(id_gen.get_id(), weight_size_2, dtype);
    auto bias_desc_2
            = utils::logical_tensor_init(id_gen.get_id(), bias_size_2, dtype);
    graph::logical_tensor_t matmul_desc_2 = create_matmul(
            id_gen, agraph, &gelu_dst, &weight_desc_2, &bias_desc_2);

    auto input_desc_3
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto weight_desc_3
            = utils::logical_tensor_init(id_gen.get_id(), weight_size_3, dtype);
    graph::logical_tensor_t matmul_desc_3 = create_matmul(
            id_gen, agraph, &input_desc_3, &weight_desc_3, nullptr);
    auto add1_dst
            = utils::create_add(id_gen, *agraph, matmul_desc_3, matmul_desc_2);
    auto add2_in
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto add2_dst = utils::create_add(id_gen, *agraph, add1_dst, add2_in);
    graph::op_t layernorm(
            id_gen.get_id(), graph::op_kind::LayerNorm, "layernorm");
    layernorm.set_attr(graph::op_attr::keep_stats, false);
    auto layernorm_gamma = utils::logical_tensor_init(
            id_gen.get_id(), bias_size_2, add2_dst.data_type);
    auto layernorm_beta = utils::logical_tensor_init(
            id_gen.get_id(), bias_size_2, add2_dst.data_type);
    auto layernorm_dst
            = utils::logical_tensor_init(id_gen.get_id(), add2_dst.data_type);
    layernorm.add_input(add2_dst);
    layernorm.add_input(layernorm_gamma);
    layernorm.add_input(layernorm_beta);
    layernorm.add_output(layernorm_dst);
    agraph->add_op(&layernorm);
    return layernorm_dst;
}

inline void construct_gpt_mlp_subgraph(graph::graph_t *agraph,
        utils::id_generator &id_gen, bool use_bf16 = false,
        bool use_int8 = false, graph::dim_t batch_size = 4,
        std::vector<graph::dim_t> hidden_sizes = {4096, 16384, 4096}) {
    auto output_lt = construct_gpt_mlp_base(agraph, id_gen,
            use_bf16 ? graph::data_type::bf16 : graph::data_type::f32,
            batch_size, hidden_sizes);
    auto ops = agraph->get_ops();
    // insert quantize / dequantize / typecast
    if (use_int8) {
        for (const auto &op : ops) {
            if (op->get_kind() == graph::op_kind::MatMul) {
                insert_quantization_before_op(
                        agraph, id_gen, op.get(), 0, use_bf16);
                insert_quantization_before_op(
                        agraph, id_gen, op.get(), 1, use_bf16);
            }
        }
        insert_quantization_after_output(agraph, id_gen, output_lt, use_bf16);
    }
}

static graph::logical_tensor_t construct_rms_norm_subgraph(
        graph::graph_t *agraph, utils::id_generator &id_gen,
        const graph::logical_tensor_t &input_desc,
        graph::dim_t mul_in_size = 4096) {
    bool is_bf16 = input_desc.data_type == graph::data_type::bf16;
    graph::logical_tensor_t input_lt = input_desc;
    if (is_bf16) {
        graph::op_t typecast_in(
                id_gen.get_id(), graph::op_kind::TypeCast, "typecast_in");
        auto typecast_in_dst = utils::logical_tensor_init(
                id_gen.get_id(), graph::data_type::f32);
        typecast_in.add_input(input_lt);
        typecast_in.add_output(typecast_in_dst);
        agraph->add_op(&typecast_in);
        input_lt = typecast_in_dst;
    }
    graph::op_t pow(id_gen.get_id(), graph::op_kind::Pow, "pow");
    pow.set_attr(graph::op_attr::beta, 2.0f);
    auto pow_dst = utils::logical_tensor_init(
            id_gen.get_id(), graph::data_type::f32);
    pow.add_input(input_lt);
    pow.add_output(pow_dst);
    agraph->add_op(&pow);

    graph::op_t reduce_mean(
            id_gen.get_id(), graph::op_kind::ReduceMean, "reduce_mean");
    reduce_mean.set_attr(graph::op_attr::axes, std::vector<int64_t> {-1});
    reduce_mean.set_attr(graph::op_attr::keep_dims, true);
    auto reduce_mean_dst = utils::logical_tensor_init(
            id_gen.get_id(), graph::data_type::f32);
    reduce_mean.add_input(pow_dst);
    reduce_mean.add_output(reduce_mean_dst);
    agraph->add_op(&reduce_mean);

    graph::logical_tensor_t add_in = utils::logical_tensor_init(id_gen.get_id(),
            std::vector<graph::dim_t> {1}, graph::data_type::f32);
    graph::logical_tensor_t add_dst
            = utils::create_add(id_gen, *agraph, reduce_mean_dst, add_in);

    graph::op_t rsqrt(id_gen.get_id(), graph::op_kind::Pow, "rsqrt");
    rsqrt.set_attr(graph::op_attr::beta, -0.5f);
    auto rsqrt_dst = utils::logical_tensor_init(
            id_gen.get_id(), graph::data_type::f32);
    rsqrt.add_input(add_dst);
    rsqrt.add_output(rsqrt_dst);
    agraph->add_op(&rsqrt);

    graph::op_t mul1(id_gen.get_id(), graph::op_kind::Multiply, "mul1");
    auto mul1_dst = utils::logical_tensor_init(
            id_gen.get_id(), graph::data_type::f32);
    mul1.add_input(input_lt);
    mul1.add_input(rsqrt_dst);
    mul1.add_output(mul1_dst);
    agraph->add_op(&mul1);

    if (is_bf16) {
        graph::op_t typecast(
                id_gen.get_id(), graph::op_kind::TypeCast, "typecast");
        auto typecast_dst = utils::logical_tensor_init(
                id_gen.get_id(), graph::data_type::bf16);
        typecast.add_input(mul1_dst);
        typecast.add_output(typecast_dst);
        agraph->add_op(&typecast);
        mul1_dst = typecast_dst;
    }

    graph::op_t mul2(id_gen.get_id(), graph::op_kind::Multiply, "mul2");
    auto mul2_in = utils::logical_tensor_init(id_gen.get_id(),
            std::vector<graph::dim_t> {mul_in_size}, mul1_dst.data_type);
    auto mul2_dst
            = utils::logical_tensor_init(id_gen.get_id(), mul1_dst.data_type);
    mul2.add_input(mul1_dst);
    mul2.add_input(mul2_in);
    mul2.add_output(mul2_dst);
    agraph->add_op(&mul2);
    return mul2_dst;
}

static graph::logical_tensor_t construct_llama_mlp_base(graph::graph_t *agraph,
        utils::id_generator &id_gen, graph::data_type_t dtype,
        graph::dim_t batch_size = 4,
        std::vector<graph::dim_t> hidden_sizes = {4096, 4096, 11008, 4096}) {
    std::vector<graph::dim_t> input_shape {batch_size, 1, hidden_sizes[0]};
    std::vector<graph::dim_t> matmul1_shape {batch_size, 1, hidden_sizes[1]};
    std::vector<graph::dim_t> weight_size_1 {hidden_sizes[0], hidden_sizes[1]};
    std::vector<graph::dim_t> weight_size_2 {hidden_sizes[1], hidden_sizes[2]};
    std::vector<graph::dim_t> weight_size_3 {hidden_sizes[2], hidden_sizes[3]};

    auto input_desc_1
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto weight_desc_1
            = utils::logical_tensor_init(id_gen.get_id(), weight_size_1, dtype);
    graph::logical_tensor_t matmul_desc_1 = create_matmul(
            id_gen, agraph, &input_desc_1, &weight_desc_1, nullptr);
    auto add2_in
            = utils::logical_tensor_init(id_gen.get_id(), matmul1_shape, dtype);
    graph::logical_tensor_t add_desc_1
            = utils::create_add(id_gen, *agraph, matmul_desc_1, add2_in);

    auto norm1_dst = construct_rms_norm_subgraph(
            agraph, id_gen, add_desc_1, hidden_sizes[1]);

    auto weight_desc_2
            = utils::logical_tensor_init(id_gen.get_id(), weight_size_2, dtype);
    auto weight_desc_3
            = utils::logical_tensor_init(id_gen.get_id(), weight_size_2, dtype);
    graph::logical_tensor_t matmul_desc_2 = create_matmul(
            id_gen, agraph, &norm1_dst, &weight_desc_2, nullptr);
    graph::logical_tensor_t matmul_desc_3 = create_matmul(
            id_gen, agraph, &norm1_dst, &weight_desc_3, nullptr);
    graph::op_t sigmoid(id_gen.get_id(), graph::op_kind::Sigmoid, "sigmoid");
    auto sigmoid_dst = utils::logical_tensor_init(id_gen.get_id(), dtype);
    sigmoid.add_input(matmul_desc_2);
    sigmoid.add_output(sigmoid_dst);
    agraph->add_op(&sigmoid);
    graph::op_t silu_mul(id_gen.get_id(), graph::op_kind::Multiply, "silu_mul");
    auto silu_mul_dst = utils::logical_tensor_init(id_gen.get_id(), dtype);
    silu_mul.add_input(matmul_desc_2);
    silu_mul.add_input(sigmoid_dst);
    silu_mul.add_output(silu_mul_dst);
    agraph->add_op(&silu_mul);
    graph::op_t mul(id_gen.get_id(), graph::op_kind::Multiply, "mul");
    auto mul_dst = utils::logical_tensor_init(id_gen.get_id(), dtype);
    mul.add_input(silu_mul_dst);
    mul.add_input(matmul_desc_3);
    mul.add_output(mul_dst);
    agraph->add_op(&mul);

    auto weight_desc_4
            = utils::logical_tensor_init(id_gen.get_id(), weight_size_3, dtype);
    graph::logical_tensor_t matmul_desc_4
            = create_matmul(id_gen, agraph, &mul_dst, &weight_desc_4, nullptr);
    graph::logical_tensor_t add_desc_2
            = utils::create_add(id_gen, *agraph, matmul_desc_4, add_desc_1);

    auto norm2_dst = construct_rms_norm_subgraph(
            agraph, id_gen, add_desc_2, hidden_sizes[3]);
    return norm2_dst;
}

inline void construct_llama_mlp_subgraph(graph::graph_t *agraph,
        utils::id_generator &id_gen, bool use_bf16 = false,
        bool use_int8 = false, graph::dim_t batch_size = 4,
        std::vector<graph::dim_t> hidden_sizes = {4096, 4096, 11008, 4096}) {
    auto output_lt = construct_llama_mlp_base(agraph, id_gen,
            use_bf16 ? graph::data_type::bf16 : graph::data_type::f32,
            batch_size, hidden_sizes);
    auto ops = agraph->get_ops();
    // insert quantize / dequantize / typecast
    if (use_int8) {
        for (const auto &op : ops) {
            if (op->get_kind() == graph::op_kind::MatMul
                    && (op->get_id() == 2 || op->get_id() == 34
                            || op->get_id() == 38)) {
                insert_quantization_before_op(
                        agraph, id_gen, op.get(), 0, use_bf16);
                insert_quantization_before_op(
                        agraph, id_gen, op.get(), 1, use_bf16);
            } else if (op->get_kind() == graph::op_kind::MatMul
                    && (op->get_id() == 23 || op->get_id() == 25
                            || op->get_id() == 27 || op->get_id() == 29)) {
                insert_quantization_before_op(
                        agraph, id_gen, op.get(), 1, use_bf16);
            }
        }
        // specially deal with horizontally merged matmul part
        auto matmul_lhs_iter = std::find_if(ops.begin(), ops.end(),
                [&](const std::shared_ptr<graph::op_t> &op) {
                    if ((!use_bf16 && op->get_id() == 23)
                            || (use_bf16 && op->get_id() == 27)) {
                        return true;
                    }
                    return false;
                });
        auto matmul_rhs_iter = std::find_if(ops.begin(), ops.end(),
                [&](const std::shared_ptr<graph::op_t> &op) {
                    if ((!use_bf16 && op->get_id() == 25)
                            || (use_bf16 && op->get_id() == 29)) {
                        return true;
                    }
                    return false;
                });
        if (matmul_lhs_iter == ops.end() || matmul_rhs_iter == ops.end()) {
            throw std::runtime_error("Cannot find op with specific id");
        }
        graph::op_t *matmul_lhs = (*matmul_lhs_iter).get();
        graph::op_t *matmul_rhs = (*matmul_rhs_iter).get();
        graph::op_t quantize_common {
                id_gen.get_id(), graph::op_kind::Quantize, "quantize"};
        graph::op_t dequantize_lhs {
                id_gen.get_id(), graph::op_kind::Dequantize, "dequantize_lhs"};
        graph::op_t dequantize_rhs {
                id_gen.get_id(), graph::op_kind::Dequantize, "dequantize_rhs"};
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize_common);
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_lhs);
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(dequantize_rhs);
        auto in_val_lhs = matmul_lhs->get_input_value(0);
        in_val_lhs->remove_consumer(*matmul_lhs, 0);
        in_val_lhs->remove_consumer(*matmul_rhs, 0);
        auto ref_lt = in_val_lhs->get_logical_tensor();
        auto quantize_common_out_lt = ref_lt;
        quantize_common_out_lt.id = id_gen.get_id();
        quantize_common_out_lt.data_type = graph::data_type::u8;
        quantize_common.add_output(quantize_common_out_lt);
        dequantize_lhs.add_input(quantize_common_out_lt);
        dequantize_rhs.add_input(quantize_common_out_lt);
        auto dequant_lhs_out_lt = ref_lt;
        dequant_lhs_out_lt.id = id_gen.get_id();
        dequant_lhs_out_lt.data_type = graph::data_type::f32;
        auto dequant_rhs_out_lt = ref_lt;
        dequant_rhs_out_lt.id = id_gen.get_id();
        dequant_rhs_out_lt.data_type = graph::data_type::f32;

        if (use_bf16) {
            graph::op_t typecast_common {id_gen.get_id(),
                    graph::op_kind::TypeCast, "typecast_common"};
            graph::op_t typecast_lhs {
                    id_gen.get_id(), graph::op_kind::TypeCast, "typecast_lhs"};
            graph::op_t typecast_rhs {
                    id_gen.get_id(), graph::op_kind::TypeCast, "typecast_rhs"};

            auto typecast_common_out_lt = ref_lt;
            typecast_common_out_lt.id = id_gen.get_id();
            typecast_common_out_lt.data_type = graph::data_type::f32;
            typecast_common.add_input(in_val_lhs);
            typecast_common.add_output(typecast_common_out_lt);

            quantize_common.add_input(typecast_common_out_lt);
            dequantize_lhs.add_output(dequant_lhs_out_lt);
            dequantize_rhs.add_output(dequant_rhs_out_lt);
            typecast_lhs.add_input(dequant_lhs_out_lt);
            typecast_rhs.add_input(dequant_rhs_out_lt);
            auto typecast_lhs_out_lt = ref_lt;
            typecast_lhs_out_lt.id = id_gen.get_id();
            typecast_lhs_out_lt.data_type = graph::data_type::bf16;
            auto typecast_rhs_out_lt = ref_lt;
            typecast_rhs_out_lt.id = id_gen.get_id();
            typecast_rhs_out_lt.data_type = graph::data_type::bf16;

            auto typecast_lhs_out_val = std::make_shared<graph::value_t>(
                    typecast_lhs_out_lt, false);
            matmul_lhs->connect_input(0, typecast_lhs_out_val);
            typecast_lhs.add_output(typecast_lhs_out_val);
            auto typecast_rhs_out_val = std::make_shared<graph::value_t>(
                    typecast_rhs_out_lt, false);
            matmul_rhs->connect_input(0, typecast_rhs_out_val);
            typecast_rhs.add_output(typecast_rhs_out_val);

            agraph->add_op(&typecast_common);
            agraph->add_op(&typecast_lhs);
            agraph->add_op(&typecast_rhs);
        } else {
            quantize_common.add_input(in_val_lhs);

            auto dequant_lhs_out_val = std::make_shared<graph::value_t>(
                    dequant_lhs_out_lt, false);
            matmul_lhs->connect_input(0, dequant_lhs_out_val);
            dequantize_lhs.add_output(dequant_lhs_out_val);
            auto dequant_rhs_out_val = std::make_shared<graph::value_t>(
                    dequant_rhs_out_lt, false);
            matmul_rhs->connect_input(0, dequant_rhs_out_val);
            dequantize_rhs.add_output(dequant_rhs_out_val);
        }
        agraph->add_op(&quantize_common);
        agraph->add_op(&dequantize_lhs);
        agraph->add_op(&dequantize_rhs);

        insert_quantization_after_output(agraph, id_gen, output_lt, use_bf16);
    }
}

/* from GPT-J
[IN0](dtype)  [IN1](dtype) [IN2](dtype)   [IN3](dtype)
      \            /            |              /
      mul1      mul2            |             /
        \       /               |            /
           add                 to1          /
            \                  /           /
                concat1                   /
                   |                     /
                permute                 /
                   \                   /
                         concat2
                            |
                           to2
                            |
                           to3
                            |
                          quant
                            |
                        [OUT0](dtype)
*/
inline void add_gptj_concat_subgraph(
        graph::graph_t *agraph, bool add_quant = false) {
    std::vector<graph::dim_t> shape0 = {32, 1024, 256};
    std::vector<graph::dim_t> shape1 = {32, 1024, 256};
    std::vector<graph::dim_t> shape2 = {32, 1024, 256};
    std::vector<graph::dim_t> shape3 = {32, 256, 2048};
    graph::dim_t concat1_axis = 1;
    std::vector<int64_t> permute_order = {0, 2, 1};
    graph::dim_t concat2_axis = 1;

    size_t op_idx = 0, logical_tensor_idx = 0;
    graph::logical_tensor_t in0, in1, in2, in3;
    in0 = utils::logical_tensor_init(
            logical_tensor_idx++, shape0, graph::data_type::f32);
    in1 = utils::logical_tensor_init(
            logical_tensor_idx++, shape1, graph::data_type::f32);
    in2 = utils::logical_tensor_init(
            logical_tensor_idx++, shape2, graph::data_type::bf16);
    in3 = utils::logical_tensor_init(
            logical_tensor_idx++, shape3, graph::data_type::f32);

    graph::op_t mul1 {op_idx++, graph::op_kind::Multiply, "mul1"};
    graph::logical_tensor_t mul1_out = utils::logical_tensor_init(
            logical_tensor_idx++, shape0, graph::data_type::f32);
    mul1.add_input(in0);
    mul1.add_input(in1);
    mul1.add_output(mul1_out);
    agraph->add_op(&mul1);

    graph::op_t mul2 {op_idx++, graph::op_kind::Multiply, "mul2"};
    graph::logical_tensor_t mul2_out = utils::logical_tensor_init(
            logical_tensor_idx++, shape0, graph::data_type::f32);
    mul2.add_input(in0);
    mul2.add_input(in1);
    mul2.add_output(mul2_out);
    agraph->add_op(&mul2);

    graph::op_t add_layer {op_idx++, graph::op_kind::Add, "add_layer"};
    graph::logical_tensor_t add_out = utils::logical_tensor_init(
            logical_tensor_idx++, shape0, graph::data_type::f32);
    add_layer.add_input(mul1_out);
    add_layer.add_input(mul2_out);
    add_layer.add_output(add_out);
    agraph->add_op(&add_layer);

    graph::op_t to_layer1 {op_idx++, graph::op_kind::TypeCast, "to_layer1"};
    graph::logical_tensor_t to1_out = utils::logical_tensor_init(
            logical_tensor_idx++, shape2, graph::data_type::f32);
    to_layer1.add_input(in2);
    to_layer1.add_output(to1_out);
    agraph->add_op(&to_layer1);

    graph::op_t concat_layer1
            = {op_idx++, graph::op_kind::Concat, "concat_layer1"};
    concat_layer1.set_attr<int64_t>(graph::op_attr::axis, concat1_axis);
    std::vector<graph::dim_t> concat1_out_shape = shape0;
    concat1_out_shape[concat1_axis] += shape2[concat1_axis];
    graph::logical_tensor_t concat1_out = utils::logical_tensor_init(
            logical_tensor_idx++, concat1_out_shape, graph::data_type::f32);
    concat_layer1.add_input(add_out);
    concat_layer1.add_input(to1_out);
    concat_layer1.add_output(concat1_out);
    agraph->add_op(&concat_layer1);

    graph::op_t permute_layer
            = {op_idx++, graph::op_kind::StaticTranspose, "permute_layer"};
    permute_layer.set_attr(graph::op_attr::order, permute_order);
    std::vector<graph::dim_t> permute_out_shape = concat1_out_shape;
    for (size_t i = 0; i < permute_order.size(); ++i) {
        auto axis = permute_order[i];
        permute_out_shape[i] = concat1_out_shape[axis];
    }
    graph::logical_tensor_t permute_out = utils::logical_tensor_init(
            logical_tensor_idx++, permute_out_shape, graph::data_type::f32);
    permute_layer.add_input(concat1_out);
    permute_layer.add_output(permute_out);
    agraph->add_op(&permute_layer);

    graph::op_t concat_layer2
            = {op_idx++, graph::op_kind::Concat, "concat_layer2"};
    concat_layer2.set_attr<int64_t>(graph::op_attr::axis, concat2_axis);
    std::vector<graph::dim_t> concat2_out_shape = permute_out_shape;
    concat2_out_shape[concat2_axis] += shape3[concat2_axis];
    graph::logical_tensor_t concat2_out = utils::logical_tensor_init(
            logical_tensor_idx++, concat2_out_shape, graph::data_type::f32);
    concat_layer2.add_input(in3);
    concat_layer2.add_input(permute_out);
    concat_layer2.add_output(concat2_out);
    agraph->add_op(&concat_layer2);

    graph::op_t to_layer2 = {op_idx++, graph::op_kind::TypeCast, "to_layer2"};
    graph::logical_tensor_t to2_out = utils::logical_tensor_init(
            logical_tensor_idx++, concat2_out_shape, graph::data_type::bf16);
    to_layer2.add_input(concat2_out);
    to_layer2.add_output(to2_out);
    agraph->add_op(&to_layer2);

    if (add_quant) {
        graph::op_t to_layer3
                = {op_idx++, graph::op_kind::TypeCast, "to_layer3"};
        graph::logical_tensor_t to3_out = utils::logical_tensor_init(
                logical_tensor_idx++, concat2_out_shape, graph::data_type::f32);
        to_layer3.add_input(to2_out);
        to_layer3.add_output(to3_out);
        agraph->add_op(&to_layer3);

        graph::op_t quant_layer {
                op_idx++, graph::op_kind::Quantize, "quant_layer"};
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quant_layer);
        graph::logical_tensor_t quant_out = utils::logical_tensor_init(
                logical_tensor_idx++, concat2_out_shape, graph::data_type::u8);
        quant_layer.add_input(to3_out);
        quant_layer.add_output(quant_out);
        agraph->add_op(&quant_layer);
    }
}

/* from Llama
[IN0](dtype)  [IN1](dtype) [IN2](dtype)
      \            /            /
           add                 /
            |                 /
           to1               /
            \               /
                concat1
                   |
                  to2
                   |
                  to3
                   |
                 quant
                   |
               [OUT0](dtype)
*/
inline void add_llama_concat_subgraph(
        graph::graph_t *agraph, bool add_quant = false) {
    std::vector<graph::dim_t> shape0 = {32, 1024, 256};
    std::vector<graph::dim_t> shape1 = {32, 1024, 256};
    std::vector<graph::dim_t> shape2 = {32, 1024, 256};
    graph::dim_t concat1_axis = 1;

    size_t op_idx = 0, logical_tensor_idx = 0;
    graph::logical_tensor_t in0, in1, in2;
    in0 = utils::logical_tensor_init(
            logical_tensor_idx++, shape0, graph::data_type::bf16);
    in1 = utils::logical_tensor_init(
            logical_tensor_idx++, shape1, graph::data_type::bf16);
    in2 = utils::logical_tensor_init(
            logical_tensor_idx++, shape2, graph::data_type::f32);

    graph::op_t add_layer {op_idx++, graph::op_kind::Add, "add_layer"};
    graph::logical_tensor_t add_out = utils::logical_tensor_init(
            logical_tensor_idx++, shape0, graph::data_type::bf16);
    add_layer.add_input(in0);
    add_layer.add_input(in1);
    add_layer.add_output(add_out);
    agraph->add_op(&add_layer);

    graph::op_t to_layer1 {op_idx++, graph::op_kind::TypeCast, "to_layer1"};
    graph::logical_tensor_t to1_out = utils::logical_tensor_init(
            logical_tensor_idx++, shape2, graph::data_type::f32);
    to_layer1.add_input(add_out);
    to_layer1.add_output(to1_out);
    agraph->add_op(&to_layer1);

    graph::op_t concat_layer1
            = {op_idx++, graph::op_kind::Concat, "concat_layer1"};
    concat_layer1.set_attr<int64_t>(graph::op_attr::axis, concat1_axis);
    std::vector<graph::dim_t> concat1_out_shape = shape0;
    concat1_out_shape[concat1_axis] += shape2[concat1_axis];
    graph::logical_tensor_t concat1_out = utils::logical_tensor_init(
            logical_tensor_idx++, concat1_out_shape, graph::data_type::f32);
    concat_layer1.add_input(in2);
    concat_layer1.add_input(to1_out);
    concat_layer1.add_output(concat1_out);
    agraph->add_op(&concat_layer1);

    graph::op_t to_layer2 = {op_idx++, graph::op_kind::TypeCast, "to_layer2"};
    graph::logical_tensor_t to2_out = utils::logical_tensor_init(
            logical_tensor_idx++, concat1_out_shape, graph::data_type::bf16);
    to_layer2.add_input(concat1_out);
    to_layer2.add_output(to2_out);
    agraph->add_op(&to_layer2);

    if (add_quant) {
        graph::op_t to_layer3
                = {op_idx++, graph::op_kind::TypeCast, "to_layer3"};
        graph::logical_tensor_t to3_out = utils::logical_tensor_init(
                logical_tensor_idx++, concat1_out_shape, graph::data_type::f32);
        to_layer3.add_input(to2_out);
        to_layer3.add_output(to3_out);
        agraph->add_op(&to_layer3);

        graph::op_t quant_layer {
                op_idx++, graph::op_kind::Quantize, "quant_layer"};
        DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quant_layer);
        graph::logical_tensor_t quant_out = utils::logical_tensor_init(
                logical_tensor_idx++, concat1_out_shape, graph::data_type::u8);
        quant_layer.add_input(to3_out);
        quant_layer.add_output(quant_out);
        agraph->add_op(&quant_layer);
    }
}

static void construct_starcoder_mha_base(graph::graph_t *agraph,
        utils::id_generator &id_gen, graph::data_type_t dtype,
        int batch_size = 1, int query_seq_len = 16, int key_seq_len = 16,
        int num_head = 32, int head_dim = 4096) {
    int size_per_head = head_dim / num_head;
    std::vector<graph::dim_t> COND_MASK_SHAPE {batch_size, 1, 1, key_seq_len};
    std::vector<graph::dim_t> QUERY_SHAPE {
            batch_size, query_seq_len, num_head, size_per_head};
    std::vector<graph::dim_t> KEY_SHAPE {
            batch_size, 1, size_per_head, key_seq_len};
    std::vector<graph::dim_t> VALUE_SHAPE {
            batch_size, query_seq_len, key_seq_len, size_per_head};
    std::vector<graph::dim_t> CONST_SHAPE {1};

    graph::logical_tensor_t query_matmul_input, key_matmul_input,
            value_matmul_input;
    query_matmul_input
            = utils::logical_tensor_init(id_gen.get_id(), QUERY_SHAPE, dtype);
    key_matmul_input
            = utils::logical_tensor_init(id_gen.get_id(), KEY_SHAPE, dtype);
    value_matmul_input
            = utils::logical_tensor_init(id_gen.get_id(), VALUE_SHAPE, dtype);

    graph::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t fscore_scale, fscore_scale_out;
    fscore_scale
            = utils::logical_tensor_init(id_gen.get_id(), CONST_SHAPE, dtype);
    fscore_scale_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t fscore_mask, fscore_mask_else, fscore_select_out,
            softmax_out;
    fscore_mask = utils::logical_tensor_init(
            id_gen.get_id(), COND_MASK_SHAPE, graph::data_type::boolean);
    fscore_mask_else
            = utils::logical_tensor_init(id_gen.get_id(), CONST_SHAPE, dtype);
    fscore_select_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);
    softmax_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            id_gen.get_id(), dtype, graph::layout_type::strided);

    graph::op_t matmul_qk {
            id_gen.get_id(), graph::op_kind::MatMul, "matmul_qk"};

    graph::op_t fscore_rescale {
            id_gen.get_id(), graph::op_kind::Multiply, "fscore_rescale"};
    fscore_rescale.set_attr(
            graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t fscore_select {
            id_gen.get_id(), graph::op_kind::Select, "fscore_select"};
    fscore_select.set_attr(
            graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t softmax {id_gen.get_id(), graph::op_kind::SoftMax, "softmax"};
    softmax.set_attr(graph::op_attr::axis, (int64_t)3);

    graph::op_t matmul_v {id_gen.get_id(), graph::op_kind::MatMul, "matmul_v"};

    matmul_qk.add_input(query_matmul_input);
    matmul_qk.add_input(key_matmul_input);
    matmul_qk.add_output(matmul_qk_out);
    fscore_rescale.add_input(matmul_qk_out);
    fscore_rescale.add_input(fscore_scale);
    fscore_rescale.add_output(fscore_scale_out);
    fscore_select.add_input(fscore_mask);
    fscore_select.add_input(fscore_scale_out);
    fscore_select.add_input(fscore_mask_else);
    fscore_select.add_output(fscore_select_out);
    softmax.add_input(fscore_select_out);
    softmax.add_output(softmax_out);
    matmul_v.add_input(softmax_out);
    matmul_v.add_input(value_matmul_input);
    matmul_v.add_output(matmul_v_out);

    agraph->add_op(&matmul_qk);
    agraph->add_op(&fscore_rescale);
    agraph->add_op(&fscore_select);
    agraph->add_op(&softmax);
    agraph->add_op(&matmul_v);
}

inline void construct_starcoder_mha_subgraph(graph::graph_t *agraph,
        utils::id_generator &id_gen, bool use_bf16 = false,
        bool use_int8 = false, int batch_size = 1, int query_seq_len = 16,
        int key_seq_len = 16, int num_head = 32, int head_dim = 4096) {
    construct_starcoder_mha_base(agraph, id_gen,
            use_bf16 ? graph::data_type::bf16 : graph::data_type::f32,
            batch_size, query_seq_len, key_seq_len, num_head, head_dim);
    auto ops = agraph->get_ops();
    // insert quantize / dequantize / typecast
    if (use_int8) {
        for (const auto &op : ops) {
            if (op->get_kind() == graph::op_kind::MatMul) {
                insert_quantization_before_op(
                        agraph, id_gen, op.get(), 0, use_bf16);
                insert_quantization_before_op(
                        agraph, id_gen, op.get(), 1, use_bf16);
            }
        }
    }
}

} // namespace utils
} // namespace compiler
} // namespace unit
} // namespace tests
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
