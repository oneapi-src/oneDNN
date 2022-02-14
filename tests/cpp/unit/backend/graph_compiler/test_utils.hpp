/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "compiler/config/context.hpp"
#include "cpp/unit/utils.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#define REQUIRE_AVX512() \
    if (!::sc::get_default_context()->machine_.cpu_flags_.fAVX512F) { \
        GTEST_SKIP(); \
        return; \
    }

#define REQUIRE_VNNI_AMXINT8() \
    REQUIRE_AVX512() \
    if (!::sc::get_default_context()->machine_.cpu_flags_.fAVX512VNNI \
            && !::sc::get_default_context() \
                        ->machine_.cpu_flags_.fAVX512AMXINT8) { \
        GTEST_SKIP(); \
        return; \
    }

#define REQUIRE_BF16_AMXBF16() \
    REQUIRE_AVX512() \
    if (!::sc::get_default_context()->machine_.cpu_flags_.fAVX512BF16 \
            && !::sc::get_default_context() \
                        ->machine_.cpu_flags_.fAVX512AMXBF16) { \
        GTEST_SKIP(); \
        return; \
    }

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

typedef enum {
    RESHAPE_INCLUDED = 0,
    RESHAPE_EXCLUDED = 1,
} quantize_position_t;

// this function can add 4 different styles of MHA graph
// 1. fp32; 2. bf16; 3. pairwise-quantized int8 graph; 4. int8 in/out int8 graph
inline void add_MHA_subgraph(impl::graph_t *agraph, bool is_quantized = true,
        bool paired_quantize = true, bool use_bf16 = false,
        int batch_size = 128, int seq_len = 384, int num_head = 16,
        int head_dim = 1024) {
    assertm(is_quantized || paired_quantize,
            "paired_quantize should be true when is_quantized is false");
    assertm((!use_bf16) || (!is_quantized),
            "is_quantized should be false when use_bf16 is true");
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    int size_per_head = head_dim / num_head;
    std::vector<impl::dim_t> MIXED_LAYER_INPUT_SHAPE {
            batch_size, seq_len, head_dim};
    std::vector<impl::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<impl::dim_t> QKV_RESHAPED_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<impl::dim_t> QKV_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<impl::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<impl::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<impl::dim_t> MATMUL_V_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<impl::dim_t> CONST_SHAPE {1};
    std::vector<impl::dim_t> OUTPUT_SHAPE {batch_size, seq_len, head_dim};

    impl::logical_tensor_t query_gemm_out_flt, qk_bmm_wei_flt,
            value_bmm_wei_flt;
    if (!paired_quantize) {
        query_gemm_out_flt = utils::logical_tensor_init(logical_tensor_idx++,
                MIXED_LAYER_INPUT_SHAPE, impl::data_type::u8);
        qk_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
                MIXED_LAYER_INPUT_SHAPE, impl::data_type::u8);
        value_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
                MIXED_LAYER_INPUT_SHAPE, impl::data_type::u8);
    } else {
        query_gemm_out_flt = utils::logical_tensor_init(logical_tensor_idx++,
                MIXED_LAYER_INPUT_SHAPE,
                use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
        qk_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
                MIXED_LAYER_INPUT_SHAPE,
                use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
        value_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
                MIXED_LAYER_INPUT_SHAPE,
                use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    }

    impl::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(logical_tensor_idx++,
            EXTENDED_ATTENTION_MASK_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t query_gemm_out_q, qk_bmm_wei_q, value_bmm_wei_q;
    query_gemm_out_q = utils::logical_tensor_init(
            logical_tensor_idx++, MIXED_LAYER_INPUT_SHAPE, impl::data_type::u8);
    qk_bmm_wei_q = utils::logical_tensor_init(
            logical_tensor_idx++, MIXED_LAYER_INPUT_SHAPE, impl::data_type::u8);
    value_bmm_wei_q = utils::logical_tensor_init(
            logical_tensor_idx++, MIXED_LAYER_INPUT_SHAPE, impl::data_type::u8);

    impl::logical_tensor_t query_gemm_out_deq, qk_bmm_wei_deq,
            value_bmm_wei_deq;
    query_gemm_out_deq = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, impl::data_type::f32);
    qk_bmm_wei_deq = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, impl::data_type::f32);
    value_bmm_wei_deq = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t query_reshape_out, query_transpose_out;
    query_reshape_out = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_RESHAPED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    query_transpose_out = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_TRANSPOSED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t key_reshape_out, key_transpose_out;
    key_reshape_out = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_RESHAPED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    key_transpose_out = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_TRANSPOSED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t key_transpose_out2;
    key_transpose_out2 = utils::logical_tensor_init(logical_tensor_idx++,
            KEY_TRANSPOSED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t fscore_scale, fscore_div_out;
    fscore_scale = utils::logical_tensor_init(logical_tensor_idx++, CONST_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    fscore_div_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t fscore_add_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    softmax_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t value_reshape_out, value_transpose_out;
    value_reshape_out = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_RESHAPED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    value_transpose_out = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_TRANSPOSED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_V_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t softmax_out_q, softmax_out_deq;
    softmax_out_q = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::u8);
    softmax_out_deq = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_RESHAPED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    context_reshape_out
            = utils::logical_tensor_init(logical_tensor_idx++, OUTPUT_SHAPE,
                    use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t MHA_out_q, MHA_out_deq;
    MHA_out_q = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, impl::data_type::u8);
    MHA_out_deq = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, impl::data_type::f32);

    // add quantize-dequantize ops for all three inputs
    impl::op_t quantize_query_gemm {
            op_idx++, impl::op_kind::Quantize, "quantize_query_gemm"};
    impl::op_t quantize_key_gemm {
            op_idx++, impl::op_kind::Quantize, "quantize_key_gemm"};
    impl::op_t quantize_value_gemm {
            op_idx++, impl::op_kind::Quantize, "quantize_value_gemm"};
    quantize_query_gemm.set_attr("scales", std::vector<float>({0.12f}));
    quantize_query_gemm.set_attr("zps", std::vector<int64_t>({2}));
    quantize_query_gemm.set_attr("qtype", std::string("per_tensor"));
    quantize_query_gemm.set_attr("axis", (int64_t)0);
    quantize_key_gemm.set_attr("scales", std::vector<float>({0.12f}));
    quantize_key_gemm.set_attr("zps", std::vector<int64_t>({2}));
    quantize_key_gemm.set_attr("qtype", std::string("per_tensor"));
    quantize_key_gemm.set_attr("axis", (int64_t)0);
    quantize_value_gemm.set_attr("scales", std::vector<float>({0.12f}));
    quantize_value_gemm.set_attr("zps", std::vector<int64_t>({2}));
    quantize_value_gemm.set_attr("qtype", std::string("per_tensor"));
    quantize_value_gemm.set_attr("axis", (int64_t)0);

    impl::op_t dequantize_query_gemm {
            op_idx++, impl::op_kind::Dequantize, "dequantize_query_gemm"};
    dequantize_query_gemm.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_query_gemm.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_query_gemm.set_attr("qtype", std::string("per_tensor"));
    dequantize_query_gemm.set_attr("axis", (int64_t)0);
    impl::op_t dequantize_key_gemm {
            op_idx++, impl::op_kind::Dequantize, "dequantize_key_gemm"};
    dequantize_key_gemm.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_key_gemm.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_key_gemm.set_attr("qtype", std::string("per_tensor"));
    dequantize_key_gemm.set_attr("axis", (int64_t)0);
    impl::op_t dequantize_value_gemm {
            op_idx++, impl::op_kind::Dequantize, "dequantize_value_gemm"};
    dequantize_value_gemm.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_value_gemm.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_value_gemm.set_attr("qtype", std::string("per_tensor"));
    dequantize_value_gemm.set_attr("axis", (int64_t)0);

    // reshape + transpose for query + key
    impl::op_t query_reshape {
            op_idx++, impl::op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr("shape",
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    query_reshape.set_attr("special_zero", false);
    impl::op_t query_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_reshape {
            op_idx++, impl::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr("shape",
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    key_reshape.set_attr("special_zero", false);
    impl::op_t key_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_transpose2 {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr("order", std::vector<int64_t> {0, 1, 3, 2});
    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};

    impl::op_t fscore_div {op_idx++, impl::op_kind::Divide, "fscore_div"};
    fscore_div.set_attr("auto_broadcast", std::string("numpy"));
    impl::op_t fscore_add {op_idx++, impl::op_kind::Add, "fscore_add"};
    fscore_add.set_attr("auto_broadcast", std::string("numpy"));
    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr("axis", (int64_t)3);

    // quantize-dequantize softmax's output
    impl::op_t quantize_softmax {
            op_idx++, impl::op_kind::Quantize, "quantize_softmax"};
    impl::op_t dequantize_softmax {
            op_idx++, impl::op_kind::Dequantize, "dequantize_softmax"};
    quantize_softmax.set_attr("scales", std::vector<float>({0.12f}));
    quantize_softmax.set_attr("zps", std::vector<int64_t>({2}));
    quantize_softmax.set_attr("qtype", std::string("per_tensor"));
    quantize_softmax.set_attr("axis", (int64_t)0);
    dequantize_softmax.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_softmax.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_softmax.set_attr("qtype", std::string("per_tensor"));
    dequantize_softmax.set_attr("axis", (int64_t)0);

    // reshape + transpose for value
    impl::op_t value_reshape {
            op_idx++, impl::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr("shape",
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    value_reshape.set_attr("special_zero", false);
    impl::op_t value_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t reshape_output {
            op_idx++, impl::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr("special_zero", false);
    reshape_output.set_attr(
            "shape", std::vector<int64_t> {batch_size, seq_len, head_dim});

    // quantize dequantize output
    impl::op_t quantize_output {
            op_idx++, impl::op_kind::Quantize, "quantize_output"};
    quantize_output.set_attr("scales", std::vector<float>({0.12f}));
    quantize_output.set_attr("zps", std::vector<int64_t>({2}));
    quantize_output.set_attr("qtype", std::string("per_tensor"));
    quantize_output.set_attr("axis", (int64_t)0);
    impl::op_t dequantize_output {
            op_idx++, impl::op_kind::Dequantize, "dequantize_output"};
    dequantize_output.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_output.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_output.set_attr("qtype", std::string("per_tensor"));
    dequantize_output.set_attr("axis", (int64_t)0);

    if (is_quantized) {
        if (paired_quantize) {
            quantize_query_gemm.add_input(query_gemm_out_flt);
            quantize_query_gemm.add_output(query_gemm_out_q);
            dequantize_query_gemm.add_input(query_gemm_out_q);
            dequantize_query_gemm.add_output(query_gemm_out_deq);
            quantize_key_gemm.add_input(qk_bmm_wei_flt);
            quantize_key_gemm.add_output(qk_bmm_wei_q);
            dequantize_key_gemm.add_input(qk_bmm_wei_q);
            dequantize_key_gemm.add_output(qk_bmm_wei_deq);
            quantize_value_gemm.add_input(value_bmm_wei_flt);
            quantize_value_gemm.add_output(value_bmm_wei_q);
            dequantize_value_gemm.add_input(value_bmm_wei_q);
            dequantize_value_gemm.add_output(value_bmm_wei_deq);
        } else {
            dequantize_query_gemm.add_input(query_gemm_out_flt);
            dequantize_query_gemm.add_output(query_gemm_out_deq);
            dequantize_key_gemm.add_input(qk_bmm_wei_flt);
            dequantize_key_gemm.add_output(qk_bmm_wei_deq);
            dequantize_value_gemm.add_input(value_bmm_wei_flt);
            dequantize_value_gemm.add_output(value_bmm_wei_deq);
        }
        query_reshape.add_input(query_gemm_out_deq);
        key_reshape.add_input(qk_bmm_wei_deq);
        value_reshape.add_input(value_bmm_wei_deq);
    } else {
        query_reshape.add_input(query_gemm_out_flt);
        key_reshape.add_input(qk_bmm_wei_flt);
        value_reshape.add_input(value_bmm_wei_flt);
    }

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

    if (is_quantized) {
        quantize_softmax.add_input(softmax_out);
        quantize_softmax.add_output(softmax_out_q);
        dequantize_softmax.add_input(softmax_out_q);
        dequantize_softmax.add_output(softmax_out_deq);
    }

    value_reshape.add_output(value_reshape_out);
    value_transpose.add_input(value_reshape_out);
    value_transpose.add_output(value_transpose_out);
    if (is_quantized) {
        matmul_v.add_input(softmax_out_deq);
    } else {
        matmul_v.add_input(softmax_out);
    }
    matmul_v.add_input(value_transpose_out);
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);
    reshape_output.add_input(context_transpose_out);
    reshape_output.add_output(context_reshape_out);
    if (is_quantized) {
        quantize_output.add_input(context_reshape_out);
        quantize_output.add_output(MHA_out_q);
        if (paired_quantize) {
            dequantize_output.add_input(MHA_out_q);
            dequantize_output.add_output(MHA_out_deq);
        }
    }

    // adding ops
    if (is_quantized) {
        if (paired_quantize) {
            agraph->add_op(&quantize_query_gemm);
            agraph->add_op(&quantize_key_gemm);
            agraph->add_op(&quantize_value_gemm);
        }
        agraph->add_op(&dequantize_query_gemm);
        agraph->add_op(&dequantize_key_gemm);
        agraph->add_op(&dequantize_value_gemm);
    }
    agraph->add_op(&query_reshape);
    agraph->add_op(&query_transpose);
    agraph->add_op(&key_reshape);
    agraph->add_op(&key_transpose);
    agraph->add_op(&key_transpose2);
    agraph->add_op(&matmul_qk);
    agraph->add_op(&fscore_div);
    agraph->add_op(&fscore_add);
    agraph->add_op(&softmax);
    if (is_quantized) {
        agraph->add_op(&quantize_softmax);
        agraph->add_op(&dequantize_softmax);
    }
    agraph->add_op(&value_reshape);
    agraph->add_op(&value_transpose);
    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
    agraph->add_op(&reshape_output);
    if (is_quantized) {
        agraph->add_op(&quantize_output);
        if (paired_quantize) { agraph->add_op(&dequantize_output); }
    }
}

// add MHA fp32/bf16/int8 graph with an extra reorder
inline void add_MHA_subgraph_alternative(impl::graph_t *agraph,
        bool use_bf16 = false, bool use_int8 = false, int batch_size = 128,
        int seq_len = 384, int num_head = 16, int head_dim = 1024) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    int size_per_head = head_dim / num_head;
    std::vector<impl::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<impl::dim_t> QKV_RESHAPED_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<impl::dim_t> QKV_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<impl::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<impl::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<impl::dim_t> MATMUL_V_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<impl::dim_t> CONST_SHAPE {1};

    impl::logical_tensor_t query_dequantize_input, key_dequantize_input,
            value_dequantize_input;
    query_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, impl::data_type::u8);
    key_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, impl::data_type::u8);
    value_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, impl::data_type::u8);

    impl::logical_tensor_t query_typecast_input, key_typecast_input,
            value_typecast_input;
    query_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, impl::data_type::f32);
    key_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, impl::data_type::f32);
    value_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t query_matmul_input, key_matmul_input,
            value_matmul_input;
    query_matmul_input = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_TRANSPOSED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    key_matmul_input = utils::logical_tensor_init(logical_tensor_idx++,
            KEY_TRANSPOSED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    value_matmul_input = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_TRANSPOSED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(logical_tensor_idx++,
            EXTENDED_ATTENTION_MASK_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t fscore_scale, fscore_div_out;
    fscore_scale = utils::logical_tensor_init(logical_tensor_idx++, CONST_SHAPE,
            impl::data_type::f32); // fscore_scale is always fp32
    fscore_div_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t fscore_add_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    softmax_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_QK_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t softmax_cast_out;
    softmax_cast_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t softmax_quantize_out;
    softmax_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, impl::data_type::u8);
    impl::logical_tensor_t softmax_dequantize_out;
    softmax_dequantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t softmax_dequantize_out_cast;
    softmax_dequantize_out_cast
            = utils::logical_tensor_init(logical_tensor_idx++,
                    MATMUL_QK_OUTPUT_SHAPE, impl::data_type::bf16);

    impl::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(logical_tensor_idx++,
            MATMUL_V_OUTPUT_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t context_transpose_out, context_reorder_out;
    context_transpose_out = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_RESHAPED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);
    context_reorder_out = utils::logical_tensor_init(logical_tensor_idx++,
            QKV_RESHAPED_SHAPE,
            use_bf16 ? impl::data_type::bf16 : impl::data_type::f32);

    impl::logical_tensor_t context_cast_out;
    context_cast_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t context_quantize_out;
    context_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, impl::data_type::u8);

    impl::op_t dequantize_query {
            op_idx++, impl::op_kind::Dequantize, "dequantize_query"};
    dequantize_query.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_query.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_query.set_attr("qtype", std::string("per_tensor"));
    dequantize_query.set_attr("axis", (int64_t)0);
    impl::op_t dequantize_key {
            op_idx++, impl::op_kind::Dequantize, "dequantize_key"};
    dequantize_key.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_key.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_key.set_attr("qtype", std::string("per_tensor"));
    dequantize_key.set_attr("axis", (int64_t)0);
    impl::op_t dequantize_value {
            op_idx++, impl::op_kind::Dequantize, "dequantize_value"};
    dequantize_value.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_value.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_value.set_attr("qtype", std::string("per_tensor"));
    dequantize_value.set_attr("axis", (int64_t)0);
    impl::op_t typecast_query {
            op_idx++, impl::op_kind::TypeCast, "typecast_query"};
    impl::op_t typecast_key {op_idx++, impl::op_kind::TypeCast, "typecast_key"};
    impl::op_t typecast_value {
            op_idx++, impl::op_kind::TypeCast, "typecast_value"};

    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};

    impl::op_t fscore_div {op_idx++, impl::op_kind::Divide, "fscore_div"};
    fscore_div.set_attr("auto_broadcast", std::string("numpy"));
    impl::op_t fscore_add {op_idx++, impl::op_kind::Add, "fscore_add"};
    fscore_add.set_attr("auto_broadcast", std::string("numpy"));
    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr("axis", (int64_t)3);

    impl::op_t softmax_cast {op_idx++, impl::op_kind::TypeCast, "softmax_cast"};
    impl::op_t quantize_softmax {
            op_idx++, impl::op_kind::Quantize, "quantize_softmax"};
    quantize_softmax.set_attr("scales", std::vector<float>({0.12f}));
    quantize_softmax.set_attr("zps", std::vector<int64_t>({2}));
    quantize_softmax.set_attr("qtype", std::string("per_tensor"));
    quantize_softmax.set_attr("axis", (int64_t)0);
    impl::op_t dequantize_softmax {
            op_idx++, impl::op_kind::Dequantize, "dequantize_softmax"};
    dequantize_softmax.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_softmax.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_softmax.set_attr("qtype", std::string("per_tensor"));
    dequantize_softmax.set_attr("axis", (int64_t)0);

    impl::op_t dequantize_softmax_cast {
            op_idx++, impl::op_kind::TypeCast, "dequantize_softmax_cast"};

    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t reorder_output {
            op_idx++, impl::op_kind::Reorder, "reorder_output"};

    impl::op_t typecast_output {
            op_idx++, impl::op_kind::TypeCast, "typecast_output"};
    impl::op_t quantize_output {
            op_idx++, impl::op_kind::Quantize, "quantize_output"};
    quantize_output.set_attr("scales", std::vector<float>({0.12f}));
    quantize_output.set_attr("zps", std::vector<int64_t>({2}));
    quantize_output.set_attr("qtype", std::string("per_tensor"));
    quantize_output.set_attr("axis", (int64_t)0);

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
    agraph->add_op(&reorder_output);

    if (use_int8) {
        agraph->add_op(&quantize_output);
        if (use_bf16) { agraph->add_op(&typecast_output); }
    }
}

// this function can add a simple MHA without intermediate tensor shape
// aiming for testing infer shape
inline void add_MHA_infer_shape(impl::graph_t *agraph, int batch_size = 128,
        int seq_len = 384, int num_head = 16, int head_dim = 1024) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    int size_per_head = head_dim / num_head;
    std::vector<impl::dim_t> MIXED_LAYER_INPUT_SHAPE {
            batch_size, seq_len, head_dim};
    std::vector<impl::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<impl::dim_t> CONST_SHAPE {1};
    std::vector<impl::dim_t> OUTPUT_SHAPE {batch_size, seq_len, head_dim};

    impl::logical_tensor_t query_gemm_out_flt, qk_bmm_wei_flt,
            value_bmm_wei_flt;
    query_gemm_out_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, impl::data_type::f32);
    qk_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, impl::data_type::f32);
    value_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(logical_tensor_idx++,
            EXTENDED_ATTENTION_MASK_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t query_reshape_out, query_transpose_out;
    query_reshape_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);
    query_transpose_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t key_reshape_out, key_transpose_out;
    key_reshape_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);
    key_transpose_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t key_transpose_out2;
    key_transpose_out2 = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t fscore_scale, fscore_div_out;
    fscore_scale = utils::logical_tensor_init(
            logical_tensor_idx++, CONST_SHAPE, impl::data_type::f32);
    fscore_div_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t fscore_add_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);
    softmax_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t value_reshape_out, value_transpose_out;
    value_reshape_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);
    value_transpose_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t softmax_out_q, softmax_out_deq;
    softmax_out_q = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);
    softmax_out_deq = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(logical_tensor_idx++,
            impl::data_type::f32, impl::layout_type::strided);
    context_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, impl::data_type::f32);

    // reshape + transpose for query + key
    impl::op_t query_reshape {
            op_idx++, impl::op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr("shape",
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    query_reshape.set_attr("special_zero", false);
    impl::op_t query_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_reshape {
            op_idx++, impl::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr("shape",
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    key_reshape.set_attr("special_zero", false);
    impl::op_t key_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_transpose2 {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr("order", std::vector<int64_t> {0, 1, 3, 2});
    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};

    impl::op_t fscore_div {op_idx++, impl::op_kind::Divide, "fscore_div"};
    fscore_div.set_attr("auto_broadcast", std::string("numpy"));
    impl::op_t fscore_add {op_idx++, impl::op_kind::Add, "fscore_add"};
    fscore_add.set_attr("auto_broadcast", std::string("numpy"));
    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr("axis", (int64_t)3);

    // reshape + transpose for value
    impl::op_t value_reshape {
            op_idx++, impl::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr("shape",
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    value_reshape.set_attr("special_zero", false);
    impl::op_t value_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t reshape_output {
            op_idx++, impl::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr("special_zero", false);
    reshape_output.set_attr(
            "shape", std::vector<int64_t> {batch_size, seq_len, head_dim});

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

inline void get_int8_MHA_subgraph_varients(impl::graph_t *agraph,
        bool use_div = true,
        const std::vector<quantize_position_t> &quantize_positions
        = std::vector<quantize_position_t>(4, RESHAPE_INCLUDED),
        int add_inport = 0, int batch_size = 128, int seq_len = 384,
        int num_head = 16, int head_dim = 1024) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    int size_per_head = head_dim / num_head;
    std::vector<impl::dim_t> MIXED_LAYER_INPUT_SHAPE {
            batch_size, seq_len, head_dim};
    std::vector<impl::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<impl::dim_t> QKV_RESHAPED_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<impl::dim_t> QKV_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<impl::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<impl::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<impl::dim_t> MATMUL_V_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<impl::dim_t> CONST_SHAPE {1};
    std::vector<impl::dim_t> OUTPUT_SHAPE {batch_size, seq_len, head_dim};

    impl::logical_tensor_t query_gemm_out_flt, qk_bmm_wei_flt,
            value_bmm_wei_flt, attention_mask_flt;
    query_gemm_out_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, impl::data_type::f32);
    qk_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, impl::data_type::f32);
    value_bmm_wei_flt = utils::logical_tensor_init(logical_tensor_idx++,
            MIXED_LAYER_INPUT_SHAPE, impl::data_type::f32);
    attention_mask_flt = utils::logical_tensor_init(logical_tensor_idx++,
            EXTENDED_ATTENTION_MASK_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t query_gemm_out_q, qk_bmm_wei_q, value_bmm_wei_q;
    query_gemm_out_q = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[0] ? QKV_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            impl::data_type::u8);
    qk_bmm_wei_q = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[1] ? KEY_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            impl::data_type::u8);
    value_bmm_wei_q = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[2] ? QKV_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            impl::data_type::u8);

    impl::logical_tensor_t query_gemm_out_deq, qk_bmm_wei_deq,
            value_bmm_wei_deq;
    query_gemm_out_deq = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[0] ? QKV_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            impl::data_type::f32);
    qk_bmm_wei_deq = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[1] ? KEY_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            impl::data_type::f32);
    value_bmm_wei_deq = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[2] ? QKV_TRANSPOSED_SHAPE
                                  : MIXED_LAYER_INPUT_SHAPE,
            impl::data_type::f32);

    impl::logical_tensor_t query_reshape_out, query_transpose_out;
    query_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, impl::data_type::f32);
    query_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t key_reshape_out, key_transpose_out;
    key_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, impl::data_type::f32);
    key_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t key_transpose_out2;
    key_transpose_out2 = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t fscore_scale, fscore_div_out;
    fscore_scale = utils::logical_tensor_init(
            logical_tensor_idx++, CONST_SHAPE, impl::data_type::f32);
    fscore_div_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t fscore_add_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, impl::data_type::f32);
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t value_reshape_out, value_transpose_out;
    value_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, impl::data_type::f32);
    value_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t softmax_out_q, softmax_out_deq;
    softmax_out_q = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, impl::data_type::u8);
    softmax_out_deq = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, impl::data_type::f32);
    context_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t context_out_q, context_out_deq;
    context_out_q = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[3] ? MATMUL_V_OUTPUT_SHAPE : OUTPUT_SHAPE,
            impl::data_type::u8);
    context_out_deq = utils::logical_tensor_init(logical_tensor_idx++,
            quantize_positions[3] ? MATMUL_V_OUTPUT_SHAPE : OUTPUT_SHAPE,
            impl::data_type::f32);

    // add quantize-dequantize
    impl::op_t quantize_query_gemm {
            op_idx++, impl::op_kind::Quantize, "quantize_query_gemm"};
    impl::op_t quantize_key_gemm {
            op_idx++, impl::op_kind::Quantize, "quantize_key_gemm"};
    impl::op_t quantize_value_gemm {
            op_idx++, impl::op_kind::Quantize, "quantize_value_gemm"};
    quantize_query_gemm.set_attr("scales", std::vector<float>({0.12f}));
    quantize_query_gemm.set_attr("zps", std::vector<int64_t>({2}));
    quantize_query_gemm.set_attr("qtype", std::string("per_tensor"));
    quantize_query_gemm.set_attr("axis", (int64_t)0);
    quantize_key_gemm.set_attr("scales", std::vector<float>({0.12f}));
    quantize_key_gemm.set_attr("zps", std::vector<int64_t>({2}));
    quantize_key_gemm.set_attr("qtype", std::string("per_tensor"));
    quantize_key_gemm.set_attr("axis", (int64_t)0);
    quantize_value_gemm.set_attr("scales", std::vector<float>({0.12f}));
    quantize_value_gemm.set_attr("zps", std::vector<int64_t>({2}));
    quantize_value_gemm.set_attr("qtype", std::string("per_tensor"));
    quantize_value_gemm.set_attr("axis", (int64_t)0);

    impl::op_t dequantize_query_gemm {
            op_idx++, impl::op_kind::Dequantize, "dequantize_query_gemm"};
    dequantize_query_gemm.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_query_gemm.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_query_gemm.set_attr("qtype", std::string("per_tensor"));
    dequantize_query_gemm.set_attr("axis", (int64_t)0);
    impl::op_t dequantize_key_gemm {
            op_idx++, impl::op_kind::Dequantize, "dequantize_key_gemm"};
    dequantize_key_gemm.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_key_gemm.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_key_gemm.set_attr("qtype", std::string("per_tensor"));
    dequantize_key_gemm.set_attr("axis", (int64_t)0);
    impl::op_t dequantize_value_gemm {
            op_idx++, impl::op_kind::Dequantize, "dequantize_value_gemm"};
    dequantize_value_gemm.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_value_gemm.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_value_gemm.set_attr("qtype", std::string("per_tensor"));
    dequantize_value_gemm.set_attr("axis", (int64_t)0);

    // reshape + transpose for query + key
    impl::op_t query_reshape {
            op_idx++, impl::op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr("special_zero", false);
    query_reshape.set_attr("shape",
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    impl::op_t query_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});

    impl::op_t key_reshape {
            op_idx++, impl::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr("special_zero", false);
    key_reshape.set_attr("shape",
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    impl::op_t key_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_transpose2 {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr("order", std::vector<int64_t> {0, 1, 3, 2});

    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};

    impl::op_t fscore_rescale {op_idx++,
            use_div ? impl::op_kind::Divide : impl::op_kind::Multiply,
            "fscore_rescale"};
    fscore_rescale.set_attr("auto_broadcast", std::string("numpy"));
    impl::op_t fscore_add {op_idx++, impl::op_kind::Add, "fscore_add"};
    fscore_add.set_attr("auto_broadcast", std::string("numpy"));
    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr("axis", (int64_t)3);
    // quantize-dequantize softmax's output
    impl::op_t quantize_softmax {
            op_idx++, impl::op_kind::Quantize, "quantize_softmax"};
    impl::op_t dequantize_softmax {
            op_idx++, impl::op_kind::Dequantize, "dequantize_softmax"};
    quantize_softmax.set_attr("scales", std::vector<float>({0.12f}));
    quantize_softmax.set_attr("zps", std::vector<int64_t>({2}));
    quantize_softmax.set_attr("qtype", std::string("per_tensor"));
    quantize_softmax.set_attr("axis", (int64_t)0);
    dequantize_softmax.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_softmax.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_softmax.set_attr("qtype", std::string("per_tensor"));
    dequantize_softmax.set_attr("axis", (int64_t)0);

    // reshape + transpose for value
    impl::op_t value_reshape {
            op_idx++, impl::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr("special_zero", false);
    value_reshape.set_attr("shape",
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    impl::op_t value_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});

    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr("order", std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t reshape_output {
            op_idx++, impl::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr("special_zero", false);
    reshape_output.set_attr(
            "shape", std::vector<int64_t> {batch_size, seq_len, head_dim});

    // quantize dequantize output
    impl::op_t quantize_output {
            op_idx++, impl::op_kind::Quantize, "quantize_output"};
    quantize_output.set_attr("scales", std::vector<float>({0.12f}));
    quantize_output.set_attr("zps", std::vector<int64_t>({2}));
    quantize_output.set_attr("qtype", std::string("per_tensor"));
    quantize_output.set_attr("axis", (int64_t)0);
    impl::op_t dequantize_output {
            op_idx++, impl::op_kind::Dequantize, "dequantize_output"};
    dequantize_output.set_attr("scales", std::vector<float>({0.12f}));
    dequantize_output.set_attr("zps", std::vector<int64_t>({2}));
    dequantize_output.set_attr("qtype", std::string("per_tensor"));
    dequantize_output.set_attr("axis", (int64_t)0);

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

#endif
