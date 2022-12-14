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
#include "runtime/config.hpp"

namespace dnnl {
namespace graph {
namespace tests {
namespace unit {
namespace compiler {
namespace utils {

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

#define REQUIRE_SINGLE_THREAD() \
    if (::sc::runtime_config_t::get().get_num_threads() != 1) { \
        GTEST_SKIP(); \
        return; \
    }

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

typedef enum {
    RESHAPE_INCLUDED = 0,
    RESHAPE_EXCLUDED = 1,
} quantize_position_t;

// this function can add fp32/bf16/int8-fp32/int8-bf16 MHA graph
// if is_itex is false, it is the default setting
// if itex is true, it will put K side's second transpose to matmul_qk
inline void add_MHA_subgraph(impl::graph_t *agraph, bool use_bf16 = false,
        bool use_int8 = false, bool is_itex = false, int batch_size = 128,
        int seq_len = 384, int num_head = 16, int head_dim = 1024) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    int size_per_head = head_dim / num_head;
    std::vector<impl::dim_t> EXTENDED_ATTENTION_MASK_SHAPE = is_itex
            ? std::vector<impl::dim_t> {batch_size, 1, seq_len, seq_len}
            : std::vector<impl::dim_t> {batch_size, 1, 1, seq_len};
    std::vector<impl::dim_t> QKV_INPUT_SHAPE = is_itex
            ? std::vector<impl::dim_t> {batch_size * seq_len, head_dim}
            : std::vector<impl::dim_t> {batch_size, seq_len, head_dim};
    std::vector<impl::dim_t> QKV_RESHAPED_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<impl::dim_t> QKV_TRANSPOSED_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<impl::dim_t> KEY_TRANSPOSED_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<impl::dim_t> MATMUL_QK_OUTPUT_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<impl::dim_t> CONST_SHAPE {1};
    auto dtype = use_bf16 ? impl::data_type::bf16 : impl::data_type::f32;

    impl::logical_tensor_t query_dequantize_input, key_dequantize_input,
            value_dequantize_input;
    query_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, impl::data_type::u8);
    key_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, impl::data_type::u8);
    value_dequantize_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, impl::data_type::u8);

    impl::logical_tensor_t query_typecast_input, key_typecast_input,
            value_typecast_input;
    query_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, impl::data_type::f32);
    key_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, impl::data_type::f32);
    value_typecast_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t query_reshape_input, key_reshape_input,
            value_reshape_input;
    query_reshape_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, dtype);
    key_reshape_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, dtype);
    value_reshape_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, dtype);

    impl::logical_tensor_t query_transpose_input, key_transpose_input,
            value_transpose_input;
    query_transpose_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    key_transpose_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    value_transpose_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);

    impl::logical_tensor_t key_transpose2_input;
    key_transpose2_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t query_matmul_input, key_matmul_input,
            value_matmul_input;
    query_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);
    key_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, dtype);
    value_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(
            logical_tensor_idx++, EXTENDED_ATTENTION_MASK_SHAPE, dtype);

    impl::logical_tensor_t fscore_scale, fscore_scale_out;
    fscore_scale = utils::logical_tensor_init(logical_tensor_idx++, CONST_SHAPE,
            is_itex || (use_bf16 && !use_int8) ? dtype : impl::data_type::f32);
    fscore_scale_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t fscore_add_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

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
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    context_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, dtype);

    impl::logical_tensor_t context_cast_out;
    context_cast_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t context_quantize_out;
    context_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_INPUT_SHAPE, impl::data_type::u8);

    impl::op_t dequantize_query {
            op_idx++, impl::op_kind::Dequantize, "dequantize_query"};
    dequantize_query.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_query.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_query.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_query.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_key {
            op_idx++, impl::op_kind::Dequantize, "dequantize_key"};
    dequantize_key.set_attr(impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_key.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_key.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_key.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_value {
            op_idx++, impl::op_kind::Dequantize, "dequantize_value"};
    dequantize_value.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_value.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_value.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_value.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t typecast_query {
            op_idx++, impl::op_kind::TypeCast, "typecast_query"};
    impl::op_t typecast_key {op_idx++, impl::op_kind::TypeCast, "typecast_key"};
    impl::op_t typecast_value {
            op_idx++, impl::op_kind::TypeCast, "typecast_value"};

    impl::op_t query_reshape {
            op_idx++, impl::op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr(impl::op_attr::shape, QKV_RESHAPED_SHAPE);
    query_reshape.set_attr(impl::op_attr::special_zero, false);
    impl::op_t query_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_reshape {
            op_idx++, impl::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(impl::op_attr::shape, QKV_RESHAPED_SHAPE);
    key_reshape.set_attr(impl::op_attr::special_zero, false);
    impl::op_t key_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_transpose2 {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 1, 3, 2});

    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};
    if (is_itex) { matmul_qk.set_attr(impl::op_attr::transpose_b, true); }

    impl::op_t fscore_rescale {op_idx++,
            is_itex ? impl::op_kind::Multiply : impl::op_kind::Divide,
            "fscore_rescale"};
    fscore_rescale.set_attr(
            impl::op_attr::auto_broadcast, std::string("numpy"));
    impl::op_t fscore_add {op_idx++, impl::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr(impl::op_attr::axis, (int64_t)3);

    impl::op_t softmax_cast {op_idx++, impl::op_kind::TypeCast, "softmax_cast"};
    impl::op_t quantize_softmax {
            op_idx++, impl::op_kind::Quantize, "quantize_softmax"};
    impl::op_t dequantize_softmax {
            op_idx++, impl::op_kind::Dequantize, "dequantize_softmax"};
    quantize_softmax.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_softmax.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_softmax.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    quantize_softmax.set_attr(impl::op_attr::axis, (int64_t)0);
    dequantize_softmax.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_softmax.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_softmax.set_attr(
            impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_softmax.set_attr(impl::op_attr::axis, (int64_t)0);

    impl::op_t dequantize_softmax_cast {
            op_idx++, impl::op_kind::TypeCast, "dequantize_softmax_cast"};

    impl::op_t value_reshape {
            op_idx++, impl::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(impl::op_attr::shape, QKV_RESHAPED_SHAPE);
    value_reshape.set_attr(impl::op_attr::special_zero, false);
    impl::op_t value_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});

    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t reshape_output {
            op_idx++, impl::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(impl::op_attr::special_zero, false);
    reshape_output.set_attr(impl::op_attr::shape, QKV_INPUT_SHAPE);

    impl::op_t typecast_output {
            op_idx++, impl::op_kind::TypeCast, "typecast_output"};
    impl::op_t quantize_output {
            op_idx++, impl::op_kind::Quantize, "quantize_output"};
    quantize_output.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_output.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_output.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    quantize_output.set_attr(impl::op_attr::axis, (int64_t)0);

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
inline void add_MHA_subgraph_alternative(impl::graph_t *agraph,
        bool use_bf16 = false, bool use_int8 = false,
        impl::op_kind_t output_op_kind = impl::op_kind::Reorder,
        int batch_size = 128, int seq_len = 384, int num_head = 16,
        int head_dim = 1024) {
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
    std::vector<impl::dim_t> RESHAPE_OUTPUT_SHAPE {
            batch_size * seq_len, num_head * size_per_head};
    std::vector<impl::dim_t> CONST_SHAPE {1};
    auto dtype = use_bf16 ? impl::data_type::bf16 : impl::data_type::f32;

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
    query_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);
    key_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, dtype);
    value_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(
            logical_tensor_idx++, EXTENDED_ATTENTION_MASK_SHAPE, dtype);

    impl::logical_tensor_t fscore_scale, fscore_div_out;
    fscore_scale = utils::logical_tensor_init(logical_tensor_idx++, CONST_SHAPE,
            impl::data_type::f32); // fscore_scale is always fp32
    fscore_div_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t fscore_add_out, softmax_out;
    fscore_add_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

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
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t context_transpose_out, context_final_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    context_final_out = utils::logical_tensor_init(logical_tensor_idx++, dtype);

    impl::logical_tensor_t context_cast_out;
    context_cast_out = utils::logical_tensor_init(
            logical_tensor_idx++, impl::data_type::f32);

    impl::logical_tensor_t context_quantize_out;
    context_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, impl::data_type::u8);

    impl::op_t dequantize_query {
            op_idx++, impl::op_kind::Dequantize, "dequantize_query"};
    dequantize_query.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_query.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_query.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_query.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_key {
            op_idx++, impl::op_kind::Dequantize, "dequantize_key"};
    dequantize_key.set_attr(impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_key.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_key.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_key.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_value {
            op_idx++, impl::op_kind::Dequantize, "dequantize_value"};
    dequantize_value.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_value.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_value.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_value.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t typecast_query {
            op_idx++, impl::op_kind::TypeCast, "typecast_query"};
    impl::op_t typecast_key {op_idx++, impl::op_kind::TypeCast, "typecast_key"};
    impl::op_t typecast_value {
            op_idx++, impl::op_kind::TypeCast, "typecast_value"};

    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};

    impl::op_t fscore_div {op_idx++, impl::op_kind::Divide, "fscore_div"};
    fscore_div.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    impl::op_t fscore_add {op_idx++, impl::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr(impl::op_attr::axis, (int64_t)3);

    impl::op_t softmax_cast {op_idx++, impl::op_kind::TypeCast, "softmax_cast"};
    impl::op_t quantize_softmax {
            op_idx++, impl::op_kind::Quantize, "quantize_softmax"};
    quantize_softmax.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_softmax.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_softmax.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    quantize_softmax.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_softmax {
            op_idx++, impl::op_kind::Dequantize, "dequantize_softmax"};
    dequantize_softmax.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_softmax.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_softmax.set_attr(
            impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_softmax.set_attr(impl::op_attr::axis, (int64_t)0);

    impl::op_t dequantize_softmax_cast {
            op_idx++, impl::op_kind::TypeCast, "dequantize_softmax_cast"};

    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t reshape_reorder_output {
            op_idx++, output_op_kind, "reshape_reorder_output"};
    if (output_op_kind == impl::op_kind::StaticReshape) {
        // if static reshape
        reshape_reorder_output.set_attr(
                impl::op_attr::shape, RESHAPE_OUTPUT_SHAPE);
        reshape_reorder_output.set_attr(impl::op_attr::special_zero, false);
    }

    impl::op_t typecast_output {
            op_idx++, impl::op_kind::TypeCast, "typecast_output"};
    impl::op_t quantize_output {
            op_idx++, impl::op_kind::Quantize, "quantize_output"};
    quantize_output.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_output.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_output.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    quantize_output.set_attr(impl::op_attr::axis, (int64_t)0);

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
    query_reshape.set_attr(impl::op_attr::shape,
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    query_reshape.set_attr(impl::op_attr::special_zero, false);
    impl::op_t query_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_reshape {
            op_idx++, impl::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(impl::op_attr::shape,
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    key_reshape.set_attr(impl::op_attr::special_zero, false);
    impl::op_t key_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_transpose2 {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 1, 3, 2});
    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};

    impl::op_t fscore_div {op_idx++, impl::op_kind::Divide, "fscore_div"};
    fscore_div.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    impl::op_t fscore_add {op_idx++, impl::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr(impl::op_attr::axis, (int64_t)3);

    // reshape + transpose for value
    impl::op_t value_reshape {
            op_idx++, impl::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(impl::op_attr::shape,
            std::vector<int64_t> {
                    batch_size, seq_len, num_head, size_per_head});
    value_reshape.set_attr(impl::op_attr::special_zero, false);
    impl::op_t value_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t reshape_output {
            op_idx++, impl::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(impl::op_attr::special_zero, false);
    reshape_output.set_attr(impl::op_attr::shape,
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
    quantize_query_gemm.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_query_gemm.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_query_gemm.set_attr(
            impl::op_attr::qtype, std::string("per_tensor"));
    quantize_query_gemm.set_attr(impl::op_attr::axis, (int64_t)0);
    quantize_key_gemm.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_key_gemm.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_key_gemm.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    quantize_key_gemm.set_attr(impl::op_attr::axis, (int64_t)0);
    quantize_value_gemm.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_value_gemm.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_value_gemm.set_attr(
            impl::op_attr::qtype, std::string("per_tensor"));
    quantize_value_gemm.set_attr(impl::op_attr::axis, (int64_t)0);

    impl::op_t dequantize_query_gemm {
            op_idx++, impl::op_kind::Dequantize, "dequantize_query_gemm"};
    dequantize_query_gemm.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_query_gemm.set_attr(
            impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_query_gemm.set_attr(
            impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_query_gemm.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_key_gemm {
            op_idx++, impl::op_kind::Dequantize, "dequantize_key_gemm"};
    dequantize_key_gemm.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_key_gemm.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_key_gemm.set_attr(
            impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_key_gemm.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_value_gemm {
            op_idx++, impl::op_kind::Dequantize, "dequantize_value_gemm"};
    dequantize_value_gemm.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_value_gemm.set_attr(
            impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_value_gemm.set_attr(
            impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_value_gemm.set_attr(impl::op_attr::axis, (int64_t)0);

    // reshape + transpose for query + key
    impl::op_t query_reshape {
            op_idx++, impl::op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr(impl::op_attr::special_zero, false);
    query_reshape.set_attr(impl::op_attr::shape, QKV_RESHAPED_SHAPE);
    impl::op_t query_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});

    impl::op_t key_reshape {
            op_idx++, impl::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(impl::op_attr::special_zero, false);
    key_reshape.set_attr(impl::op_attr::shape, QKV_RESHAPED_SHAPE);
    impl::op_t key_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_transpose2 {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 1, 3, 2});

    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};

    impl::op_t fscore_rescale {op_idx++,
            use_div ? impl::op_kind::Divide : impl::op_kind::Multiply,
            "fscore_rescale"};
    fscore_rescale.set_attr(
            impl::op_attr::auto_broadcast, std::string("numpy"));
    impl::op_t fscore_add {op_idx++, impl::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr(impl::op_attr::axis, (int64_t)3);
    // quantize-dequantize softmax's output
    impl::op_t quantize_softmax {
            op_idx++, impl::op_kind::Quantize, "quantize_softmax"};
    impl::op_t dequantize_softmax {
            op_idx++, impl::op_kind::Dequantize, "dequantize_softmax"};
    quantize_softmax.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_softmax.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_softmax.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    quantize_softmax.set_attr(impl::op_attr::axis, (int64_t)0);
    dequantize_softmax.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_softmax.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_softmax.set_attr(
            impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_softmax.set_attr(impl::op_attr::axis, (int64_t)0);

    // reshape + transpose for value
    impl::op_t value_reshape {
            op_idx++, impl::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(impl::op_attr::special_zero, false);
    value_reshape.set_attr(impl::op_attr::shape, QKV_RESHAPED_SHAPE);
    impl::op_t value_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});

    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t reshape_output {
            op_idx++, impl::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(impl::op_attr::special_zero, false);
    reshape_output.set_attr(impl::op_attr::shape, OUTPUT_SHAPE);

    // quantize dequantize output
    impl::op_t quantize_output {
            op_idx++, impl::op_kind::Quantize, "quantize_output"};
    quantize_output.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_output.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_output.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    quantize_output.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_output {
            op_idx++, impl::op_kind::Dequantize, "dequantize_output"};
    dequantize_output.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_output.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_output.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_output.set_attr(impl::op_attr::axis, (int64_t)0);

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

inline void add_MHA_training_subgraph(impl::graph_t *agraph,
        bool use_bf16 = false, bool has_mul_from_select = false,
        int batch_size = 128, int seq_len = 384, int num_head = 16,
        int head_dim = 1024) {
    auto dtype = use_bf16 ? impl::data_type::bf16 : impl::data_type::f32;
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
    std::vector<impl::dim_t> SOFTMAX_SUM_SHAPE {
            batch_size, num_head, seq_len, 1};
    std::vector<impl::dim_t> CONST_SHAPE {1};
    std::vector<impl::dim_t> OUTPUT_SHAPE {batch_size, seq_len, head_dim};

    // start constructing forward graph
    impl::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(
            logical_tensor_idx++, EXTENDED_ATTENTION_MASK_SHAPE, dtype);

    impl::logical_tensor_t query_transpose_out;
    query_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t key_transpose_out;
    key_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t fscore_scale, fscore_div_out;
    fscore_scale = utils::logical_tensor_init(
            logical_tensor_idx++, CONST_SHAPE, impl::data_type::f32);
    fscore_div_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t fscore_add_out, softmax_out, dropout, dropout_out,
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

    impl::logical_tensor_t value_transpose_out;
    value_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    context_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, dtype);

    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};
    matmul_qk.set_attr(impl::op_attr::transpose_b, true);
    matmul_qk.add_input(query_transpose_out);
    matmul_qk.add_input(key_transpose_out);
    matmul_qk.add_output(matmul_qk_out);

    impl::op_t fscore_div {op_idx++, impl::op_kind::Divide, "fscore_div"};
    fscore_div.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    fscore_div.add_input(matmul_qk_out);
    fscore_div.add_input(fscore_scale);
    fscore_div.add_output(fscore_div_out);

    impl::op_t fscore_add {op_idx++, impl::op_kind::Add, "fscore_add"};
    fscore_add.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    fscore_add.add_input(fscore_div_out);
    fscore_add.add_input(attention_mask_flt);
    fscore_add.add_output(fscore_add_out);

    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr(impl::op_attr::axis, (int64_t)3);
    softmax.add_input(fscore_add_out);
    softmax.add_output(softmax_out);

    impl::op_t mul_dropout {op_idx++, impl::op_kind::Multiply, "mul_dropout"};
    mul_dropout.add_input(softmax_out);
    mul_dropout.add_input(dropout);
    mul_dropout.add_output(dropout_out);

    impl::op_t mul_select {op_idx++, impl::op_kind::Multiply, "mul_select"};
    mul_select.add_input(dropout_out);
    mul_select.add_input(select);
    mul_select.add_output(select_out);

    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};
    matmul_v.add_input(has_mul_from_select ? select_out : dropout_out);
    matmul_v.add_input(value_transpose_out);
    matmul_v.add_output(matmul_v_out);

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);

    impl::op_t reshape_output {
            op_idx++, impl::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(impl::op_attr::special_zero, false);
    reshape_output.set_attr(impl::op_attr::shape, OUTPUT_SHAPE);
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
    impl::logical_tensor_t backward_in;
    backward_in = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, dtype);
    impl::logical_tensor_t in_reshape;
    in_reshape = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    impl::logical_tensor_t in_transpose;
    in_transpose = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t bmm_v_grad_weight;
    bmm_v_grad_weight = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t value_transpose;
    value_transpose = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t bmm_v_grad_data;
    bmm_v_grad_data = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t select_grad;
    select_grad = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t dropout_grad;
    dropout_grad = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t softmax_mul;
    softmax_mul = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t softmax_sum;
    softmax_sum = utils::logical_tensor_init(
            logical_tensor_idx++, SOFTMAX_SUM_SHAPE, dtype);

    impl::logical_tensor_t softmax_sub;
    softmax_sub = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t fscore_grad;
    fscore_grad = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t bmm_q_grad_weight;
    bmm_q_grad_weight = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t bmm_k_grad_weight;
    bmm_k_grad_weight = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    impl::op_t reshape_bwd {
            op_idx++, impl::op_kind::StaticReshape, "reshape_bwd"};
    reshape_bwd.set_attr(impl::op_attr::shape, QKV_RESHAPED_SHAPE);
    reshape_bwd.set_attr(impl::op_attr::special_zero, false);
    reshape_bwd.add_input(backward_in);
    reshape_bwd.add_output(in_reshape);

    impl::op_t transpose_bwd {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_bwd"};
    transpose_bwd.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    transpose_bwd.add_input(in_reshape);
    transpose_bwd.add_output(in_transpose);

    impl::op_t grad_v {op_idx++, impl::op_kind::MatMul, "grad_v"};
    grad_v.set_attr(impl::op_attr::transpose_a, true);
    grad_v.add_input(dropout_out);
    grad_v.add_input(in_transpose);
    grad_v.add_output(bmm_v_grad_weight);

    impl::op_t grad_dropout {op_idx++, impl::op_kind::MatMul, "grad_dropout"};
    grad_dropout.set_attr(impl::op_attr::transpose_b, true);
    grad_dropout.add_input(in_transpose);
    grad_dropout.add_input(value_transpose);
    grad_dropout.add_output(bmm_v_grad_data);

    impl::op_t grad_select {op_idx++, impl::op_kind::Multiply, "grad_select"};
    grad_select.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    grad_select.add_input(bmm_v_grad_data);
    grad_select.add_input(select);
    grad_select.add_output(select_grad);

    impl::op_t mul {op_idx++, impl::op_kind::Multiply, "mul"};
    mul.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    mul.add_input(has_mul_from_select ? select_grad : bmm_v_grad_data);
    mul.add_input(dropout);
    mul.add_output(dropout_grad);

    impl::op_t mul_2 {op_idx++, impl::op_kind::Multiply, "mul_2"};
    mul_2.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    mul_2.add_input(dropout_grad);
    mul_2.add_input(softmax_out);
    mul_2.add_output(softmax_mul);

    impl::op_t reduce_sum {op_idx++, impl::op_kind::ReduceSum, "reduce_sum"};
    reduce_sum.set_attr(impl::op_attr::keep_dims, true);
    reduce_sum.set_attr(impl::op_attr::axes, std::vector<int64_t> {-1});
    reduce_sum.add_input(softmax_mul);
    reduce_sum.add_output(softmax_sum);

    impl::op_t sub {op_idx++, impl::op_kind::Subtract, "sub"};
    sub.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    sub.add_input(dropout_grad);
    sub.add_input(softmax_sum);
    sub.add_output(softmax_sub);

    impl::op_t div {op_idx++, impl::op_kind::Divide, "div"};
    div.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    div.add_input(softmax_sub);
    div.add_input(fscore_scale);
    div.add_output(fscore_grad);

    impl::op_t grad_q {op_idx++, impl::op_kind::MatMul, "grad_q"};
    grad_q.add_input(fscore_grad);
    grad_q.add_input(key_transpose_out);
    grad_q.add_output(bmm_q_grad_weight);

    impl::op_t grad_k {op_idx++, impl::op_kind::MatMul, "grad_k"};
    grad_k.set_attr(impl::op_attr::transpose_a, true);
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

inline void add_distill_bert_MHA(impl::graph_t *agraph, bool use_bf16 = false,
        bool use_int8 = false,
        impl::op_kind_t output_op_kind = impl::op_kind::Reorder,
        int batch_size = 224, int seq_len = 128, int num_head = 12,
        int head_dim = 768) {
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
    std::vector<impl::dim_t> RESHAPE_OUTPUT_SHAPE {
            batch_size * seq_len, num_head * size_per_head};
    std::vector<impl::dim_t> CONST_SHAPE {1};
    auto dtype = use_bf16 ? impl::data_type::bf16 : impl::data_type::f32;

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
    query_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);
    key_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, KEY_TRANSPOSED_SHAPE, dtype);
    value_matmul_input = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_TRANSPOSED_SHAPE, dtype);

    impl::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t attention_mask_flt;
    attention_mask_flt = utils::logical_tensor_init(logical_tensor_idx++,
            EXTENDED_ATTENTION_MASK_SHAPE, impl::data_type::boolean);

    impl::logical_tensor_t fscore_scale, select_out;
    fscore_scale = utils::logical_tensor_init(
            logical_tensor_idx++, CONST_SHAPE, dtype);
    select_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t softmax_out;
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_OUTPUT_SHAPE, dtype);

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
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_OUTPUT_SHAPE, dtype);

    impl::logical_tensor_t context_transpose_out, context_final_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPED_SHAPE, dtype);
    context_final_out = utils::logical_tensor_init(logical_tensor_idx++, dtype);

    impl::logical_tensor_t context_cast_out;
    context_cast_out = utils::logical_tensor_init(
            logical_tensor_idx++, impl::data_type::f32);

    impl::logical_tensor_t context_quantize_out;
    context_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, impl::data_type::u8);

    impl::op_t dequantize_query {
            op_idx++, impl::op_kind::Dequantize, "dequantize_query"};
    dequantize_query.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_query.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_query.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_query.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_key {
            op_idx++, impl::op_kind::Dequantize, "dequantize_key"};
    dequantize_key.set_attr(impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_key.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_key.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_key.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_value {
            op_idx++, impl::op_kind::Dequantize, "dequantize_value"};
    dequantize_value.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_value.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_value.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_value.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t typecast_query {
            op_idx++, impl::op_kind::TypeCast, "typecast_query"};
    impl::op_t typecast_key {op_idx++, impl::op_kind::TypeCast, "typecast_key"};
    impl::op_t typecast_value {
            op_idx++, impl::op_kind::TypeCast, "typecast_value"};

    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};

    impl::op_t select {op_idx++, impl::op_kind::Select, "select"};
    select.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr(impl::op_attr::axis, (int64_t)3);

    impl::op_t softmax_cast {op_idx++, impl::op_kind::TypeCast, "softmax_cast"};
    impl::op_t quantize_softmax {
            op_idx++, impl::op_kind::Quantize, "quantize_softmax"};
    quantize_softmax.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_softmax.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_softmax.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    quantize_softmax.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequantize_softmax {
            op_idx++, impl::op_kind::Dequantize, "dequantize_softmax"};
    dequantize_softmax.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    dequantize_softmax.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequantize_softmax.set_attr(
            impl::op_attr::qtype, std::string("per_tensor"));
    dequantize_softmax.set_attr(impl::op_attr::axis, (int64_t)0);

    impl::op_t dequantize_softmax_cast {
            op_idx++, impl::op_kind::TypeCast, "dequantize_softmax_cast"};

    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t reshape_reorder_output {
            op_idx++, output_op_kind, "reshape_reorder_output"};
    if (output_op_kind == impl::op_kind::StaticReshape) {
        // if static reshape
        reshape_reorder_output.set_attr(
                impl::op_attr::shape, RESHAPE_OUTPUT_SHAPE);
        reshape_reorder_output.set_attr(impl::op_attr::special_zero, false);
    }

    impl::op_t typecast_output {
            op_idx++, impl::op_kind::TypeCast, "typecast_output"};
    impl::op_t quantize_output {
            op_idx++, impl::op_kind::Quantize, "quantize_output"};
    quantize_output.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    quantize_output.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quantize_output.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    quantize_output.set_attr(impl::op_attr::axis, (int64_t)0);

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

inline void add_mlp_subgraph(impl::graph_t *agraph, bool use_bf16 = false,
        int batch_size = 1, int layer = 1,
        std::vector<int> hidden_size = {13, 512},
        std::vector<impl::op_kind_t> act_type = {impl::op_kind::ReLU},
        bool separate_bias_add = false) {
    size_t lt_idx = 0;
    size_t op_idx = 0;

    auto dtype = use_bf16 ? impl::data_type::bf16 : impl::data_type::f32;

    std::vector<impl::dim_t> layer_input_size {batch_size, hidden_size[0]};
    impl::logical_tensor_t input
            = utils::logical_tensor_init(lt_idx++, layer_input_size, dtype);

    for (int i = 0; i < layer; ++i) {
        std::vector<impl::dim_t> layer_weight_size {
                hidden_size[i], hidden_size[i + 1]};
        std::vector<impl::dim_t> layer_bias_size {hidden_size[i + 1]};
        std::vector<impl::dim_t> layer_dst_size {
                batch_size, hidden_size[i + 1]};

        impl::logical_tensor_t weight, bias, matmul_dst, add_dst, act_dst;

        weight = utils::logical_tensor_init(lt_idx++, layer_weight_size, dtype);
        weight.property = impl::property_type::constant;
        bias = utils::logical_tensor_init(lt_idx++, layer_bias_size, dtype);
        bias.property = impl::property_type::constant;
        matmul_dst
                = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
        add_dst = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
        act_dst = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);

        std::string layer_suffix = "_layer" + std::to_string(i);
        impl::op_t matmul {
                op_idx++, impl::op_kind::MatMul, "matmul" + layer_suffix};
        impl::op_t add {op_idx++, impl::op_kind::Add, "add" + layer_suffix};
        impl::op_t activation {
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

        if (act_type[i] == impl::op_kind::Wildcard) {
            input = separate_bias_add ? add_dst : matmul_dst;
        } else {
            activation.add_input(separate_bias_add ? add_dst : matmul_dst);
            activation.add_output(act_dst);
            input = act_dst;
        }

        agraph->add_op(&matmul);
        if (separate_bias_add) { agraph->add_op(&add); }
        if (act_type[i] != impl::op_kind::Wildcard) {
            agraph->add_op(&activation);
        }
    }
}

inline void add_int8_mlp_subgraph(impl::graph_t *agraph, int batch_size = 1,
        int layer = 1, std::vector<int> hidden_size = {13, 512},
        std::vector<impl::op_kind_t> act_type = {impl::op_kind::ReLU}) {
    size_t lt_idx = 0;
    size_t op_idx = 0;

    std::vector<impl::dim_t> layer_input_size {batch_size, hidden_size[0]};
    impl::logical_tensor_t input_desc = utils::logical_tensor_init(
            lt_idx++, layer_input_size, impl::data_type::f32);

    for (int i = 0; i < layer; ++i) {
        std::vector<impl::dim_t> layer_weight_size {
                hidden_size[i], hidden_size[i + 1]};
        std::vector<impl::dim_t> layer_bias_size {hidden_size[i + 1]};
        std::vector<impl::dim_t> layer_dst_size {
                batch_size, hidden_size[i + 1]};

        impl::logical_tensor_t quant_input_desc, dequant_input_desc,
                weight_desc, quant_weight_desc, dequant_weight_desc, bias_desc,
                matmul_dst_desc, act_dst_desc;

        quant_input_desc = utils::logical_tensor_init(
                lt_idx++, layer_input_size, impl::data_type::u8);
        dequant_input_desc = utils::logical_tensor_init(
                lt_idx++, layer_input_size, impl::data_type::f32);
        weight_desc = utils::logical_tensor_init(
                lt_idx++, layer_weight_size, impl::data_type::f32);
        quant_weight_desc = utils::logical_tensor_init(
                lt_idx++, layer_weight_size, impl::data_type::s8);
        quant_weight_desc.property = impl::property_type::constant;
        dequant_weight_desc = utils::logical_tensor_init(
                lt_idx++, layer_weight_size, impl::data_type::f32);
        bias_desc = utils::logical_tensor_init(
                lt_idx++, layer_bias_size, impl::data_type::f32);
        bias_desc.property = impl::property_type::constant;
        matmul_dst_desc = utils::logical_tensor_init(
                lt_idx++, layer_dst_size, impl::data_type::f32);
        act_dst_desc = utils::logical_tensor_init(
                lt_idx++, layer_dst_size, impl::data_type::f32);

        std::string layer_suffix = "_layer" + std::to_string(i);
        impl::op_t quant_input {op_idx++, impl::op_kind::Quantize,
                "quantize_input" + layer_suffix};
        quant_input.set_attr(
                impl::op_attr::scales, std::vector<float>({0.12f}));
        quant_input.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
        quant_input.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
        quant_input.set_attr(impl::op_attr::axis, (int64_t)0);
        impl::op_t dequant_input {op_idx++, impl::op_kind::Dequantize,
                "dequantize_input" + layer_suffix};
        dequant_input.set_attr(
                impl::op_attr::scales, std::vector<float>({0.12f}));
        dequant_input.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
        dequant_input.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
        dequant_input.set_attr(impl::op_attr::axis, (int64_t)0);
        impl::op_t quant_weight {op_idx++, impl::op_kind::Quantize,
                "quantize_weight" + layer_suffix};
        quant_weight.set_attr(impl::op_attr::scales,
                std::vector<float>(hidden_size[i + 1], 0.12f));
        quant_weight.set_attr(impl::op_attr::zps,
                std::vector<int64_t>(hidden_size[i + 1], 0));
        quant_weight.set_attr(impl::op_attr::qtype, std::string("per_channel"));
        quant_weight.set_attr(impl::op_attr::axis, (int64_t)1);
        impl::op_t dequant_weight {op_idx++, impl::op_kind::Dequantize,
                "dequantize_weight" + layer_suffix};
        dequant_weight.set_attr(impl::op_attr::scales,
                std::vector<float>(hidden_size[i + 1], 0.12f));
        dequant_weight.set_attr(impl::op_attr::zps,
                std::vector<int64_t>(hidden_size[i + 1], 0));
        dequant_weight.set_attr(
                impl::op_attr::qtype, std::string("per_channel"));
        dequant_weight.set_attr(impl::op_attr::axis, (int64_t)1);
        impl::op_t matmul {
                op_idx++, impl::op_kind::MatMul, "matmul" + layer_suffix};
        impl::op_t activation {
                op_idx++, act_type[i], "activation" + layer_suffix};

        quant_input.add_input(input_desc);
        quant_input.add_output(quant_input_desc);
        dequant_input.add_input(quant_input_desc);
        dequant_input.add_output(dequant_input_desc);
        quant_weight.add_input(weight_desc);
        quant_weight.add_output(quant_weight_desc);
        dequant_weight.add_input(quant_weight_desc);
        dequant_weight.add_output(dequant_weight_desc);
        matmul.add_input(dequant_input_desc);
        matmul.add_input(dequant_weight_desc);
        matmul.add_input(bias_desc);
        matmul.add_output(matmul_dst_desc);

        if (act_type[i] == impl::op_kind::Wildcard) {
            input_desc = matmul_dst_desc;
        } else {
            activation.add_input(matmul_dst_desc);
            activation.add_output(act_dst_desc);
            input_desc = act_dst_desc;
        }

        agraph->add_op(&quant_input);
        agraph->add_op(&dequant_input);
        agraph->add_op(&quant_weight);
        agraph->add_op(&dequant_weight);
        agraph->add_op(&matmul);

        if (act_type[i] != impl::op_kind::Wildcard) {
            agraph->add_op(&activation);
        }

        layer_input_size = layer_dst_size;
    }

    impl::logical_tensor_t quant_output_desc = utils::logical_tensor_init(
            lt_idx++, layer_input_size, impl::data_type::u8);
    impl::logical_tensor_t dequant_output_desc = utils::logical_tensor_init(
            lt_idx++, layer_input_size, impl::data_type::f32);

    impl::op_t quant_output {
            op_idx++, impl::op_kind::Quantize, "quantize_output"};
    quant_output.set_attr(impl::op_attr::scales, std::vector<float>({0.12f}));
    quant_output.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    quant_output.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    quant_output.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t dequant_output {
            op_idx++, impl::op_kind::Dequantize, "dequantize_output"};
    dequant_output.set_attr(impl::op_attr::scales, std::vector<float>({0.12f}));
    dequant_output.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    dequant_output.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    dequant_output.set_attr(impl::op_attr::axis, (int64_t)0);

    quant_output.add_input(input_desc);
    quant_output.add_output(quant_output_desc);
    dequant_output.add_input(quant_output_desc);
    dequant_output.add_output(dequant_output_desc);

    agraph->add_op(&quant_output);
    agraph->add_op(&dequant_output);
}

static bool apply_use_dst(bool use_dst, impl::op_kind_t op_kind) {
    if (!use_dst
            && (op_kind == impl::op_kind::ReLUBackprop
                    || op_kind == impl::op_kind::SigmoidBackprop)) {
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
inline void add_mlp_training_graph(impl::graph_t *agraph, int batch_size,
        int layer, std::vector<int> hidden_size,
        std::vector<impl::op_kind_t> act_type,
        std::vector<impl::op_kind_t> act_backprop_type, bool use_bf16 = false,
        bool use_dst = true, bool weight_transposed = false) {
    impl::data_type_t dtype
            = use_bf16 ? impl::data_type::bf16 : impl::data_type::f32;
    size_t lt_idx = 0;
    size_t op_idx = 0;
    std::vector<impl::logical_tensor_t> z_lts;
    std::vector<impl::logical_tensor_t> weight_lts;
    std::vector<impl::logical_tensor_t> x_lts;

    std::vector<impl::dim_t> layer_input_size {batch_size, hidden_size[0]};
    impl::logical_tensor_t input
            = utils::logical_tensor_init(lt_idx++, layer_input_size, dtype);
    x_lts.push_back(input);

    // constructing the forward calculations
    for (int i = 0; i < layer; ++i) {
        std::vector<impl::dim_t> layer_weight_size {
                hidden_size[i], hidden_size[i + 1]};
        std::vector<impl::dim_t> layer_bias_size {hidden_size[i + 1]};
        std::vector<impl::dim_t> layer_dst_size {
                batch_size, hidden_size[i + 1]};

        impl::logical_tensor_t weight, bias, matmul_dst, act_dst;

        weight = utils::logical_tensor_init(lt_idx++, layer_weight_size, dtype);
        weight_lts.push_back(weight);
        bias = utils::logical_tensor_init(lt_idx++, layer_bias_size, dtype);
        matmul_dst
                = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
        z_lts.push_back(matmul_dst);
        act_dst = utils::logical_tensor_init(lt_idx++, layer_dst_size, dtype);
        x_lts.push_back(act_dst);

        std::string layer_suffix = "_layer" + std::to_string(i);
        impl::op_t matmul {
                op_idx++, impl::op_kind::MatMul, "matmul" + layer_suffix};
        impl::op_t activation {
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

    impl::logical_tensor_t grad_y = utils::logical_tensor_init(
            lt_idx++, {batch_size, hidden_size[layer]}, dtype);

    // constructing backward calculations
    for (int i = 0; i < layer; ++i) {
        std::vector<impl::dim_t> grad_z_size {
                batch_size, hidden_size[layer - i]};
        std::vector<impl::dim_t> transposed_grad_z_size {
                hidden_size[layer - i], batch_size};
        std::vector<impl::dim_t> grad_x_size {
                batch_size, hidden_size[layer - i - 1]};
        std::vector<impl::dim_t> grad_w_size = weight_transposed
                ? std::vector<impl::dim_t> {hidden_size[layer - i],
                        hidden_size[layer - i - 1]}
                : std::vector<impl::dim_t> {
                        hidden_size[layer - i - 1], hidden_size[layer - i]};
        std::vector<impl::dim_t> transposed_x_size {
                hidden_size[layer - i - 1], batch_size};
        std::vector<impl::dim_t> transposed_weight_size {
                hidden_size[layer - i], hidden_size[layer - i - 1]};
        std::vector<impl::dim_t> grad_bias_size {1, hidden_size[layer - i]};

        impl::logical_tensor_t activation_backprop_out,
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
        impl::op_t activation_backward {op_idx++, act_backprop_type[i],
                "activation_backprop" + layer_suffix};
        if (!apply_use_dst(use_dst, act_backprop_type[i])) {
            activation_backward.set_attr(impl::op_attr::use_dst, false);
        }
        impl::op_t transpose_activation_backward {op_idx++,
                impl::op_kind::StaticTranspose,
                "transpose_activation_backward" + layer_suffix};
        transpose_activation_backward.set_attr(
                impl::op_attr::order, std::vector<int64_t> {1, 0});
        impl::op_t transpose_weight {op_idx++, impl::op_kind::StaticTranspose,
                "transpose_weight" + layer_suffix};
        transpose_weight.set_attr(
                impl::op_attr::order, std::vector<int64_t> {1, 0});
        impl::op_t transpose_x {op_idx++, impl::op_kind::StaticTranspose,
                "transpose_x" + layer_suffix};
        transpose_x.set_attr(impl::op_attr::order, std::vector<int64_t> {1, 0});
        impl::op_t matmul_weight {
                op_idx++, impl::op_kind::MatMul, "grad_weight" + layer_suffix};
        impl::op_t matmul_x {
                op_idx++, impl::op_kind::MatMul, "grad_x" + layer_suffix};
        impl::op_t reduce_bias {
                op_idx++, impl::op_kind::ReduceSum, "grad_bias" + layer_suffix};
        reduce_bias.set_attr(impl::op_attr::axes, std::vector<int64_t> {0});
        reduce_bias.set_attr(impl::op_attr::keep_dims, true);

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

static inline void add_MHA_subgraph_alternative2(impl::graph_t *agraph,
        bool has_quantize = false, int batch_size = 64, int seq_len = 384,
        int num_head = 16, int head_dim = 1024) {
    size_t logical_tensor_idx = 0;
    size_t op_idx = 0;
    int size_per_head = head_dim / num_head;
    std::vector<impl::dim_t> MIXED_LAYER_INPUT_SHAPE {
            batch_size, seq_len, head_dim};
    std::vector<impl::dim_t> QKV_RESHAPE_SHAPE {
            batch_size, seq_len, num_head, size_per_head};
    std::vector<impl::dim_t> QV_TRANSPOSE_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<impl::dim_t> K_TRANSPOSE_SHAPE {
            batch_size, num_head, size_per_head, seq_len};
    std::vector<impl::dim_t> MATMUL_QK_SHAPE {
            batch_size, num_head, seq_len, seq_len};
    std::vector<impl::dim_t> SOFTMAX_SHAPE {
            batch_size * num_head * seq_len, seq_len};
    std::vector<impl::dim_t> MATMUL_V_SHAPE {
            batch_size, num_head, seq_len, size_per_head};
    std::vector<impl::dim_t> EXTENDED_ATTENTION_MASK_SHAPE {
            batch_size, 1, 1, seq_len};
    std::vector<impl::dim_t> OUTPUT_SHAPE {batch_size, seq_len, head_dim};
    std::vector<impl::dim_t> CONST_SHAPE {1};

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

    impl::logical_tensor_t scale;
    scale = utils::logical_tensor_init(
            logical_tensor_idx++, CONST_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t query_reshape_out, query_transpose_out;
    query_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPE_SHAPE, impl::data_type::f32);
    query_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QV_TRANSPOSE_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t key_reshape_out, key_transpose_out;
    key_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPE_SHAPE, impl::data_type::f32);
    key_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, K_TRANSPOSE_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t matmul_qk_out;
    matmul_qk_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t fake_quantize_out, fake_dequantize_out;
    fake_quantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, impl::data_type::u8);
    fake_dequantize_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t add_out;
    add_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, impl::data_type::f32);
    impl::logical_tensor_t mul_out;
    mul_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t pre_softmax_reshape_out;
    pre_softmax_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, SOFTMAX_SHAPE, impl::data_type::f32);
    impl::logical_tensor_t softmax_out;
    softmax_out = utils::logical_tensor_init(
            logical_tensor_idx++, SOFTMAX_SHAPE, impl::data_type::f32);
    impl::logical_tensor_t softmax_reshape_out;
    softmax_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_QK_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t value_reshape_out, value_transpose_out;
    value_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPE_SHAPE, impl::data_type::f32);
    value_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QV_TRANSPOSE_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t matmul_v_out;
    matmul_v_out = utils::logical_tensor_init(
            logical_tensor_idx++, MATMUL_V_SHAPE, impl::data_type::f32);

    impl::logical_tensor_t context_transpose_out, context_reshape_out;
    context_transpose_out = utils::logical_tensor_init(
            logical_tensor_idx++, QKV_RESHAPE_SHAPE, impl::data_type::f32);
    context_reshape_out = utils::logical_tensor_init(
            logical_tensor_idx++, OUTPUT_SHAPE, impl::data_type::f32);

    // reshape + transpose for query + key
    impl::op_t query_reshape {
            op_idx++, impl::op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr(impl::op_attr::shape, QKV_RESHAPE_SHAPE);
    query_reshape.set_attr(impl::op_attr::special_zero, false);
    impl::op_t query_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t key_reshape {
            op_idx++, impl::op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(impl::op_attr::shape, QKV_RESHAPE_SHAPE);
    key_reshape.set_attr(impl::op_attr::special_zero, false);
    impl::op_t key_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 3, 1});

    impl::op_t matmul_qk {op_idx++, impl::op_kind::MatMul, "matmul_qk"};

    impl::op_t fake_quantize {
            op_idx++, impl::op_kind::Quantize, "fake_quantize"};
    fake_quantize.set_attr(impl::op_attr::scales, std::vector<float>({0.12f}));
    fake_quantize.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    fake_quantize.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    fake_quantize.set_attr(impl::op_attr::axis, (int64_t)0);
    impl::op_t fake_dequantize {
            op_idx++, impl::op_kind::Dequantize, "fake_dequantize"};
    fake_dequantize.set_attr(
            impl::op_attr::scales, std::vector<float>({0.12f}));
    fake_dequantize.set_attr(impl::op_attr::zps, std::vector<int64_t>({2}));
    fake_dequantize.set_attr(impl::op_attr::qtype, std::string("per_tensor"));
    fake_dequantize.set_attr(impl::op_attr::axis, (int64_t)0);

    impl::op_t add {op_idx++, impl::op_kind::Add, "add"};
    add.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    impl::op_t mul {op_idx++, impl::op_kind::Multiply, "mul"};
    mul.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));

    impl::op_t pre_softmax_reshape {
            op_idx++, impl::op_kind::StaticReshape, "pre_softmax_reshape"};
    pre_softmax_reshape.set_attr(impl::op_attr::shape, SOFTMAX_SHAPE);
    pre_softmax_reshape.set_attr(impl::op_attr::special_zero, false);
    impl::op_t softmax {op_idx++, impl::op_kind::SoftMax, "softmax"};
    softmax.set_attr(impl::op_attr::axis, (int64_t)1);
    impl::op_t softmax_reshape {
            op_idx++, impl::op_kind::StaticReshape, "softmax_reshape"};
    softmax_reshape.set_attr(impl::op_attr::shape, MATMUL_QK_SHAPE);
    softmax_reshape.set_attr(impl::op_attr::special_zero, false);

    // reshape + transpose for value
    impl::op_t value_reshape {
            op_idx++, impl::op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(impl::op_attr::shape, QKV_RESHAPE_SHAPE);
    value_reshape.set_attr(impl::op_attr::special_zero, false);
    impl::op_t value_transpose {
            op_idx++, impl::op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t matmul_v {op_idx++, impl::op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    impl::op_t transpose_output {
            op_idx++, impl::op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr(
            impl::op_attr::order, std::vector<int64_t> {0, 2, 1, 3});
    impl::op_t reshape_output {
            op_idx++, impl::op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(impl::op_attr::special_zero, false);
    reshape_output.set_attr(impl::op_attr::shape, OUTPUT_SHAPE);

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

// returned vector with shape {ic, ks, oc}
static std::vector<impl::dim_t> extract_filter_info(
        const impl::dims &shape, const std::string &filter_format) {
    size_t ndims = shape.size();
    return filter_format == "OIX"
            ? std::vector<impl::dim_t> {shape[1], shape[2], shape[0]}
            : std::vector<impl::dim_t> {
                    shape[ndims - 2], shape[0], shape[ndims - 1]};
}

// filter_shape in the order of 1x1; 1x1 + 3x3 + 1x1
// strides and paddings also start with the single conv branch
inline void construct_convolutional_bottleneck_resblock(
        dnnl::graph::impl::graph_t *agraph, utils::id_generator &id_gen,
        const impl::dims &input_shape,
        const std::vector<impl::dims> &filter_shapes, bool is_bf16 = false,
        const std::vector<std::vector<int64_t>> &strides
        = {{2, 2}, {1, 1}, {2, 2}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NCX",
        const std::string &filter_format = "OIX") {
    auto dtype = is_bf16 ? impl::data_type::bf16 : impl::data_type::f32;
    auto input
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);

    std::vector<std::vector<impl::dim_t>> filter_infos;
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

inline void construct_identical_bottleneck_resblock(
        dnnl::graph::impl::graph_t *agraph, utils::id_generator &id_gen,
        const impl::dims &input_shape,
        const std::vector<impl::dims> &filter_shapes, bool is_bf16 = false,
        const std::vector<std::vector<int64_t>> &strides
        = {{1, 1}, {1, 1}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NCX",
        const std::string &filter_format = "OIX") {
    auto dtype = is_bf16 ? impl::data_type::bf16 : impl::data_type::f32;
    auto input
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    std::vector<std::vector<impl::dim_t>> filter_infos;
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

inline void construct_int8_identical_bottleneck_resblock(
        dnnl::graph::impl::graph_t *agraph, utils::id_generator &id_gen,
        const impl::dims &input_shape,
        const std::vector<impl::dims> &filter_shapes,
        const std::vector<std::vector<int64_t>> &strides
        = {{1, 1}, {1, 1}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NCX",
        const std::string &filter_format = "OIX") {
    float scale_src = 1 / 255.f, scale_out = 1;
    int64_t zp_src = 0, zp_out = 78;

    auto src = utils::logical_tensor_init(
            id_gen.get_id(), input_shape, impl::data_type::u8);

    std::vector<std::vector<impl::dim_t>> filter_infos;
    for (auto shape : filter_shapes) {
        filter_infos.push_back(extract_filter_info(shape, filter_format));
    }
    auto temp_src = src;
    for (size_t i = 0; i < filter_shapes.size(); ++i) {
        std::vector<float> scale_wei(filter_infos[i][2], 1 / 127.f);
        temp_src = utils::create_int8_convolution(id_gen, *agraph, temp_src,
                filter_infos[i][0], filter_infos[i][1], filter_infos[i][2], 1,
                strides[i], {1, 1}, paddings[i], paddings[i], data_format,
                filter_format, true, false, 1e-6f,
                /*has_relu? */ i < filter_shapes.size() - 1, scale_src, zp_src,
                scale_out, zp_out, scale_wei, impl::data_type::u8,
                /*is_quantize_dst? */ i < filter_shapes.size() - 1);
    }
    auto dq3 = create_dequantize(
            id_gen, *agraph, src, "per_tensor", {zp_src}, {scale_src}, 0);
    auto add0 = create_add(id_gen, *agraph, temp_src, dq3);
    auto relu0 = create_relu(id_gen, *agraph, add0);
    auto q2 = create_quantize(id_gen, *agraph, relu0, impl::data_type::u8,
            "per_tensor", std::vector<int64_t> {zp_out},
            std::vector<float> {scale_out}, 0);
    (void)(q2);
}

// filter_shape in the order of 1x1; 1x1+3x3+1x1
// strides and paddings also first consider the single conv branch
inline void construct_int8_convolutional_bottleneck_resblock(
        dnnl::graph::impl::graph_t *agraph, utils::id_generator &id_gen,
        const impl::dims &input_shape,
        const std::vector<impl::dims> &filter_shapes,
        const std::vector<std::vector<int64_t>> &strides
        = {{2, 2}, {1, 1}, {2, 2}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NCX",
        const std::string &filter_format = "OIX") {
    float scale_src = 1 / 255.f, scale_out = 1;
    int64_t zp_src = 0, zp_out = 78;
    auto src = utils::logical_tensor_init(
            id_gen.get_id(), input_shape, impl::data_type::u8);

    std::vector<std::vector<impl::dim_t>> filter_infos;
    for (auto shape : filter_shapes) {
        filter_infos.push_back(extract_filter_info(shape, filter_format));
    }
    std::vector<float> scale_wei(filter_infos[0][2], 1 / 127.f);
    auto left = utils::create_int8_convolution(id_gen, *agraph, src,
            filter_infos[0][0], filter_infos[0][1], filter_infos[0][2], 1,
            strides[0], {1, 1}, paddings[0], paddings[0], data_format,
            filter_format, true, false, 1e-6f,
            /*no relu*/ false, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);

    auto right = src;
    for (size_t i = 1; i < filter_shapes.size(); ++i) {
        scale_wei = std::vector<float>(filter_infos[i][2], 1 / 127.f);
        right = utils::create_int8_convolution(id_gen, *agraph, right,
                filter_infos[i][0], filter_infos[i][1], filter_infos[i][2], 1,
                strides[i], {1, 1}, paddings[i], paddings[i], data_format,
                filter_format, true, false, 1e-6f,
                /*has relu*/ i < filter_shapes.size() - 1, scale_src, zp_src,
                scale_out, zp_out, scale_wei, impl::data_type::u8,
                /*is_quantize_dst? */ i < filter_shapes.size() - 1);
    }
    auto dq3 = create_dequantize(
            id_gen, *agraph, left, "per_tensor", {zp_src}, {scale_src}, 0);
    auto add0 = create_add(id_gen, *agraph, right, dq3);
    auto relu0 = create_relu(id_gen, *agraph, add0);
    auto q2 = create_quantize(id_gen, *agraph, relu0, impl::data_type::u8,
            "per_tensor", std::vector<int64_t> {zp_out},
            std::vector<float> {scale_out}, 0);
    (void)(q2);
}

inline impl::logical_tensor_t create_relu_bwd(utils::id_generator &id_gen,
        impl::graph_t &agraph, const impl::logical_tensor_t &dst,
        const impl::logical_tensor_t &delta_in, const impl::dims &dims) {
    impl::op_t relu_bwd(
            id_gen.get_id(), impl::op_kind::ReLUBackprop, "relu_bwd");
    auto delta
            = utils::logical_tensor_init(id_gen.get_id(), dims, dst.data_type);
    relu_bwd.add_input(dst);
    relu_bwd.add_input(delta);
    relu_bwd.add_output(delta_in);
    agraph.add_op(&relu_bwd);
    return delta;
}

inline impl::logical_tensor_t create_relu_bwd2(utils::id_generator &id_gen,
        impl::graph_t &agraph, const impl::logical_tensor_t &dst,
        const impl::logical_tensor_t &delta) {
    impl::op_t relu_bwd(
            id_gen.get_id(), impl::op_kind::ReLUBackprop, "relu_bwd");
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
inline std::vector<impl::logical_tensor_t> create_convolution_training(
        utils::id_generator &id_gen, impl::graph_t &agraph,
        const impl::logical_tensor_t &src, impl::logical_tensor_t &delta_in,
        const impl::dims &filter_shape, const impl::dims &strides,
        const impl::dims &pads_begin, const impl::dims &pads_end,
        const std::string &data_format, const std::string &filter_format,
        bool with_bn = false, float epsilon = 1e-6f, bool with_momentum = false,
        float momentum = 0.0f, bool with_relu = false,
        impl::logical_tensor_t *known_delta_in_next = nullptr,
        bool is_bn_bf16 = false) {
    auto bn_dtype = is_bn_bf16 ? impl::data_type::bf16 : impl::data_type::f32;
    impl::op_t conv_fwd(
            id_gen.get_id(), impl::op_kind::Convolution, "conv_fwd");
    conv_fwd.set_attr<impl::dims>(impl::op_attr::strides, strides);
    conv_fwd.set_attr<impl::dims>(impl::op_attr::dilations, impl::dims {1, 1});
    if (std::all_of(strides.begin(), strides.end(),
                [](size_t x) { return x == 1; })) {
        conv_fwd.set_attr<impl::dims>(
                impl::op_attr::pads_begin, impl::dims {0, 0});
        conv_fwd.set_attr<impl::dims>(
                impl::op_attr::pads_end, impl::dims {0, 0});
        conv_fwd.set_attr<std::string>(impl::op_attr::auto_pad, "SAME_UPPER");
    } else {
        conv_fwd.set_attr<impl::dims>(impl::op_attr::pads_begin, pads_begin);
        conv_fwd.set_attr<impl::dims>(impl::op_attr::pads_end, pads_end);
    }
    conv_fwd.set_attr<std::string>(impl::op_attr::data_format, data_format);
    conv_fwd.set_attr<std::string>(impl::op_attr::filter_format, filter_format);

    impl::dim_t ih = (data_format == "NCX") ? src.dims[2] : src.dims[1];
    impl::dim_t iw = (data_format == "NCX") ? src.dims[3] : src.dims[2];
    impl::dim_t ks
            = (filter_format == "OIX") ? filter_shape[2] : filter_shape[0];
    impl::dim_t ic
            = (filter_format == "OIX") ? filter_shape[1] : filter_shape[2];
    impl::dim_t oc
            = (filter_format == "OIX") ? filter_shape[0] : filter_shape[3];
    impl::dim_t oh = (ih + pads_begin[0] + pads_end[0] - ks) / strides[0] + 1;
    impl::dim_t ow = (iw + pads_begin[1] + pads_end[1] - ks) / strides[1] + 1;

    impl::dims src_shape = (data_format == "NCX")
            ? impl::dims {src.dims[0], ic, ih, iw}
            : impl::dims {src.dims[0], ih, iw, ic};
    impl::dims dst_shape = (data_format == "NCX")
            ? impl::dims {src.dims[0], oc, oh, ow}
            : impl::dims {src.dims[0], oh, ow, oc};

    auto wei = utils::logical_tensor_init(
            id_gen.get_id(), filter_shape, src.data_type);
    auto dst = utils::logical_tensor_init(
            id_gen.get_id(), dst_shape, src.data_type);

    conv_fwd.add_input(src);
    conv_fwd.add_input(wei);
    conv_fwd.add_output(dst);
    agraph.add_op(&conv_fwd);

    // gamma, mean, var will be reused in bwd part
    impl::logical_tensor_t bn_src, gamma, mean, var;
    if (with_bn) {
        impl::op_t bn_fwd(id_gen.get_id(),
                impl::op_kind::BatchNormForwardTraining, "bn_fwd");
        bn_fwd.set_attr<std::string>(impl::op_attr::data_format, data_format);
        bn_fwd.set_attr<float>(impl::op_attr::epsilon, epsilon);
        if (with_momentum) {
            bn_fwd.set_attr<float>(impl::op_attr::momentum, momentum);
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
        impl::op_t relu_fwd(id_gen.get_id(), impl::op_kind::ReLU, "relu_fwd");
        auto relu_src = dst;
        dst = utils::logical_tensor_init(
                id_gen.get_id(), dst_shape, src.data_type);
        relu_fwd.add_input(relu_src);
        relu_fwd.add_output(dst);
        agraph.add_op(&relu_fwd);
    }

    // start constructing bwd part
    impl::logical_tensor_t output_delta = utils::logical_tensor_init(
            id_gen.get_id(), dst_shape, src.data_type);
    impl::logical_tensor_t delta_in_next;
    if (known_delta_in_next) {
        output_delta = *known_delta_in_next;
        delta_in_next = *known_delta_in_next;
    } else {
        delta_in_next = output_delta;
    }
    if (with_relu) {
        // bwd relu
        impl::op_t relu_bwd(
                id_gen.get_id(), impl::op_kind::ReLUBackprop, "relu_bwd");
        relu_bwd.set_attr<bool>(impl::op_attr::use_dst, true);
        relu_bwd.add_input(dst); // dst is dst of fwd
        relu_bwd.add_input(output_delta);
        output_delta = utils::logical_tensor_init(
                id_gen.get_id(), dst_shape, src.data_type);
        relu_bwd.add_output(output_delta);
        agraph.add_op(&relu_bwd);
    }
    if (with_bn) {
        impl::op_t bn_bwd(id_gen.get_id(),
                impl::op_kind::BatchNormTrainingBackprop, "bn_bwd");
        bn_bwd.set_attr<std::string>(impl::op_attr::data_format, data_format);
        bn_bwd.set_attr<float>(impl::op_attr::epsilon, epsilon);

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

    impl::op_t conv_bwd_data(id_gen.get_id(),
            impl::op_kind::ConvolutionBackpropData, "conv_bwd_data");
    conv_bwd_data.set_attr<impl::dims>(impl::op_attr::strides, strides);
    conv_bwd_data.set_attr<impl::dims>(
            impl::op_attr::dilations, impl::dims {1, 1});
    if (std::all_of(strides.begin(), strides.end(),
                [](size_t x) { return x == 1; })) {
        conv_bwd_data.set_attr<impl::dims>(
                impl::op_attr::pads_begin, impl::dims {0, 0});
        conv_bwd_data.set_attr<impl::dims>(
                impl::op_attr::pads_end, impl::dims {0, 0});
        conv_bwd_data.set_attr<std::string>(
                impl::op_attr::auto_pad, "SAME_UPPER");
    } else {
        conv_bwd_data.set_attr<impl::dims>(
                impl::op_attr::pads_begin, pads_begin);
        conv_bwd_data.set_attr<impl::dims>(impl::op_attr::pads_end, pads_end);
    }
    conv_bwd_data.set_attr<std::string>(
            impl::op_attr::data_format, data_format);
    conv_bwd_data.set_attr<std::string>(
            impl::op_attr::filter_format, filter_format);
    conv_bwd_data.set_attr<impl::dims>(impl::op_attr::output_shape, src_shape);
    conv_bwd_data.add_input(output_delta);
    conv_bwd_data.add_input(wei);
    conv_bwd_data.add_output(delta_in);
    agraph.add_op(&conv_bwd_data);

    impl::op_t conv_bwd_weight(id_gen.get_id(),
            impl::op_kind::ConvolutionBackpropFilters, "conv_bwd_weight");
    conv_bwd_weight.set_attr<impl::dims>(impl::op_attr::strides, strides);
    conv_bwd_weight.set_attr<impl::dims>(
            impl::op_attr::dilations, impl::dims {1, 1});
    if (std::all_of(strides.begin(), strides.end(),
                [](size_t x) { return x == 1; })) {
        conv_bwd_weight.set_attr<impl::dims>(
                impl::op_attr::pads_begin, impl::dims {0, 0});
        conv_bwd_weight.set_attr<impl::dims>(
                impl::op_attr::pads_end, impl::dims {0, 0});
        conv_bwd_weight.set_attr<std::string>(
                impl::op_attr::auto_pad, "SAME_UPPER");
    } else {
        conv_bwd_weight.set_attr<impl::dims>(
                impl::op_attr::pads_begin, pads_begin);
        conv_bwd_weight.set_attr<impl::dims>(impl::op_attr::pads_end, pads_end);
    }
    conv_bwd_weight.set_attr<std::string>(
            impl::op_attr::data_format, data_format);
    conv_bwd_weight.set_attr<std::string>(
            impl::op_attr::filter_format, filter_format);
    conv_bwd_weight.set_attr<impl::dims>(
            impl::op_attr::filter_shape, filter_shape);
    conv_bwd_weight.add_input(src);
    conv_bwd_weight.add_input(output_delta);
    auto delta_weight = utils::logical_tensor_init(
            id_gen.get_id(), filter_shape, src.data_type);
    conv_bwd_weight.add_output(delta_weight);
    agraph.add_op(&conv_bwd_weight);

    return {dst, delta_in_next};
}

inline void construct_identical_bottleneck_training_subgraph(
        impl::graph_t *agraph, utils::id_generator &id_gen,
        const impl::dims &input_shape,
        const std::vector<impl::dims> &filter_shapes, bool is_bf16 = false,
        bool is_bn_bf16 = false,
        const std::vector<std::vector<int64_t>> &strides
        = {{1, 1}, {1, 1}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NXC",
        const std::string &filter_format = "XIO") {
    auto dtype = is_bf16 ? impl::data_type::bf16 : impl::data_type::f32;
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
        impl::graph_t *agraph, utils::id_generator &id_gen,
        const impl::dims &input_shape,
        const std::vector<impl::dims> &filter_shapes, bool is_bf16 = false,
        bool is_bn_bf16 = false,
        const std::vector<std::vector<int64_t>> &strides
        = {{2, 2}, {1, 1}, {2, 2}, {1, 1}},
        const std::vector<std::vector<int64_t>> &paddings
        = {{0, 0}, {0, 0}, {1, 1}, {0, 0}},
        const std::string &data_format = "NXC",
        const std::string &filter_format = "XIO") {
    auto infer_output_shape = [&](impl::dims shape) {
        if (data_format == "NCX")
            return impl::dims {
                    shape[0], filter_shapes[0][0], shape[2] / 2, shape[3] / 2};
        else
            return impl::dims {
                    shape[0], shape[1] / 2, shape[2] / 2, filter_shapes[0][3]};
    };

    auto dtype = is_bf16 ? impl::data_type::bf16 : impl::data_type::f32;
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

inline void construct_instance_norm_subgraph(impl::graph_t &agraph,
        utils::id_generator &id_gen, const impl::dims &input_shape,
        const impl::dims &strides, const impl::dims &filter_shape,
        const impl::dims &bias_shape, const std::string &auto_pad,
        const impl::dims &pads_begin, const impl::dims &pads_end,
        const impl::dims &reduce_axes, const std::string &relu_type = "",
        bool is_bf16 = false) {
    auto dtype = is_bf16 ? impl::data_type::bf16 : impl::data_type::f32;

    impl::op_t conv_fwd(
            id_gen.get_id(), impl::op_kind::Convolution, "conv_fwd");
    conv_fwd.set_attr<std::string>(impl::op_attr::auto_pad, auto_pad);
    conv_fwd.set_attr<std::string>(impl::op_attr::data_format, "NXC");
    conv_fwd.set_attr<std::string>(impl::op_attr::filter_format, "XIO");
    conv_fwd.set_attr<impl::dims>(impl::op_attr::pads_begin, pads_begin);
    conv_fwd.set_attr<impl::dims>(impl::op_attr::pads_end, pads_end);
    conv_fwd.set_attr<impl::dims>(impl::op_attr::strides, strides);
    conv_fwd.set_attr<impl::dims>(
            impl::op_attr::dilations, impl::dims {1, 1, 1});

    // infered based on NXC & XIO
    impl::dims dst_shape = {input_shape[0], input_shape[1], input_shape[2],
            input_shape[3], filter_shape[4]};
    auto src = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    auto weight
            = utils::logical_tensor_init(id_gen.get_id(), filter_shape, dtype);
    auto conv_dst
            = utils::logical_tensor_init(id_gen.get_id(), dst_shape, dtype);

    conv_fwd.add_input(src);
    conv_fwd.add_input(weight);
    conv_fwd.add_output(conv_dst);
    agraph.add_op(&conv_fwd);

    impl::op_t bias_add(id_gen.get_id(), impl::op_kind::Add, "bias_add");
    bias_add.set_attr<std::string>(impl::op_attr::auto_broadcast, "numpy");
    auto bias_add_lhs
            = utils::logical_tensor_init(id_gen.get_id(), bias_shape, dtype);
    auto bias_add_rhs = conv_dst;
    auto bias_add_dst
            = utils::logical_tensor_init(id_gen.get_id(), dst_shape, dtype);
    bias_add.add_input(bias_add_lhs);
    bias_add.add_input(bias_add_rhs);
    bias_add.add_output(bias_add_dst);
    agraph.add_op(&bias_add);

    impl::dims reduce_dst_shape = dst_shape;
    for (auto i : reduce_axes) {
        reduce_dst_shape[i] = 1;
    }
    impl::op_t reduce_mean_1 {
            id_gen.get_id(), impl::op_kind::ReduceMean, "reduce_mean_1"};
    reduce_mean_1.set_attr(impl::op_attr::keep_dims, true);
    reduce_mean_1.set_attr(impl::op_attr::axes, reduce_axes);

    auto reduce_mean_1_dst = utils::logical_tensor_init(
            id_gen.get_id(), reduce_dst_shape, dtype);
    reduce_mean_1.add_input(bias_add_dst);
    reduce_mean_1.add_output(reduce_mean_1_dst);
    agraph.add_op(&reduce_mean_1);

    impl::op_t sqd_diff {
            id_gen.get_id(), impl::op_kind::SquaredDifference, "squared_diff"};
    auto sqd_diff_dst
            = utils::logical_tensor_init(id_gen.get_id(), dst_shape, dtype);
    sqd_diff.add_input(bias_add_dst);
    sqd_diff.add_input(reduce_mean_1_dst);
    sqd_diff.add_output(sqd_diff_dst);
    agraph.add_op(&sqd_diff);

    impl::op_t reduce_mean_2 {
            id_gen.get_id(), impl::op_kind::ReduceMean, "reduce_mean_2"};
    reduce_mean_2.set_attr(impl::op_attr::keep_dims, true);
    reduce_mean_2.set_attr(impl::op_attr::axes, reduce_axes);
    auto reduce_mean_2_dst = utils::logical_tensor_init(
            id_gen.get_id(), reduce_dst_shape, dtype);
    reduce_mean_2.add_input(sqd_diff_dst);
    reduce_mean_2.add_output(reduce_mean_2_dst);
    agraph.add_op(&reduce_mean_2);

    impl::op_t bias_add_2(id_gen.get_id(), impl::op_kind::Add, "bias_add_2");
    bias_add_2.set_attr<std::string>(impl::op_attr::auto_broadcast, "numpy");
    auto bias_add_2_lhs = reduce_mean_2_dst;
    auto bias_add_2_rhs
            = utils::logical_tensor_init(id_gen.get_id(), bias_shape, dtype);
    auto bias_add_2_dst = utils::logical_tensor_init(
            id_gen.get_id(), reduce_dst_shape, dtype);
    bias_add_2.add_input(bias_add_2_lhs);
    bias_add_2.add_input(bias_add_2_rhs);
    bias_add_2.add_output(bias_add_2_dst);
    agraph.add_op(&bias_add_2);

    impl::op_t rsqrt(id_gen.get_id(), impl::op_kind::Rsqrt, "rsqrt");
    auto rsqrt_dst = utils::logical_tensor_init(
            id_gen.get_id(), reduce_dst_shape, dtype);
    rsqrt.add_input(bias_add_2_dst);
    rsqrt.add_output(rsqrt_dst);
    agraph.add_op(&rsqrt);

    impl::op_t mul_1 {id_gen.get_id(), impl::op_kind::Multiply, "mul_1"};
    auto mul_1_rhs = utils::logical_tensor_init(
            id_gen.get_id(), reduce_dst_shape, dtype);
    auto mul_1_dst = utils::logical_tensor_init(
            id_gen.get_id(), reduce_dst_shape, dtype);
    mul_1.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    mul_1.add_input(rsqrt_dst);
    mul_1.add_input(mul_1_rhs);
    mul_1.add_output(mul_1_dst);
    agraph.add_op(&mul_1);

    impl::op_t mul_2 {id_gen.get_id(), impl::op_kind::Multiply, "mul_2"};
    auto mul_2_dst = utils::logical_tensor_init(
            id_gen.get_id(), reduce_dst_shape, dtype);
    mul_2.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    mul_2.add_input(reduce_mean_1_dst);
    mul_2.add_input(mul_1_dst);
    mul_2.add_output(mul_2_dst);
    agraph.add_op(&mul_2);

    impl::op_t mul_3 {id_gen.get_id(), impl::op_kind::Multiply, "mul_3"};
    auto mul_3_dst
            = utils::logical_tensor_init(id_gen.get_id(), dst_shape, dtype);
    mul_3.set_attr(impl::op_attr::auto_broadcast, std::string("numpy"));
    mul_3.add_input(bias_add_dst);
    mul_3.add_input(mul_1_dst);
    mul_3.add_output(mul_3_dst);
    agraph.add_op(&mul_3);

    impl::op_t sub(id_gen.get_id(), impl::op_kind::Subtract, "subtract");
    auto sub_lhs = utils::logical_tensor_init(
            id_gen.get_id(), reduce_dst_shape, dtype);
    auto sub_dst = utils::logical_tensor_init(
            id_gen.get_id(), reduce_dst_shape, dtype);
    sub.add_input(sub_lhs);
    sub.add_input(mul_2_dst);
    sub.add_output(sub_dst);
    agraph.add_op(&sub);

    impl::op_t bias_add_3(id_gen.get_id(), impl::op_kind::Add, "bias_add_3");
    bias_add_3.set_attr<std::string>(impl::op_attr::auto_broadcast, "numpy");
    auto bias_add_3_dst
            = utils::logical_tensor_init(id_gen.get_id(), dst_shape, dtype);
    bias_add_3.add_input(mul_3_dst);
    bias_add_3.add_input(sub_dst);
    bias_add_3.add_output(bias_add_3_dst);
    agraph.add_op(&bias_add_3);

    if (relu_type == "ReLU") {
        impl::op_t relu(id_gen.get_id(), impl::op_kind::ReLU, "relu");
        auto relu_dst
                = utils::logical_tensor_init(id_gen.get_id(), dst_shape, dtype);
        relu.add_input(bias_add_3_dst);
        relu.add_output(relu_dst);
        agraph.add_op(&relu);
    } else if (relu_type == "LeakyReLU") {
        impl::op_t leaky_relu(
                id_gen.get_id(), impl::op_kind::LeakyReLU, "leaky_relu");
        leaky_relu.set_attr<float>(impl::op_attr::alpha, 0.01);
        auto leaky_relu_dst
                = utils::logical_tensor_init(id_gen.get_id(), dst_shape, dtype);
        leaky_relu.add_input(bias_add_3_dst);
        leaky_relu.add_output(leaky_relu_dst);
        agraph.add_op(&leaky_relu);
    }
}

} // namespace utils
} // namespace compiler
} // namespace unit
} // namespace tests
} // namespace graph
} // namespace dnnl

#endif
