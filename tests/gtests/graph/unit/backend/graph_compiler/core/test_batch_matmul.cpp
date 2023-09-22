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

#include <iostream>
#include <utility>
#include "context.hpp"
#include "test_utils.hpp"
#include "util/bf16.hpp"
#include "util/fp16.hpp"
#include "gtest/gtest.h"
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/index_flatten.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/templates/matmul_core.hpp>
#include <reference/act_ref.hpp>
#include <reference/bias_ref.hpp>
#include <reference/gemm_ref.hpp>
#include <util/parallel.hpp>
#include <util/reflection.hpp>

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::ops;
using namespace dnnl::impl::graph::gc::test_utils;
struct batch_gemm_params_t {
    batch_gemm_params_t(sc_dims input_dims, sc_dims weight_dims,
            sc_dims out_dims, sc_data_format_t in_format,
            sc_data_format_t weight_format,
            sc_data_type_t input_dtype = datatypes::f32,
            sc_data_type_t weight_dtype = datatypes::f32,
            bool is_input_constant = true)
        : input_dims_(std::move(input_dims))
        , weight_dims_(std::move(weight_dims))
        , out_dims_(std::move(out_dims))
        , input_format_(in_format)
        , weight_format_(weight_format)
        , input_dtype_(input_dtype)
        , weight_dtype_(weight_dtype)
        , is_input_constant_(is_input_constant) {}
    sc_dims input_dims_;
    sc_dims weight_dims_;
    sc_dims out_dims_;
    sc_data_format_t input_format_;
    sc_data_format_t weight_format_;
    sc_data_type_t input_dtype_;
    sc_data_type_t weight_dtype_;
    bool is_input_constant_;
};

template <typename Reftype>
void transpose_ref(std::vector<Reftype> &ref, const sc_dims in_dims,
        const sc_data_format_t in_format) {
    if (in_format.is_any() || in_format.format_code_ == format_kinds::ABC
            || in_format.format_code_ == format_kinds::ABCD
            || in_dims.size() == 2) {
        return;
    }
    COMPILE_ASSERT(!in_format.is_blocking(),
            "transpose_ref does not support blocking input in test");
    const std::vector<int64_t> stride = {in_dims[1] * in_dims[2] * in_dims[3],
            in_dims[2] * in_dims[3], in_dims[3], 1};
    std::vector<int64_t> orig_stride;
    orig_stride.reserve(in_dims.size());

    for (uint8_t i = 0; i < in_dims.size(); i++) {
        int64_t temp_stride = 1;
        for (uint8_t j = 0; j < in_dims.size(); j++) {
            if (in_format.format_code_.get(j) == i) {
                uint8_t ii = j + 1;
                while (ii < in_dims.size()) {
                    temp_stride *= in_dims[in_format.format_code_.get(ii)];
                    ii++;
                }
                break;
            }
        }
        orig_stride.push_back(temp_stride);
    }
    std::vector<Reftype> ref_ = ref;
    utils::parallel_for(0, in_dims[0], 1, [&](int64_t i) {
        for (int64_t j = 0; j < in_dims[1]; ++j) {
            for (int64_t ii = 0; ii < in_dims[2]; ++ii) {
                for (int64_t jj = 0; jj < in_dims[3]; ++jj) {
                    ref_[i * stride[0] + j * stride[1] + ii * stride[2] + jj]
                            = ref[i * orig_stride[0] + j * orig_stride[1]
                                    + ii * orig_stride[2] + jj];
                }
            }
        }
    });
    ref.swap(ref_);
}

template <typename Atype, typename Btype>
void alloc_sc_input_and_weight(test_buffer<Atype> &sc_input,
        test_buffer<Btype> &sc_weight, const sc_dims input_dims,
        const sc_dims weight_dims) {
    sc_input = alloc_array<Atype>(cal_size(input_dims));
    sc_weight = alloc_array<Btype>(cal_size(weight_dims));
}

template <typename Atype, typename Btype, typename Ctype>
void run_bmm_test(const std::shared_ptr<jit_function_t> &fptr, int M, int N,
        int K, int batch_size, const sc_dims input_dims,
        const sc_data_format_t in_format, const sc_dims weight_dims,
        const sc_data_format_t weight_format, const sc_dims out_dims) {
    test_buffer<Atype> sc_input;
    test_buffer<Btype> sc_weight;
    alloc_sc_input_and_weight(sc_input, sc_weight, input_dims, weight_dims);
    auto sc_output = alloc_array<Ctype>(cal_size(out_dims));
    auto ref_input = std::vector<Ctype>(sc_input.begin(), sc_input.end());
    auto ref_weight = std::vector<Ctype>(sc_weight.begin(), sc_weight.end());
    auto ref_output = std::vector<Ctype>(cal_size(out_dims));

    fptr->call_default(&sc_output[0], &sc_input[0], &sc_weight[0]);

    transpose_ref(ref_input, input_dims, in_format);
    transpose_ref(ref_weight, weight_dims, weight_format);

    int input_size = M * K;
    int weight_size = K * N;
    int out_size = M * N;
    // Reorder dst from block format to plain format
    for (int b = 0; b < batch_size; b++) {
        int b_i = input_dims.size() >= weight_dims.size() ? b
                : input_dims.size() == 2                  ? 0
                                                          : b
                        % cal_size(sc_dims {
                                input_dims.begin(), input_dims.end() - 2});
        int b_w = weight_dims.size() >= input_dims.size() ? b
                : weight_dims.size() == 2                 ? 0
                                                          : b
                        % cal_size(sc_dims {
                                weight_dims.begin(), weight_dims.end() - 2});
        gemm_params gemm_param {false, false, M, N, K, 1.0, 0.0, K, N, N};
        ref_gemm(gemm_param, &ref_input[b_i * input_size],
                &ref_weight[b_w * weight_size], &ref_output[b * out_size]);
        test_utils::compare_data(sc_output.data() + b * out_size,
                ref_output.data() + b * out_size, out_size, 1e-4f, 1e-4f);
    }
}

static void check_batch_matmul(
        const batch_gemm_params_t &param, const matmul_core_config_t &cfg) {
    REQUIRE_AVX2();
    const sc_dims input_dims = param.input_dims_;
    const sc_dims weight_dims = param.weight_dims_;
    const sc_dims out_dims = param.out_dims_;
    const sc_data_format_t input_format = param.input_format_;
    const sc_data_format_t weight_format = param.weight_format_;
    const sc_data_type_t input_dtype = param.input_dtype_;
    const sc_data_type_t weight_dtype = param.weight_dtype_;
    sc_dims batch_dims;
    sc_graph_t graph;
    int M, N, K;
    bool is_quantized
            = utils::is_one_of(input_dtype, datatypes::u8, datatypes::s8);
    bool is_s8s8
            = input_dtype == datatypes::s8 && weight_dtype == datatypes::s8;
    bool is_u8s8
            = input_dtype == datatypes::u8 && weight_dtype == datatypes::s8;
    bool is_bf16
            = input_dtype == datatypes::bf16 && weight_dtype == datatypes::bf16;
    bool is_f16
            = input_dtype == datatypes::f16 && weight_dtype == datatypes::f16;

    auto ctx = get_test_ctx();

    batch_dims = input_dims.size() > weight_dims.size()
            ? sc_dims {input_dims.begin(), input_dims.end() - 2}
            : sc_dims {weight_dims.begin(), weight_dims.end() - 2};
    M = input_dims[input_dims.size() - 2];
    N = weight_dims[weight_dims.size() - 1];
    K = input_dims[input_dims.size() - 1];
    const int batch_size = cal_size(batch_dims);

    auto data = graph.make_input(
            {graph_tensor::make(input_dims, input_format, input_dtype)});
    auto weight = graph.make_input(
            {graph_tensor::make(weight_dims, weight_format, weight_dtype)});
    auto bmm = graph.make("matmul_core",
            {data->get_outputs()[0], weight->get_outputs()[0]},
            {graph_tensor::make(out_dims, sc_data_format_t(),
                    is_quantized ? datatypes::s32 : datatypes::f32)},
            {});
    bmm->stc_cast<tunable_op_t>()->set_config(
            reflection::general_object_t::make(cfg));
    bmm->dyn_cast<op_traits::may_quantize_t>()->is_quantized_ = is_quantized;
    auto output = graph.make_output(bmm->get_outputs());
    graph.attrs_[sc_graph_t::attr_key_t::quantize] = is_quantized;
    if (param.is_input_constant_) {
        data->attrs_.set("constant", const_kind::local_const);
        weight->attrs_.set("constant", const_kind::local_const);
    }
    graph_driver(graph, ctx);

    auto f = lower_graph(
            ctx, graph, std::vector<sc_op_ptr> {output, data, weight});
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);

    if (is_quantized && is_s8s8) { // s8s8
        run_bmm_test<int8_t, int8_t, int32_t>(fptr, M, N, K, batch_size,
                input_dims, input_format, weight_dims, weight_format, out_dims);
    } else if (is_quantized && is_u8s8) { // u8s8
        run_bmm_test<uint8_t, int8_t, int32_t>(fptr, M, N, K, batch_size,
                input_dims, input_format, weight_dims, weight_format, out_dims);
    } else if (is_bf16) { // bf16
        run_bmm_test<bf16_t, bf16_t, float>(fptr, M, N, K, batch_size,
                input_dims, input_format, weight_dims, weight_format, out_dims);
    } else if (is_f16) { // f16
        run_bmm_test<fp16_t, fp16_t, float>(fptr, M, N, K, batch_size,
                input_dims, input_format, weight_dims, weight_format, out_dims);
    } else { // f32
        run_bmm_test<float, float, float>(fptr, M, N, K, batch_size, input_dims,
                input_format, weight_dims, weight_format, out_dims);
    }
}

const matmul_core_config_t cfg_fwd = {
        16, // M_block
        8, // N_block
        4, // K_block
};
// f32
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD1) {
    check_batch_matmul({{32, 2, 64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABCD)},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD2) {
    check_batch_matmul({{2, 2048, 512}, {2, 512, 4096}, {2, 2048, 4096},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABC)},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD3) {
    check_batch_matmul({{8, 4, 64, 512}, {8, 4, 512, 32}, {8, 4, 64, 32},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABCD)},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD4) {
    check_batch_matmul({{2, 4, 64, 256}, {2, 4, 256, 128}, {2, 4, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABCD)},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD5) {
    check_batch_matmul({{2, 256, 64}, {2, 64, 128}, {2, 256, 128},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABC)},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD6) {
    check_batch_matmul({{32, 2, 64, 256}, {256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t::KN()},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD7) {
    check_batch_matmul({{32, 2, 64, 256}, {2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABC)},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD8) {
    check_batch_matmul({{64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t::MK(),
                               sc_data_format_t(format_kinds::ABCD)},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD9) {
    check_batch_matmul({{2, 64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABCD)},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD10) {
    check_batch_matmul({{2, 2048, 512}, {2, 512, 4096}, {2, 2048, 4096},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABC),
                               datatypes::f32, datatypes::f32, false},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWD11) {
    check_batch_matmul({{8, 4, 64, 512}, {8, 4, 512, 32}, {8, 4, 64, 32},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::f32, datatypes::f32, false},
            cfg_fwd);
}
// s8s8
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemms8s8FWD1) {
    REQUIRE_VNNI();
    check_batch_matmul({{2, 256, 64}, {2, 64, 128}, {2, 256, 128},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABC),
                               datatypes::s8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemms8s8FWD2) {
    REQUIRE_VNNI();
    check_batch_matmul({{32, 2, 64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::s8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemms8s8FWD3) {
    REQUIRE_VNNI();
    check_batch_matmul(
            {{32, 2, 64, 256}, {256, 128}, {32, 2, 64, 128},
                    sc_data_format_t(format_kinds::ABCD),
                    sc_data_format_t::KN(), datatypes::s8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemms8s8FWD4) {
    REQUIRE_VNNI();
    check_batch_matmul({{32, 2, 64, 256}, {2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABC),
                               datatypes::s8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemms8s8FWD5) {
    REQUIRE_VNNI();
    check_batch_matmul({{64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t::MK(),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::s8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemms8s8FWD6) {
    REQUIRE_VNNI();
    check_batch_matmul({{2, 64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::s8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemms8s8FWD7) {
    REQUIRE_VNNI();
    check_batch_matmul({{2, 256, 64}, {2, 64, 128}, {2, 256, 128},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABC),
                               datatypes::s8, datatypes::s8, false},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemms8s8FWD8) {
    REQUIRE_VNNI();
    check_batch_matmul({{32, 2, 64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::s8, datatypes::s8, false},
            cfg_fwd);
}
// u8s8
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmu8s8FWD1) {
    REQUIRE_VNNI();
    check_batch_matmul({{2, 256, 64}, {2, 64, 128}, {2, 256, 128},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABC),
                               datatypes::u8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmu8s8FWD2) {
    REQUIRE_VNNI();
    check_batch_matmul({{32, 2, 64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::u8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmu8s8FWD3) {
    REQUIRE_VNNI();
    check_batch_matmul(
            {{32, 2, 64, 256}, {256, 128}, {32, 2, 64, 128},
                    sc_data_format_t(format_kinds::ABCD),
                    sc_data_format_t::KN(), datatypes::u8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmu8s8FWD4) {
    REQUIRE_VNNI();
    check_batch_matmul({{32, 2, 64, 256}, {2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABC),
                               datatypes::u8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmu8s8FWD5) {
    REQUIRE_VNNI();
    check_batch_matmul({{64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t::MK(),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::u8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmu8s8FWD6) {
    REQUIRE_VNNI();
    check_batch_matmul({{2, 64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::u8, datatypes::s8},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmu8s8FWD7) {
    REQUIRE_VNNI();
    check_batch_matmul({{2, 256, 64}, {2, 64, 128}, {2, 256, 128},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABC),
                               datatypes::u8, datatypes::s8, false},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmu8s8FWD8) {
    REQUIRE_VNNI();
    check_batch_matmul({{32, 2, 64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::u8, datatypes::s8, false},
            cfg_fwd);
}
// bf16
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmbf16FWD1) {
    REQUIRE_VNNI();
    REQUIRE_BF16();
    check_batch_matmul({{2, 256, 64}, {2, 64, 128}, {2, 256, 128},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABC),
                               datatypes::bf16, datatypes::bf16},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmbf16FWD2) {
    REQUIRE_VNNI();
    REQUIRE_BF16();
    check_batch_matmul({{32, 2, 64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::bf16, datatypes::bf16},
            cfg_fwd);
}
// f16
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmf16FWD1) {
    REQUIRE_FP16();
    check_batch_matmul({{2, 256, 64}, {2, 64, 128}, {2, 256, 128},
                               sc_data_format_t(format_kinds::ABC),
                               sc_data_format_t(format_kinds::ABC),
                               datatypes::f16, datatypes::f16},
            cfg_fwd);
}
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmf16FWD2) {
    REQUIRE_FP16();
    check_batch_matmul({{32, 2, 64, 256}, {32, 2, 256, 128}, {32, 2, 64, 128},
                               sc_data_format_t(format_kinds::ABCD),
                               sc_data_format_t(format_kinds::ABCD),
                               datatypes::f16, datatypes::f16},
            cfg_fwd);
}
// bert case
TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERT_QK) {
    check_batch_matmul({{2, 16, 384, 64}, {2, 16, 64, 384}, {2, 16, 384, 384},
                               sc_data_format_t(), sc_data_format_t()},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERT_QK2) {
    check_batch_matmul({{2, 16, 384, 64}, {2, 16, 64, 384}, {2, 16, 384, 384},
                               sc_data_format_t(), sc_data_format_t(),
                               datatypes::f32, datatypes::f32, false},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERTs8s8_QK) {
    REQUIRE_VNNI();
    check_batch_matmul({{2, 16, 384, 64}, {2, 16, 64, 384}, {2, 16, 384, 384},
                               sc_data_format_t(), sc_data_format_t(),
                               datatypes::s8, datatypes::s8},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERTs8s8_QK2) {
    REQUIRE_VNNI();
    check_batch_matmul({{2, 16, 384, 64}, {2, 16, 64, 384}, {2, 16, 384, 384},
                               sc_data_format_t(), sc_data_format_t(),
                               datatypes::s8, datatypes::s8, false},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERTu8s8_QK) {
    REQUIRE_VNNI();
    check_batch_matmul({{2, 16, 384, 64}, {2, 16, 64, 384}, {2, 16, 384, 384},
                               sc_data_format_t(), sc_data_format_t(),
                               datatypes::u8, datatypes::s8},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERTu8s8_QK2) {
    REQUIRE_VNNI();
    check_batch_matmul({{2, 16, 384, 64}, {2, 16, 64, 384}, {2, 16, 384, 384},
                               sc_data_format_t(), sc_data_format_t(),
                               datatypes::u8, datatypes::s8, false},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERT_V) {
    check_batch_matmul(
            {{2, 16, 384, 384}, {2, 16, 384, 64}, {2, 16, 384, 64},
                    sc_data_format_t(), sc_data_format_t(format_kinds::ACBD)},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERT_V2) {
    check_batch_matmul(
            {{2, 16, 384, 384}, {2, 16, 384, 64}, {2, 16, 384, 64},
                    sc_data_format_t(), sc_data_format_t(format_kinds::ACBD),
                    datatypes::f32, datatypes::f32, false},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERTs8s8_V) {
    REQUIRE_VNNI();
    check_batch_matmul(
            {{2, 16, 384, 384}, {2, 16, 384, 64}, {2, 16, 384, 64},
                    sc_data_format_t(), sc_data_format_t(format_kinds::ACBD),
                    datatypes::s8, datatypes::s8},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERTs8s8_V2) {
    REQUIRE_VNNI();
    check_batch_matmul(
            {{2, 16, 384, 384}, {2, 16, 384, 64}, {2, 16, 384, 64},
                    sc_data_format_t(), sc_data_format_t(format_kinds::ACBD),
                    datatypes::s8, datatypes::s8, false},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERTu8s8_V) {
    REQUIRE_VNNI();
    check_batch_matmul(
            {{2, 16, 384, 384}, {2, 16, 384, 64}, {2, 16, 384, 64},
                    sc_data_format_t(), sc_data_format_t(format_kinds::ACBD),
                    datatypes::u8, datatypes::s8},
            cfg_fwd);
}

TEST(GCCore_CPU_batch_matmul_test, TestBatchGemmFWDBERTu8s8_V2) {
    REQUIRE_VNNI();
    check_batch_matmul(
            {{2, 16, 384, 384}, {2, 16, 384, 64}, {2, 16, 384, 64},
                    sc_data_format_t(), sc_data_format_t(format_kinds::ACBD),
                    datatypes::u8, datatypes::s8, false},
            cfg_fwd);
}
