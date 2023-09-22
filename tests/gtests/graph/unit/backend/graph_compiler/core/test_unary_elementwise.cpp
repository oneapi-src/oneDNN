/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include "act_ref.hpp"
#include "compiler/ir/graph/driver.hpp"
#include "compiler/ir/graph/graph.hpp"
#include "compiler/ir/graph/lowering.hpp"
#include "compiler/ir/sc_data_format.hpp"
#include "context.hpp"
#include "gtest/gtest.h"
#include <compiler/jit/jit.hpp>

using namespace dnnl::impl::graph::gc;
static bool verbose = false;
static float alpha = 0.35f;
static float thebeta = 1.47f;

TEST(GCCore_CPU_unary_elemwise, TestStridedRelu) {
    REQUIRE_AVX2();
    sc_dims input_dims = {128, 64, 56, 56};
    sc_graph_t g;
    auto ins = g.make_input({graph_tensor::make(input_dims)});
    auto reorderop = g.make("reorder", ins->get_outputs(),
            {graph_tensor::make(input_dims, sc_data_format_t::NCHWc(16))}, {});
    auto out = g.make_output(g.make("relu", reorderop->get_outputs(),
                                      {graph_tensor::make(input_dims)}, {})
                                     ->get_outputs());

    const auto input_size = test_utils::product(input_dims);
    std::vector<float> input_data(input_size), sc_output(input_size),
            ref_output(input_size);
    test_utils::fill_data(&input_data[0], input_size);

    graph_driver(g);
    auto f = lower_graph(get_test_ctx(), g, {ins, out});
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f);
    fptr->call_default(&input_data[0], &sc_output[0]);

    ref_relu(ref_output.data(), input_data.data(), input_size);
    test_utils::compare_data(sc_output, ref_output, 1e-4f, 1e-5f);
}

template <typename T>
static void check_unary_elementwise(const std::string &op_name,
        const sc_dims &input_dims,
        std::function<void(T *, const T *, size_t)> &ref_func) {
    sc_graph_t g;
    sc_op_ptr ins;
    bool is_bf16 = std::is_same<typename std::decay<T>::type, bf16_t>::value;
    if (is_bf16
            && !::dnnl::impl::graph::gc::get_default_context()
                        ->machine_.cpu_flags_.fAVX512F) {
        return;
    }
    ins = g.make_input({graph_tensor::make(input_dims, sc_data_format_t(),
            is_bf16 ? datatypes::bf16 : datatypes::f32)});
    auto op = g.make(op_name, ins->get_outputs(), {},
            {{"alpha", alpha}, {"beta", thebeta}});
    g.make_output(op->get_outputs());
    graph_driver(g, get_test_ctx());

    auto f = lower_graph(get_test_ctx(), g, {});
    if (verbose) std::cout << f << std::endl;
    if (op_name == "pow" && thebeta == -0.5f) {
        std::stringstream ss;
        ss << f;
        EXPECT_TRUE(ss.str().find("rsqrt") != std::string::npos);
    }
    const auto input_size = test_utils::product(input_dims);
    std::vector<T> input_data(input_size);
    if (utils::is_one_of(op_name, std::string("log"), std::string("pow"))) {
        test_utils::fill_data(&input_data[0], input_size, static_cast<T>(1e-4f),
                static_cast<T>(1.f));
    } else {
        test_utils::fill_data(&input_data[0], input_size);
    }
    std::vector<T> ref_output(input_size);
    ref_func(ref_output.data(), input_data.data(), input_size);
    std::vector<T> sc_output(input_size);
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f);
    fptr->call_default(&input_data[0], &sc_output[0]);
    if (!is_bf16) {
        test_utils::compare_data(sc_output, ref_output, 1e-4f, 1e-5f);
    } else {
        float sum = 0.f;
        std::for_each(sc_output.begin(), sc_output.end(),
                [&sum](const T &n) { sum += std::abs(float(n)); });
        EXPECT_TRUE(test_utils::cal_rmse(sc_output, ref_output) / sum < 1e-3f);
    }
}

template <typename T,
        typename dummy = typename std::enable_if<
                std::is_same<typename std::decay<T>::type, float>::value
                || std::is_same<typename std::decay<T>::type, bf16_t>::value>>
static void check_unary_elementwise(const std::string &op_name,
        const sc_dims &input_dims, void (*ref_func)(T *, const T *, size_t)) {
    std::function<void(T *, const T *, size_t)> ref_func_obj = ref_func;
    check_unary_elementwise(op_name, input_dims, ref_func_obj);
}

static void check_qnan_log(const sc_dims &input_dims) {
    sc_graph_t g;
    sc_op_ptr ins;
    ins = g.make_input({graph_tensor::make(
            input_dims, sc_data_format_t(), datatypes::f32)});
    auto op = g.make("log", ins->get_outputs(), {},
            {{"alpha", alpha}, {"beta", thebeta}});
    g.make_output(op->get_outputs());
    graph_driver(g, get_test_ctx());

    auto f = lower_graph(get_test_ctx(), g, {});
    if (verbose) std::cout << f << std::endl;
    const auto input_size = test_utils::product(input_dims);
    std::vector<float> input_data(input_size);
    test_utils::fill_data(&input_data[0], input_size,
            std::numeric_limits<float>::quiet_NaN());
    std::vector<float> sc_output(input_size);
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f);
    fptr->call_default(&input_data[0], &sc_output[0]);
    for (auto &it : sc_output) {
        EXPECT_TRUE(std::isnan(it));
    }
}

static std::vector<sc_dims> test_shapes
        = {{16, 63}, {2, 8, 4}, {4, 16, 256, 1024}};
TEST(GCCore_CPU_unary_elementwise_test, TestAbsOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>("abs", shape, ref_abs);
        check_unary_elementwise<bf16_t>("abs", shape, ref_abs);
    }
}

template <typename T>
void ref_elu_func(T *out, const T *in, size_t size) {
    auto func = std::bind(ref_elu<T>, std::placeholders::_1,
            std::placeholders::_2, std::placeholders::_3, alpha);
    func(out, in, size);
}
TEST(GCCore_CPU_unary_elementwise_test, TestEluOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>("elu", shape, ref_elu_func);
        check_unary_elementwise<bf16_t>("elu", shape, ref_elu_func);
    }
}

template <typename T>
void ref_hardsigmoid_func(T *out, const T *in, size_t size) {
    auto func = std::bind(ref_hardsigmoid<T>, std::placeholders::_1,
            std::placeholders::_2, std::placeholders::_3, alpha, thebeta);
    func(out, in, size);
}
TEST(GCCore_CPU_unary_elementwise_test, TestHardSigmoidOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>(
                "hardsigmoid", shape, ref_hardsigmoid_func);
        check_unary_elementwise<bf16_t>(
                "hardsigmoid", shape, ref_hardsigmoid_func);
    }
}

template <typename T>
void ref_hardswish_func(T *out, const T *in, size_t size) {
    auto func = std::bind(ref_hardswish<T>, std::placeholders::_1,
            std::placeholders::_2, std::placeholders::_3, alpha, thebeta);
    func(out, in, size);
}
TEST(GCCore_CPU_unary_elementwise_test, TestHardSwishOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>("hardswish", shape, ref_hardswish_func);
        check_unary_elementwise<bf16_t>("hardswish", shape, ref_hardswish_func);
    }
}

TEST(GCCore_CPU_unary_elementwise_test, TestLogOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>("log", shape, ref_log);
        check_unary_elementwise<bf16_t>("log", shape, ref_log);
        check_qnan_log(shape);
    }
}

TEST(GCCore_CPU_unary_elementwise_test, TestMishOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>("mish", shape, ref_mish);
        check_unary_elementwise<bf16_t>("mish", shape, ref_mish);
    }
}

template <typename T>
void ref_soft_plus_func(T *out, const T *in, size_t size) {
    auto func = std::bind(ref_soft_plus<T>, std::placeholders::_1,
            std::placeholders::_2, std::placeholders::_3, thebeta);
    func(out, in, size);
}
TEST(GCCore_CPU_unary_elementwise_test, TestSoftPlusOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>("soft_plus", shape, ref_soft_plus_func);
        check_unary_elementwise<bf16_t>("soft_plus", shape, ref_soft_plus_func);
    }
}

TEST(GCCore_CPU_unary_elementwise_test, TestSquareOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>("square", shape, ref_square);
        check_unary_elementwise<bf16_t>("square", shape, ref_square);
    }
}

template <typename T>
void ref_swish_func(T *out, const T *in, size_t size) {
    auto func = std::bind(ref_swish<T>, std::placeholders::_1,
            std::placeholders::_2, std::placeholders::_3, alpha);
    func(out, in, size);
}
TEST(GCCore_CPU_unary_elementwise_test, TestSwishOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>("swish", shape, ref_swish_func);
        check_unary_elementwise<bf16_t>("swish", shape, ref_swish_func);
    }
}

TEST(GCCore_CPU_unary_elementwise_test, TestTanhOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>("tanh", shape, ref_tanh);
        check_unary_elementwise<bf16_t>("tanh", shape, ref_tanh);
    }
}

TEST(GCCore_CPU_unary_elementwise_test, TestErfOp) {
    REQUIRE_AVX2();
    for (auto &shape : test_shapes) {
        check_unary_elementwise<float>("erf", shape, ref_erf);
        check_unary_elementwise<bf16_t>("erf", shape, ref_erf);
    }
}

TEST(GCCore_CPU_unary_elementwise_test, TestPowOp) {
    REQUIRE_AVX2(); // SSE no vgatherdps
    BUILTIN_REQUIRE_AVX512(); // AVX2 accuracy fail (rsqrt)
    std::vector<float> beta_candidates
            = {thebeta, 0.f, 1.f, 2.f, 3.f, 0.5f, -0.5f, -1.f};
    auto old_beta = thebeta;
    for (auto cur_beta : beta_candidates) {
        thebeta = cur_beta;
        for (auto &shape : test_shapes) {
            std::function<void(float *, const float *, size_t)> f32_func(
                    std::bind(ref_pow<float>, std::placeholders::_1,
                            std::placeholders::_2, std::placeholders::_3,
                            cur_beta));
            std::function<void(bf16_t *, const bf16_t *, size_t)> bf16_func(
                    std::bind(ref_pow<bf16_t>, std::placeholders::_1,
                            std::placeholders::_2, std::placeholders::_3,
                            bf16_t(cur_beta)));
            check_unary_elementwise<float>("pow", shape, f32_func);
            check_unary_elementwise<bf16_t>("pow", shape, bf16_func);
        }
    }
    thebeta = old_beta;
}
