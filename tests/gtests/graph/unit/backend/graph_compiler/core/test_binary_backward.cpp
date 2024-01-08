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

#include <iostream>
#include <numeric>
#include <vector>

#include "context.hpp"
#include "reference/act_ref.hpp"
#include "test_utils.hpp"
#include "util/bf16.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fusion_anchor.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/trait/may_broadcast.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/jit/jit.hpp>
#include <util/any_map.hpp>
#include <util/math_utils.hpp>

using namespace dnnl::impl::graph::gc;

template <class T>
static void cmp_result(bool is_bf16, const std::vector<T> &sc_out_1,
        const std::vector<T> &sc_out_2, const std::vector<T> ref_output1,
        const std::vector<T> ref_output2, const size_t input_size,
        const float rtol = 1e-4f, const float atol = 1e-5,
        bool fused_test = false) {
    if (!is_bf16) {
        test_utils::compare_data(sc_out_1, ref_output1, rtol, atol);
        if (!fused_test) {
            test_utils::compare_data(sc_out_2, ref_output2, rtol, atol);
        }
    } else {
        auto test_f = [&rtol](const std::vector<T> &dst_inp,
                              const std::vector<T> &ref_inp) {
            float sum = 0.f;
            auto ff = [&sum](std::vector<T> sc_output) {
                std::for_each(sc_output.begin(), sc_output.end(),
                        [&sum](const T &n) { sum += std::abs(float(n)); });
            };

            auto rmse = test_utils::cal_rmse(dst_inp, ref_inp);
            ff(dst_inp);
            EXPECT_TRUE(((sum != 0.f) ? rmse / sum : rmse) < rtol);
        };
        test_f(sc_out_1, ref_output1);
        if (!fused_test) { test_f(sc_out_2, ref_output2); }
    }
}

template <class T>
static void prepare_input_value(
        std::vector<std::vector<T>> &inp_data, const size_t input_size) {
    auto &input_data1 = inp_data[0];
    auto &input_data2 = inp_data[1];
    auto &input_data3 = inp_data[2];
    auto &elt_add = inp_data[3];

    test_utils::fill_data(&input_data1[0], input_size);
    test_utils::fill_data(&input_data2[0], input_size);
    test_utils::fill_data(&input_data3[0], input_size);
    test_utils::fill_data(&elt_add[0], input_size);
}

template <class T>
static void test_gc(const sc_dims &input_dims, const sc_data_type_t &dtype,
        const std::string &op_name, const sc_data_format_t &infmt,
        std::vector<std::vector<T>> &inp_data, std::vector<T> &sc_output1,
        std::vector<T> &sc_output2, const any_map_t &attrs,
        const std::function<sc_op_ptr(sc_graph_t &, graph_tensor_ptr,
                const sc_dims &, const sc_data_format_t &,
                const sc_data_type_t &)> &graph_func
        = nullptr,
        const std::function<void(std::vector<generic_val> &, std::vector<T> &)>
                &args_func
        = nullptr) {
    auto &input_data1 = inp_data[0];
    auto &input_data2 = inp_data[1];
    auto &input_data3 = inp_data[2];
    auto &elt_add = inp_data[3];

    sc_graph_t g;
    sc_op_ptr ins;
    ins = g.make_input({graph_tensor::make(input_dims,
                                graph_func ? infmt : sc_data_format_t(), dtype),
            graph_tensor::make(
                    input_dims, graph_func ? infmt : sc_data_format_t(), dtype),
            graph_tensor::make(input_dims,
                    graph_func ? infmt : sc_data_format_t(), dtype)});
    auto op = g.make(op_name, ins->get_outputs(), {}, attrs);
    if (graph_func) {
        op = graph_func(g, op->get_outputs()[0], input_dims, infmt, dtype);
    }
    g.make_output(op->get_outputs());
    g.attrs_[sc_graph_t::attr_key_t::is_input_plain] = false;
    g.attrs_[sc_graph_t::attr_key_t::is_output_plain] = false;
    graph_driver(g, get_test_ctx());
    auto f = lower_graph(get_test_ctx(), g, {});
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f);

    std::vector<generic_val> args;
    if (args_func) { args_func(args, elt_add); }
    args.emplace_back(input_data1.data());
    args.emplace_back(input_data2.data());
    args.emplace_back(input_data3.data());
    args.emplace_back(sc_output1.data());
    if (!args_func) { args.emplace_back(sc_output2.data()); }

    fptr->call_generic_default(args.data());
}

template <class T>
static void prepare_ref_value(void (*ref_func)(T *, T *, const T *, const T *,
                                      const T *, size_t, any_map_t &attrs),
        std::vector<T> &ref_output1, std::vector<T> &ref_output2,
        std::vector<std::vector<T>> &inp_data, any_map_t &attrs,
        const size_t input_size, bool need_fused = false) {
    auto &input_data1 = inp_data[0];
    auto &input_data2 = inp_data[1];
    auto &input_data3 = inp_data[2];
    auto &elt_add = inp_data[3];
    ref_func(ref_output1.data(), ref_output2.data(), input_data1.data(),
            input_data2.data(), input_data3.data(), input_size, attrs);
    if (need_fused) {
        std::transform(ref_output1.begin(), ref_output1.end(), elt_add.begin(),
                ref_output1.begin(), std::plus<T>());
    }
}

template <typename T,
        typename dummy = typename std::enable_if<
                std::is_same<typename std::decay<T>::type, float>::value
                || std::is_same<typename std::decay<T>::type, bf16_t>::value>>
static void check_binary_backward(const std::string &op_name,
        const sc_dims &input_dims,
        void (*ref_func)(T *, T *, const T *, const T *, const T *, size_t,
                any_map_t &attrs),
        const sc_data_format_t &inp_format = format_kinds::A,
        const std::function<sc_op_ptr(sc_graph_t &, graph_tensor_ptr,
                const sc_dims &, const sc_data_format_t &,
                const sc_data_type_t &)> &graph_func
        = nullptr,
        const std::function<void(std::vector<generic_val> &, std::vector<T> &)>
                &args_func
        = nullptr) {
    bool is_bf16 = std::is_same<typename std::decay<T>::type, bf16_t>::value;
    if (is_bf16
            && !::dnnl::impl::graph::gc::get_default_context()
                        ->machine_.cpu_flags_.fAVX512F) {
        return;
    }

    // define used var
    any_map_t attrs;
    float rtol = 1e-4f, atol = 1e-5f;
    auto dtype = is_bf16 ? datatypes::bf16 : datatypes::f32;
    const auto input_size = test_utils::product(input_dims);
    std::vector<T> input_data1(input_size), input_data2(input_size),
            input_data3(input_size), elt_add(input_size);
    std::vector<T> sc_out_1(input_size), sc_out_2(input_size);
    std::vector<T> ref_output1(input_size), ref_output2(input_size);

    // prepare input value
    std::vector<std::vector<T>> inp_data
            = {input_data1, input_data2, input_data3, elt_add};
    prepare_input_value(inp_data, input_size);

    // test gc
    test_gc<T>(input_dims, dtype, op_name, inp_format, inp_data, sc_out_1,
            sc_out_2, attrs, graph_func, args_func);

    // prepare ref result
    prepare_ref_value<T>(ref_func, ref_output1, ref_output2, inp_data, attrs,
            input_size, graph_func != nullptr);

    // compare test result
    cmp_result<T>(is_bf16, sc_out_1, sc_out_2, ref_output1, ref_output2,
            input_size, rtol, atol, graph_func != nullptr);
}

template <class T>
static void check_fused(const std::string &op_name, const sc_dims &test_shapes,
        void (*ref_func)(T *, T *, const T *, const T *, const T *, size_t,
                any_map_t &attrs),
        const sc_data_format_t &inp_format) {
    check_binary_backward<T>(
            op_name, test_shapes, ref_func, inp_format,
            [](sc_graph_t &g, graph_tensor_ptr op, const sc_dims &inputdims,
                    const sc_data_format_t &outfmt,
                    const sc_data_type_t &dtype) {
                auto extra_in = g.make_input({graph_tensor::make(
                                                     inputdims, outfmt, dtype)})
                                        ->get_outputs()[0];
                return g.make("add", {std::move(op), extra_in}, {}, {});
            },
            [](std::vector<generic_val> &args, std::vector<T> &eltadd) {
                args.emplace_back(eltadd.data());
            });
}

static std::vector<sc_dims> test_shapes
        = {{16, 63}, {2, 8, 4}, {4, 16, 256, 1024}};
static std::vector<sc_data_format_t> test_inp_formats
        = {format_kinds::AB, format_kinds::ABC, format_kinds::ABCD};
template <class T>
static void make_binary_backward_test(const std::string &op_name,
        void (*ref_func)(T *, T *, const T *, const T *, const T *, size_t,
                any_map_t &attrs)) {
    for (size_t idx = 0; idx < test_shapes.size(); idx++) {
        check_binary_backward<T>(op_name, test_shapes[idx], ref_func);
        check_fused<T>(
                op_name, test_shapes[idx], ref_func, test_inp_formats[idx]);
    }
}
TEST(GCCore_CPU_binary_backward_test, TestPreluBwdOp) {
    REQUIRE_AVX2();
    make_binary_backward_test<float>("prelu_bwd", ref_prelu_bwd);
    make_binary_backward_test<bf16_t>("prelu_bwd", ref_prelu_bwd);
}
