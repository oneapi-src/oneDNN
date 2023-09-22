
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
#include <iostream>
#include "context.hpp"
#include "reference/conv_ref.hpp"
#include "reference/gemm_ref.hpp"
#include "gtest/gtest.h"
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/jit/jit.hpp>
#include <test_utils.hpp>
#include <util/any_map.hpp>
#include <util/utils.hpp>
// #define DO_PERF_IN_UT
#ifdef DO_PERF_IN_UT
#include <tuner/time_evaluator.hpp>
#endif

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;
static bool verbose = false;
static bool ut_use_random_data = true;

static sc_data_format_kind_t ABCcbc {0, 1, 2, 2, 1, 2};
static sc_data_format_t ABCcb4c(int c, int b) {
    return sc_data_format_t(ABCcbc, {c, b, 4});
}
static sc_data_format_kind_t ABCDcdc {0, 1, 2, 3, 2, 3, 2};
static sc_data_format_t ABCDcd2c(int c, int d) {
    return sc_data_format_t(ABCDcdc, {c, d, 2});
}
static sc_data_format_kind_t ABDC {0, 1, 3, 2};
static sc_data_format_kind_t ABCD {0, 1, 2, 3};
static sc_data_format_kind_t ABDCcdc {0, 1, 3, 2, 2, 3, 2};
static sc_data_format_t ABDCcd2c(int c, int d) {
    return sc_data_format_t(ABDCcdc, {c, d, 2});
}
static sc_data_format_kind_t ABCbabc {0, 1, 2, 1, 0, 1, 2};
static sc_data_format_t ABCba2bc(int a, int b, int c) {
    return sc_data_format_t(ABCbabc, {b, a, 2, c});
}
static sc_data_format_kind_t ABCabc {0, 1, 2, 0, 1, 2};
template <typename T>
static void compute_elementwise_add(T *src_m1, T *src_m2, sc_dims dims,
        T *dst_m = nullptr, bool inplace = true, char op = '+') {
    int64_t ranges = 1;
    T *ref;
    for (auto d : dims)
        ranges *= d;
    if (inplace)
        ref = src_m1;
    else
        ref = dst_m;

    utils::parallel_for(
            0, ranges, 1, [&](int64_t i) { ref[i] = src_m1[i] + src_m2[i]; });
}

template <typename T>
void compute_reorder_op(T &in, T &out, sc_dims in_dims, sc_data_format_t infmt,
        sc_data_format_t outfmt, sc_data_type_t dtype,
        bool use_input_loop = false) {
    sc_graph_t g;
    auto ins = g.make_input({graph_tensor::make(in_dims, infmt, dtype)});
    auto op = g.make("reorder", ins->get_outputs(), {},
            {{"out_format", outfmt}, {"internal", true},
                    {"use_input_loop", use_input_loop}});
    g.make_output(op->get_outputs());
    g.attrs_[sc_graph_t::attr_key_t::is_input_plain] = false;
    g.attrs_[sc_graph_t::attr_key_t::is_output_plain] = false;

    graph_driver(g, get_test_ctx());

    std::vector<sc_op_ptr> args;
    auto outputs = g.get_output_ops();
    args.insert(args.end(), outputs.begin(), outputs.end());
    auto inputs = g.get_input_ops();
    args.insert(args.end(), inputs.begin(), inputs.end());

    auto f = lower_graph(get_test_ctx(), g, {});

    std::vector<generic_val> generic_args = {
            &out[0],
            &in[0],
    };
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);
    fptr->call_generic_default(generic_args.data());
}

void test_simple_instructions(
        int in_N, int in_K, int N, int K, int k, int n, int k2) {
    ir_builder_t builder;
    _function_(datatypes::void_t, aaa,
            _arg_("in0", datatypes::u8, {in_K, in_N}),
            _arg_("out0", datatypes::u8, {N, K, k / k2, n, k2})) {
        _bind_(in0, out0);
        // using input for loop
        _for_(i, 0, in_K, 4) {
            _for_(j, 0, in_N, 16) {
                _var_(zmm0, sc_data_type_t::u8(16));
                _var_(zmm1, sc_data_type_t::u8(16));
                _var_(zmm2, sc_data_type_t::u8(16));
                _var_(zmm3, sc_data_type_t::u8(16));
                _var_(zmmx, sc_data_type_t::u8(16));

                zmm0 = in0[span_t({i + 0, j}, 16)];
                zmm1 = in0[span_t({i + 1, j}, 16)];
                zmm2 = in0[span_t({i + 2, j}, 16)];
                zmm3 = in0[span_t({i + 3, j}, 16)];

                any_map_t reinterpret_attr;
                reinterpret_attr[intrin_attr::out_dtype]
                        = sc_data_type_t::u8(16);

                zmmx = make_unpack_low(zmm0, zmm2, 8);
                zmm2 = make_unpack_high(zmm0, zmm2, 8);
                zmm0 = zmmx;
                zmmx = make_unpack_low(zmm1, zmm3, 8);
                zmm3 = make_unpack_high(zmm1, zmm3, 8);
                zmm1 = zmmx;
                zmmx = make_unpack_low(zmm0, zmm1, 8);
                zmm1 = make_unpack_high(zmm0, zmm1, 8);
                zmm0 = zmmx;
                zmmx = make_unpack_low(zmm2, zmm3, 8);
                zmm3 = make_unpack_high(zmm2, zmm3, 8);
                zmm2 = zmmx;
                out0[span_t(
                        {j / n, i / k, (i % k) / k2, (j % n + 0), (i % k) % k2},
                        16)]
                        = make_reinterpret(zmm0, sc_data_type_t::u8(16));
                out0[span_t(
                        {j / n, i / k, (i % k) / k2, (j % n + 4), (i % k) % k2},
                        16)]
                        = make_reinterpret(zmm1, sc_data_type_t::u8(16));
                out0[span_t(
                        {j / n, i / k, (i % k) / k2, (j % n + 8), (i % k) % k2},
                        16)]
                        = make_reinterpret(zmm2, sc_data_type_t::u8(16));
                out0[span_t({j / n, i / k, (i % k) / k2, (j % n + 12),
                                    (i % k) % k2},
                        16)]
                        = make_reinterpret(zmm3, sc_data_type_t::u8(16));
            }
        }
    }
    context_ptr ctx = std::make_shared<context_t>(*get_default_context());
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(
            ir_module_t::from_entry_func(get_default_context(), aaa));

    ASSERT_TRUE(fptr);

    auto in = alloc_array<uint8_t>(in_N * in_K, init_action::INIT_RANDOM);

    if (!ut_use_random_data) {
        for (size_t i = 0; i < in.size(); i++) {
            in[i] = uint8_t(i % 256);
        }
    }

    if (verbose) {
        for (size_t i = 0; i < in.size(); i++) {
            std::cout << int(in[i]) << " ";
            if ((i + 1) % in_K == 0) std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }

    auto out = alloc_array<uint8_t>(in_N * in_K, init_action::INIT_NOOP);
    auto ref_out = alloc_array<uint8_t>(in_N * in_K, init_action::INIT_NOOP);
    auto reorder_op_out
            = alloc_array<uint8_t>(in_N * in_K, init_action::INIT_NOOP);

    fptr->call_default(in.data(), out.data());

    ref_out = KN2NKkn(in, N, K, k, n, in_K, in_N, k2);
    if (verbose) {
        for (size_t i = 0; i < ref_out.size(); i++) {
            std::cout << int(ref_out[i]) << " ";
            if ((i + 1) % in_K == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
        for (size_t i = 0; i < out.size(); i++) {
            std::cout << int(out[i]) << " ";
            if ((i + 1) % in_K == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    test_utils::compare_data(out.data(), ref_out.data(), in_N * in_K);

    compute_reorder_op(reorder_op_out, in, sc_dims({in_K, in_N}),
            sc_data_format_t::KN(), sc_data_format_t::NKkn4k(k, n),
            sc_data_type_t::u8());

    test_utils::compare_data(reorder_op_out, ref_out);
}

TEST(GCCore_CPU_vnni_reorder_test, TestSimpleInstructions1) {
    REQUIRE_AVX2()
    test_simple_instructions(64, 16, 4, 1, 16, 16, 4);
}

TEST(GCCore_CPU_vnni_reorder_test, TestSimpleInstructions2) {
    REQUIRE_AVX2()
    test_simple_instructions(1024, 512, 8, 4, 128, 128, 4);
}

bool is_vnni_reorder_triggered(ir_module_ptr ir) {
    // vnni fast reorder should contain unpack intrinsic in IR.
    // Some of the unit tests in this file is NOT vnni reorder.
    std::stringstream ss;
    ss << ir;
    return ss.str().find("_vnni_reorder_") != std::string::npos;
}

template <typename T>
static void check(const sc_dims &inputdims, const sc_data_format_t &infmt,
        const sc_data_format_t &outfmt, sc_data_type_t dtype,
        bool is_vnni_reorder, bool use_input_loop,
        const std::function<test_buffer<T>(test_buffer<T> &)> &ref_func,
        const std::function<sc_op_ptr(sc_graph_t &, graph_tensor_ptr)>
                &graph_func
        = nullptr,
        std::function<void(std::vector<generic_val> &)> args_func = nullptr,
        bool fuse = true) {
    sc_graph_t g;
    auto ins = g.make_input({graph_tensor::make(inputdims, infmt, dtype)});
    auto op = g.make("reorder", ins->get_outputs(), {},
            {{"out_format", outfmt}, {"internal", true},
                    {"use_input_loop", use_input_loop}});
    if (graph_func) { op = graph_func(g, op->get_outputs()[0]); }
    auto g_out = g.make_output(op->get_outputs());
    g.attrs_[sc_graph_t::attr_key_t::is_input_plain] = false;
    g.attrs_[sc_graph_t::attr_key_t::is_output_plain] = false;
    if (!fuse) { g.attrs_["temp.disable_graph_fusion"] = 1; }
    size_t input_total = test_utils::product(
            ins->get_outputs()[0]->details_.get_blocking_dims());
    size_t output_total = test_utils::product(
            op->get_outputs()[0]->details_.get_blocking_dims());
    auto x = op->get_outputs()[0]->details_.get_blocking_dims();
    graph_driver(g, get_test_ctx());
    auto f = lower_graph(get_test_ctx(), g, {});

    auto input = alloc_array<T>(input_total, init_action::INIT_RANDOM);
    auto sc_output = alloc_array<T>(output_total, init_action::INIT_NOOP);
    auto ref_output = alloc_array<T>(output_total, init_action::INIT_NOOP);

    if (!ut_use_random_data) {
        for (size_t i = 0; i < input.size(); i++) {
            input[i] = T(i % 256);
        }
    }

    std::vector<generic_val> args;
    if (args_func) { args_func(args); }
    args.emplace_back(input.data());
    args.emplace_back(sc_output.data());
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);

    auto exec = [&]() { fptr->call_generic_default(args.data()); };

#ifdef DO_PERF_IN_UT
    const int warm_up = 5000;
    const int repeat = 3;
    const int loop = 10000;
    double cost1 = 1e12;
    for (int r = 0; r < repeat; r++) {
        double cost = 0.f;
        for (int t = 0; t < loop + warm_up; t++) {
            auto time = evaluate_time(exec);
            if (t >= warm_up) cost += time;
        }
        cost1 = std::min(cost, cost1);
    }
    printf("\ncost %f ms\n", cost1 / loop);
#else
    exec();
#endif
    ref_output = ref_func(input);
    if (verbose) {
        std::cout << "input:" << std::endl;
        for (size_t i = 0; i < input.size(); i++) {
            std::cout << int(input[i]) << " ";
            if ((i + 1) % 16 == 0) std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;

        std::cout << "ref_output:" << std::endl;
        for (size_t i = 0; i < ref_output.size(); i++) {
            std::cout << int(ref_output[i]) << " ";
            if ((i + 1) % 16 == 0) std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;

        std::cout << "sc_output:" << std::endl;
        for (size_t i = 0; i < sc_output.size(); i++) {
            std::cout << int(sc_output[i]) << " ";
            if ((i + 1) % 16 == 0) std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
    test_utils::compare_data(sc_output, ref_output);
    EXPECT_EQ(is_vnni_reorder_triggered(f), is_vnni_reorder);
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose1) {
    REQUIRE_AVX2()
    check<uint8_t>({128, 64}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn4k(16 * 4, 16), sc_data_type_t::u8(), true,
            false, [](test_buffer<uint8_t> &input) {
                return NK2NKknk(input, 4, 2, 16, 16, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose2) {
    REQUIRE_AVX2()
    check<uint8_t>({64, 384}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn4k(8 * 4, 32), sc_data_type_t::u8(), true,
            true, [](test_buffer<uint8_t> &input) {
                return NK2NKknk(input, 12, 2, 8, 32, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose3) {
    REQUIRE_AVX2()
    check<uint8_t>({1024, 4096}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn4k(32 * 4, 64), sc_data_type_t::u8(), true,
            false, [](test_buffer<uint8_t> &input) {
                return NK2NKknk(input, 64, 8, 32, 64, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose4) {
    REQUIRE_AVX2()
    check<bf16_t>({128, 64}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn2k(16 * 2, 16), sc_data_type_t::bf16(), true,
            true, [](test_buffer<bf16_t> &input) {
                return NK2NKknk(input, 4, 4, 16, 16, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose5) {
    REQUIRE_AVX2()
    check<bf16_t>({64, 384}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn2k(8 * 2, 32), sc_data_type_t::bf16(), true,
            false, [](test_buffer<bf16_t> &input) {
                return NK2NKknk(input, 12, 4, 8, 32, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose6) {
    REQUIRE_AVX2()
    check<bf16_t>({1024, 4096}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn2k(32 * 2, 64), sc_data_type_t::bf16(), true,
            false, [](test_buffer<bf16_t> &input) {
                return NK2NKknk(input, 64, 16, 32, 64, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose7) {
    REQUIRE_AVX2()
    check<uint8_t>({32, 16}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn4k(4 * 4, 8), sc_data_type_t::u8(), true,
            true, [](test_buffer<uint8_t> &input) {
                return NK2NKknk(input, 2, 2, 4, 8, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose8) {
    REQUIRE_AVX2()
    check<uint8_t>({128, 16, 32}, sc_data_format_t(format_kinds::ABC),
            ABCcb4c(4 * 4, 8), sc_data_type_t::u8(), true, false,
            [](test_buffer<uint8_t> &input) {
                return ABC2ABCcb4c(input, 128, 2, 2, 4, 8, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose9) {
    REQUIRE_AVX2()
    check<bf16_t>({64, 384}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn2k(8 * 2, 32), sc_data_type_t::bf16(), true,
            true, [](test_buffer<bf16_t> &input) {
                return NK2NKknk(input, 12, 4, 8, 32, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose10) {
    REQUIRE_AVX2()
    check<int8_t>({16, 16}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn4k(4 * 4, 16), sc_data_type_t::u8(), true,
            false, [](test_buffer<int8_t> &input) {
                return NK2NKknk(input, 1, 1, 4, 16, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder1) {
    REQUIRE_AVX2()
    check<int8_t>({16, 16}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(4 * 4, 16), sc_data_type_t::s8(), true,
            true, [](test_buffer<int8_t> &input) {
                return KN2NKkn(input, 1, 1, 16, 16, 16, 16, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder2) {
    REQUIRE_AVX2()
    check<uint8_t>({16, 64}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(16, 16), sc_data_type_t::u8(), true, false,
            [](test_buffer<uint8_t> &input) {
                return KN2NKkn(input, 4, 1, 16, 16, 16, 64, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder3) {
    REQUIRE_AVX2()
    check<uint8_t>({64, 64}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(16, 16), sc_data_type_t::u8(), true, true,
            [](test_buffer<uint8_t> &input) {
                return KN2NKkn(input, 4, 4, 16, 16, 64, 64, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder4) {
    REQUIRE_AVX2()
    check<uint8_t>({4, 16}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(4, 16), sc_data_type_t::u8(), true, false,
            [](test_buffer<uint8_t> &input) {
                return KN2NKkn(input, 1, 1, 4, 16, 4, 16, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder5) {
    REQUIRE_AVX2()
    check<uint8_t>({64, 384}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(16, 32), sc_data_type_t::u8(), true, true,
            [](test_buffer<uint8_t> &input) {
                return KN2NKkn(input, 12, 4, 16, 32, 64, 384, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder6) {
    REQUIRE_AVX2()
    check<uint8_t>({1024, 4096}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(128, 64), sc_data_type_t::u8(), true,
            false, [](test_buffer<uint8_t> &input) {
                return KN2NKkn(input, 64, 8, 128, 64, 1024, 4096, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder7) {
    REQUIRE_AVX2()
    check<bf16_t>({4, 8}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn2k(4, 8), sc_data_type_t::bf16(), true, true,
            [](test_buffer<bf16_t> &input) {
                return KN2NKkn(input, 1, 1, 4, 8, 4, 8, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder8) {
    REQUIRE_AVX2()
    check<bf16_t>({16, 16}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn2k(16, 16), sc_data_type_t::bf16(), true,
            false, [](test_buffer<bf16_t> &input) {
                return KN2NKkn(input, 1, 1, 16, 16, 16, 16, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder9) {
    REQUIRE_AVX2()
    check<bf16_t>({64, 384}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn2k(16, 32), sc_data_type_t::bf16(), true,
            true, [](test_buffer<bf16_t> &input) {
                return KN2NKkn(input, 12, 4, 16, 32, 64, 384, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder10) {
    REQUIRE_AVX2()
    check<bf16_t>({1024, 4096}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn2k(64, 64), sc_data_type_t::bf16(), true,
            false, [](test_buffer<bf16_t> &input) {
                return KN2NKkn(input, 64, 16, 64, 64, 1024, 4096, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderPadding1) {
    REQUIRE_AVX2()
    check<uint8_t>({63, 64}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(16, 16), sc_data_type_t::u8(), true, false,
            [](test_buffer<uint8_t> &input) {
                return KN2NKkn(input, 4, 4, 16, 16, 63, 64, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderPadding2) {
    REQUIRE_AVX2()
    check<uint8_t>({479, 1024}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(64, 64), sc_data_type_t::u8(), true, false,
            [](test_buffer<uint8_t> &input) {
                return KN2NKkn(input, 16, 8, 64, 64, 479, 1024, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderPadding3) {
    REQUIRE_AVX2()
    check<bf16_t>({63, 64}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn2k(16, 16), sc_data_type_t::bf16(), true,
            false, [](test_buffer<bf16_t> &input) {
                return KN2NKkn(input, 4, 4, 16, 16, 63, 64, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderPadding4) {
    REQUIRE_AVX2()
    check<bf16_t>({479, 1024}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn2k(64, 64), sc_data_type_t::bf16(), true,
            false, [](test_buffer<bf16_t> &input) {
                return KN2NKkn(input, 16, 8, 64, 64, 479, 1024, 2);
            });
}

// not vnni reorder
TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose11) {
    REQUIRE_AVX2()
    check<uint8_t>({256, 128}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn4k(1 * 4, 8), sc_data_type_t::u8(), false,
            true, [](test_buffer<uint8_t> &input) {
                return NK2NKknk(input, 16, 64, 1, 8, 4);
            });
}

// not vnni reorder
TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose12) {
    REQUIRE_AVX2()
    check<uint8_t>({64, 64, 3, 3}, sc_data_format_t::KCRS(),
            sc_data_format_t::KCRSck4c(32, 32), sc_data_type_t::u8(), false,
            false, [](test_buffer<uint8_t> &input) {
                return KCRS2KCRSckc(input, 2, 2, 3, 3, 8, 32, 4);
            });
} // ABCD->ABCDba4

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder13) {
    REQUIRE_AVX2()
    check<bf16_t>({28, 12, 128, 64}, sc_data_format_t(format_kinds::ACBD),
            ABCDcd2c(64, 64), sc_data_type_t::bf16(), true, true,
            [](test_buffer<bf16_t> &input) {
                // 28 12 2 1 32 64 2
                return ACBD2ABCDcd(
                        input, 28, 12, 2, 1, 64, 64, 28, 128, 12, 64, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder14) {
    REQUIRE_AVX2()
    check<bf16_t>({1, 1, 384, 64}, sc_data_format_t(format_kinds::ACBD),
            ABCDcd2c(64, 64), sc_data_type_t::bf16(), true, false,
            [](test_buffer<bf16_t> &input) {
                return ACBD2ABCDcd(input, 1, 1, 6, 1, 64, 64, 1, 384, 1, 64, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder15) {
    REQUIRE_AVX2()
    check<bf16_t>({1, 16, 384, 64}, sc_data_format_t(format_kinds::ACBD),
            ABCDcd2c(64, 64), sc_data_type_t::bf16(), true, true,
            [](test_buffer<bf16_t> &input) {
                return ACBD2ABCDcd(
                        input, 1, 16, 6, 1, 64, 64, 1, 384, 16, 64, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder16) {
    REQUIRE_AVX2()
    check<bf16_t>({1, 1, 16, 16}, sc_data_format_t(ABCD), ABDCcd2c(16, 16),
            sc_data_type_t::bf16(), true, false,
            [](test_buffer<bf16_t> &input) {
                return ABCD2ABDCcd(input, 1, 1, 1, 1, 16, 16, 1, 1, 16, 16, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose17) {
    REQUIRE_AVX2()
    check<bf16_t>({1, 1, 64, 384}, sc_data_format_t(ABDC), ABCDcd2c(64, 64),
            sc_data_type_t::bf16(), true, true, [](test_buffer<bf16_t> &input) {
                return ABDC2ABCDcd(input, 1, 1, 1, 6, 64, 64, 1, 1, 384, 64, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder18) {
    REQUIRE_AVX2()
    check<uint8_t>({768, 2304}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(256, 256), sc_data_type_t::s8(), true,
            false, [](test_buffer<uint8_t> &input) {
                return KN2NKkn(input, 9, 3, 256, 256, 768, 2304, 4);
            });
}
TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder19) {
    REQUIRE_AVX2()
    check<uint8_t>({768, 2304}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(256, 18), sc_data_type_t::s8(), false,
            false, [](test_buffer<uint8_t> &input) {
                return KN2NKkn(input, 128, 3, 256, 18, 768, 2304, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder20) {
    REQUIRE_AVX2()
    check<bf16_t>({13, 512}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn2k(18, 32), sc_data_type_t::bf16(), false,
            false, [](test_buffer<bf16_t> &input) {
                return KN2NKkn(input, 16, 1, 18, 32, 13, 512, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder21) {
    REQUIRE_AVX2()
    check<bf16_t>({1, 12, 15, 64}, format_kinds::ACBD, ABDCcd2c(64, 64),
            sc_data_type_t::bf16(), true, false,
            [](test_buffer<bf16_t> &input) {
                return ACBD2ABDCcd(
                        input, 1, 12, 1, 1, 64, 64, 1, 15, 12, 64, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder22) {
    REQUIRE_AVX2()
    check<uint8_t>({64, 15}, sc_data_format_t::KN(),
            sc_data_format_t::NKkn4k(16, 32), sc_data_type_t::u8(), true, false,
            [](test_buffer<uint8_t> &input) {
                return KN2NKkn(input, 1, 4, 16, 32, 64, 15, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder23) {
    REQUIRE_AVX2()
    check<uint8_t>({64, 64, 1, 1}, sc_data_format_t::KCRS(),
            sc_data_format_t::KCRSck4c(32, 32), sc_data_type_t::u8(), true,
            false, [](test_buffer<uint8_t> &input) {
                return KCRS2KCRSckc(input, 2, 2, 1, 1, 8, 32, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder25) {
    REQUIRE_AVX2()
    check<uint8_t>({64, 64, 2, 1}, sc_data_format_t::KCRS(),
            sc_data_format_t::KCRSck4c(32, 32), sc_data_type_t::u8(), false,
            false, [](test_buffer<uint8_t> &input) {
                return KCRS2KCRSckc(input, 2, 2, 2, 1, 8, 32, 4);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder26) {
    REQUIRE_AVX2()
    check<bf16_t>({16, 128, 1}, sc_data_format_t(format_kinds::ABC),
            ABCba2bc(16, 16, 1), sc_data_type_t::bf16(), true, true,
            [](test_buffer<bf16_t> &input) {
                return ABC2ABCbac(input, 1, 8, 1, 16, 16, 1, 16, 128, 1, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder27) {
    REQUIRE_AVX2()
    check<bf16_t>({16, 128, 8}, sc_data_format_t(format_kinds::ABC),
            ABCba2bc(16, 16, 8), sc_data_type_t::bf16(), false, true,
            [](test_buffer<bf16_t> &input) {
                return ABC2ABCbac(input, 1, 8, 1, 16, 16, 8, 16, 128, 8, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder28) {
    REQUIRE_AVX2()
    check<bf16_t>({16, 128, 1}, sc_data_format_t(format_kinds::ABC),
            sc_data_format_t(ABCabc, {16, 2, 1}), sc_data_type_t::bf16(), true,
            true, [](test_buffer<bf16_t> &input) {
                return ABC2ABCabc(input, 1, 64, 1, 16, 2, 1, 16, 128, 1);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder29) {
    REQUIRE_AVX2()
    check<bf16_t>({16, 128, 16}, sc_data_format_t(format_kinds::ABC),
            sc_data_format_t(ABCabc, {16, 4, 2}), sc_data_type_t::bf16(), true,
            true, [](test_buffer<bf16_t> &input) {
                return ABC2ABCabc(input, 1, 32, 8, 16, 4, 2, 16, 128, 16);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorder30) {
    REQUIRE_AVX2()
    check<bf16_t>({16, 128, 16}, sc_data_format_t(format_kinds::ABC),
            sc_data_format_t(ABCabc, {16, 4, 4}), sc_data_type_t::bf16(), false,
            true, [](test_buffer<bf16_t> &input) {
                return ABC2ABCabc(input, 1, 32, 4, 16, 4, 4, 16, 128, 16);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose22) {
    REQUIRE_AVX2()
    check<bf16_t>({1, 12, 64, 15},
            sc_data_format_t(sc_data_format_kind_t {0, 3, 1, 2}),
            ABDCcd2c(64, 16), sc_data_type_t::bf16(), true, false,
            [](test_buffer<bf16_t> &input) {
                return ADBC2ABDCcd(
                        input, 1, 12, 1, 1, 64, 16, 1, 15, 12, 64, 2);
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestVNNIReorderTranspose23) {
    REQUIRE_AVX2()
    check<bf16_t>({1, 12, 64, 16},
            sc_data_format_t(sc_data_format_kind_t {0, 3, 1, 2}),
            ABDCcd2c(64, 16), sc_data_type_t::bf16(), true, false,
            [](test_buffer<bf16_t> &input) {
                return ADBC2ABDCcd(
                        input, 1, 12, 1, 1, 64, 16, 1, 16, 12, 64, 2);
            });
}

template <typename T>
static void check_fuse_add(const sc_dims &inputdims,
        const sc_data_format_t &infmt, const sc_data_format_t &outfmt,
        sc_data_type_t dtype, bool is_vnni_reorder,
        std::function<test_buffer<T>(test_buffer<T> &, test_buffer<T> &)>
                ref_func) {
    auto eltadd = alloc_array<T>(test_utils::product(inputdims));
    check<T>(
            inputdims, infmt, outfmt, dtype, is_vnni_reorder, false,
            [&ref_func, &eltadd](
                    test_buffer<T> &input) { return ref_func(input, eltadd); },
            [&inputdims, &outfmt](sc_graph_t &g, graph_tensor_ptr op) {
                auto extra_in = g.make_input({graph_tensor::make(
                        inputdims, outfmt, sc_data_traits_t<T>().type())});
                return g.make("add",
                        {std::move(op), extra_in->get_outputs()[0]}, {}, {});
            },
            [&eltadd](std::vector<generic_val> &args) {
                args.emplace_back(eltadd.data());
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestUInt8VNNIReorderTransposeFuseAdd1) {
    REQUIRE_AVX2()
    check_fuse_add<uint8_t>({128, 64}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn4k(16 * 4, 16), sc_data_type_t::u8(), true,
            [](test_buffer<uint8_t> &input, test_buffer<uint8_t> &eltadd) {
                auto ref_output = NK2NKknk(input, 4, 2, 16, 16, 4);
                compute_elementwise_add<uint8_t>(
                        &ref_output[0], &eltadd[0], {4, 2, 16, 16, 4});
                return ref_output;
            });
}

TEST(GCCore_CPU_vnni_reorder_test, TestUInt8VNNIReorderTransposeFuseAdd2) {
    REQUIRE_AVX2()
    check_fuse_add<uint8_t>({64, 384}, sc_data_format_t::NK(),
            sc_data_format_t::NKkn4k(8 * 2, 32), sc_data_type_t::u8(), true,
            [](test_buffer<uint8_t> &input, test_buffer<uint8_t> &eltadd) {
                auto ref_output = NK2NKknk(input, 12, 2, 8, 32, 4);
                compute_elementwise_add<uint8_t>(
                        &ref_output[0], &eltadd[0], {12, 2, 8, 32, 4});
                return ref_output;
            });
}
