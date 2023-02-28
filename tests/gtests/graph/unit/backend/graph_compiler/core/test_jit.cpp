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

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/graph.hpp>
#if SC_BUILTIN_JIT_ENABLED
#include <compiler/jit/xbyak/xbyak_jit_engine.hpp>
#include <compiler/jit/xbyak/xbyak_jit_module.hpp>
#endif
#if SC_CFAKE_JIT_ENABLED
#include <compiler/jit/cfake/cfake_jit.hpp>
#endif
#if defined(SC_LLVM_BACKEND)
#include <compiler/jit/llvm/llvm_jit.hpp>
#endif
#include <cfenv>
#include <cmath>
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/builtin.hpp>
#include <runtime/config.hpp>

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

static bool path_exists(const std::string &path) {
    return !std::ifstream(path).fail();
}

static const char *get_engine_name(std::unique_ptr<jit_engine_t> &engine) {
    auto &engineobj = *engine;
    return typeid(engineobj).name();
}

static std::vector<std::unique_ptr<jit_engine_t>> get_engines() {
    std::vector<std::unique_ptr<jit_engine_t>> ret;
#if SC_CFAKE_JIT_ENABLED
    ret.emplace_back(utils::make_unique<cfake_jit>());
#endif
#if defined(SC_LLVM_BACKEND)
    ret.emplace_back(utils::make_unique<llvm_jit>());
#endif
#if SC_BUILTIN_JIT_ENABLED
    if (get_default_context()->machine_.cpu_flags_.fAVX512F) {
        ret.emplace_back(utils::make_unique<sc_xbyak::xbyak_jit_engine>());
    }
#endif
    return ret;
}

TEST(GCCore_jit_cpp, TestJIT) {
    ir_builder_t builder;
    auto m = std::make_shared<ir_module_t>(get_default_context());
    _global_tensor_(m, gtsr, datatypes::f32, 10);
    _global_var_(m, func_ptr, datatypes::pointer, expr());
    _function_(datatypes::s32, proto_func, _arg_("ii", datatypes::s32)) {}
    func_ptr->attr()["prototype"] = proto_func;
    _function_(datatypes::s32, aaa, _arg_("ii", datatypes::s32),
            _arg_("jj", datatypes::s32), _arg_("buf", datatypes::s32, {100})) {
        _bind_(ii, jj, buf);
        _for_(i, 0, 10) {
            buf[i] = ii + jj + builder::make_cast(datatypes::s32, i);
            _if_(i < 2) {
                builtin::print_str("hahah");
                builtin::print_int(buf[i]);
            }
        }
        gtsr[0] = 1.0f;
        gtsr[1] = builder::make_reinterpret(jj, datatypes::f32);
        _return_(make_expr<call_node>(func_ptr, std::vector<expr> {122}));
    }
    m->add_func({aaa});

    auto engines = get_engines();
    std::vector<std::string> temp_files;
    auto the_call_back = +[](int32_t c) { return c + 1; };
    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        auto mod = engine->make_jit_module(m, false);
        auto fptr = mod->get_function("aaa");

        void **gfunc_ptr = (void **)mod->get_address_of_symbol("func_ptr");
        ASSERT_TRUE(gfunc_ptr);
        *gfunc_ptr = (void *)the_call_back;
        EXPECT_TRUE(fptr);
        int buf[100];
        testing::internal::CaptureStdout();
        EXPECT_EQ(fptr->call<int>(1, 23, buf), 123);
        float *gtsrptr = (float *)mod->get_address_of_symbol("gtsr");
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_EQ(output, "hahah24\nhahah25\n");
        EXPECT_EQ(buf[4], 28);
        ASSERT_TRUE(gtsrptr);
        EXPECT_EQ(gtsrptr[0], 1.0f);
        // now test reinterpret
        union {
            int v;
            float v2;
        } reint;
        reint.v = 23;
        EXPECT_EQ(gtsrptr[1], reint.v2);
        // reinterpret test done
        temp_files = fptr->get_module()->get_temp_filenames();
        for (const auto &file : temp_files) {
            EXPECT_TRUE(path_exists(file));
        }
        fptr = nullptr;
        mod = nullptr;

        // auto removal check
        for (const auto &file : temp_files) {
            EXPECT_FALSE(path_exists(file));
        }
    }

    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        func_c faaa = aaa;
        // test loading after unloading a module
        auto mod = engine->make_jit_module(m, true);
        auto fptr = mod->get_function("aaa");
        void **gfunc_ptr = (void **)mod->get_address_of_symbol("func_ptr");
        ASSERT_TRUE(gfunc_ptr);
        *gfunc_ptr = (void *)the_call_back;
        int buf[100];
        testing::internal::CaptureStdout();
        fptr->call<float>(3, 23, buf);
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_EQ(output, "hahah26\nhahah27\n");
        EXPECT_EQ(buf[8], 34);

        temp_files = fptr->get_module()->get_temp_filenames();
        // generic calls
        generic_val args[3];
        args[0].v_int32_t = 12;
        args[1].v_int32_t = 14;
        args[2].v_ptr = buf;
        testing::internal::CaptureStdout();
        fptr->call_generic_default(args);
        output = testing::internal::GetCapturedStdout();
        EXPECT_EQ(output, "hahah26\nhahah27\n");
        EXPECT_EQ(buf[9], 35);
        for (const auto &file : temp_files) {
            EXPECT_TRUE(path_exists(file));
        }
        mod = nullptr;
        fptr = nullptr;
        // auto removal check
        for (const auto &file : temp_files) {
            EXPECT_FALSE(path_exists(file));
        }
    }
}

TEST(GCCore_jit_cpp, TestJITCast) {
    ir_builder_t builder;
    auto make_module = [](int lanes) {
        _function_(datatypes::s32, aaa, _arg_("buf", datatypes::f32, {1024}),
                _arg_("out", datatypes::s32, {1024}),
                _arg_("out2", datatypes::s32, {1024})) {
            _bind_(buf, out, out2);
            _for_(i, 0, 1024, lanes) {
                out[span_t({i}, lanes)] = builder::make_cast(
                        sc_data_type_t::s32(lanes), buf[span_t({i}, lanes)]);
                out2[span_t({i}, lanes)] = builder::make_round_and_cast(
                        buf[span_t({i}, lanes)], sc_data_type_t::s32(lanes));
            }
            _return_(123);
        }
        return ir_module_t::from_entry_func(get_default_context(), aaa);
    };

    auto engines = get_engines();
    std::vector<int> lanes_v = {1, 4, 8, 16};
    if (!get_default_context()->machine_.cpu_flags_.fAVX512F) {
        lanes_v.pop_back();
    }
    for (int lanes : lanes_v) {
        for (auto &engine : engines) {
            SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine)
                    + " lanes=" + std::to_string(lanes));
            auto fptr = engine->get_entry_func(make_module(lanes));
            ASSERT_TRUE(fptr);
            auto inbuf = alloc_array<float>(1024);
            uint32_t seed = rand(); // NOLINT
            for (auto &v : inbuf) {
                v = v * 10 + test_utils::rand_for_test<float>(seed, -1.0, 1.0);
            }
            auto out = alloc_array<int32_t>(1024, INIT_NOOP);
            auto out2 = alloc_array<int32_t>(1024, INIT_NOOP);
            EXPECT_EQ(fptr->call<int>(inbuf.data(), out.data(), out2.data()),
                    123);

            for (int i = 0; i < 1024; i++) {
                ASSERT_EQ(static_cast<int32_t>(inbuf[i]), out[i]);
                ASSERT_EQ(static_cast<int32_t>(std::roundf(inbuf[i])), out2[i]);
            }
        }
    }
}

TEST(GCCore_jit_cpp, TestJITCastToBF16) {
    REQUIRE_BF16();
    ir_builder_t builder;
    auto make_module = [](int lanes) {
        _function_(datatypes::s32, aaa, _arg_("buf", datatypes::f32, {1024}),
                _arg_("out", datatypes::bf16, {1024})) {
            _bind_(buf, out);
            _for_(i, 0, 1024, lanes) {
                out[span_t({i}, lanes)] = builder::make_cast(
                        sc_data_type_t::bf16(lanes), buf[span_t({i}, lanes)]);
            }
            _return_(123);
        }
        return ir_module_t::from_entry_func(get_default_context(), aaa);
    };

    auto engines = get_engines();
    std::vector<int> lanes_v = {1, 4, 8, 16};
    if (!get_default_context()->machine_.cpu_flags_.fAVX512F) {
        lanes_v.pop_back();
    }
    for (int lanes : lanes_v) {
        for (auto &engine : engines) {
            SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine)
                    + " lanes=" + std::to_string(lanes));
            auto fptr = engine->get_entry_func(make_module(lanes));
            ASSERT_TRUE(fptr);
            auto inbuf = alloc_array<float>(1024);
            uint32_t seed = rand(); // NOLINT
            for (auto &v : inbuf) {
                v = v * 10 + test_utils::rand_for_test<float>(seed);
            }
            auto out = alloc_array<bf16_t>(1024, INIT_NOOP);

            EXPECT_EQ(fptr->call<int>(inbuf.data(), out.data()), 123);

            for (int i = 0; i < 1024; i++) {
                ASSERT_EQ(static_cast<bf16_t>(inbuf[i]), out[i]);
            }
        }
    }
}

TEST(GCCore_jit_cpp, TestJITParallelFor) {
    ir_builder_t builder;
    _function_(datatypes::s32, aaa, _arg_("ii", datatypes::s32),
            _arg_("jj", datatypes::s32),
            _arg_("buf", datatypes::s32, {10000})) {
        _bind_(ii, jj, buf);
        _for_(i, 0, 10000, 1, for_type::PARALLEL) {
            buf[i] = ii + builder::make_abs(jj)
                    + builder::make_cast(datatypes::s32, i);

            buf[builtin::get_thread_id_func()()] = 333;
        }
        buf[9998] = builtin::get_thread_id_func()();
        _return_(123);
    }

    int threads = runtime_config_t::get().get_num_threads();
    ASSERT_LT(threads, 1000);
    auto engines = get_engines();
    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        auto fptr = engine->get_entry_func(
                ir_module_t::from_entry_func(get_default_context(), aaa));
        ASSERT_TRUE(fptr);
        std::vector<int> buf(10000);
        fptr->call_default(1, 23, buf.data());
        int collected_num_threads = 0;
        for (int i = 0; i < threads; i++) {
            if (buf[i] == 333) { collected_num_threads++; }
        }
        EXPECT_GT(collected_num_threads, 0);
        for (int i = threads; i < 9998; i++) {
            EXPECT_EQ(buf[i], 1 + 23 + i);
        }
        EXPECT_EQ(buf[9998], 0);
    }
}

TEST(GCCore_jit_cpp, TestJITVector) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {1024}),
            _arg_("B", datatypes::f32, {1024}),
            _arg_("C", datatypes::f32, {1024}),
            _arg_("D", datatypes::f32, {1024}),
            _arg_("bias", datatypes::f32, {8}),
            _arg_("bias1", datatypes::f32, {8}),
            _arg_("bias2", datatypes::f32, {8})) {
        _bind_(A, B, C, D, bias, bias1, bias2);
        _var_(bia, sc_data_type_t::f32(8));
        _var_(bia1, sc_data_type_t::f32(8));
        _var_(bia2, sc_data_type_t::f32(8));
        bia = bias[span_t({expr(0)}, 8)];
        bia1 = bias1[span_t({expr(0)}, 8)];
        bia2 = bias2[span_t({expr(0)}, 8)];
        _for_(i, 0, 1024, 8) {
            D[span_t({i}, 8)] = A[span_t({i}, 8)]
                    + B[span_t({i}, 8)] * C[span_t({i}, 8)] + bia
                    + builder::make_floor(bia1) + builder::make_ceil(bia2);
        }
    }

    auto engines = get_engines();
    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        auto fptr = engine->get_entry_func(
                ir_module_t::from_entry_func(get_default_context(), aaa));
        ASSERT_TRUE(fptr);
        auto getA = []() {
            std::vector<float> A(2048);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = i;
            }
            return A;
        };
        auto getB = []() {
            std::vector<float> A(2048);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = 2 * i - 100;
            }
            return A;
        };
        auto getC = []() {
            std::vector<float> A(2048);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = -0.1f * i;
            }
            return A;
        };
        auto getBias = []() {
            std::vector<float> A(8);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = i;
            }
            return A;
        };
        auto getBias1 = []() {
            std::vector<float> A(8, -12.3f);
            return A;
        };
        auto getBias2 = []() {
            std::vector<float> A(8, -45.6f);
            return A;
        };
        std::vector<float> D(1024);
        auto A = getA();
        auto B = getB();
        auto C = getC();
        auto bias = getBias();
        auto bias1 = getBias1();
        auto bias2 = getBias2();
        fptr->call<void>(A.data(), B.data(), C.data(), D.data(), bias.data(),
                bias1.data(), bias2.data());
        for (int i = 0; i < 1024; i++) {
            auto expected = float(i) + float(2 * i - 100) * float(-0.1f * i)
                    + float(i % 8) + float(-13) + float(-45);
            EXPECT_NEAR(D[i], expected, 0.1f);
        }
    }
}

TEST(GCCore_jit_cpp, TestJITVectorShift) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::s32, {1024}),
            _arg_("B", datatypes::s32, {1024}),
            _arg_("C", datatypes::s32, {1024}),
            _arg_("D", datatypes::s32, {1024})) {
        _bind_(A, B, C, D);
        _for_(i, 0, 1024, 8) {
            C[span_t({i}, 8)] = A[span_t({i}, 8)] << B[span_t({i}, 8)];
            D[span_t({i}, 8)] = A[span_t({i}, 8)] >> B[span_t({i}, 8)];
        }
    }

    auto engines = get_engines();
    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        auto fptr = engine->get_entry_func(
                ir_module_t::from_entry_func(get_default_context(), aaa));
        ASSERT_TRUE(fptr);
        auto getA = []() {
            std::vector<int32_t> A(2048);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = i;
            }
            return A;
        };
        auto getB = []() {
            std::vector<int32_t> A(2048, 2);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = 2;
            }
            return A;
        };
        std::vector<int32_t> C(1024);
        std::vector<int32_t> D(1024);
        auto A = getA();
        auto B = getB();
        fptr->call<void>(A.data(), B.data(), C.data(), D.data());
        for (int i = 0; i < 1024; i++) {
            auto expected1 = i * 4;
            auto expected2 = i / 4;
            EXPECT_NEAR(C[i], expected1, 1.0f);
            EXPECT_NEAR(D[i], expected2, 1.0f);
        }
    }
}

TEST(GCCore_jit_cpp, TestJITVectorShuffle) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {1024}),
            _arg_("B", datatypes::f32, {1024}),
            _arg_("C", datatypes::f32, {1024}),
            _arg_("D", datatypes::f32, {1024}),
            _arg_("E", datatypes::f32, {1024}),
            _arg_("F", datatypes::f32, {1024})) {
        _bind_(A, B, C, D, E, F);
        _for_(i, 0, 1024, 8) {
            C[span_t({i}, 8)] = builder::make_unpack_high(
                    A[span_t({i}, 8)], B[span_t({i}, 8)]);
            D[span_t({i}, 8)] = builder::make_unpack_low(
                    A[span_t({i}, 8)], B[span_t({i}, 8)]);
            E[span_t({i}, 8)] = builder::make_shuffle(
                    A[span_t({i}, 8)], B[span_t({i}, 8)], 68);
            F[span_t({i}, 8)] = builder::make_permute(
                    A[span_t({i}, 8)], B[span_t({i}, 8)], 32);
        }
    }

    auto engines = get_engines();
    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        auto fptr = engine->get_entry_func(
                ir_module_t::from_entry_func(get_default_context(), aaa));
        ASSERT_TRUE(fptr);
        auto getA = []() {
            std::vector<float> A(1024);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = 3 * i;
            }
            return A;
        };
        auto getB = []() {
            std::vector<float> A(1024);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = 2 * i;
            }
            return A;
        };
        std::vector<float> C(1024);
        std::vector<float> D(1024);
        std::vector<float> E(1024);
        std::vector<float> F(1024);
        auto A = getA();
        auto B = getB();
        fptr->call<void>(
                A.data(), B.data(), C.data(), D.data(), E.data(), F.data());
        for (int i = 0; i < 1024; i += 8) {
            EXPECT_NEAR(C[i + 0], A[i + 2], 1e-5);
            EXPECT_NEAR(C[i + 1], B[i + 2], 1e-5);
            EXPECT_NEAR(C[i + 2], A[i + 3], 1e-5);
            EXPECT_NEAR(C[i + 3], B[i + 3], 1e-5);
            EXPECT_NEAR(C[i + 4], A[i + 6], 1e-5);
            EXPECT_NEAR(C[i + 5], B[i + 6], 1e-5);
            EXPECT_NEAR(C[i + 6], A[i + 7], 1e-5);
            EXPECT_NEAR(C[i + 7], B[i + 7], 1e-5);

            EXPECT_NEAR(D[i + 0], A[i + 0], 1e-5);
            EXPECT_NEAR(D[i + 1], B[i + 0], 1e-5);
            EXPECT_NEAR(D[i + 2], A[i + 1], 1e-5);
            EXPECT_NEAR(D[i + 3], B[i + 1], 1e-5);
            EXPECT_NEAR(D[i + 4], A[i + 4], 1e-5);
            EXPECT_NEAR(D[i + 5], B[i + 4], 1e-5);
            EXPECT_NEAR(D[i + 6], A[i + 5], 1e-5);
            EXPECT_NEAR(D[i + 7], B[i + 5], 1e-5);

            // 0x01000100
            EXPECT_NEAR(E[i + 0], A[i + 0], 1e-5);
            EXPECT_NEAR(E[i + 1], A[i + 1], 1e-5);
            EXPECT_NEAR(E[i + 2], B[i + 0], 1e-5);
            EXPECT_NEAR(E[i + 3], B[i + 1], 1e-5);
            EXPECT_NEAR(E[i + 4], A[i + 4], 1e-5);
            EXPECT_NEAR(E[i + 5], A[i + 5], 1e-5);
            EXPECT_NEAR(E[i + 6], B[i + 4], 1e-5);
            EXPECT_NEAR(E[i + 7], B[i + 5], 1e-5);
            // 0x00100000
            EXPECT_NEAR(F[i + 0], A[i + 0], 1e-5);
            EXPECT_NEAR(F[i + 1], A[i + 1], 1e-5);
            EXPECT_NEAR(F[i + 2], A[i + 2], 1e-5);
            EXPECT_NEAR(F[i + 3], A[i + 3], 1e-5);
            EXPECT_NEAR(F[i + 4], B[i + 0], 1e-5);
            EXPECT_NEAR(F[i + 5], B[i + 1], 1e-5);
            EXPECT_NEAR(F[i + 6], B[i + 2], 1e-5);
            EXPECT_NEAR(F[i + 7], B[i + 3], 1e-5);
        }
    }
}

TEST(GCCore_jit_cpp, TestJITVectorExp) {
    builder::ir_builder_t builder;
    int lanes = get_default_context()->get_max_vector_lanes(sc_data_etype::F32);
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {4 * 16}),
            _arg_("out", datatypes::f32, {4 * 16})) {
        _bind_(A, out);
        _for_(i, 0, 4 * 16, lanes) {
            out[span_t({i}, lanes)] = builder::make_exp(A[span_t({i}, lanes)]);
        }
    }

    auto engines = get_engines();
    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        auto fptr = engine->get_entry_func(
                ir_module_t::from_entry_func(get_default_context(), aaa));
        ASSERT_TRUE(fptr);
        std::vector<float> A {-400.0, -300.0, -200.0, 0.0, 88.54, -87.32736,
                -0.0000000023431783, 0.00000537133123132, -77.36143377854597,
                -31.265935302161033, -44.375920170147786, 53.45083212459927,
                -46.00751423020855, -77.80392243036792, -51.47062321531107,
                12.29778060970466, -73.16798893622726, -6.448146416082253,
                29.355474974724515, 9.493752483855133, -18.690698248686658,
                22.2158171736952, 12.127730168284998, 34.77565690140503,
                -36.56118977835629, -0.048786066934596306, 86.30609073702799,
                -64.70384703267281, 35.43627610059812, 67.76313421869965,
                -64.51974340337865, -39.850006944087596, 1.492907139628315,
                -84.2008668172895, -46.281348832117345, 84.87902235985125,
                -83.21309421146465, -38.3112098289613, -15.55520431275707,
                22.24779132743943, -60.47383900817228, -3.2541373529870867,
                9.050757135279753, -37.443981734529984, 5.023265684648337,
                -79.69942646301337, 39.922178527841936, -33.99523678044496,
                10.047838688566046, -1.5905743454625707, 19.752474224382297,
                54.520029453447435, -85.21571279171789, -79.80216815226852,
                83.35114654903332, -9.600846495008582, 35.06101727500716,
                24.70281125550754, -0.9431269946899761, -33.21298977151045,
                -10.254364349128764, 32.98638056767258, -49.648058417016884,
                -35.939207939124046};

        std::vector<float> out(16 * 4);
        fptr->call<void>(A.data(), out.data());
        for (int i = 0; i < 16 * 4; i++) {
            float expected = expf(A[i]);
            EXPECT_NEAR(out[i], expected, 0.00001 * expected);
        }
    }
}

TEST(GCCore_jit_cpp, TestJITVectorLoad) {
    builder::ir_builder_t builder;
    {
        int lanes = get_default_context()->get_max_vector_lanes(
                sc_data_etype::F32);
        _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {4 * 16}),
                _arg_("out", datatypes::f32, {4 * 16})) {
            _bind_(A, out);
            _for_(i, 0, 4 * 16, lanes) {
                out[span_t({i}, lanes)]
                        = builder::make_exp(A[span_t({i}, lanes)]);
            }
        }

        auto engines = get_engines();
        for (auto &engine : engines) {
            SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
            auto fptr = engine->get_entry_func(
                    ir_module_t::from_entry_func(get_default_context(), aaa));
            ASSERT_TRUE(fptr);
            std::vector<float> A {-400.0, -300.0, -200.0, 0.0, 88.54, -87.32736,
                    -0.0000000023431783, 0.00000537133123132,
                    -77.36143377854597, -31.265935302161033,
                    -44.375920170147786, 53.45083212459927, -46.00751423020855,
                    -77.80392243036792, -51.47062321531107, 12.29778060970466,
                    -73.16798893622726, -6.448146416082253, 29.355474974724515,
                    9.493752483855133, -18.690698248686658, 22.2158171736952,
                    12.127730168284998, 34.77565690140503, -36.56118977835629,
                    -0.048786066934596306, 86.30609073702799,
                    -64.70384703267281, 35.43627610059812, 67.76313421869965,
                    -64.51974340337865, -39.850006944087596, 1.492907139628315,
                    -84.2008668172895, -46.281348832117345, 84.87902235985125,
                    -83.21309421146465, -38.3112098289613, -15.55520431275707,
                    22.24779132743943, -60.47383900817228, -3.2541373529870867,
                    9.050757135279753, -37.443981734529984, 5.023265684648337,
                    -79.69942646301337, 39.922178527841936, -33.99523678044496,
                    10.047838688566046, -1.5905743454625707, 19.752474224382297,
                    54.520029453447435, -85.21571279171789, -79.80216815226852,
                    83.35114654903332, -9.600846495008582, 35.06101727500716,
                    24.70281125550754, -0.9431269946899761, -33.21298977151045,
                    -10.254364349128764, 32.98638056767258, -49.648058417016884,
                    -35.939207939124046};

            std::vector<float> out(16 * 4);
            fptr->call<void>(A.data(), out.data());
            for (int i = 0; i < 16 * 4; i++) {
                float expected = expf(A[i]);
                EXPECT_NEAR(out[i], expected, 0.00001 * expected);
            }
        }
    }
}

#ifdef __AVX512F__
TEST(GCCore_jit_cpp, TestJITCondition) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {1024}),
            _arg_("B", datatypes::f32, {1024}),
            _arg_("C", datatypes::f32, {1024}),
            _arg_("D", datatypes::f32, {1024}),
            _arg_("E", datatypes::f32, {1024}),
            _arg_("F", datatypes::f32, {1024})) {
        _bind_(A, B, C, D, E, F);
        _for_(i, 0, 1024, 8) {
            E[span_t({i}, 8)] = builder::make_select(
                    C[span_t({i}, 8)] > D[span_t({i}, 8)], A[span_t({i}, 8)],
                    B[span_t({i}, 8)]);
            F[span_t({i}, 8)] = builder::make_select(
                    builder::make_constant({UINT64_C(0x03)}, datatypes::u8),
                    A[span_t({i}, 8)], B[span_t({i}, 8)]);
        }
    }

    auto engines = get_engines();
    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        auto fptr = engine->get_entry_func(
                ir_module_t::from_entry_func(get_default_context(), aaa));
        ASSERT_TRUE(fptr);
        auto getC = []() {
            std::vector<float> A(2048);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = i;
            }
            return A;
        };
        auto getD = []() {
            std::vector<float> A(2048);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = 2 * i - 100;
            }
            return A;
        };
        auto getA = []() {
            std::vector<float> A(2048);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = 2 * i + 100;
            }
            return A;
        };
        auto getB = []() {
            std::vector<float> A(2048);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = 2 * i - 100;
            }
            return A;
        };
        std::vector<float> E(1024);
        std::vector<float> F(1024);
        auto A = getA();
        auto B = getB();
        auto C = getC();
        auto D = getD();
        fptr->call<void>(
                A.data(), B.data(), C.data(), D.data(), E.data(), F.data());
        for (int i = 0; i < 1024; i++) {
            auto expected_e = C[i] > D[i] ? A[i] : B[i];
            auto expected_f = (i % 8) < 2 ? A[i] : B[i];
            EXPECT_NEAR(E[i], expected_e, 1e-5);
            EXPECT_NEAR(F[i], expected_f, 1e-5);
        }
    }
}

TEST(GCCore_jit_cpp, TestJITVectorUnpackElemLanes) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::u16, {1024}),
            _arg_("B", datatypes::u16, {1024}),
            _arg_("C", datatypes::u16, {1024}),
            _arg_("D", datatypes::u16, {1024}),
            _arg_("E", datatypes::u16, {1024}),
            _arg_("F", datatypes::u16, {1024}),
            _arg_("G", datatypes::u16, {1024}),
            _arg_("H", datatypes::u16, {1024})) {
        _bind_(A, B, C, D, E, F, G, H);
        _for_(i, 0, 1024, 32) {
            C[span_t({i}, 32)] = builder::make_unpack_high(
                    A[span_t({i}, 32)], B[span_t({i}, 32)], 16);
            D[span_t({i}, 32)] = builder::make_unpack_low(
                    A[span_t({i}, 32)], B[span_t({i}, 32)], 16);
            E[span_t({i}, 32)] = builder::make_unpack_high(
                    A[span_t({i}, 32)], B[span_t({i}, 32)], 32);
            F[span_t({i}, 32)] = builder::make_unpack_low(
                    A[span_t({i}, 32)], B[span_t({i}, 32)], 32);
            G[span_t({i}, 32)] = builder::make_unpack_high(
                    A[span_t({i}, 32)], B[span_t({i}, 32)], 64);
            H[span_t({i}, 32)] = builder::make_unpack_low(
                    A[span_t({i}, 32)], B[span_t({i}, 32)], 64);
        }
    }

    auto engines = get_engines();
    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        auto fptr = engine->get_entry_func(
                ir_module_t::from_entry_func(get_default_context(), aaa));
        ASSERT_TRUE(fptr);
        auto getA = []() {
            std::vector<uint16_t> A(1024);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = 3 * i;
            }
            return A;
        };
        auto getB = []() {
            std::vector<uint16_t> A(1024);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = 2 * i;
            }
            return A;
        };
        std::vector<uint16_t> C(1024);
        std::vector<uint16_t> D(1024);
        std::vector<uint16_t> E(1024);
        std::vector<uint16_t> F(1024);
        std::vector<uint16_t> G(1024);
        std::vector<uint16_t> H(1024);
        auto A = getA();
        auto B = getB();
        fptr->call<void>(A.data(), B.data(), C.data(), D.data(), E.data(),
                F.data(), G.data(), H.data());
        for (int i = 0; i < 1024; i += 32) {
            EXPECT_NEAR(C[i + 0], A[i + 4], 1e-5);
            EXPECT_NEAR(C[i + 1], B[i + 4], 1e-5);
            EXPECT_NEAR(C[i + 6], A[i + 7], 1e-5);
            EXPECT_NEAR(C[i + 7], B[i + 7], 1e-5);
            EXPECT_NEAR(C[i + 22], A[i + 23], 1e-5);
            EXPECT_NEAR(C[i + 23], B[i + 23], 1e-5);
            EXPECT_NEAR(C[i + 30], A[i + 31], 1e-5);
            EXPECT_NEAR(C[i + 31], B[i + 31], 1e-5);

            EXPECT_NEAR(D[i + 0], A[i + 0], 1e-5);
            EXPECT_NEAR(D[i + 1], B[i + 0], 1e-5);
            EXPECT_NEAR(D[i + 6], A[i + 3], 1e-5);
            EXPECT_NEAR(D[i + 7], B[i + 3], 1e-5);
            EXPECT_NEAR(D[i + 22], A[i + 19], 1e-5);
            EXPECT_NEAR(D[i + 23], B[i + 19], 1e-5);
            EXPECT_NEAR(D[i + 30], A[i + 27], 1e-5);
            EXPECT_NEAR(D[i + 31], B[i + 27], 1e-5);

            EXPECT_NEAR(E[i + 0], A[i + 4], 1e-5);
            EXPECT_NEAR(E[i + 1], A[i + 5], 1e-5);
            EXPECT_NEAR(E[i + 6], B[i + 6], 1e-5);
            EXPECT_NEAR(E[i + 7], B[i + 7], 1e-5);
            EXPECT_NEAR(E[i + 22], B[i + 22], 1e-5);
            EXPECT_NEAR(E[i + 23], B[i + 23], 1e-5);
            EXPECT_NEAR(E[i + 30], B[i + 30], 1e-5);
            EXPECT_NEAR(E[i + 31], B[i + 31], 1e-5);

            EXPECT_NEAR(F[i + 0], A[i + 0], 1e-5);
            EXPECT_NEAR(F[i + 1], A[i + 1], 1e-5);
            EXPECT_NEAR(F[i + 6], B[i + 2], 1e-5);
            EXPECT_NEAR(F[i + 7], B[i + 3], 1e-5);
            EXPECT_NEAR(F[i + 22], B[i + 18], 1e-5);
            EXPECT_NEAR(F[i + 23], B[i + 19], 1e-5);
            EXPECT_NEAR(F[i + 30], B[i + 26], 1e-5);
            EXPECT_NEAR(F[i + 31], B[i + 27], 1e-5);

            EXPECT_NEAR(G[i + 0], A[i + 4], 1e-5);
            EXPECT_NEAR(G[i + 1], A[i + 5], 1e-5);
            EXPECT_NEAR(G[i + 6], B[i + 6], 1e-5);
            EXPECT_NEAR(G[i + 7], B[i + 7], 1e-5);
            EXPECT_NEAR(G[i + 18], A[i + 22], 1e-5);
            EXPECT_NEAR(G[i + 19], A[i + 23], 1e-5);
            EXPECT_NEAR(G[i + 30], B[i + 30], 1e-5);
            EXPECT_NEAR(G[i + 31], B[i + 31], 1e-5);

            EXPECT_NEAR(H[i + 0], A[i + 0], 1e-5);
            EXPECT_NEAR(H[i + 1], A[i + 1], 1e-5);
            EXPECT_NEAR(H[i + 6], B[i + 2], 1e-5);
            EXPECT_NEAR(H[i + 7], B[i + 3], 1e-5);
            EXPECT_NEAR(H[i + 18], A[i + 18], 1e-5);
            EXPECT_NEAR(H[i + 19], A[i + 19], 1e-5);
            EXPECT_NEAR(H[i + 30], B[i + 26], 1e-5);
            EXPECT_NEAR(H[i + 31], B[i + 27], 1e-5);
        }
    }
}

TEST(GCCore_jit_cpp, TestJITVectorBroadcast) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::u16, {256}),
            _arg_("B", datatypes::u16, {1024})) {
        _bind_(A, B);
        _for_(i, 0, 256, 8) {
            B[span_t({i * 4}, 32)]
                    = builder::make_broadcast(A[span_t({i}, 8)], 32);
        }
    }

    auto engines = get_engines();
    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        auto fptr = engine->get_entry_func(
                ir_module_t::from_entry_func(get_default_context(), aaa));
        ASSERT_TRUE(fptr);
        auto getA = []() {
            std::vector<uint16_t> A(256);
            for (int i = 0; i < (int)A.size(); i++) {
                A[i] = 3 * i;
            }
            return A;
        };
        std::vector<uint16_t> B(1024);
        auto A = getA();
        fptr->call<void>(A.data(), B.data());
        for (int i = 0; i < 1024; i++) {
            EXPECT_NEAR(B[i], A[i % 8 + i / 32 * 8], 1e-5);
        }
    }
}
#endif

TEST(GCCore_jit_cpp, TestJITGlobalTensor) {
    builder::ir_builder_t builder;
    auto m = std::make_shared<ir_module_t>(get_default_context());
    _global_tensor_(m, gv, datatypes::s32, 2, 2);
    _global_var_(m, gvar, datatypes::s32, 2);
    int32_t values[] = {1456, 2, 3, 4};
    gv.static_as<tensor>()->init_value_
            = std::make_shared<static_data_t>(values, sizeof(values));

    _function_(datatypes::s32, bbb) {
        gv[{0, 1}] = 123;
        gvar = gvar + 1;
        _return_(gv[{0, 0}]);
    }
    m->add_func({bbb});

    auto engines = get_engines();
    for (auto &engine : engines) {
        SCOPED_TRACE(std::string("Testing ") + get_engine_name(engine));
        auto jitmod = engine->make_jit_module(m, false);
        statics_table_t *pglobals = nullptr;
#if SC_BUILTIN_JIT_ENABLED
        if (auto jmod
                = dynamic_cast<sc_xbyak::xbyak_jit_module *>(jitmod.get())) {
            pglobals = &jmod->globals_;
        }
#else
        if (false) {
            // make compiler happy
        }
#endif
#if SC_CFAKE_JIT_ENABLED
        else if (auto jmod = dynamic_cast<cfake_jit_module_t *>(jitmod.get())) {
            pglobals = &jmod->globals_;
        }
#else
        if (false) {
            // make compiler happy
        }
#endif
#if defined(SC_LLVM_BACKEND)
        else if (auto jmod = dynamic_cast<llvm_jit_module *>(jitmod.get())) {
            pglobals = &jmod->globals_;
        }
#endif
        ASSERT_TRUE(pglobals);
        auto &globals = *pglobals;
        void *entry = globals.get_or_null("gv");
        ASSERT_NE(entry, nullptr);
        int32_t *real_gv = reinterpret_cast<int32_t *>(entry);
        ASSERT_EQ(real_gv, jitmod->get_address_of_symbol("gv"));
        // makes sure real_gv is a copy of values[]
        ASSERT_EQ(real_gv[3], 4);

        int32_t *pvar = reinterpret_cast<int32_t *>(
                jitmod->get_address_of_symbol("gvar"));
        ASSERT_TRUE(pvar);
        ASSERT_EQ(*pvar, 2);

        auto jitfunc = jitmod->get_function("bbb");
        ASSERT_EQ(jitfunc->call<int32_t>(), 1456);
        ASSERT_EQ(real_gv[1], 123);
        ASSERT_EQ(*pvar, 3);

        EXPECT_EQ(globals.impl_["gvar"], 0UL);
        EXPECT_EQ(globals.impl_["gv"], 4UL);
    }
}

TEST(GCCore_jit_cpp, TestJITDispatchTable) {
    builder::ir_builder_t builder;
    auto m = std::make_shared<ir_module_t>(get_default_context());
    _function_(datatypes::s32, bbb_0) { _return_(0); }
    _function_(datatypes::s32, bbb_1) { _return_(1); }
    _function_(datatypes::s32, bbb_2) { _return_(2); }
    _function_(datatypes::s32, ccc_0) { _return_(3); }
    _function_(datatypes::s32, ccc_1) { _return_(4); }
    m->add_func({bbb_0, bbb_1, bbb_2, ccc_0, ccc_1});
    auto bbb_table = std::make_shared<op_dispatch_tables_t>();
    auto dispatch_key = op_dispatch_key_t();
    dispatch_key.in_out_formats_ = {sc_data_format_t::MK()};
    add_dispatch_symbol_to_kernel_table(bbb_table, &dispatch_key, "bbb_0");
    dispatch_key.in_out_formats_ = {sc_data_format_t::NK()};
    add_dispatch_symbol_to_kernel_table(bbb_table, &dispatch_key, "bbb_1");
    dispatch_key.in_out_formats_ = {sc_data_format_t::NCHW()};
    add_dispatch_symbol_to_kernel_table(bbb_table, &dispatch_key, "bbb_2");
    m->add_op_table(std::make_pair("bbb_table", bbb_table));
    m->make_global_var(
            datatypes::pointer, "bbb_table", linkage::private_global);
    auto ccc_table = std::make_shared<op_dispatch_tables_t>();
    dispatch_key.in_out_formats_ = {sc_data_format_t::MK()};
    add_dispatch_symbol_to_kernel_table(ccc_table, &dispatch_key, "ccc_0");
    dispatch_key.in_out_formats_ = {sc_data_format_t::NK()};
    add_dispatch_symbol_to_kernel_table(ccc_table, &dispatch_key, "ccc_1");
    m->add_op_table(std::make_pair("ccc_table", ccc_table));
    m->make_global_var(
            datatypes::pointer, "ccc_table", linkage::private_global);
    auto engines = get_engines();
    for (auto &engine : engines) {
        auto jitm = engine->make_jit_module(m, false);
        auto bbb_runtime_table = jitm->op_tables_["bbb_table"];
        auto ccc_runtime_table = jitm->op_tables_["ccc_table"];
        EXPECT_TRUE(bbb_runtime_table->kernel_dispatch_func_);
        auto runtime_format = uint64_t(sc_data_format_t::MK().to_runtime());
        EXPECT_TRUE(bbb_runtime_table->kernel_table_->get(&runtime_format, 1));
        runtime_format = uint64_t(sc_data_format_t::NK().to_runtime());
        EXPECT_TRUE(bbb_runtime_table->kernel_table_->get(&runtime_format, 1));
        runtime_format = uint64_t(sc_data_format_t::NCHW().to_runtime());
        EXPECT_TRUE(bbb_runtime_table->kernel_table_->get(&runtime_format, 1));
        runtime_format = uint64_t(sc_data_format_t::MK().to_runtime());
        EXPECT_TRUE(ccc_runtime_table->kernel_table_->get(&runtime_format, 1));
        runtime_format = uint64_t(sc_data_format_t::NK().to_runtime());
        EXPECT_TRUE(ccc_runtime_table->kernel_table_->get(&runtime_format, 1));
        runtime_format = uint64_t(sc_data_format_t::NCHW().to_runtime());
        EXPECT_TRUE(ccc_runtime_table->kernel_table_->get(&runtime_format, 1)
                == nullptr);

        EXPECT_EQ(*reinterpret_cast<runtime::op_dispatch_tables_t **>(
                          jitm->globals_.get("bbb_table")),
                bbb_runtime_table.get());
        EXPECT_EQ(*reinterpret_cast<runtime::op_dispatch_tables_t **>(
                          jitm->globals_.get("ccc_table")),
                ccc_runtime_table.get());
    }
}

#if 0

/*
The following test check that the alias info is correctly propagated to LLVM
codegen. It heavily depends on LLVM's optimization result, so we decide not add
it to the test set. The result LLVM IR should be:

define i32 @aaa(i8* nocapture readnone %__stream_arg, i8* noalias nocapture
nonnull readnone %__module_data_arg, i32* nocapture nonnull %buf_arg, i32*
nocapture nonnull %out_arg, i32* noalias nocapture nonnull %out2_arg, i32*
noalias nocapture nonnull %out3_arg) local_unnamed_addr #0 { entry: store i32 1,
i32* %buf_arg, align 4, !alias.scope !0, !noalias !3 store i32 3, i32* %out_arg,
align 4, !alias.scope !0, !noalias !3 %0 = load i32, i32* %buf_arg, align 4,
!alias.scope !0, !noalias !3 %1 = add i32 %0, 1 store i32 %1, i32* %buf_arg,
align 4, !alias.scope !0, !noalias !3 %2 = getelementptr i32, i32* %buf_arg, i64
1 store i32 3, i32* %out2_arg, align 4, !alias.scope !7, !noalias !8 store i32
2, i32* %2, align 4, !alias.scope !0, !noalias !3 %3 = getelementptr i32, i32*
%out2_arg, i64 1 store i32 3, i32* %3, align 4, !alias.scope !7, !noalias !8
  store i32 2, i32* %out3_arg, align 4, !alias.scope !9, !noalias !10
  ret i32 123
}

Note that out and buf has alias, so no optimization on buf[0] = buf[0] + 1;
out & buf2, buf3 & buf2 has no alias, so LLVM can optimize it
*/

#include <compiler/ir/transform/pointer_alias_info.hpp>
TEST(GCCore_jit_cpp, TestJITLLVMAlias) {
    ir_builder_t builder;
    _function_(datatypes::s32, aaa, _arg_("buf", datatypes::s32, {2}),
            _arg_("out", datatypes::s32, {2}),
            _arg_("out2", datatypes::s32, {2}),
            _arg_("out3", datatypes::s32, {2})) {
        _bind_(buf, out, out2, out3);
        {
            auto clique = std::make_shared<alias_info::alias_set_t>();
            alias_info::get_or_create_alias_info(*buf.get())
                    ->add_to_clique(clique);
            alias_info::get_or_create_alias_info(*out.get())
                    ->add_to_clique(clique);
        }
        buf[0] = 1;
        out[0] = 3;
        buf[0] = buf[0] + 1;

        buf[1] = 1;
        out2[0] = 3;
        buf[1] = buf[1] + 1;

        out3[0] = 1;
        out2[1] = 3;
        out3[0] = out3[0] + 1;
        _return_(123);
    }
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->flags_.index2var_ = false;
    auto mod = ir_module_t::from_entry_func(ctx, aaa);

    auto jitf = llvm_jit(ctx).get_entry_func(mod, false);
    float A[2], B[2], C[2], D[2];
    jitf->call<int32_t>(A, B, C, D);
}

#endif
