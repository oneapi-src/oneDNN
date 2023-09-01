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

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include "compiler/config/context.hpp"
#include "context.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#if SC_CFAKE_JIT_ENABLED
#include <compiler/jit/cfake/cfake_jit.hpp>
#endif
#include "test_utils.hpp"
#if defined(SC_LLVM_BACKEND)
#include <compiler/jit/llvm/llvm_jit.hpp>
#endif
#include <compiler/ir/builtin.hpp>
#if SC_BUILTIN_JIT_ENABLED
#include <compiler/jit/xbyak/xbyak_jit.hpp>
#endif
#include "util/bf16.hpp"
#include "util/fp16.hpp"
#include <runtime/generic_val.hpp>
#include <runtime/runtime.hpp>
#include <util/string_utils.hpp>
#include <util/utils.hpp>

#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;
using std::endl;
using std::make_shared;
using std::map;
using std::ostringstream;
using std::shared_ptr;
using std::string;
using std::vector;

static const map<string, shared_ptr<jit_engine_t>> &get_engines() {
    auto f = []() {
        map<string, shared_ptr<jit_engine_t>> ret;
#if SC_CFAKE_JIT_ENABLED
        ret["cfake_jit"] = make_shared<cfake_jit>();
#endif
#if defined(SC_LLVM_BACKEND)
        ret["llvm_jit"] = make_shared<llvm_jit>();
#endif
#if SC_BUILTIN_JIT_ENABLED
        if (get_default_context()->machine_.cpu_flags_.fAVX2) {
            ret["xbyak_jit"] = make_shared<xbyak_jit>();
        }
#endif
        return ret;
    };
    static auto ret = f();
    return ret;
}

static const runtime::cpu_flags_t &test_cpu_flags() {
    static auto ret = get_default_context()->machine_.cpu_flags_;
    return ret;
}
static const bool is_cpu_support_fp16() {
    return get_default_context()->machine_.cpu_flags_.fAVX512FP16;
}
static bool use_cfake = true;

static bool is_cfake(const std::string &jitname) {
    return jitname == "cfake_jit";
}

static bool is_cfake_support_fp16() {
    // Our cfake uses _Float16, which needs to be supported by g++.
#if SC_CFAKE_JIT_ENABLED
    auto f = []() {
        auto tm = get_default_context()->machine_;
        cfake_jit::set_target_machine(tm);
        auto ret = tm.cpu_flags_.fAVX512FP16;
        return ret;
    };
    static bool ret = f();
    return ret;
#else
    return false;
#endif
}

static uint16_t get_lanes(
        const sc_data_etype &etype, const uint16_t data_lanes = 16) {
    static context_ptr cptr = get_default_context();
    return std::min(data_lanes, cptr->get_max_vector_lanes(etype));
}

//===========================================================================
// Pre-defined dataset
//===========================================================================
#define DATA_LEN_8 8
#define DATA_LEN_16 16
#define DATA_LEN_64 64

#define DATASET_I1 \
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }
#define DATASET_I2 \
    { 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 }
#define DATASET_I3 \
    { -16, -15, -14, -13, -12, -11, -10, -9, 8, 7, 6, 5, 4, 3, 2, 1 }
#define DATASET_I1_8 \
    { 1, 2, 3, 4, 5, 6, 7, 8 }
#define DATASET_I1_4 \
    { 1, 2, 3, 4 }
#define DATASET_I1_64 \
    { \
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, \
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, \
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64 \
    }
#define DATASET_F1 \
    { \
        1.16f, 2.15f, 3.14f, 4.13f, 5.12f, 6.11f, 7.10f, 8.9f, 9.8f, 10.7f, \
                11.6f, 12.5f, 13.4f, 14.3f, 15.2f, 16.1f \
    }
#define DATASET_F2 \
    { \
        16.1f, 15.2f, 14.3f, 13.4f, 12.5f, 11.6f, 10.7f, 9.8f, 8.9f, 7.10f, \
                6.11f, 5.12f, 4.13f, 3.14f, 2.15f, 1.16f \
    }
#define DATASET_F3 \
    { \
        -64.1f, -32.9f, -16.2f, -8.8f, -4.3f, -2.7f, -1.4f, 0.6f, 0.5f, 1.5f, \
                2.6f, 4.4f, 8.7f, 16.3f, 32.8f, 64.2f \
    }
#define DATASET_F4 \
    { \
        -64.1f, -32.9f, -16.2f, -8.8f, -4.3f, -2.7f, -1.4f, 0.6f, 1.16f, \
                2.15f, 3.14f, 4.13f, 5.12f, 6.11f, 7.10f, 8.9f \
    }
#define DATASET_BF1 \
    { \
        0x3f94, 0x400a, 0x4049, 0x4084, 0x40a4, 0x40c4, 0x40e3, 0x410e, \
                0x411d, 0x412b, 0x413a, 0x4148, 0x4156, 0x4165, 0x4173, 0x4181 \
    }
#define DATASET_BF2 \
    { \
        0x4181, 0x4173, 0x4165, 0x4156, 0x4148, 0x413a, 0x412b, 0x411d, \
                0x410e, 0x40e3, 0x40c4, 0x40a4, 0x4084, 0x4049, 0x400a, 0x3f94 \
    }
#define DATASET_BF3 \
    { \
        0xc280, 0xc204, 0xc182, 0xc10d, 0xc08a, 0xc02d, 0xbfb3, 0x3f1a, \
                0x3f00, 0x3fc0, 0x4026, 0x408d, 0x410b, 0x4182, 0x4203, 0x4280 \
    }

//===========================================================================
// Pre-defined cast
//===========================================================================

#define F32_TO_BF16(X) (bf16_t(X).storage_)
#define BF16_TO_F32(X) ((float)bf16_t::from_storage(X))

//===========================================================================
// Test op kind
//===========================================================================
#define UNARY 1
#define BINARY 2
#define TRINARY 3

//===========================================================================
// Test options
//===========================================================================
#define EXACT true
#define APPROX false
#define TEST_SCALAR true
#define SKIP_SCALAR false
#define TEST_SIMD true
#define SKIP_SIMD false
static float precision_threshold = 1e-4F;
//===========================================================================
// MAKE_OP wrapper
//===========================================================================
#define MAKE_UNARY_OP(MAKE_OP) \
    [](expr buf, expr idx, int lanes) { \
        return MAKE_OP(buf[span_t({0, idx}, lanes)]); \
    }
#define MAKE_BINARY_OP(MAKE_OP) \
    [](expr buf, expr idx, int lanes) { \
        return MAKE_OP( \
                buf[span_t({0, idx}, lanes)], buf[span_t({1, idx}, lanes)]); \
    }
#define MAKE_TRINARY_OP(MAKE_OP) \
    [](expr buf, expr idx, int lanes) { \
        return MAKE_OP(buf[span_t({0, idx}, lanes)], \
                buf[span_t({1, idx}, lanes)], buf[span_t({2, idx}, lanes)]); \
    }
#define MAKE_CAST(DTYPE) \
    [](expr buf, expr idx, int lanes) { \
        return make_cast(sc_data_type_t(DTYPE.type_code_, lanes), \
                buf[span_t({0, idx}, lanes)]); \
    }
#define MAKE_MASK_MOV(MASK_LOCAL) \
    [MASK_LOCAL](expr buf, expr idx, int lanes) { \
        return (expr)buf[span_t({0, idx}, lanes, MASK_LOCAL)]; \
    }

using MAKE_EXPR_OP = std::function<expr(expr, expr, int)>;

//===========================================================================
// Macro for element-wise operation tests
//---------------------------------------------------------------------------
// INPUT: int(UNARY/BINARY/TRINARY); determine op kind and input size
// PRECISION: bool(EXACT/APPROX); determine use EXPECT_EQ or EXPECT_NEAR
// TYPE_IN: c++ types; input array type
// TYPE_OUT: c++ types; output and ref array type
// DTYPE_IN: sc_data_type_t; input data type
// DTYPE_OUT: sc_data_type_t; output data type
// DATA_LEN: int(16/32/64); dataset length want to be tested
// NUM_LANES: int(2/4/8/16); simd lanes want to be tested
// SCALAR: bool(TEST_SCALAR/SKIP_SCALAR); determine if test scalar op
// SIMD: bool(TEST_SIMD/SKIP_SIMD); determine if test simd op
// REF_OP: Macro; provide calculation for reference
// MAKE_OP: MAKE_EXPR_OP; provide builder::make_xxx() func for op maker on ir
// __VA_ARGS__: pre defined datasets for testing, num must match INPUT
//===========================================================================
#define TEST_OP(INPUT, PRECISION, TYPE_IN, TYPE_OUT, DTYPE_IN, DTYPE_OUT, \
        DATA_LEN, NUM_LANES, SCALAR, SIMD, REF_OP, MAKE_OP, ...) \
    { \
        TYPE_IN tensor_in[INPUT][DATA_LEN] = {__VA_ARGS__}; \
        TYPE_OUT tensor_out[DATA_LEN]; \
        TYPE_OUT tensor_ref[DATA_LEN]; \
        generic_val generic_args[] = { \
                tensor_in[0], \
                tensor_out, \
        }; \
        for (int i = 0; i < DATA_LEN; ++i) { \
            tensor_ref[i] = REF_OP(tensor_in, NUM_LANES, i); \
        } \
        TEST_IR_OP<TYPE_OUT, PRECISION, INPUT, SCALAR, SIMD>(MAKE_OP, \
                tensor_out, tensor_ref, generic_args, DTYPE_IN, DTYPE_OUT, \
                DATA_LEN, NUM_LANES); \
    }

//===========================================================================
// Template function for element-wise operation tests
//===========================================================================
template <typename TYPE, bool PRECISION, int INPUT, bool SCALAR, bool SIMD>
void TEST_IR_OP(MAKE_EXPR_OP make_op, TYPE *out, TYPE *ref,
        generic_val *generic_args, sc_data_type_t type_in,
        sc_data_type_t type_out, int data_len, int lanes) {
// Make ir function with op
#define DEFINE_IR_FUNC(NAME, LANES) \
    _function_(datatypes::void_t, NAME, \
            _arg_("buf_I", type_in, {INPUT, data_len}), \
            _arg_("buf_O", type_out, {data_len})) { \
        _bind_(buf_I, buf_O); \
        _for_(idx, 0, data_len, LANES) { \
            buf_O[span_t({idx}, LANES)] = make_op(buf_I, idx, LANES); \
        } \
    } \
    ir_mod->add_func({NAME});
// Test ir function with op
#define TEST_IR_FUNC(NAME) \
    shared_ptr<jit_function_t> jf = jm->get_function(NAME); \
    EXPECT_NE(jf, nullptr); \
    if (!jf) { continue; } \
    for (int i = 0; i < data_len; i++) { \
        out[i] = 0; \
    } \
    jf->call_generic_default(generic_args); \
    if (PRECISION) { \
        for (int i = 0; i < data_len; i++) { \
            EXPECT_EQ(out[i], ref[i]); \
        } \
    } else { \
        for (int i = 0; i < data_len; i++) { \
            EXPECT_NEAR( \
                    out[i], ref[i], std::abs(precision_threshold *ref[i])); \
        } \
    }
    // Make ir module
    ir_builder_t builder;
    auto ctx = get_default_context();
    auto ir_mod = std::make_shared<ir_module_t>(ctx);
    // Add test functions
    if (SCALAR) { DEFINE_IR_FUNC(test_scalar, 1); }
    if (SIMD) { DEFINE_IR_FUNC(test_simd, lanes); }
    // Run the test
    for (auto &kv : get_engines()) {
        const string &je_name = kv.first;
        if (!use_cfake) {
            if (je_name == "cfake_jit") continue;
        }
        if (is_cpu_support_fp16() && is_cfake(je_name)
                && !is_cfake_support_fp16()) {
            continue;
        }
        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());
        // jit_engine_t
        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }
        // jit_module
        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);
        if (!jm) { continue; }
        if (SCALAR) { TEST_IR_FUNC("test_scalar"); }
        if (SIMD) { TEST_IR_FUNC("test_simd"); }
    }
#undef DEFINE_IR_FUNC
#undef TEST_IR_FUNC
}

/// ==================================
/// Test Group 1: functionality & stmt
/// ==================================

TEST(GCCore_CPU_jit_engine_equivalence, TestLocalTensor) {
    ir_builder_t builder;

    _function_(datatypes::void_t, foo, _arg_("x_in", datatypes::s32),
            _arg_("x_out", datatypes::s32, {1})) {
        _bind_(x_in, x_out);
        _tensor_(my_tensor, datatypes::s32, {2});
        my_tensor[0] = x_in;
        x_out[0] = my_tensor[0];
    }

    ir_module_ptr ir_mod
            = ir_module_t::from_entry_func(get_default_context(), foo);

    for (auto &kv : get_engines()) {
        const int32_t host_x_in = 42;
        int32_t host_x_out[1] = {0};

        generic_val generic_args[] = {host_x_in, &host_x_out};

        const string &je_name = kv.first;

        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> jf = je->get_entry_func(ir_mod, true);
        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        void *native_func = jf->get_function_pointer();
        EXPECT_NE(native_func, nullptr);
        if (!native_func) { continue; }

        jf->call_generic_default(generic_args);
        EXPECT_EQ(host_x_in, host_x_out[0]);
    }
}

/// Verifies that the address of a caller-supplied tensor is accurately
/// passed down to JIT-generated code that accesses that tensor.
TEST(GCCore_CPU_jit_engine_equivalence, TestTensorAddrPassing) {
    ir_builder_t builder;

    _function_(datatypes::void_t, foo, _arg_("tensor_in", datatypes::f32, {1}),
            _arg_("result", datatypes::index, {1})) {
        _bind_(tensor_in, result);
        result[0] = make_cast(datatypes::index, tensor_in);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    float host_tensor_in[1];
    void *host_result;

    generic_val generic_args[] = {(void *)(host_tensor_in), &host_result};

    for (auto &kv : get_engines()) {
        host_result = nullptr;

        const string &je_name = kv.first;
        if (je_name == "llvm_jit") continue;

        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> jf = jm->get_function("foo");

        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        // TODO(xxx): In the cfake_jit, the generic wrapper has TWO parameters
        // if 'is_parallel' is true. (See 'write_cpp_generic_wrapper(...)' in
        // codegen/codegen_c.cpp) For now we're assuming 'is_parallel' is false.
        jf->call_generic_default(generic_args);

        ASSERT_EQ(&host_tensor_in, host_result);
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestSequentialElwiseAdd) {
    ir_builder_t builder;

    const int d1_size = 2;
    const int d2_size = 2;

    // Compute C = A + B
    _function_(datatypes::void_t, foo,
            _arg_("buf_A", datatypes::f32, {d1_size, d2_size}),
            _arg_("buf_B", datatypes::f32, {d1_size, d2_size}),
            _arg_("buf_C", datatypes::f32, {d1_size, d2_size})) {
        _bind_(buf_A, buf_B, buf_C);
        _for_(idx1, 0, d1_size) {
            _for_(idx2, 0, d2_size) {
                buf_C[{idx1, idx2}] = buf_A[{idx1, idx2}] + buf_B[{idx1, idx2}];
            }
        }
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    const int num_elem = d1_size * d2_size;
    float A[num_elem];
    float B[num_elem];
    float C[num_elem];
    float C_expected[num_elem];

    for (int i = 0; i < num_elem; ++i) {
        A[i] = float(i);
        B[i] = float(i + num_elem);
        C_expected[i] = A[i] + B[i];
    }

    generic_val generic_args[] = {(void *)(A), (void *)(B), (void *)(C)};

    for (auto &kv : get_engines()) {
        for (int i = 0; i < num_elem; ++i) {
            C[i] = float(-1);
        }

        const string &je_name = kv.first;

        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> jf = jm->get_function("foo");

        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        // TODO(xxx): In the cfake_jit, the generic wrapper has TWO parameters
        // if 'is_parallel' is true. (See 'write_cpp_generic_wrapper(...)' in
        // codegen/codegen_c.cpp) For now we're assuming 'is_parallel' is false.
        jf->call_generic_default(generic_args);

        for (int i = 0; i < num_elem; ++i) {
            ASSERT_EQ(C[i], C_expected[i]);
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestTrivialGenericWrapper) {
    ir_builder_t builder;
    _function_(datatypes::void_t, foo) {}

    ir_module_ptr ir_mod
            = ir_module_t::from_entry_func(get_default_context(), foo);

    for (auto &kv : get_engines()) {
        const string &je_name = kv.first;

        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_function_t> jf = je->get_entry_func(ir_mod, true);
        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        jf->call_generic_default(nullptr);
    }
}

// Disabled because the WIP Xbyak backend currently produces only
// the generic wrapper entry function, not the stronger-typed
// entry function.
TEST(GCCore_CPU_jit_engine_equivalence, DISABLED_TestSimpleEntryFunction) {
    ir_builder_t builder;
    _function_(datatypes::s32, foo) { _return_(42); }

    ir_module_ptr ir_mod
            = ir_module_t::from_entry_func(get_default_context(), foo);

    for (auto &kv : get_engines()) {
        const string &je_name = kv.first;

        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_function_t> jf = je->get_entry_func(ir_mod, true);
        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        void *native_func = jf->get_function_pointer();
        EXPECT_NE(native_func, nullptr);
        if (!native_func) { continue; }

        EXPECT_EQ(jf->call<int32_t>(), 42);
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestFuncAddrNode) {
    const void *host_print_int_addr = (void *)(&print_int);

    ir_builder_t builder;

    _function_(datatypes::void_t, foo,
            _arg_("jit_print_int_addr", datatypes::index, {1})) {
        _bind_(jit_print_int_addr);
        jit_print_int_addr[0] = make_cast(datatypes::index,
                make_func_addr(make_func("print_int",
                        {_arg_("x", datatypes::s32)}, {}, datatypes::void_t)));
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    const void *returned_print_int_addr;
    generic_val generic_args[] = {&returned_print_int_addr};

    for (auto &kv : get_engines()) {
        const string &je_name = kv.first;
        if (je_name == "llvm_jit") continue;

        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> jf = jm->get_function("foo");

        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        returned_print_int_addr = nullptr;
        jf->call_generic_default(generic_args);

        ASSERT_EQ(host_print_int_addr, returned_print_int_addr);
    }
}

// Verify that a named-for-loop's index variable has the expected
// sequence of values.
TEST(GCCore_CPU_jit_engine_equivalence, TestNamedForLoop) {
    constexpr int expected_num_iter = 4;
    const uint64_t expected_idx_values[] = {0, 3, 6, 9};

    uint64_t host_num_iter;
    uint64_t host_idx_values[expected_num_iter]; // NOLINT

    ir_builder_t builder;

    _function_(datatypes::void_t, foo, _arg_("num_iter", datatypes::index, {1}),
            _arg_("idx_values", datatypes::index, {expected_num_iter}), ) {
        _bind_(num_iter, idx_values);
        num_iter[0] = 0;
        for_loop l;
        _named_for_(l, idx, 0, 10, 3) {
            idx_values[num_iter[0]] = idx;
            num_iter[0] = num_iter[0] + 1;
        }
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    generic_val gv0 = &host_num_iter;
    generic_val gv1 = &host_idx_values;
    generic_val generic_args[] = {gv0, gv1};

    for (auto &kv : get_engines()) {
        // Initialize buffers to incorrect values...
        host_num_iter = 42;
        host_idx_values[0] = 42;
        host_idx_values[1] = 42;
        host_idx_values[2] = 42;
        host_idx_values[3] = 42;

        ostringstream err_context;
        const string &je_name = kv.first;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        j_foo->call_generic_default(generic_args);

        auto exp_num_iter = static_cast<uint64_t>(expected_num_iter);
        EXPECT_EQ(exp_num_iter, host_num_iter);
        for (int i = 0; i < expected_num_iter; ++i) {
            EXPECT_EQ(expected_idx_values[i], host_idx_values[i]);
        }
    }
}

// Verify that an if_else statement flows into the correct branch,
// and DOES NOT flow into the incorrect branch.
TEST(GCCore_CPU_jit_engine_equivalence, TestIfElse) {
    uint64_t host_basic_blocks_visited[5];
    const uint64_t host_first_tested_value = 1;
    const uint64_t host_second_tested_value = 1;

    ir_builder_t builder;

    _function_(datatypes::void_t, foo,
            _arg_("basic_blocks_visited", datatypes::index, {5}),
            _arg_("first_tested_value", datatypes::index),
            _arg_("second_tested_value", datatypes::index), ) {
        _bind_(basic_blocks_visited, first_tested_value, second_tested_value);
        _if_(first_tested_value == 1) { basic_blocks_visited[0] = 1; }
        else {
            basic_blocks_visited[1] = 1;
        }

        _if_(second_tested_value != 1) { basic_blocks_visited[2] = 1; }
        else {
            basic_blocks_visited[3] = 1;
        }

        _if_(!(first_tested_value == 0)) { basic_blocks_visited[4] = 2; }
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    generic_val gv0 = &host_basic_blocks_visited;
    generic_val gv1 = host_first_tested_value;
    generic_val gv2 = host_second_tested_value;
    generic_val generic_args[] = {gv0, gv1, gv2};

    for (auto &kv : get_engines()) {
        // Mark all 4 basic blocks as not-visited...
        host_basic_blocks_visited[0] = 0;
        host_basic_blocks_visited[1] = 0;
        host_basic_blocks_visited[2] = 0;
        host_basic_blocks_visited[3] = 0;
        host_basic_blocks_visited[4] = 0;

        ostringstream err_context;
        const string &je_name = kv.first;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        j_foo->call_generic_default(generic_args);

        EXPECT_EQ(host_basic_blocks_visited[0], 1UL);
        EXPECT_EQ(host_basic_blocks_visited[1], 0UL);
        EXPECT_EQ(host_basic_blocks_visited[2], 0UL);
        EXPECT_EQ(host_basic_blocks_visited[3], 1UL);
        EXPECT_EQ(host_basic_blocks_visited[4], 2UL);
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestCMPExprSint) {
    ir_builder_t builder;

    const int num_elems = 16;

    _function_(datatypes::void_t, foo,
            _arg_("first", datatypes::s32, {num_elems}),
            _arg_("second", datatypes::s32, {num_elems}),
            _arg_("result_eq", datatypes::boolean, {num_elems}),
            _arg_("result_le", datatypes::boolean, {num_elems}),
            _arg_("result_lt", datatypes::boolean, {num_elems}),
            _arg_("result_ne", datatypes::boolean, {num_elems}),
            _arg_("result_ge", datatypes::boolean, {num_elems}),
            _arg_("result_gt", datatypes::boolean, {num_elems}), ) {
        _bind_(first, second, result_eq, result_le, result_lt, result_ne,
                result_ge, result_gt);
        _for_(i, 0, num_elems) {
            result_eq[i] = first[i] == second[i];
            result_le[i] = first[i] <= second[i];
            result_lt[i] = first[i] < second[i];
            result_ne[i] = first[i] != second[i];
            result_ge[i] = first[i] >= second[i];
            result_gt[i] = first[i] > second[i];
        }
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        const int32_t INT_MIN_ = -2147483647 - 1;
        // first (prev 8 positions) test positive number as : == <= < != >= > ==
        // > second test negative number as : == <= < != >= > == >
        int32_t host_input_first[num_elems]
                = {100, 0, 3000, 5, 6, 8, 0x7fffffff, 0x7ffffff1, -1, INT_MIN_,
                        -2, -3000, -200, -1, INT_MIN_ + 3, INT_MIN_ + 2};
        int32_t host_input_second[num_elems]
                = {100, 0, 4000, 4, 5, 6, 0x7fffffff, 0x7ffffff0, -1, INT_MIN_,
                        -1, -4000, -200, -2, INT_MIN_ + 3, INT_MIN_ + 1};
        bool host_out_result_eq[num_elems] = {false};
        bool host_out_result_le[num_elems] = {false};
        bool host_out_result_lt[num_elems] = {false};
        bool host_out_result_ne[num_elems] = {false};
        bool host_out_result_ge[num_elems] = {false};
        bool host_out_result_gt[num_elems] = {false};
        bool expected_result_eq[num_elems]
                = {true, true, false, false, false, false, true, false, true,
                        true, false, false, true, false, true, false};
        bool expected_result_le[num_elems]
                = {true, true, true, false, false, false, true, false, true,
                        true, true, false, true, false, true, false};
        bool expected_result_lt[num_elems]
                = {false, false, true, false, false, false, false, false, false,
                        false, true, false, false, false, false, false};
        bool expected_result_ne[num_elems]
                = {false, false, true, true, true, true, false, true, false,
                        false, true, true, false, true, false, true};
        bool expected_result_gt[num_elems]
                = {false, false, false, true, true, true, false, true, false,
                        false, false, true, false, true, false, true};
        bool expected_result_ge[num_elems]
                = {true, true, false, true, true, true, true, true, true, true,
                        false, true, true, true, true, true};

        generic_val generic_args[] = {&host_input_first, &host_input_second,
                &host_out_result_eq, &host_out_result_le, &host_out_result_lt,
                &host_out_result_ne, &host_out_result_ge, &host_out_result_gt};

        j_foo->call_generic_default(generic_args);

        for (int i = 0; i < num_elems; ++i) {
            EXPECT_EQ(host_out_result_eq[i], expected_result_eq[i]);
            EXPECT_EQ(host_out_result_lt[i], expected_result_lt[i]);
            EXPECT_EQ(host_out_result_le[i], expected_result_le[i]);
            EXPECT_EQ(host_out_result_ne[i], expected_result_ne[i]);
            EXPECT_EQ(host_out_result_ge[i], expected_result_ge[i]);
            EXPECT_EQ(host_out_result_gt[i], expected_result_gt[i]);
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestCMPExprUint) {
    ir_builder_t builder;

    const int num_elems = 16;

    _function_(datatypes::void_t, foo,
            _arg_("first", datatypes::u32, {num_elems}),
            _arg_("second", datatypes::u32, {num_elems}),
            _arg_("result_eq", datatypes::boolean, {num_elems}),
            _arg_("result_le", datatypes::boolean, {num_elems}),
            _arg_("result_lt", datatypes::boolean, {num_elems}),
            _arg_("result_ne", datatypes::boolean, {num_elems}),
            _arg_("result_ge", datatypes::boolean, {num_elems}),
            _arg_("result_gt", datatypes::boolean, {num_elems}), ) {
        _bind_(first, second, result_eq, result_le, result_lt, result_ne,
                result_ge, result_gt);
        _for_(i, 0, num_elems) {
            result_eq[i] = first[i] == second[i];
            result_le[i] = first[i] <= second[i];
            result_lt[i] = first[i] < second[i];
            result_ne[i] = first[i] != second[i];
            result_ge[i] = first[i] >= second[i];
            result_gt[i] = first[i] > second[i];
        }
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        // fist (prev eight position) == <= < != >= > == >
        // == <= < != >= > == >
        uint32_t host_input_first[num_elems] = {0xfffffff0, 0xfffffff1,
                0xfffffff1, 0xfffffff1, 0xfffffff2, 0xfffffff1, 0, 100011,
                0xffff0000, 0xfff00000, 0xff000000, 0xf0000000, 0xf0000000,
                0xf00000ff, 0xff888888, 0xfff88888};
        uint32_t host_input_second[num_elems] = {0xfffffff0, 0xfffffff1,
                0xfffffff2, 0xfffffff0, 0xfffffff2, 0xfffffff0, 0, 100010,
                0xffff0000, 0xffffffff, 0xfff00000, 0xf000000f, 0xf0000000,
                0xf0000000, 0xff888888, 0xfff88880};
        bool host_out_result_eq[num_elems] = {false};
        bool host_out_result_le[num_elems] = {false};
        bool host_out_result_lt[num_elems] = {false};
        bool host_out_result_ne[num_elems] = {false};
        bool host_out_result_ge[num_elems] = {false};
        bool host_out_result_gt[num_elems] = {false};
        bool expected_result_eq[num_elems]
                = {true, true, false, false, true, false, true, false, true,
                        false, false, false, true, false, true, false};
        bool expected_result_le[num_elems]
                = {true, true, true, false, true, false, true, false, true,
                        true, true, true, true, false, true, false};
        bool expected_result_lt[num_elems]
                = {false, false, true, false, false, false, false, false, false,
                        true, true, true, false, false, false, false};
        bool expected_result_ne[num_elems]
                = {false, false, true, true, false, true, false, true, false,
                        true, true, true, false, true, false, true};
        bool expected_result_gt[num_elems]
                = {false, false, false, true, false, true, false, true, false,
                        false, false, false, false, true, false, true};
        bool expected_result_ge[num_elems]
                = {true, true, false, true, true, true, true, true, true, false,
                        false, false, true, true, true, true};

        generic_val generic_args[] = {&host_input_first, &host_input_second,
                &host_out_result_eq, &host_out_result_le, &host_out_result_lt,
                &host_out_result_ne, &host_out_result_ge, &host_out_result_gt};

        j_foo->call_generic_default(generic_args);

        for (int i = 0; i < num_elems; ++i) {
            EXPECT_EQ(host_out_result_eq[i], expected_result_eq[i]);
            EXPECT_EQ(host_out_result_lt[i], expected_result_lt[i]);
            EXPECT_EQ(host_out_result_le[i], expected_result_le[i]);
            EXPECT_EQ(host_out_result_ne[i], expected_result_ne[i]);
            EXPECT_EQ(host_out_result_ge[i], expected_result_ge[i]);
            EXPECT_EQ(host_out_result_gt[i], expected_result_gt[i]);
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestConstantBroadcast) {
    REQUIRE_AVX2();
    ir_builder_t builder;

    // We'll use over-sized tensors to help us notice errors in the indexing
    // calculations.
    const int num_elems = 1 + DATA_LEN_16 + 1;
    const int num_lanes_i8 = get_lanes(sc_data_etype::U8, DATA_LEN_16);
    const int num_lanes_i16 = get_lanes(sc_data_etype::U16, DATA_LEN_16);
    const int num_lanes_i32 = get_lanes(sc_data_etype::U32, DATA_LEN_16);
    const int num_lanes_f32 = get_lanes(sc_data_etype::F32, DATA_LEN_16);
    const int num_lanes_f16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);

    _function_(datatypes::void_t, foo,
            _arg_("tensor_out", datatypes::f32, {num_elems}),
            _arg_("tensor_int8", datatypes::s8, {num_elems}),
            _arg_("tensor_uint8", datatypes::u8, {num_elems}),
            _arg_("tensor_int32", datatypes::s32, {num_elems}),
            _arg_("tensor_uint32", datatypes::u32, {num_elems}),
            _arg_("tensor_uint16", datatypes::u16, {num_elems}), ) {
        _bind_(tensor_out, tensor_int8, tensor_uint8, tensor_int32,
                tensor_uint32, tensor_uint16);
        union_val vali(UINT64_C(42));
        union_val valf(42.f);
        _for_(i, 0, DATA_LEN_16, num_lanes_i8) {
            tensor_int8[span_t({1 + i}, num_lanes_i8)]
                    = make_expr<constant_node>(
                            vali, sc_data_type_t::s8(num_lanes_i8));
            tensor_uint8[span_t({1 + i}, num_lanes_i8)]
                    = make_expr<constant_node>(
                            vali, sc_data_type_t::u8(num_lanes_i8));
        }
        _for_(i, 0, DATA_LEN_16, num_lanes_i16) {
            tensor_uint16[span_t({1 + i}, num_lanes_i16)]
                    = make_expr<constant_node>(
                            vali, sc_data_type_t::u16(num_lanes_i16));
        }
        _for_(i, 0, DATA_LEN_16, num_lanes_i32) {
            tensor_int32[span_t({1 + i}, num_lanes_i32)]
                    = make_expr<constant_node>(
                            vali, sc_data_type_t::s32(num_lanes_i32));
            tensor_uint32[span_t({1 + i}, num_lanes_i32)]
                    = make_expr<constant_node>(
                            vali, sc_data_type_t::u32(num_lanes_i32));
        }
        _for_(i, 0, DATA_LEN_16, num_lanes_f32) {
            tensor_out[span_t({1 + i}, num_lanes_f32)]
                    = make_expr<constant_node>(
                            valf, sc_data_type_t::f32(num_lanes_f32));
        }
    }

    _function_(datatypes::void_t, foofp16,
            _arg_("tensor_f16", datatypes::f16, {num_elems}), ) {
        _bind_(tensor_f16);
        union_val valf(42.f);
        _for_(i, 0, DATA_LEN_16, num_lanes_f16) {
            tensor_f16[span_t({1 + i}, num_lanes_f16)]
                    = make_expr<constant_node>(
                            valf, sc_data_type_t::f16(num_lanes_f16));
        }
    }
    auto ctx = get_default_context();
    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(ctx);
    ir_mod->add_func({foo});
    if (is_cpu_support_fp16()) { ir_mod->add_func({foofp16}); }
    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        if (is_cpu_support_fp16() && is_cfake(je_name)
                && !is_cfake_support_fp16()) {
            continue;
        }
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");
        shared_ptr<jit_function_t> j_foo_fp16 = jm->get_function("foofp16");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        int8_t host_tensor_int8[num_elems] = {
                -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1};
        const int8_t expected_int8[num_elems] = {-1, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, -1};

        uint8_t host_tensor_uint8[num_elems]
                = {1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1};
        const uint8_t expected_uint8[num_elems] = {1, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 1};

        uint16_t host_tensor_uint16[num_elems]
                = {1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1};
        const uint16_t expected_uint16[num_elems] = {1, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 1};

        int32_t host_tensor_int32[num_elems] = {
                -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1};
        const int32_t expected_int32[num_elems] = {-1, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42, -1};

        uint32_t host_tensor_uint32[num_elems]
                = {1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1};
        const uint32_t expected_uint32[num_elems] = {1, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 1};

        float host_tensor_out[num_elems] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1};
        const float expected_result[num_elems] = {-1, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42, -1};

        fp16_t host_tensor_f16[num_elems] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1};
        const fp16_t expected_result_f16[num_elems] = {-1, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, -1};
        generic_val generic_args[] = {&host_tensor_out, &host_tensor_int8,
                &host_tensor_uint8, &host_tensor_int32, &host_tensor_uint32,
                &host_tensor_uint16};
        generic_val generic_args_fp16[] = {&host_tensor_f16};
        j_foo->call_generic_default(generic_args);
        if (is_cpu_support_fp16()) {
            j_foo_fp16->call_generic_default(generic_args_fp16);
        }
        for (int i = 0; i < num_elems; ++i) {
            EXPECT_EQ(host_tensor_out[i], expected_result[i]);
            EXPECT_EQ(host_tensor_int8[i], expected_int8[i]);
            EXPECT_EQ(host_tensor_uint8[i], expected_uint8[i]);
            EXPECT_EQ(host_tensor_int32[i], expected_int32[i]);
            EXPECT_EQ(host_tensor_uint32[i], expected_uint32[i]);
            EXPECT_EQ(host_tensor_uint16[i], expected_uint16[i]);
            if (is_cpu_support_fp16()) {
                EXPECT_EQ(host_tensor_f16[i], expected_result_f16[i]);
            }
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestIntrinsicBroadcast) {
    REQUIRE_AVX2();
    ir_builder_t builder;

    // We'll use over-sized tensors to help us notice errors in the indexing
    // calculations.
    const int num_elems = 1 + DATA_LEN_16 + 1;
    const int num_lanes_f32 = get_lanes(sc_data_etype::F32, DATA_LEN_16);

    _function_(datatypes::void_t, foo, _arg_("x", datatypes::f32),
            _arg_("tensor_out", datatypes::f32, {num_elems}), ) {
        _bind_(x, tensor_out);
        _for_(i, 0, DATA_LEN_16, num_lanes_f32) {
            tensor_out[span_t({1 + i}, num_lanes_f32)]
                    = make_broadcast(x, num_lanes_f32);
        }
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        float host_x = 42;
        float host_tensor_out[num_elems] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1};
        const float expected_result[num_elems] = {-1, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42, -1};

        generic_val generic_args[] = {
                host_x,
                &host_tensor_out,
        };

        j_foo->call_generic_default(generic_args);

        for (int i = 0; i < num_elems; ++i) {
            EXPECT_EQ(host_tensor_out[i], expected_result[i]);
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestIntrinsicGather) {
    REQUIRE_AVX2();
    ir_builder_t builder;

    // We'll use over-sized tensors to help us notice errors in the indexing
    // calculations.
    const int num_elems = DATA_LEN_16;
    const int num_lanes_f32 = get_lanes(sc_data_etype::F32, DATA_LEN_16);

    _function_(datatypes::void_t, foo,
            _arg_("tensor_out", datatypes::f32, {num_elems}),
            _arg_("tensor_in", datatypes::f32, {num_elems}),
            _arg_("tensor_idx", datatypes::s32, {num_elems})) {
        _bind_(tensor_out, tensor_in, tensor_idx);
        _for_(i, 0, DATA_LEN_16, num_lanes_f32) {
            tensor_out[span_t({0 + i}, num_lanes_f32)] = make_gather(
                    tensor_in, tensor_idx[span_t({0 + i}, num_lanes_f32)]);
        }
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        float host_tensor_out[num_elems] = {
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        float host_tensor_in[num_elems] = DATASET_I1;
        int host_tensor_idx[num_elems] = DATASET_I2;
        host_tensor_idx[0] = 0;
        const float expected_result[num_elems]
                = {1, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2};

        generic_val generic_args[]
                = {&host_tensor_out, &host_tensor_in, &host_tensor_idx};

        j_foo->call_generic_default(generic_args);

        for (int i = 0; i < num_elems; ++i) {
            EXPECT_EQ(host_tensor_out[i], expected_result[i]);
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestIntrinsicpermute2) {
    REQUIRE_AVX512();
    ir_builder_t builder;

    const int simd_lanes = 16;
    const int num_elems = simd_lanes;
    _function_(datatypes::void_t, foo,
            _arg_("tensor_out1", datatypes::u16, {num_elems}),
            _arg_("tensor_out2", datatypes::u16, {num_elems}),
            _arg_("tensor_src1", datatypes::u16, {num_elems}),
            _arg_("tensor_src2", datatypes::u16, {num_elems})) {
        _bind_(tensor_out1, tensor_out2, tensor_src1, tensor_src2);
        tensor_out1[span_t({0}, simd_lanes)]
                = make_permute(tensor_src1[span_t({0}, simd_lanes)],
                        tensor_src2[span_t({0}, simd_lanes)], 0x20, 128);
        tensor_out2[span_t({0}, simd_lanes)]
                = make_permute(tensor_src1[span_t({0}, simd_lanes)],
                        tensor_src2[span_t({0}, simd_lanes)], 0x31, 128);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        uint16_t host_tensor_out1[num_elems] = {0};
        uint16_t host_tensor_out2[num_elems] = {0};
        uint16_t host_tensor_src1[num_elems] = DATASET_I1;
        uint16_t host_tensor_src2[num_elems] = DATASET_I2;
        const uint16_t expected_result1[num_elems]
                = {1, 2, 3, 4, 5, 6, 7, 8, 16, 15, 14, 13, 12, 11, 10, 9};
        const uint16_t expected_result2[num_elems]
                = {9, 10, 11, 12, 13, 14, 15, 16, 8, 7, 6, 5, 4, 3, 2, 1};

        generic_val generic_args[] = {&host_tensor_out1, &host_tensor_out2,
                &host_tensor_src1, &host_tensor_src2};

        j_foo->call_generic_default(generic_args);

        for (int i = 0; i < num_elems; ++i) {
            EXPECT_EQ(host_tensor_out1[i], expected_result1[i]);
            EXPECT_EQ(host_tensor_out2[i], expected_result2[i]);
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestModuleVar) {
    ir_builder_t builder;

    auto ir_mod = std::make_shared<ir_module_t>(get_default_context());

    _module_var_(ir_mod, x, datatypes::f32, expr());

    _function_(datatypes::void_t, foo_set_x, _arg_("new_x", datatypes::f32)) {
        _bind_(new_x);
        x = new_x;
    }

    _function_(
            datatypes::void_t, foo_get_x, _arg_("out_x", datatypes::f32, {1})) {
        _bind_(out_x);
        out_x[0] = x;
    }

    ir_mod->add_func({foo_set_x, foo_get_x});

    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo_set_x = jm->get_function("foo_set_x");
        EXPECT_NE(j_foo_set_x, nullptr);
        if (!j_foo_set_x) { continue; }

        shared_ptr<jit_function_t> j_foo_get_x = jm->get_function("foo_get_x");
        EXPECT_NE(j_foo_get_x, nullptr);
        if (!j_foo_get_x) { continue; }

        const float host_new_x1 = 0;
        const float host_new_x2 = 42;
        float host_out_x;

        generic_val generic_args_set1[] = {
                host_new_x1,
        };

        generic_val generic_args_set2[] = {
                host_new_x2,
        };

        generic_val generic_args_get[] = {
                &host_out_x,
        };

        j_foo_set_x->call_generic_default(generic_args_set1);
        j_foo_get_x->call_generic_default(generic_args_get);
        EXPECT_EQ(host_new_x1, host_out_x);

        j_foo_set_x->call_generic_default(generic_args_set2);
        j_foo_get_x->call_generic_default(generic_args_get);
        EXPECT_EQ(host_new_x2, host_out_x);
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestSubtract8Args) {
    ir_builder_t builder;

    _function_(datatypes::void_t, foo, _arg_("x1", datatypes::index),
            _arg_("x2", datatypes::index), _arg_("x3", datatypes::index),
            _arg_("x4", datatypes::index), _arg_("x5", datatypes::index),
            _arg_("x6", datatypes::index), _arg_("x7", datatypes::index),
            _arg_("result", datatypes::index, {1})) {
        _bind_(x1, x2, x3, x4, x5, x6, x7, result);
        result[0] = x1 - x2 - x3 - x4 - x5 - x6 - x7;
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    int64_t expected_result = 1UL;
    int64_t result;

    generic_val generic_args[] = {UINT64_C(22), UINT64_C(1), UINT64_C(2),
            UINT64_C(3), UINT64_C(4), UINT64_C(5), UINT64_C(6), &result};

    for (auto &kv : get_engines()) {
        const string &je_name = kv.first;

        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> jf = jm->get_function("foo");

        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        result = 0;
        jf->call_generic_default(generic_args);

        ASSERT_EQ(result, expected_result);
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestIntrinsicReduceAdd) {
    REQUIRE_AVX2();
    ir_builder_t builder;

    // We'll use over-sized tensors to help us notice errors in the indexing
    // calculations.
    const int num_elems = 1 + DATA_LEN_16 + 1;
    const int num_lanes_i32 = get_lanes(sc_data_etype::U32, DATA_LEN_16);
    const int num_lanes_f32 = get_lanes(sc_data_etype::F32, DATA_LEN_16);
    const int num_lanes_f16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
    _function_(datatypes::void_t, foo,
            _arg_("tensor_f32_in", datatypes::f32, {num_elems}),
            _arg_("out_f32", datatypes::f32, {1}),
            _arg_("tensor_s32_in", datatypes::s32, {num_elems}),
            _arg_("out_s32", datatypes::s32, {1}), ) {
        _bind_(tensor_f32_in, out_f32, tensor_s32_in, out_s32);

        _var_init_(
                out_var_s32, sc_data_type_t::s32(1), make_constant(INT32_C(0)));
        _for_(i, 0, DATA_LEN_16, num_lanes_i32) {
            _var_(local_temp_int32, sc_data_type_t::s32(num_lanes_i32));
            local_temp_int32 = tensor_s32_in[span_t({1 + i}, num_lanes_i32)];
            out_var_s32 = out_var_s32 + make_reduce_add(local_temp_int32);
        }
        out_s32[0] = out_var_s32;

        _var_init_(out_var_f32, sc_data_type_t::f32(1), make_constant(0.f));
        _for_(i, 0, DATA_LEN_16, num_lanes_f32) {
            _var_(local_temp, sc_data_type_t::f32(num_lanes_f32));
            local_temp = tensor_f32_in[span_t({1 + i}, num_lanes_f32)];
            out_var_f32 = out_var_f32 + make_reduce_add(local_temp);
        }
        out_f32[0] = out_var_f32;
    }

    _function_(datatypes::void_t, foofp16,
            _arg_("tensor_f16_in", datatypes::f16, {num_elems}),
            _arg_("out_f16", datatypes::f16, {1}), ) {
        _bind_(tensor_f16_in, out_f16);
        _var_init_(out_var_f16, sc_data_type_t::f16(1),
                make_constant({0.f}, datatypes::f16));
        _for_(i, 0, DATA_LEN_16, num_lanes_f16) {
            _var_(local_temp, sc_data_type_t::f16(num_lanes_f16));
            local_temp = tensor_f16_in[span_t({1 + i}, num_lanes_f16)];
            out_var_f16 = out_var_f16 + make_reduce_add(local_temp);
        }
        out_f16[0] = out_var_f16;
    }

    auto ctx = get_default_context();
    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(ctx);
    ir_mod->add_func({foo});
    if (is_cpu_support_fp16()) { ir_mod->add_func({foofp16}); }
    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        if (is_cpu_support_fp16() && is_cfake(je_name)
                && !is_cfake_support_fp16()) {
            continue;
        }
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");
        shared_ptr<jit_function_t> j_foo_fp16 = jm->get_function("foofp16");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        float host_in[num_elems] = {
                -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1};
        float host_out = 0;
        const float expected_out = 136;
        int32_t host_int32_in[num_elems] = {
                -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1};
        int32_t host_int32_out = 0;
        const float expected_int32_out = 136;

        fp16_t host_f16_in[num_elems] = {
                -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1};
        fp16_t host_f16_out = 0;
        const fp16_t expected_f16_out = 136;

        generic_val generic_args[] = {
                host_in,
                &host_out,
                host_int32_in,
                &host_int32_out,
        };
        generic_val generic_args_fp16[] = {
                host_f16_in,
                &host_f16_out,
        };

        j_foo->call_generic_default(generic_args);
        if (is_cpu_support_fp16()) {
            j_foo_fp16->call_generic_default(generic_args_fp16);
        }

        EXPECT_EQ(host_out, expected_out);
        EXPECT_EQ(host_int32_out, expected_int32_out);
        if (is_cpu_support_fp16()) {
            EXPECT_EQ(host_f16_out, expected_f16_out);
        }
    }
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestUnpackHighLow) {
    REQUIRE_AVX2();
    const int array_len = 32;
    ir_builder_t builder;
    const int data_len = 8;
    const int data_len_1 = 16;

    _function_(datatypes::void_t, foo, _arg_("x", datatypes::bf16, {data_len}),
            _arg_("y", datatypes::bf16, {data_len}),
            _arg_("k", datatypes::bf16, {data_len}),
            _arg_("z", datatypes::bf16, {data_len}),
            _arg_("result", datatypes::bf16, {array_len}),
            _arg_("x_1", datatypes::u8, {data_len_1}),
            _arg_("y_1", datatypes::u8, {data_len_1}),
            _arg_("result_1", datatypes::u8, {array_len})) {
        _bind_(x, y, k, z, result, x_1, y_1, result_1);
        result[span_t({0}, data_len)] = builder::make_unpack_low(
                x[span_t({0}, data_len)], y[span_t({0}, data_len)], 16);
        result[span_t({8}, data_len)] = builder::make_unpack_high(
                x[span_t({0}, data_len)], y[span_t({0}, data_len)], 16);
        result[span_t({16}, data_len)] = builder::make_unpack_low(
                k[span_t({0}, data_len)], z[span_t({0}, data_len)], 16);
        result[span_t({24}, data_len)] = builder::make_unpack_high(
                k[span_t({0}, data_len)], z[span_t({0}, data_len)], 16);
        result_1[span_t({0}, data_len_1)] = builder::make_unpack_low(
                x_1[span_t({0}, data_len_1)], y_1[span_t({0}, data_len_1)], 8);
        result_1[span_t({16}, data_len_1)] = builder::make_unpack_high(
                x_1[span_t({0}, data_len_1)], y_1[span_t({0}, data_len_1)], 8);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    uint16_t x[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    uint16_t y[8] = {2, 4, 6, 8, 10, 12, 14, 16};
    uint16_t k[8] = {17, 19, 21, 23, 25, 27, 29, 31};
    uint16_t z[8] = {18, 20, 22, 24, 26, 28, 30, 32};
    uint8_t x_1[16]
            = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    uint8_t y_1[16]
            = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    uint16_t result[array_len] = {0};
    uint16_t expected_result[array_len] = {0};
    uint8_t result_1[array_len] = {0};
    for (int i = 0; i < array_len; i++) {
        expected_result[i] = i + 1;
    }

    generic_val generic_args[] = {x, y, k, z, &result, x_1, y_1, &result_1};

    for (auto &kv : get_engines()) {
        const string &je_name = kv.first;

        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> jf = jm->get_function("foo");

        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        jf->call_generic_default(generic_args);
        for (int i = 0; i < array_len; i++) {
            EXPECT_EQ(result[i], expected_result[i]);
            EXPECT_EQ(result_1[i], expected_result[i]);
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestIntrinsicPermutexvar) {
    REQUIRE_AVX512VBMI();
    ir_builder_t builder;

    any_map_t reinterpret_attr;
    expr idx0;
#define MAKE_IDX_NEW(name, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, \
        v12, v13, v14, v15) \
    idx##name = make_expr<intrin_call_node>(intrin_type::reinterpret, \
            std::vector<expr> {make_expr<constant_node>( \
                    std::vector<union_val> {UINT64_C(v0), UINT64_C(v1), \
                            UINT64_C(v2), UINT64_C(v3), UINT64_C(v4), \
                            UINT64_C(v5), UINT64_C(v6), UINT64_C(v7), \
                            UINT64_C(v8), UINT64_C(v9), UINT64_C(v10), \
                            UINT64_C(v11), UINT64_C(v12), UINT64_C(v13), \
                            UINT64_C(v14), UINT64_C(v15)}, \
                    sc_data_type_t::u32(16))}, \
            reinterpret_attr);
    const int simd_lanes = 64;

    _function_(datatypes::void_t, foo,
            _arg_("tensor_in", datatypes::u8, {simd_lanes}),
            _arg_("tensor_in_1", datatypes::u8, {simd_lanes / 2}), ) {
        _bind_(tensor_in, tensor_in_1);
        _var_(local_temp, sc_data_type_t::u8(64));
        _var_(local_temp_1, sc_data_type_t::u8(32));
        local_temp = tensor_in[span_t({0}, simd_lanes)];
        local_temp_1 = tensor_in_1[span_t({0}, simd_lanes / 2)];
        reinterpret_attr[intrin_attr::out_dtype] = sc_data_type_t::u8(64);
        MAKE_IDX_NEW(0, 0x30201000, 0x31211101, 0x32221202, 0x33231303,
                0x34241404, 0x35251505, 0x36261606, 0x37271707, 0x38281808,
                0x39291909, 0x3a2a1a0a, 0x3b2b1b0b, 0x3c2c1c0c, 0x3d2d1d0d,
                0x3e2e1e0e, 0x3f2f1f0f)
        tensor_in[span_t({0}, simd_lanes)] = make_permutexvar(idx0, local_temp);
        tensor_in_1[span_t({0}, simd_lanes / 2)] = make_permutexvar(
                builder::make_constant(UINT64_C(0b11011000)), local_temp_1, 8);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }
        int rows = 4, cols = 16;
        int init = 1;
        uint8_t host_in[simd_lanes] = {0};
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                host_in[i * cols + j] = init++;
            }
        }
        uint8_t host_in_1[simd_lanes / 2] = {0};
        init = 1;
        for (int i = 0; i < simd_lanes / 2; i++) {
            host_in_1[i] = init++;
        }
        uint8_t expected_out[simd_lanes] = {1, 17, 33, 49, 2, 18, 34, 50, 3, 19,
                35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39,
                55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43,
                59, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47,
                63, 16, 32, 48, 64};
        uint8_t expected_out_1[simd_lanes / 2] = {1, 2, 3, 4, 5, 6, 7, 8, 17,
                18, 19, 20, 21, 22, 23, 24, 9, 10, 11, 12, 13, 14, 15, 16, 25,
                26, 27, 28, 29, 30, 31, 32};

        generic_val generic_args[] = {&host_in, &host_in_1};

        j_foo->call_generic_default(generic_args);
        for (int i = 0; i < simd_lanes; i++) {
            EXPECT_EQ((int)host_in[i], (int)expected_out[i]);
            if (i < simd_lanes / 2) {
                EXPECT_EQ((int)host_in_1[i], (int)expected_out_1[i]);
            }
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestIntrinsicInsert) {
    REQUIRE_AVX512();
    ir_builder_t builder;
    const int simd_lanes = 32;
#define INSERT_CMP_DATA(type, imm) \
    type host_in_1[32] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
            16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; \
    type host_in_2[16] \
            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}; \
    const type expected_out_1[32] \
            = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, \
                    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}; \
    const type expected_out_0[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, \
            13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; \
    generic_val generic_args[] = { \
            &host_in_1, \
            &host_in_2, \
    }; \
    j_foo->call_generic_default(generic_args); \
    for (int i = 0; i < simd_lanes; i++) { \
        if ((imm)&1) { \
            EXPECT_EQ(host_in_1[i], expected_out_1[i]); \
        } else { \
            EXPECT_EQ(host_in_1[i], expected_out_0[i]); \
        } \
    }

    auto test_func = [&](const sc_data_type_t &dtype, const int imm) {
        _function_(datatypes::void_t, foo,
                _arg_("tensor_in_1", dtype, {simd_lanes}),
                _arg_("tensor_in_2", dtype, {simd_lanes / 2}), ) {
            _bind_(tensor_in_1, tensor_in_2);
            _var_(local_temp_1, sc_data_type_t(dtype.type_code_, simd_lanes));
            local_temp_1->attr()["can_promote_to_f32"] = false;
            _var_(local_temp_2,
                    sc_data_type_t(dtype.type_code_, simd_lanes / 2));
            local_temp_2->attr()["can_promote_to_f32"] = false;
            local_temp_1 = tensor_in_1[span_t({0}, simd_lanes)];
            local_temp_2 = tensor_in_2[span_t({0}, simd_lanes / 2)];
            tensor_in_1[span_t({0}, simd_lanes)]
                    = make_insert(local_temp_1, local_temp_2, imm);
        }
        for (auto &kv : get_engines()) {
            ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
                    get_default_context(), vector<func_t> {foo}, 0);
            ostringstream err_context;
            const string &je_name = kv.first;
            err_context << "jit_engine_t class '" << je_name << "'";
            SCOPED_TRACE(err_context.str());
            shared_ptr<jit_engine_t> je = kv.second;
            EXPECT_NE(je, nullptr);
            if (!je) { continue; }
            shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
            EXPECT_NE(jm, nullptr);
            shared_ptr<jit_function_t> j_foo = jm->get_function("foo");
            EXPECT_NE(j_foo, nullptr);
            if (!j_foo) { continue; }
            switch (dtype.as_etype()) {
                case sc_data_etype::U8: {
                    INSERT_CMP_DATA(uint8_t, imm);
                } break;
                case sc_data_etype::S8: {
                    INSERT_CMP_DATA(int8_t, imm);
                } break;
                case sc_data_etype::U16: {
                    INSERT_CMP_DATA(uint16_t, imm);
                } break;
                case sc_data_etype::BF16: {
                    INSERT_CMP_DATA(uint16_t, imm);
                } break;
                default: {
                    assert(0 && "Do not support this type.");
                } break;
            }
        }
    };
    test_func(datatypes::s8, 1); // insert imm = 1
    test_func(datatypes::s8, 0); // insert imm = 0
    test_func(datatypes::u8, 1); // insert imm = 1
    test_func(datatypes::u8, 0); // insert imm = 0
    test_func(datatypes::bf16, 1); // insert imm = 1
    test_func(datatypes::bf16, 1); // insert imm = 0
}

TEST(GCCore_CPU_jit_engine_equivalence, TestIntrinsicExtractAVX2) {
    REQUIRE_AVX2();
    ir_builder_t builder;
    const int simd_lanes = 8;
#define EXTRACT_CMP_DATA(type, imm) \
    type host_in_1[8] = {1, 2, 3, 4, 5, 6, 7, 8}; \
    type host_in_2[8] = {0}; \
    const type expected_out_1 = 1; \
    const type expected_out_2 = 2; \
    generic_val generic_args[] = { \
            &host_in_1, \
            &host_in_2, \
    }; \
    j_foo->call_generic_default(generic_args); \
    for (int i = 0; i < simd_lanes; i++) { \
        if ((imm)&1) { \
            EXPECT_EQ(host_in_2[i], expected_out_2); \
        } else { \
            EXPECT_EQ(host_in_2[i], expected_out_1); \
        } \
    }

    auto test_func = [&](sc_data_type_t dtype, const int imm) {
        _function_(datatypes::void_t, foo,
                _arg_("tensor_in_1", dtype, {simd_lanes}),
                _arg_("tensor_in_2", dtype, {simd_lanes})) {
            _bind_(tensor_in_1, tensor_in_2);
            _var_(local_temp_1, sc_data_type_t(dtype.type_code_, simd_lanes));
            local_temp_1->attr()["can_promote_to_f32"] = false;
            local_temp_1 = tensor_in_1[span_t({0}, simd_lanes)];
            _for_(idx, 0, simd_lanes) {
                tensor_in_2[span_t({idx}, 1)] = make_extract(local_temp_1, imm);
            }
        }
        for (auto &kv : get_engines()) {
            ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
                    get_default_context(), vector<func_t> {foo}, 0);
            ostringstream err_context;
            const string &je_name = kv.first;
            err_context << "jit_engine_t class '" << je_name << "'";
            SCOPED_TRACE(err_context.str());
            shared_ptr<jit_engine_t> je = kv.second;
            EXPECT_NE(je, nullptr);
            if (!je) { continue; }
            shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
            EXPECT_NE(jm, nullptr);
            shared_ptr<jit_function_t> j_foo = jm->get_function("foo");
            EXPECT_NE(j_foo, nullptr);
            if (!j_foo) { continue; }
            switch (dtype.as_etype()) {
                case sc_data_etype::U8: {
                    EXTRACT_CMP_DATA(uint8_t, imm);
                } break;
                case sc_data_etype::S8: {
                    EXTRACT_CMP_DATA(int8_t, imm);
                } break;
                case sc_data_etype::U16: {
                    EXTRACT_CMP_DATA(uint16_t, imm);
                } break;
                case sc_data_etype::BF16: {
                    EXTRACT_CMP_DATA(uint16_t, imm);
                } break;
                default: {
                    assert(0 && "Do not support this type.");
                } break;
            }
        }
    };

    test_func(datatypes::u8, 1); // extract imm = 1
    test_func(datatypes::u8, 0); // extract imm = 0
    test_func(datatypes::u16, 1); // extract imm = 1
    test_func(datatypes::u16, 0); // extract imm = 0
    test_func(datatypes::bf16, 1); // extract imm = 1
    test_func(datatypes::bf16, 0); // extract imm = 0
}

TEST(GCCore_CPU_jit_engine_equivalence, TestIntrinsicInsertAVX2) {
    REQUIRE_AVX2();
    ir_builder_t builder;
    const int simd_lanes = 8;
#define INSERT_AVX2_CMP_DATA(type) \
    type x = 1; \
    type expected_result[8] = {0, 1, 2, 2, 2, 2, 0, 0}; \
    type result[8] = {0}; \
    generic_val generic_args[] = {x, &result}; \
    jf->call_generic_default(generic_args); \
    for (int i = 0; i < 8; i++) { \
        ASSERT_EQ(result[i], expected_result[i]); \
    }

    auto test_func = [&](const sc_data_type_t &dtype) {
        _function_(datatypes::void_t, foo, _arg_("x", dtype),
                _arg_("result", dtype, {simd_lanes})) {
            _bind_(x, result);
            result[span_t({0}, simd_lanes)] = builder::make_insert(
                    result[span_t({0}, simd_lanes)], x, 1);
            x = x + builder::make_cast(dtype, 1);
            result[span_t({0}, simd_lanes)] = builder::make_insert(
                    result[span_t({0}, simd_lanes)], x, 2);
            result[span_t({0}, simd_lanes)] = builder::make_insert(
                    result[span_t({0}, simd_lanes)], x, 3);
            result[span_t({0}, simd_lanes)] = builder::make_insert(
                    result[span_t({0}, simd_lanes)], x, 4);
            result[span_t({0}, simd_lanes)] = builder::make_insert(
                    result[span_t({0}, simd_lanes)], x, 5);
        }

        ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
                get_default_context(), vector<func_t> {foo}, 0);

        for (auto &kv : get_engines()) {
            const string &je_name = kv.first;
            if (je_name != "xbyak_jit") continue;

            ostringstream err_context;
            err_context << "jit_engine_t class '" << je_name << "'";
            SCOPED_TRACE(err_context.str());

            shared_ptr<jit_engine_t> je = kv.second;
            EXPECT_NE(je, nullptr);
            if (!je) { continue; }

            shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
            EXPECT_NE(jm, nullptr);

            shared_ptr<jit_function_t> jf = jm->get_function("foo");

            EXPECT_NE(jf, nullptr);
            if (!jf) { continue; }
            switch (dtype.as_etype()) {
                case sc_data_etype::U8: {
                    INSERT_AVX2_CMP_DATA(uint8_t);
                } break;
                case sc_data_etype::S8: {
                    INSERT_AVX2_CMP_DATA(int8_t);
                } break;
                case sc_data_etype::U16: {
                    INSERT_AVX2_CMP_DATA(uint16_t);
                } break;
                case sc_data_etype::BF16: {
                    INSERT_AVX2_CMP_DATA(uint16_t);
                } break;
                default: {
                    assert(0 && "Do not support this type.");
                } break;
            }
        }
    };
    test_func(datatypes::s8);
    test_func(datatypes::u8);
    test_func(datatypes::u16);
}

TEST(GCCore_CPU_jit_engine_equivalence, TestConstantBF16) {
    REQUIRE_AVX512();
    ir_builder_t builder;

    // We'll use over-sized tensors to help us notice errors in the indexing
    // calculations.
    const int simd_lanes = 16;
    const int num_elems = simd_lanes;

    auto make_constf = [](std::initializer_list<float> v, sc_data_type_t type) {
        std::vector<union_val> val;
        for (auto i : v) {
            val.push_back(i);
        }
        return make_expr<constant_node>(val, type);
    };

    const float v1[num_elems] = DATASET_F2;
    const float v2[num_elems] = DATASET_F1;

    _function_(datatypes::void_t, foo,
            _arg_("tensor_in", datatypes::f32, {num_elems}),
            _arg_("tensor_out", datatypes::f32, {num_elems}),
            _arg_("single_out", datatypes::f32, {num_elems}), ) {
        _bind_(tensor_in, tensor_out, single_out);

        _var_(local_temp, sc_data_type_t::bf16(simd_lanes));
        local_temp = make_constf(DATASET_F2, sc_data_type_t::bf16(simd_lanes));
        local_temp = local_temp
                + make_cast(sc_data_type_t::bf16(simd_lanes),
                        tensor_in[span_t({0}, simd_lanes)]);
        local_temp = local_temp
                + make_constf(DATASET_F1, sc_data_type_t::bf16(simd_lanes));
        tensor_out[span_t({0}, simd_lanes)]
                = make_cast(sc_data_type_t::f32(simd_lanes), local_temp);

        _var_(single_tmp, datatypes::bf16);
        for (int i = 0; i < simd_lanes; i++) {
            single_tmp = make_expr<constant_node>(v1[i], datatypes::bf16);
            single_tmp = single_tmp + make_cast(datatypes::bf16, tensor_in[i]);
            single_tmp = single_tmp
                    + make_expr<constant_node>(v2[i], datatypes::bf16);
            single_out[i] = make_cast(datatypes::f32, single_tmp);
        }
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        float host_tensor_in[num_elems] = DATASET_F3;
        float host_tensor_out[num_elems] = {
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        float host_tensor_out_single[num_elems] = {
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        const float expected_result[num_elems] = {-46.815f, -15.5625f, 1.2525f,
                8.705f, 13.32f, 15.035f, 16.387f, 19.3125f, 19.175f, 19.294f,
                20.325f, 22.025f, 26.225f, 33.7406f, 50.1562f, 81.4562f};

        generic_val generic_args[] = {
                &host_tensor_in,
                &host_tensor_out,
                &host_tensor_out_single,
        };

        j_foo->call_generic_default(generic_args);

        for (int i = 0; i < num_elems; ++i) {
            EXPECT_NEAR(host_tensor_out[i], expected_result[i],
                    std::abs(1e-2 * expected_result[i]));
            EXPECT_NEAR(host_tensor_out_single[i], expected_result[i],
                    std::abs(1e-2 * expected_result[i]));
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestConstantFP16) {
    REQUIRE_AVX512FP16();
    ir_builder_t builder;

    // We'll use over-sized tensors to help us notice errors in the indexing
    // calculations.
    const int simd_lanes = 16;
    const int num_elems = simd_lanes;

    auto make_constf = [](std::initializer_list<float> v, sc_data_type_t type) {
        std::vector<union_val> val;
        for (auto i : v) {
            val.push_back(i);
        }
        return make_expr<constant_node>(val, type);
    };

    const float v1[num_elems] = DATASET_F2;
    const float v2[num_elems] = DATASET_F1;

    _function_(datatypes::void_t, foo,
            _arg_("tensor_in", datatypes::f32, {num_elems}),
            _arg_("tensor_out", datatypes::f32, {num_elems}),
            _arg_("single_out", datatypes::f32, {num_elems}), ) {
        _bind_(tensor_in, tensor_out, single_out);

        _var_(local_temp, sc_data_type_t::f16(simd_lanes));
        local_temp = make_constf(DATASET_F2, sc_data_type_t::f16(simd_lanes));
        local_temp = local_temp
                + make_cast(sc_data_type_t::f16(simd_lanes),
                        tensor_in[span_t({0}, simd_lanes)]);
        local_temp = local_temp
                + make_constf(DATASET_F1, sc_data_type_t::f16(simd_lanes));
        tensor_out[span_t({0}, simd_lanes)]
                = make_cast(sc_data_type_t::f32(simd_lanes), local_temp);

        _var_(single_tmp, datatypes::f16);
        for (int i = 0; i < simd_lanes; i++) {
            single_tmp = make_expr<constant_node>(v1[i], datatypes::f16);
            single_tmp = single_tmp + make_cast(datatypes::f16, tensor_in[i]);
            single_tmp = single_tmp
                    + make_expr<constant_node>(v2[i], datatypes::f16);
            single_out[i] = make_cast(datatypes::f32, single_tmp);
        }
    }
    auto ctx = get_default_context();
    ir_module_ptr ir_mod
            = std::make_shared<ir_module_t>(ctx, vector<func_t> {foo}, 0);

    for (auto &kv : get_engines()) {
        ostringstream err_context;
        const string &je_name = kv.first;
        if (is_cpu_support_fp16() && is_cfake(je_name)
                && !is_cfake_support_fp16()) {
            continue;
        }
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        float host_tensor_in[num_elems] = DATASET_F3;
        float host_tensor_out[num_elems] = {
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        float host_tensor_out_single[num_elems] = {
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        const float expected_result[num_elems] = {-46.815f, -15.5625f, 1.2525f,
                8.705f, 13.32f, 15.035f, 16.387f, 19.3125f, 19.175f, 19.294f,
                20.325f, 22.025f, 26.225f, 33.7406f, 50.1562f, 81.4562f};

        generic_val generic_args[] = {
                &host_tensor_in,
                &host_tensor_out,
                &host_tensor_out_single,
        };

        j_foo->call_generic_default(generic_args);

        for (int i = 0; i < num_elems; ++i) {
            EXPECT_NEAR(host_tensor_out[i], (expected_result[i]),
                    std::abs(1e-1 * (expected_result[i])));
            EXPECT_NEAR(host_tensor_out_single[i], expected_result[i],
                    std::abs(1e-1 * expected_result[i]));
        }
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestConstDivModMul) {
    ir_builder_t builder;

    _function_(datatypes::void_t, foo, _arg_("x", datatypes::index),
            _arg_("y", datatypes::index),
            _arg_("result", datatypes::index, {1})) {
        _bind_(x, y, result);
        result[0] = ((x % UINT64_C(16)) + (y / UINT64_C(64))) * UINT64_C(32);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    uint64_t x = 431;
    uint64_t y = 975;
    uint64_t expected_result = 960;
    uint64_t result;

    generic_val generic_args[] = {x, y, &result};

    for (auto &kv : get_engines()) {
        const string &je_name = kv.first;

        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> jf = jm->get_function("foo");

        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        result = 0;
        jf->call_generic_default(generic_args);

        ASSERT_EQ(result, expected_result);
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestConstExceed32bit) {
    ir_builder_t builder;

    _function_(datatypes::void_t, foo, _arg_("result", datatypes::index, {1})) {
        _bind_(result);
        result[0] = UINT64_C(0x1FFFFFFFFF);
        _var_init_(x, datatypes::index, UINT64_C(0xFFFFFFFFF));
        result[0] = x;
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    uint64_t expected_result = 0xFFFFFFFFF;
    uint64_t result;

    generic_val generic_args[] = {&result};

    for (auto &kv : get_engines()) {
        const string &je_name = kv.first;

        ostringstream err_context;
        err_context << "jit_engine_t class '" << je_name << "'";
        SCOPED_TRACE(err_context.str());

        shared_ptr<jit_engine_t> je = kv.second;
        EXPECT_NE(je, nullptr);
        if (!je) { continue; }

        shared_ptr<jit_module> jm = je->make_jit_module(ir_mod, true);
        EXPECT_NE(jm, nullptr);

        shared_ptr<jit_function_t> jf = jm->get_function("foo");

        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        result = 0;
        jf->call_generic_default(generic_args);

        ASSERT_EQ(result, expected_result);
    }
}

TEST(GCCore_CPU_jit_engine_equivalence, TestJITCondition) {
    REQUIRE_AVX2()
    builder::ir_builder_t builder;
    auto is_avx512 = test_cpu_flags().fAVX512F;
#define TEST_FUNC( \
        test_type, test_step, test_dtype, test_const_type, test_func_name) \
    auto test_func_name = [&](const int step, sc_data_type_t dtype_sc, \
                                  sc_data_type_t sc_const_type) { \
        _function_(datatypes::void_t, aaa, _arg_("A", dtype_sc, {1024}), \
                _arg_("B", dtype_sc, {1024}), _arg_("C", dtype_sc, {1024}), \
                _arg_("D", dtype_sc, {1024}), _arg_("E", dtype_sc, {1024}), \
                _arg_("F", dtype_sc, {1024})) { \
            _bind_(A, B, C, D, E, F); \
            _for_(i, 0, 1024, step) { \
                E[span_t({i}, step)] = builder::make_select( \
                        C[span_t({i}, step)] > D[span_t({i}, step)], \
                        A[span_t({i}, step)], B[span_t({i}, step)]); \
                F[span_t({i}, step)] = builder::make_select( \
                        builder::make_constant( \
                                {UINT64_C(0x03)}, sc_const_type), \
                        A[span_t({i}, step)], B[span_t({i}, step)]); \
            } \
        } \
        auto ctx = get_default_context(); \
        for (auto &kv : get_engines()) { \
            const string &je_name = kv.first; \
            ostringstream err_context; \
            err_context << "jit_engine_t class '" << je_name << "'"; \
            if (is_cfake(je_name) && !use_cfake) { continue; } \
            if (is_cpu_support_fp16() && is_cfake(je_name) \
                    && !is_cfake_support_fp16()) { \
                continue; \
            } \
            shared_ptr<jit_engine_t> engine = kv.second; \
            SCOPED_TRACE(err_context.str()); \
            auto fptr = engine->get_entry_func( \
                    ir_module_t::from_entry_func(ctx, aaa)); \
            ASSERT_TRUE(fptr); \
            auto getC = []() { \
                std::vector<test_type> A(2048); \
                for (int i = 0; i < (int)A.size(); i++) { \
                    A[i] = i % 2; \
                } \
                return A; \
            }; \
            auto getD = [&]() { \
                std::vector<test_type> A(2048); \
                for (int i = 0; i < (int)A.size(); i++) { \
                    A[i] = 2 * i % step; \
                } \
                return A; \
            }; \
            auto getA = []() { \
                std::vector<test_type> A(2048); \
                for (int i = 0; i < (int)A.size(); i++) { \
                    A[i] = 2 * i + 100; \
                } \
                return A; \
            }; \
            auto getB = []() { \
                std::vector<test_type> A(2048); \
                for (int i = 0; i < (int)A.size(); i++) { \
                    A[i] = 2 * i - 100; \
                } \
                return A; \
            }; \
            std::vector<test_type> E(1024); \
            std::vector<test_type> F(1024); \
            auto A = getA(); \
            auto B = getB(); \
            auto C = getC(); \
            auto D = getD(); \
            fptr->call<void>(A.data(), B.data(), C.data(), D.data(), E.data(), \
                    F.data()); \
            for (int i = 0; i < 1024; i++) { \
                auto expected_e = C[i] > D[i] ? A[i] : B[i]; \
                auto expected_f = (i % step) < 2 ? A[i] : B[i]; \
                EXPECT_NEAR(E[i], expected_e, 1e-5); \
                EXPECT_NEAR(F[i], expected_f, 1e-5); \
            } \
        } \
    }; \
    test_func_name(test_step, test_dtype, test_const_type);
    if (is_avx512) {
        TEST_FUNC(float, 16, datatypes::f32, datatypes::u16, test_floatx16)
        TEST_FUNC(int32_t, 16, datatypes::s32, datatypes::u16, test_s32x16)
        TEST_FUNC(uint32_t, 16, datatypes::u32, datatypes::u16, test_u32x16)
        TEST_FUNC(uint16_t, 32, datatypes::u16, datatypes::u32, test_u16x32)
        TEST_FUNC(uint8_t, 64, datatypes::u8, datatypes::index, test_u8x64)
        TEST_FUNC(int8_t, 64, datatypes::s8, datatypes::index, test_s8x64)
    }
    // todo our cfake need avx2 refactor
    if (!is_avx512) { use_cfake = false; }
    TEST_FUNC(float, 8, datatypes::f32, datatypes::u8, test_floatx8)
    TEST_FUNC(int32_t, 8, datatypes::s32, datatypes::u8, test_s32x8)
    TEST_FUNC(uint32_t, 8, datatypes::u32, datatypes::u8, test_u32x8)
    TEST_FUNC(uint16_t, 8, datatypes::u16, datatypes::u8, test_u16x8)
    TEST_FUNC(uint16_t, 16, datatypes::u16, datatypes::u16, test_u16x16)
    TEST_FUNC(uint8_t, 8, datatypes::u8, datatypes::u8, test_u8x8)
    TEST_FUNC(uint8_t, 16, datatypes::u8, datatypes::u16, test_u8x16)
    TEST_FUNC(uint8_t, 32, datatypes::u8, datatypes::u32, test_u8x32)
    TEST_FUNC(int8_t, 8, datatypes::s8, datatypes::u8, test_s8x8)
    TEST_FUNC(int8_t, 16, datatypes::s8, datatypes::u16, test_s8x16)
    TEST_FUNC(int8_t, 32, datatypes::s8, datatypes::u32, test_s8x32)
    if (!is_avx512) { use_cfake = true; }
    if (is_cpu_support_fp16()) {
        TEST_FUNC(fp16_t, 8, datatypes::f16, datatypes::u8, test_f16x8)
        TEST_FUNC(fp16_t, 16, datatypes::f16, datatypes::u16, test_f16x16)
        TEST_FUNC(fp16_t, 32, datatypes::f16, datatypes::u32, test_f16x32)
    }
}

/// ===================================
/// Test Group 2: oprations & data type
/// ===================================

TEST(GCCore_CPU_test_jit_engine_equivalence, TestOpCast) {
    REQUIRE_AVX512();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_16);
    //-----------------------------
    // Cast to datatypes::s8
    //-----------------------------
#define REF_CAST_TO_S8(IN, LANES, I) (static_cast<int8_t>(IN[0][I]))
    // data_type: sint_32
    TEST_OP(UNARY, EXACT, int32_t, int8_t, datatypes::s32, datatypes::s8,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_S8,
            MAKE_CAST(datatypes::s8), DATASET_I3);
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, int8_t, datatypes::f32, datatypes::s8,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_S8,
            MAKE_CAST(datatypes::s8), DATASET_F3);
#undef REF_CAST_TO_S8
    //-----------------------------
    // Cast to datatypes::u8
    //-----------------------------
#define REF_CAST_TO_U8(IN, LANES, I) (static_cast<uint8_t>(IN[0][I]))
    // data_type: uint_32
    TEST_OP(UNARY, EXACT, uint32_t, uint8_t, datatypes::s32, datatypes::u8,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_U8,
            MAKE_CAST(datatypes::u8), DATASET_I1);
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, uint8_t, datatypes::f32, datatypes::u8,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_U8,
            MAKE_CAST(datatypes::u8), DATASET_F1);
#undef REF_CAST_TO_U8
    //-----------------------------
    // Cast to datatypes::u16
    //-----------------------------
#define REF_CAST_TO_U16(IN, LANES, I) (static_cast<uint16_t>(IN[0][I]))
    // data_type: uint_32
    TEST_OP(UNARY, EXACT, uint32_t, uint16_t, datatypes::u32, datatypes::u16,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_U16,
            MAKE_CAST(datatypes::u16), DATASET_I1);
    // data_type: index
    TEST_OP(UNARY, EXACT, uint64_t, uint16_t, datatypes::index, datatypes::u16,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_CAST_TO_U16,
            MAKE_CAST(datatypes::u16), DATASET_I1);
    if (is_cpu_support_fp16()) {
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        // data_type: fp16_t
        TEST_OP(UNARY, EXACT, fp16_t, uint16_t, datatypes::f16, datatypes::u16,
                DATA_LEN_16, num_lanes_fp16, TEST_SCALAR, TEST_SIMD,
                REF_CAST_TO_U16, MAKE_CAST(datatypes::u16), DATASET_I1);
    }
#undef REF_CAST_TO_U16
    //-----------------------------
    // Cast to datatypes::u32
    //-----------------------------
#define REF_CAST_TO_U32(IN, LANES, I) (static_cast<uint32_t>(IN[0][I]))
    TEST_OP(UNARY, EXACT, uint16_t, uint32_t, datatypes::u16, datatypes::u32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_U32,
            MAKE_CAST(datatypes::u32), DATASET_I1);
    if (is_cpu_support_fp16()) {
        TEST_OP(UNARY, EXACT, fp16_t, uint32_t, datatypes::f16, datatypes::u32,
                DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_U32,
                MAKE_CAST(datatypes::u32), DATASET_I1);
    }
#undef REF_CAST_TO_U32
    //-----------------------------
    // Cast to datatypes::index
    //-----------------------------
#define REF_CAST_TO_INDEX(IN, LANES, I) (static_cast<uint64_t>(IN[0][I]))
    TEST_OP(UNARY, EXACT, uint8_t, uint64_t, datatypes::u8, datatypes::index,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_CAST_TO_INDEX,
            MAKE_CAST(datatypes::index), DATASET_I1);
    if (is_cpu_support_fp16()) {
        TEST_OP(UNARY, EXACT, fp16_t, uint64_t, datatypes::f16,
                datatypes::index, DATA_LEN_16, num_lanes, TEST_SCALAR,
                SKIP_SIMD, REF_CAST_TO_INDEX, MAKE_CAST(datatypes::index),
                DATASET_I1);
    }
#undef REF_CAST_TO_INDEX
    //-----------------------------
    // Cast to datatypes::s32
    //-----------------------------
#define REF_CAST_TO_S32(IN, LANES, I) (static_cast<int32_t>(IN[0][I]))
    // data_type: uint_8
    TEST_OP(UNARY, EXACT, uint8_t, int32_t, datatypes::u8, datatypes::s32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_CAST_TO_S32,
            MAKE_CAST(datatypes::s32), DATASET_I1);
    // data_type: sint_8
    TEST_OP(UNARY, EXACT, int8_t, int32_t, datatypes::s8, datatypes::s32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_S32,
            MAKE_CAST(datatypes::s32), DATASET_I3);
    if (is_cpu_support_fp16()) {
        TEST_OP(UNARY, EXACT, fp16_t, int32_t, datatypes::f16, datatypes::s32,
                DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_CAST_TO_S32,
                MAKE_CAST(datatypes::s32), DATASET_I3);
    }
    // data_type: uint_16
    TEST_OP(UNARY, EXACT, uint16_t, int32_t, datatypes::u16, datatypes::s32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_S32,
            MAKE_CAST(datatypes::s32), DATASET_I1);
#undef REF_CAST_TO_S32
    //-----------------------------
    // Cast to datatypes::f16
    //-----------------------------
    if (is_cpu_support_fp16()) {
#define REF_CAST_TO_F16(IN, LANES, I) (static_cast<fp16_t>(IN[0][I]))
        TEST_OP(UNARY, EXACT, float, fp16_t, datatypes::f32, datatypes::f16,
                DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_F16,
                MAKE_CAST(datatypes::f16), DATASET_F3);
        TEST_OP(UNARY, EXACT, uint32_t, fp16_t, datatypes::u32, datatypes::f16,
                DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_F16,
                MAKE_CAST(datatypes::f16), DATASET_I1);
        TEST_OP(UNARY, EXACT, int32_t, fp16_t, datatypes::s32, datatypes::f16,
                DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_F16,
                MAKE_CAST(datatypes::f16), DATASET_I1);
        TEST_OP(UNARY, EXACT, uint64_t, fp16_t, datatypes::index,
                datatypes::f16, DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD,
                REF_CAST_TO_F16, MAKE_CAST(datatypes::f16), DATASET_I1);
        TEST_OP(UNARY, EXACT, uint8_t, fp16_t, datatypes::u8, datatypes::f16,
                DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_CAST_TO_F16,
                MAKE_CAST(datatypes::f16), DATASET_I1);
        TEST_OP(UNARY, EXACT, uint16_t, fp16_t, datatypes::u16, datatypes::f16,
                DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_CAST_TO_F16,
                MAKE_CAST(datatypes::f16), DATASET_I1);
#undef REF_CAST_TO_F32
    }
    //-----------------------------
    // Cast to datatypes::f32
    //-----------------------------
#define REF_CAST_TO_F32(IN, LANES, I) (static_cast<float>(IN[0][I]))
    // data_type: uint_8
    TEST_OP(UNARY, EXACT, uint8_t, float, datatypes::u8, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_F32,
            MAKE_CAST(datatypes::f32), DATASET_I1);
    // data_type: sint_8
    TEST_OP(UNARY, EXACT, int8_t, float, datatypes::s8, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_F32,
            MAKE_CAST(datatypes::f32), DATASET_I3);
    // data_type: sint_32
    TEST_OP(UNARY, EXACT, int32_t, float, datatypes::s32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_F32,
            MAKE_CAST(datatypes::f32), DATASET_I3);
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_F32,
            MAKE_CAST(datatypes::f32), DATASET_F3);
    if (is_cpu_support_fp16()) {
        TEST_OP(UNARY, EXACT, fp16_t, float, datatypes::f16, datatypes::f32,
                DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_F32,
                MAKE_CAST(datatypes::f32), DATASET_F3);
    }
#undef REF_CAST_TO_F32
    //-----------------------------
    // Cast to datatypes::generic
    //-----------------------------
#define REF_CAST_TO_GENERIC(IN, LANES, I) \
    ((uint32_t)generic_val(IN[0][I]).v_uint64_t)
    // data_type: sint_32, truncate generic_val to only check lower bits
    TEST_OP(UNARY, EXACT, int32_t, uint64_t, datatypes::s32, datatypes::generic,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_CAST_TO_GENERIC,
            MAKE_CAST(datatypes::generic), DATASET_I3);
#undef REF_CAST_TO_GENERIC
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestOpCastAVX2) {
    REQUIRE_AVX2();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_8);
    //-----------------------------
    // Cast to datatypes::s8
    //-----------------------------
#define REF_CAST_TO_S8(IN, LANES, I) (static_cast<int8_t>(IN[0][I]))
    use_cfake = false;
    // data_type: sint_32
    TEST_OP(UNARY, EXACT, int32_t, int8_t, datatypes::s32, datatypes::s8,
            DATA_LEN_8, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_S8,
            MAKE_CAST(datatypes::s8), DATASET_I1_8);
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, int8_t, datatypes::f32, datatypes::s8,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_S8,
            MAKE_CAST(datatypes::s8), DATASET_I1_8);
    use_cfake = true;
#undef REF_CAST_TO_S8
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestOpAdd) {
    REQUIRE_AVX2();
#define REF_ADD(IN, LANES, I) (IN[0][I] + IN[1][I])
    // data_type: sint_32
    const int num_lanes_s32 = get_lanes(sc_data_etype::S32, DATA_LEN_16);
    TEST_OP(BINARY, EXACT, int32_t, int32_t, datatypes::s32, datatypes::s32,
            DATA_LEN_16, num_lanes_s32, TEST_SCALAR, SKIP_SIMD, REF_ADD,
            MAKE_BINARY_OP(make_add), DATASET_I1, DATASET_I3);
    // data_type: float_32
    const int num_lanes_f32 = get_lanes(sc_data_etype::F32, DATA_LEN_16);
    TEST_OP(BINARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes_f32, TEST_SCALAR, TEST_SIMD, REF_ADD,
            MAKE_BINARY_OP(make_add), DATASET_F1, DATASET_F3);
    // data_type: uint_64
    const int num_lanes_u64 = get_lanes(sc_data_etype::INDEX, DATA_LEN_16);
    TEST_OP(BINARY, EXACT, uint64_t, uint64_t, datatypes::index,
            datatypes::index, DATA_LEN_16, num_lanes_u64, TEST_SCALAR,
            SKIP_SIMD, REF_ADD, MAKE_BINARY_OP(make_add), DATASET_I1,
            DATASET_I2);
    if (is_cpu_support_fp16()) {
        // data_type: fp16
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(BINARY, EXACT, uint16_t, uint16_t, datatypes::f16,
                datatypes::f16, DATA_LEN_16, num_lanes_fp16, TEST_SCALAR,
                TEST_SIMD, REF_ADD, MAKE_BINARY_OP(make_add), DATASET_I1,
                DATASET_I2);
    }

#undef REF_ADD
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestOpSub) {
    REQUIRE_AVX2();
#define REF_SUB(IN, LANES, I) (IN[0][I] - IN[1][I])
    // data_type: float_32
    const int num_lanes_f32 = get_lanes(sc_data_etype::F32, DATA_LEN_16);
    TEST_OP(BINARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes_f32, TEST_SCALAR, TEST_SIMD, REF_SUB,
            MAKE_BINARY_OP(make_sub), DATASET_F1, DATASET_F3);
    // data_type: uint_64
    const int num_lanes_u64 = get_lanes(sc_data_etype::INDEX, DATA_LEN_16);
    TEST_OP(BINARY, EXACT, uint64_t, uint64_t, datatypes::index,
            datatypes::index, DATA_LEN_16, num_lanes_u64, TEST_SCALAR,
            SKIP_SIMD, REF_SUB, MAKE_BINARY_OP(make_sub), DATASET_I1,
            DATASET_I2);
    if (is_cpu_support_fp16()) {
        // data_type:: float16
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(BINARY, EXACT, fp16_t, fp16_t, datatypes::f16, datatypes::f16,
                DATA_LEN_16, num_lanes_fp16, TEST_SCALAR, TEST_SIMD, REF_SUB,
                MAKE_BINARY_OP(make_sub), DATASET_I1, DATASET_I2);
    }
#undef REF_SUB
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestOpMul) {
    REQUIRE_AVX2();
#define REF_MUL(IN, LANES, I) (IN[0][I] * IN[1][I])
    // data_type: float_32
    const int num_lanes_f32 = get_lanes(sc_data_etype::F32, DATA_LEN_16);
    TEST_OP(BINARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes_f32, TEST_SCALAR, TEST_SIMD, REF_MUL,
            MAKE_BINARY_OP(make_mul), DATASET_F1, DATASET_F3);
    // data_type: uint_64
    const int num_lanes_u64 = get_lanes(sc_data_etype::INDEX, DATA_LEN_16);
    TEST_OP(BINARY, EXACT, uint64_t, uint64_t, datatypes::index,
            datatypes::index, DATA_LEN_16, num_lanes_u64, TEST_SCALAR,
            SKIP_SIMD, REF_MUL, MAKE_BINARY_OP(make_mul), DATASET_I1,
            DATASET_I2);
    if (is_cpu_support_fp16()) {
        // data_type:: float_16
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(BINARY, EXACT, fp16_t, fp16_t, datatypes::f16, datatypes::f16,
                DATA_LEN_16, num_lanes_fp16, TEST_SCALAR, SKIP_SIMD, REF_MUL,
                MAKE_BINARY_OP(make_mul), DATASET_F1, DATASET_F3);
    }
#undef REF_MUL
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestOpDiv) {
    REQUIRE_AVX2();
#define REF_DIV(IN, LANES, I) (IN[0][I] / IN[1][I])
    // data_type: float_32
    const int num_lanes_f32 = get_lanes(sc_data_etype::F32, DATA_LEN_16);
    TEST_OP(BINARY, APPROX, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes_f32, TEST_SCALAR, TEST_SIMD, REF_DIV,
            MAKE_BINARY_OP(make_div), DATASET_F1, DATASET_F3);
    // data_type: uint_64
    const int num_lanes_u64 = get_lanes(sc_data_etype::INDEX, DATA_LEN_16);
    TEST_OP(BINARY, APPROX, uint64_t, uint64_t, datatypes::index,
            datatypes::index, DATA_LEN_16, num_lanes_u64, TEST_SCALAR,
            SKIP_SIMD, REF_DIV, MAKE_BINARY_OP(make_div), DATASET_I1,
            DATASET_I2);
    if (test_cpu_flags().fAVX512F) {
        const int num_lanes_bf16 = get_lanes(sc_data_etype::BF16, DATA_LEN_16);
        TEST_OP(BINARY, APPROX, bf16_t, bf16_t, datatypes::bf16,
                datatypes::bf16, DATA_LEN_16, num_lanes_bf16, SKIP_SCALAR,
                TEST_SIMD, REF_DIV, MAKE_BINARY_OP(make_div), DATASET_F1,
                DATASET_F2);
    }
    if (is_cpu_support_fp16()) {
        // fp16 div in llvm use fast math, precision 1e-3 can meet require.
        precision_threshold = 1e-3F;
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(BINARY, APPROX, fp16_t, fp16_t, datatypes::f16, datatypes::f16,
                DATA_LEN_16, num_lanes_fp16, SKIP_SCALAR, TEST_SIMD, REF_DIV,
                MAKE_BINARY_OP(make_div), DATASET_F1, DATASET_F2);
        precision_threshold = 1e-4F;
    }
#undef REF_DIV
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestOpMod) {
    REQUIRE_AVX2();
#define REF_MOD(IN, LANES, I) (IN[0][I] % IN[1][I])
    // data_type: uint_64
    const int num_lanes = get_lanes(sc_data_etype::INDEX, DATA_LEN_16);
    TEST_OP(BINARY, EXACT, uint64_t, uint64_t, datatypes::index,
            datatypes::index, DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD,
            REF_MOD, MAKE_BINARY_OP(make_mod), DATASET_I1, DATASET_I2);
#undef REF_MOD
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestOpCmp) {
    REQUIRE_AVX2();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_16);
#define REF_EQ(IN, LANES, I) (IN[0][I] == IN[1][I])
    // data_type: float32
    TEST_OP(BINARY, EXACT, float, bool, datatypes::f32, datatypes::boolean,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_EQ,
            MAKE_BINARY_OP(make_cmp_eq), DATASET_F3, DATASET_F4);
    if (is_cpu_support_fp16()) {
        TEST_OP(BINARY, EXACT, fp16_t, bool, datatypes::f16, datatypes::boolean,
                DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_EQ,
                MAKE_BINARY_OP(make_cmp_eq), DATASET_F3, DATASET_F4);
    }
#undef REF_EQ
#define REF_NE(IN, LANES, I) (IN[0][I] != IN[1][I])
    // data_type: float32
    TEST_OP(BINARY, EXACT, float, bool, datatypes::f32, datatypes::boolean,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_NE,
            MAKE_BINARY_OP(make_cmp_ne), DATASET_F3, DATASET_F4);
    if (is_cpu_support_fp16()) {
        TEST_OP(BINARY, EXACT, fp16_t, bool, datatypes::f16, datatypes::boolean,
                DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_NE,
                MAKE_BINARY_OP(make_cmp_ne), DATASET_F3, DATASET_F4);
    }
#undef REF_NE
#define REF_LT(IN, LANES, I) (IN[0][I] < IN[1][I])
    // data_type: float32
    TEST_OP(BINARY, EXACT, float, bool, datatypes::f32, datatypes::boolean,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_LT,
            MAKE_BINARY_OP(make_cmp_lt), DATASET_F3, DATASET_F4);
    if (is_cpu_support_fp16()) {
        TEST_OP(BINARY, EXACT, fp16_t, bool, datatypes::f16, datatypes::boolean,
                DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_LT,
                MAKE_BINARY_OP(make_cmp_lt), DATASET_F3, DATASET_F4);
    }
#undef REF_LT
#define REF_LE(IN, LANES, I) (IN[0][I] <= IN[1][I])
    // data_type: float32
    TEST_OP(BINARY, EXACT, float, bool, datatypes::f32, datatypes::boolean,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_LE,
            MAKE_BINARY_OP(make_cmp_le), DATASET_F3, DATASET_F4);
    if (is_cpu_support_fp16()) {
        TEST_OP(BINARY, EXACT, fp16_t, bool, datatypes::f16, datatypes::boolean,
                DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_LE,
                MAKE_BINARY_OP(make_cmp_le), DATASET_F3, DATASET_F4);
    }
#undef REF_LE
#define REF_GT(IN, LANES, I) (IN[0][I] > IN[1][I])
    // data_type: float32
    TEST_OP(BINARY, EXACT, float, bool, datatypes::f32, datatypes::boolean,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_GT,
            MAKE_BINARY_OP(make_cmp_gt), DATASET_F3, DATASET_F4);
    if (is_cpu_support_fp16()) {
        TEST_OP(BINARY, EXACT, fp16_t, bool, datatypes::f16, datatypes::boolean,
                DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_GT,
                MAKE_BINARY_OP(make_cmp_gt), DATASET_F3, DATASET_F4);
    }
#undef REF_GT
#define REF_GE(IN, LANES, I) (IN[0][I] >= IN[1][I])
    // data_type: float32
    TEST_OP(BINARY, EXACT, float, bool, datatypes::f32, datatypes::boolean,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_GE,
            MAKE_BINARY_OP(make_cmp_ge), DATASET_F3, DATASET_F4);
    if (is_cpu_support_fp16()) {
        TEST_OP(BINARY, EXACT, fp16_t, bool, datatypes::f16, datatypes::boolean,
                DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_GE,
                MAKE_BINARY_OP(make_cmp_ge), DATASET_F3, DATASET_F4);
    }
#undef REF_GE
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestIntrinMin) {
    REQUIRE_AVX2();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_16);
#define REF_MIN(IN, LANES, I) (std::min(IN[0][I], IN[1][I]))
    // data_type: float_32
    TEST_OP(BINARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_MIN,
            MAKE_BINARY_OP(make_min), DATASET_F1, DATASET_F2);
    if (is_cpu_support_fp16()) {
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        // data_type: fp16
        TEST_OP(BINARY, EXACT, fp16_t, fp16_t, datatypes::f16, datatypes::f16,
                DATA_LEN_16, num_lanes_fp16, TEST_SCALAR, TEST_SIMD, REF_MIN,
                MAKE_BINARY_OP(make_min), DATASET_F1, DATASET_F2);
    }
#undef REF_MIN
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestIntrinMax) {
    REQUIRE_AVX2();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_16);
#define REF_MAX(IN, LANES, I) (std::max(IN[0][I], IN[1][I]))
    // data_type: float_32
    TEST_OP(BINARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_MAX,
            MAKE_BINARY_OP(make_max), DATASET_F1, DATASET_F2);
    if (is_cpu_support_fp16()) {
        // data_type: fp16
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(BINARY, EXACT, fp16_t, fp16_t, datatypes::f16, datatypes::f16,
                DATA_LEN_16, num_lanes_fp16, TEST_SCALAR, TEST_SIMD, REF_MAX,
                MAKE_BINARY_OP(make_max), DATASET_F1, DATASET_F2);
    }
#undef REF_MAX
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestIntrinFloor) {
    REQUIRE_AVX2();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_16);
#define REF_FLOOR(IN, LANES, I) (floor(IN[0][I]))
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_FLOOR,
            MAKE_UNARY_OP(make_floor), DATASET_F3);
    if (is_cpu_support_fp16()) {
        // data_type: fp16
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(UNARY, EXACT, fp16_t, fp16_t, datatypes::f16, datatypes::f16,
                DATA_LEN_16, num_lanes_fp16, SKIP_SCALAR, TEST_SIMD, REF_FLOOR,
                MAKE_UNARY_OP(make_floor), DATASET_F3);
    }
#undef REF_FLOOR
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestIntrinCeil) {
    REQUIRE_AVX2();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_16);
#define REF_CEIL(IN, LANES, I) (ceil(IN[0][I]))
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_CEIL,
            MAKE_UNARY_OP(make_ceil), DATASET_F3);
    if (is_cpu_support_fp16()) {
        // data_type: fp16
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(UNARY, EXACT, fp16_t, fp16_t, datatypes::f16, datatypes::f16,
                DATA_LEN_16, num_lanes_fp16, SKIP_SCALAR, TEST_SIMD, REF_CEIL,
                MAKE_UNARY_OP(make_ceil), DATASET_F3);
    }
#undef REF_CEIL
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestIntrinExp) {
    REQUIRE_AVX2();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_16);
#define REF_EXP(IN, LANES, I) (expf(IN[0][I]))
    // data_type: float_32
    TEST_OP(UNARY, APPROX, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_EXP,
            MAKE_UNARY_OP(make_exp), DATASET_F3);
#undef REF_EXP
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestIntrinLog) {
    REQUIRE_AVX2();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_16);
#define REF_LOG(IN, LANES, I) (logf(IN[0][I]))
    // data_type: float_32
    TEST_OP(UNARY, APPROX, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_LOG,
            MAKE_UNARY_OP(make_log), DATASET_F1);
#undef REF_LOG
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestIntrinFmadd) {
    REQUIRE_AVX2();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_16);
#define REF_FMADD(IN, LANES, I) (IN[0][I] * IN[1][I] + IN[2][I])
    // data_type: float_32
    TEST_OP(TRINARY, APPROX, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_FMADD,
            MAKE_TRINARY_OP(make_fmadd), DATASET_F1, DATASET_F2, DATASET_F3);
    if (is_cpu_support_fp16()) {
        // data_type: fp16
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(TRINARY, APPROX, fp16_t, fp16_t, datatypes::f16, datatypes::f16,
                DATA_LEN_16, num_lanes_fp16, TEST_SCALAR, TEST_SIMD, REF_FMADD,
                MAKE_TRINARY_OP(make_fmadd), DATASET_F1, DATASET_F2,
                DATASET_F3);
    }
#undef REF_FMADD
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestIntrinAbs) {
    REQUIRE_AVX2();
#define REF_ABS(IN, LANES, I) (std::abs(IN[0][I]))
    // data_type: sint_8
    const int num_lanes_s8 = DATA_LEN_16;
    TEST_OP(UNARY, EXACT, int8_t, int8_t, datatypes::s8, datatypes::s8,
            DATA_LEN_16, num_lanes_s8, SKIP_SCALAR, TEST_SIMD, REF_ABS,
            MAKE_UNARY_OP(make_abs), DATASET_I3);
    // data_type: sint_32
    const int num_lanes_s32 = get_lanes(sc_data_etype::S32, DATA_LEN_16);
    TEST_OP(UNARY, EXACT, int32_t, int32_t, datatypes::s32, datatypes::s32,
            DATA_LEN_16, num_lanes_s32, SKIP_SCALAR, TEST_SIMD, REF_ABS,
            MAKE_UNARY_OP(make_abs), DATASET_I3);
    // data_type: float_32
    const int num_lanes_f32 = get_lanes(sc_data_etype::F32, DATA_LEN_16);
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes_f32, TEST_SCALAR, TEST_SIMD, REF_ABS,
            MAKE_UNARY_OP(make_abs), DATASET_F3);
    if (test_cpu_flags().fAVX512F) {
        const int num_lanes_bf16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(UNARY, EXACT, bf16_t, bf16_t, datatypes::bf16, datatypes::bf16,
                DATA_LEN_16, num_lanes_bf16, SKIP_SCALAR, TEST_SIMD, REF_ABS,
                MAKE_UNARY_OP(make_abs), DATASET_F3);
    }
    if (is_cpu_support_fp16()) {
        // data_type: fp16
        const int num_lanes_f16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(UNARY, EXACT, fp16_t, fp16_t, datatypes::f16, datatypes::f16,
                DATA_LEN_16, num_lanes_f16, SKIP_SCALAR, TEST_SIMD, REF_ABS,
                MAKE_UNARY_OP(make_abs), DATASET_F3);
    }
#undef REF_ABS
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestIntrinRound) {
    REQUIRE_AVX2();
    const int num_lanes = get_lanes(sc_data_etype::F32, DATA_LEN_16);
#define REF_ROUND(IN, LANES, I) (std::rint(IN[0][I]))
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_ROUND,
            MAKE_UNARY_OP(make_round), DATASET_F3);
    if (is_cpu_support_fp16()) {
        // data_type: fp16
        const int num_lanes_fp16 = get_lanes(sc_data_etype::F16, DATA_LEN_16);
        TEST_OP(UNARY, EXACT, fp16_t, fp16_t, datatypes::f16, datatypes::f16,
                DATA_LEN_16, num_lanes_fp16, SKIP_SCALAR, TEST_SIMD, REF_ROUND,
                MAKE_UNARY_OP(make_round), DATASET_F3);
    }
#undef REF_ROUND
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestAVX2MaskMovx4) {
    REQUIRE_AVX2();
    const int num_lanes = 4;
    const uint64_t mask_val = 0xf >> 1;
    const expr mask = make_constant({mask_val}, datatypes::u8);
#define REF_MASK_MOV(IN, LANES, I) \
    ((mask_val & UINT64_C(1) << (I % LANES)) ? IN[0][I] : 0)
    // data_type: sint_32
    TEST_OP(UNARY, EXACT, int32_t, int32_t, datatypes::s32, datatypes::s32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I3);
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_F1);
#undef REF_MASK_MOV
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestAVX2MaskMovx8) {
    REQUIRE_AVX2();
    const int num_lanes = 8;
    const uint64_t mask_val = 0xff >> 1;
    const expr mask = make_constant({mask_val}, datatypes::u8);
#define REF_MASK_MOV(IN, LANES, I) \
    ((mask_val & UINT64_C(1) << (I % LANES)) ? IN[0][I] : 0)
    // data_type: sint_32
    TEST_OP(UNARY, EXACT, int32_t, int32_t, datatypes::s32, datatypes::s32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I3);
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_F1);
#undef REF_MASK_MOV
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestMaskMovx4) {
    REQUIRE_AVX512();
    const int num_lanes = 4;
    const uint64_t mask_val = 0xf >> 1;
    const expr mask = make_constant({mask_val}, datatypes::u8);
#define REF_MASK_MOV(IN, LANES, I) \
    ((mask_val & UINT64_C(1) << (I % LANES)) ? IN[0][I] : 0)
    // data_type: sint_32
    TEST_OP(UNARY, EXACT, int32_t, int32_t, datatypes::s32, datatypes::s32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I3);
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_F1);
#undef REF_MASK_MOV
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestMaskMovx8) {
    REQUIRE_AVX512();
    const int num_lanes = 8;
    const uint64_t mask_val = 0xff >> 1;
    const expr mask = make_constant({mask_val}, datatypes::u8);
#define REF_MASK_MOV(IN, LANES, I) \
    ((mask_val & UINT64_C(1) << (I % LANES)) ? IN[0][I] : 0)
    // data_type: sint_8
    TEST_OP(UNARY, EXACT, int8_t, int8_t, datatypes::s8, datatypes::s8,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I1);
    // data_type: uint_8
    TEST_OP(UNARY, EXACT, uint8_t, uint8_t, datatypes::u8, datatypes::u8,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I1);
    // data_type: uint_16
    TEST_OP(UNARY, EXACT, uint16_t, uint16_t, datatypes::u16, datatypes::u16,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I1);
    // data_type: sint_32
    TEST_OP(UNARY, EXACT, int32_t, int32_t, datatypes::s32, datatypes::s32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I3);
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_F1);
#undef REF_MASK_MOV
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestMaskMovx16) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
    const uint64_t mask_val = 0xffff >> 1;
    const expr mask = make_constant({mask_val}, datatypes::u16);
#define REF_MASK_MOV(IN, LANES, I) \
    ((mask_val & UINT64_C(1) << (I % LANES)) ? IN[0][I] : 0)
    // data_type: uint_8
    TEST_OP(UNARY, EXACT, uint8_t, uint8_t, datatypes::u8, datatypes::u8,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I1);
    // data_type: uint_16
    TEST_OP(UNARY, EXACT, uint16_t, uint16_t, datatypes::u16, datatypes::u16,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I2);
    // data_type: sint_32
    TEST_OP(UNARY, EXACT, int32_t, int32_t, datatypes::s32, datatypes::s32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I3);
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_F1);
#undef REF_MASK_MOV
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestMaskMovx32) {
    REQUIRE_AVX512();
    const int num_lanes = 32;
    const uint64_t mask_val = 0xffffffff >> 1;
    const expr mask = make_constant({mask_val}, datatypes::u32);
#define REF_MASK_MOV(IN, LANES, I) \
    ((mask_val & UINT64_C(1) << (I % LANES)) ? IN[0][I] : 0)
    // data_type: uint_8
    TEST_OP(UNARY, EXACT, uint8_t, uint8_t, datatypes::u8, datatypes::u8,
            DATA_LEN_64, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I1_64);
    // data_type: uint_16
    TEST_OP(UNARY, EXACT, uint16_t, uint16_t, datatypes::u16, datatypes::u16,
            DATA_LEN_64, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I1_64);
#undef REF_MASK_MOV
}

TEST(GCCore_CPU_test_jit_engine_equivalence, TestMaskMovx64) {
    REQUIRE_AVX512();
    const int num_lanes = 64;
    const uint64_t mask_val = 0xffffffffffffffff >> 1;
    const expr mask = make_constant({mask_val}, datatypes::index);
#define REF_MASK_MOV(IN, LANES, I) \
    ((mask_val & UINT64_C(1) << (I % LANES)) ? IN[0][I] : 0)
    // data_type: uint_8
    TEST_OP(UNARY, EXACT, uint8_t, uint8_t, datatypes::u8, datatypes::u8,
            DATA_LEN_64, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_MASK_MOV,
            MAKE_MASK_MOV(mask), DATASET_I1_64);
#undef REF_MASK_MOV
}
