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
#include <compiler/jit/xbyak/xbyak_jit_engine.hpp>
#endif
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

static map<string, shared_ptr<jit_engine_t>> get_engines() {
    map<string, shared_ptr<jit_engine_t>> ret;
#if SC_CFAKE_JIT_ENABLED
    ret["cfake_jit"] = make_shared<cfake_jit>();
#endif
#if defined(SC_LLVM_BACKEND)
    ret["llvm_jit"] = make_shared<llvm_jit>();
#endif
#if SC_BUILTIN_JIT_ENABLED
    ret["xbyak_jit_engine"] = make_shared<sc_xbyak::xbyak_jit_engine>();
#endif
    return ret;
}

static map<string, shared_ptr<jit_engine_t>> test_jit_engines = get_engines();

//===========================================================================
// Pre-defined dataset
//===========================================================================
#define DATA_LEN_16 16
#define DATA_LEN_64 64

#define DATASET_I1 \
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }
#define DATASET_I2 \
    { 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 }
#define DATASET_I3 \
    { -16, -15, -14, -13, -12, -11, -10, -9, 8, 7, 6, 5, 4, 3, 2, 1 }
#define DATASET_I1_64 \
    { \
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, \
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, \
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64 \
    }
#define DATASET_F1 \
    { \
        1.16, 2.15, 3.14, 4.13, 5.12, 6.11, 7.10, 8.9, 9.8, 10.7, 11.6, 12.5, \
                13.4, 14.3, 15.2, 16.1 \
    }
#define DATASET_F2 \
    { \
        16.1, 15.2, 14.3, 13.4, 12.5, 11.6, 10.7, 9.8, 8.9, 7.10, 6.11, 5.12, \
                4.13, 3.14, 2.15, 1.16 \
    }
#define DATASET_F3 \
    { \
        -64.1, -32.9, -16.2, -8.8, -4.3, -2.7, -1.4, 0.6, 0.5, 1.5, 2.6, 4.4, \
                8.7, 16.3, 32.8, 64.2 \
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
            EXPECT_NEAR(out[i], ref[i], std::abs(1e-4 * ref[i])); \
        } \
    }
    // Make ir module
    ir_builder_t builder;
    auto ir_mod = std::make_shared<ir_module_t>(get_default_context());
    // Add test functions
    if (SCALAR) { DEFINE_IR_FUNC(test_scalar, 1); }
    if (SIMD) { DEFINE_IR_FUNC(test_simd, lanes); }
    // Run the test
    for (auto &kv : test_jit_engines) {
        const string &je_name = kv.first;
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

TEST(GCCore_jit_engine_equivalence, TestLocalTensor) {
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

    for (auto &kv : test_jit_engines) {
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
TEST(GCCore_jit_engine_equivalence, TestTensorAddrPassing) {
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

    for (auto &kv : test_jit_engines) {
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

TEST(GCCore_jit_engine_equivalence, TestSequentialElwiseAdd) {
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

    for (auto &kv : test_jit_engines) {
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

TEST(GCCore_jit_engine_equivalence, TestTrivialGenericWrapper) {
    ir_builder_t builder;
    _function_(datatypes::void_t, foo) {}

    ir_module_ptr ir_mod
            = ir_module_t::from_entry_func(get_default_context(), foo);

    for (auto &kv : test_jit_engines) {
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
TEST(GCCore_jit_engine_equivalence, DISABLED_TestSimpleEntryFunction) {
    ir_builder_t builder;
    _function_(datatypes::s32, foo) { _return_(42); }

    ir_module_ptr ir_mod
            = ir_module_t::from_entry_func(get_default_context(), foo);

    for (auto &kv : test_jit_engines) {
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

TEST(GCCore_jit_engine_equivalence, TestFuncAddrNode) {
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

    for (auto &kv : test_jit_engines) {
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
TEST(GCCore_jit_engine_equivalence, TestNamedForLoop) {
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

    for (auto &kv : test_jit_engines) {
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
TEST(GCCore_jit_engine_equivalence, TestIfElse) {
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

    for (auto &kv : test_jit_engines) {
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

TEST(GCCore_jit_engine_equivalence, TestCMPExprSint) {
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

    for (auto &kv : test_jit_engines) {
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

TEST(GCCore_jit_engine_equivalence, TestCMPExprUint) {
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

    for (auto &kv : test_jit_engines) {
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

TEST(GCCore_jit_engine_equivalence, TestConstantBroadcast) {
    REQUIRE_AVX512();
    ir_builder_t builder;

    // We'll use over-sized tensors to help us notice errors in the indexing
    // calculations.
    const int simd_lanes = 16;
    const int num_elems = 1 + simd_lanes + 1;

    _function_(datatypes::void_t, foo,
            _arg_("tensor_out", datatypes::f32, {num_elems}),
            _arg_("tensor_int8", datatypes::s8, {num_elems}),
            _arg_("tensor_uint8", datatypes::u8, {num_elems}),
            _arg_("tensor_int32", datatypes::s32, {num_elems}),
            _arg_("tensor_uint32", datatypes::u32, {num_elems}),
            _arg_("tensor_uint16", datatypes::u16, {num_elems}), ) {
        _bind_(tensor_out, tensor_int8, tensor_uint8, tensor_int32,
                tensor_uint32, tensor_uint16);
        tensor_out[span_t({1}, simd_lanes)] = make_expr<constant_node>(
                42.f, sc_data_type_t::f32(simd_lanes));
        union_val val;
        val.u64 = 42;
        tensor_int8[span_t({1}, simd_lanes)]
                = make_expr<constant_node>(val, sc_data_type_t::s8(simd_lanes));
        tensor_uint8[span_t({1}, simd_lanes)]
                = make_expr<constant_node>(val, sc_data_type_t::u8(simd_lanes));
        tensor_int32[span_t({1}, simd_lanes)] = make_expr<constant_node>(
                val, sc_data_type_t::s32(simd_lanes));
        tensor_uint32[span_t({1}, simd_lanes)] = make_expr<constant_node>(
                val, sc_data_type_t::u32(simd_lanes));
        tensor_uint16[span_t({1}, simd_lanes)] = make_expr<constant_node>(
                val, sc_data_type_t::u16(simd_lanes));
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : test_jit_engines) {
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

        generic_val generic_args[] = {&host_tensor_out, &host_tensor_int8,
                &host_tensor_uint8, &host_tensor_int32, &host_tensor_uint32,
                &host_tensor_uint16};

        j_foo->call_generic_default(generic_args);

        for (int i = 0; i < num_elems; ++i) {
            EXPECT_EQ(host_tensor_out[i], expected_result[i]);
            EXPECT_EQ(host_tensor_int8[i], expected_int8[i]);
            EXPECT_EQ(host_tensor_uint8[i], expected_uint8[i]);
            EXPECT_EQ(host_tensor_int32[i], expected_int32[i]);
            EXPECT_EQ(host_tensor_uint32[i], expected_uint32[i]);
            EXPECT_EQ(host_tensor_uint16[i], expected_uint16[i]);
        }
    }
}

TEST(GCCore_jit_engine_equivalence, TestIntrinsicBroadcast) {
    REQUIRE_AVX512();
    ir_builder_t builder;

    // We'll use over-sized tensors to help us notice errors in the indexing
    // calculations.
    const int simd_lanes = 16;
    const int num_elems = 1 + simd_lanes + 1;

    _function_(datatypes::void_t, foo, _arg_("x", datatypes::f32),
            _arg_("tensor_out", datatypes::f32, {num_elems}), ) {
        _bind_(x, tensor_out);
        tensor_out[span_t({1}, simd_lanes)] = make_broadcast(x, simd_lanes);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : test_jit_engines) {
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

TEST(GCCore_jit_engine_equivalence, TestModuleVar) {
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

    for (auto &kv : test_jit_engines) {
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

TEST(GCCore_jit_engine_equivalence, TestSubtract8Args) {
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

    for (auto &kv : test_jit_engines) {
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

TEST(GCCore_jit_engine_equivalence, TestIntrinsicReduceAdd) {
    REQUIRE_AVX512();
    ir_builder_t builder;

    // We'll use over-sized tensors to help us notice errors in the indexing
    // calculations.
    const int simd_lanes = 16;
    const int num_elems = 1 + simd_lanes + 1;

    _function_(datatypes::void_t, foo,
            _arg_("tensor_in", datatypes::f32, {num_elems}),
            _arg_("out", datatypes::f32, {1}),
            _arg_("tensor_int32_in", datatypes::s32, {num_elems}),
            _arg_("out_int32", datatypes::s32, {1})) {
        _bind_(tensor_in, out, tensor_int32_in, out_int32);

        _var_(local_temp, sc_data_type_t::f32(16));
        local_temp = tensor_in[span_t({1}, simd_lanes)];
        out[0] = make_reduce_add(local_temp);
        _var_(local_temp_int32, sc_data_type_t::s32(16));
        local_temp_int32 = tensor_int32_in[span_t({1}, simd_lanes)];
        out_int32[0] = make_reduce_add(local_temp_int32);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : test_jit_engines) {
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

        float host_in[num_elems] = {
                -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1};
        float host_out = 0;
        const float expected_out = 136;
        int32_t host_int32_in[num_elems] = {
                -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1};
        int32_t host_int32_out = 0;
        const float expected_int32_out = 136;

        generic_val generic_args[] = {
                host_in,
                &host_out,
                host_int32_in,
                &host_int32_out,
        };

        j_foo->call_generic_default(generic_args);

        EXPECT_EQ(host_out, expected_out);
        EXPECT_EQ(host_int32_out, expected_int32_out);
    }
}

TEST(GCCore_jit_engine_equivalence, TestConstantBF16) {
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

    _function_(datatypes::void_t, foo,
            _arg_("tensor_in", datatypes::f32, {num_elems}),
            _arg_("tensor_out", datatypes::f32, {num_elems}), ) {
        _bind_(tensor_in, tensor_out);

        _var_(local_temp, sc_data_type_t::bf16(simd_lanes));
        local_temp = make_constf(DATASET_F2, sc_data_type_t::bf16(simd_lanes));
        local_temp = local_temp
                + make_cast(sc_data_type_t::bf16(simd_lanes),
                        tensor_in[span_t({0}, simd_lanes)]);
        local_temp = local_temp
                + make_constf(DATASET_F1, sc_data_type_t::bf16(simd_lanes));
        tensor_out[span_t({0}, simd_lanes)]
                = make_cast(sc_data_type_t::f32(simd_lanes), local_temp);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    for (auto &kv : test_jit_engines) {
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
        const float expected_result[num_elems] = {-46.815, -15.5625, 1.2525,
                8.705, 13.32, 15.035, 16.387, 19.3125, 19.175, 19.294, 20.325,
                22.025, 26.225, 33.7406, 50.1562, 81.4562};

        generic_val generic_args[] = {
                &host_tensor_in,
                &host_tensor_out,
        };

        j_foo->call_generic_default(generic_args);

        for (int i = 0; i < num_elems; ++i) {
            EXPECT_NEAR(host_tensor_out[i], expected_result[i],
                    std::abs(1e-2 * expected_result[i]));
        }
    }
}

TEST(GCCore_jit_engine_equivalence, TestConstDivModMul) {
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

    for (auto &kv : test_jit_engines) {
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

/// ===================================
/// Test Group 2: oprations & data type
/// ===================================

TEST(GCCore_test_jit_engine_equivalence, TestOpCast) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
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
#undef REF_CAST_TO_U16
    //-----------------------------
    // Cast to datatypes::u32
    //-----------------------------
#define REF_CAST_TO_U32(IN, LANES, I) (static_cast<uint32_t>(IN[0][I]))
    TEST_OP(UNARY, EXACT, uint16_t, uint32_t, datatypes::u16, datatypes::u32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_CAST_TO_U32,
            MAKE_CAST(datatypes::u32), DATASET_I1);
#undef REF_CAST_TO_U32
    //-----------------------------
    // Cast to datatypes::index
    //-----------------------------
#define REF_CAST_TO_INDEX(IN, LANES, I) (static_cast<uint64_t>(IN[0][I]))
    TEST_OP(UNARY, EXACT, uint8_t, uint64_t, datatypes::u8, datatypes::index,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_CAST_TO_INDEX,
            MAKE_CAST(datatypes::index), DATASET_I1);
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
#undef REF_CAST_TO_S32
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

TEST(GCCore_test_jit_engine_equivalence, TestOpAdd) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_ADD(IN, LANES, I) (IN[0][I] + IN[1][I])
    // data_type: sint_32
    TEST_OP(BINARY, EXACT, int32_t, int32_t, datatypes::s32, datatypes::s32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD, REF_ADD,
            MAKE_BINARY_OP(make_add), DATASET_I1, DATASET_I3);
    // data_type: float_32
    TEST_OP(BINARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_ADD,
            MAKE_BINARY_OP(make_add), DATASET_F1, DATASET_F3);
    // data_type: uint_64
    TEST_OP(BINARY, EXACT, uint64_t, uint64_t, datatypes::index,
            datatypes::index, DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD,
            REF_ADD, MAKE_BINARY_OP(make_add), DATASET_I1, DATASET_I2);
#undef REF_ADD
}

TEST(GCCore_test_jit_engine_equivalence, TestOpSub) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_SUB(IN, LANES, I) (IN[0][I] - IN[1][I])
    // data_type: float_32
    TEST_OP(BINARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_SUB,
            MAKE_BINARY_OP(make_sub), DATASET_F1, DATASET_F3);
    // data_type: uint_64
    TEST_OP(BINARY, EXACT, uint64_t, uint64_t, datatypes::index,
            datatypes::index, DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD,
            REF_SUB, MAKE_BINARY_OP(make_sub), DATASET_I1, DATASET_I2);
#undef REF_SUB
}

TEST(GCCore_test_jit_engine_equivalence, TestOpMul) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_MUL(IN, LANES, I) (IN[0][I] * IN[1][I])
    // data_type: float_32
    TEST_OP(BINARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_MUL,
            MAKE_BINARY_OP(make_mul), DATASET_F1, DATASET_F3);
    // data_type: uint_64
    TEST_OP(BINARY, EXACT, uint64_t, uint64_t, datatypes::index,
            datatypes::index, DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD,
            REF_MUL, MAKE_BINARY_OP(make_mul), DATASET_I1, DATASET_I2);
#undef REF_MUL
}

TEST(GCCore_test_jit_engine_equivalence, TestOpDiv) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_DIV(IN, LANES, I) (IN[0][I] / IN[1][I])
    // data_type: float_32
    TEST_OP(BINARY, APPROX, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_DIV,
            MAKE_BINARY_OP(make_div), DATASET_F1, DATASET_F3);
    // data_type: uint_64
    TEST_OP(BINARY, APPROX, uint64_t, uint64_t, datatypes::index,
            datatypes::index, DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD,
            REF_DIV, MAKE_BINARY_OP(make_div), DATASET_I1, DATASET_I2);
#undef REF_DIV
}

TEST(GCCore_test_jit_engine_equivalence, TestOpMod) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_MOD(IN, LANES, I) (IN[0][I] % IN[1][I])
    // data_type: uint_64
    TEST_OP(BINARY, EXACT, uint64_t, uint64_t, datatypes::index,
            datatypes::index, DATA_LEN_16, num_lanes, TEST_SCALAR, SKIP_SIMD,
            REF_MOD, MAKE_BINARY_OP(make_mod), DATASET_I1, DATASET_I2);
#undef REF_MOD
}

TEST(GCCore_test_jit_engine_equivalence, TestIntrinMin) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_MIN(IN, LANES, I) (std::min(IN[0][I], IN[1][I]))
    // data_type: float_32
    TEST_OP(BINARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_MIN,
            MAKE_BINARY_OP(make_min), DATASET_F1, DATASET_F2);
#undef REF_MIN
}

TEST(GCCore_test_jit_engine_equivalence, TestIntrinMax) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_MAX(IN, LANES, I) (std::max(IN[0][I], IN[1][I]))
    // data_type: float_32
    TEST_OP(BINARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_MAX,
            MAKE_BINARY_OP(make_max), DATASET_F1, DATASET_F2);
#undef REF_MAX
}

TEST(GCCore_test_jit_engine_equivalence, TestIntrinFloor) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_FLOOR(IN, LANES, I) (floor(IN[0][I]))
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_FLOOR,
            MAKE_UNARY_OP(make_floor), DATASET_F3);
#undef REF_FLOOR
}

TEST(GCCore_test_jit_engine_equivalence, TestIntrinCeil) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_CEIL(IN, LANES, I) (ceil(IN[0][I]))
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_CEIL,
            MAKE_UNARY_OP(make_ceil), DATASET_F3);
#undef REF_CEIL
}

TEST(GCCore_test_jit_engine_equivalence, TestIntrinExp) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_EXP(IN, LANES, I) (expf(IN[0][I]))
    // data_type: float_32
    TEST_OP(UNARY, APPROX, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_EXP,
            MAKE_UNARY_OP(make_exp), DATASET_F3);
#undef REF_EXP
}

TEST(GCCore_test_jit_engine_equivalence, TestIntrinFmadd) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_FMADD(IN, LANES, I) (IN[0][I] * IN[1][I] + IN[2][I])
    // data_type: float_32
    TEST_OP(TRINARY, APPROX, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_FMADD,
            MAKE_TRINARY_OP(make_fmadd), DATASET_F1, DATASET_F2, DATASET_F3);
#undef REF_FMADD
}

TEST(GCCore_test_jit_engine_equivalence, TestIntrinAbs) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_ABS(IN, LANES, I) (std::abs(IN[0][I]))
    // data_type: sint_8
    TEST_OP(UNARY, EXACT, int8_t, int8_t, datatypes::s8, datatypes::s8,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_ABS,
            MAKE_UNARY_OP(make_abs), DATASET_I3);
    // data_type: sint_32
    TEST_OP(UNARY, EXACT, int32_t, int32_t, datatypes::s32, datatypes::s32,
            DATA_LEN_16, num_lanes, SKIP_SCALAR, TEST_SIMD, REF_ABS,
            MAKE_UNARY_OP(make_abs), DATASET_I3);
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_ABS,
            MAKE_UNARY_OP(make_abs), DATASET_F3);
#undef REF_ABS
}

TEST(GCCore_test_jit_engine_equivalence, TestIntrinRound) {
    REQUIRE_AVX512();
    const int num_lanes = 16;
#define REF_ROUND(IN, LANES, I) (std::rint(IN[0][I]))
    // data_type: float_32
    TEST_OP(UNARY, EXACT, float, float, datatypes::f32, datatypes::f32,
            DATA_LEN_16, num_lanes, TEST_SCALAR, TEST_SIMD, REF_ROUND,
            MAKE_UNARY_OP(make_round), DATASET_F3);
#undef REF_ROUND
}

TEST(GCCore_test_jit_engine_equivalence, TestMaskMovx4) {
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

TEST(GCCore_test_jit_engine_equivalence, TestMaskMovx8) {
    REQUIRE_AVX512();
    const int num_lanes = 8;
    const uint64_t mask_val = 0xff >> 1;
    const expr mask = make_constant({mask_val}, datatypes::u8);
#define REF_MASK_MOV(IN, LANES, I) \
    ((mask_val & UINT64_C(1) << (I % LANES)) ? IN[0][I] : 0)
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

TEST(GCCore_test_jit_engine_equivalence, TestMaskMovx16) {
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

TEST(GCCore_test_jit_engine_equivalence, TestMaskMovx32) {
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

TEST(GCCore_test_jit_engine_equivalence, TestMaskMovx64) {
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
