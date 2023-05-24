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
#include <map>
#include <memory>
#include <string>
#include "test_utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#if SC_CFAKE_JIT_ENABLED
#include <compiler/jit/cfake/cfake_jit.hpp>
#endif
#include "gtest/gtest.h"
#include <compiler/ir/builtin.hpp>
#if SC_BUILTIN_JIT_ENABLED
#include <compiler/jit/xbyak/xbyak_jit.hpp>
#endif
#include <compiler/jit/jit.hpp>
#include <runtime/generic_val.hpp>
#include <util/string_utils.hpp>
#include <util/utils.hpp>

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
#if SC_BUILTIN_JIT_ENABLED
    if (get_default_context()->machine_.cpu_flags_.fAVX2) {
        ret["xbyak_jit"] = make_shared<xbyak_jit>();
    }
#endif
    return ret;
}

static map<string, shared_ptr<jit_engine_t>> test_jit_engines = get_engines();

//===========================================================================

TEST(GCCore_CPU_jit_workload_for_debugging, TestAssignConstToTensorElem) {
    SKIP_BOUNDARY_CHECK();
    ir_builder_t builder;

    // Has no observable behavior; just used to generate object code snippets
    // for debugging purposes.
    _function_(datatypes::void_t, foo, _arg_("buf_A", datatypes::f32, {10})) {
        _bind_(buf_A);
        _var_(idx_B, datatypes::s32);
        idx_B = 1;
        buf_A[idx_B] = 42.0f;
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    float A[10];

    for (size_t i = 0; i < 10; ++i) {
        A[i] = float(i);
    }

    generic_val generic_args[] = {(void *)(A)};

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

        // TODO(xxx): In the cfake_jit, the generic wrapper has TWO
        // parameters if 'is_parallel' is true. (See
        // 'write_cpp_generic_wrapper(...)' in codegen/codegen_c.cpp) For now
        // we're assuming 'is_parallel' is false.
        jf->call_generic_default(generic_args);
    }
}

TEST(GCCore_CPU_jit_workload_for_debugging, TestAssignS32ConstToVar) {
    ir_builder_t builder;

    // Has no observable behavior; just used to generate object code snippets
    // for debugging purposes.
    _function_(datatypes::void_t, foo) {
        _var_(my_int, datatypes::s32);
        my_int = 42;
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    generic_val generic_args[] = {1.0f};

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

        // TODO(xxx): In the cfake_jit, the generic wrapper has TWO
        // parameters if 'is_parallel' is true. (See
        // 'write_cpp_generic_wrapper(...)' in codegen/codegen_c.cpp) For now
        // we're assuming 'is_parallel' is false.
        jf->call_generic_default(generic_args);
    }
}

TEST(GCCore_CPU_jit_workload_for_debugging, TestAssignS32VarToVar) {
    ir_builder_t builder;

    // Has no observable behavior; just used to generate object code snippets
    // for debugging purposes.
    _function_(datatypes::void_t, foo) {
        _var_(my_src, datatypes::s32);
        _var_(my_dest, datatypes::s32);
        my_src = 42;
        my_dest = my_src;
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    generic_val generic_args[] = {1.0f};

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

        // TODO(xxx): In the cfake_jit, the generic wrapper has TWO
        // parameters if 'is_parallel' is true. (See
        // 'write_cpp_generic_wrapper(...)' in codegen/codegen_c.cpp) For now
        // we're assuming 'is_parallel' is false.
        jf->call_generic_default(generic_args);
    }
}

TEST(GCCore_CPU_jit_workload_for_debugging, TestIndexingRvalue) {
    ir_builder_t builder;

    // Has no observable behavior; just used to generate object code snippets
    // for debugging purposes.
    //    _function_(
    //            datatypes::void_t, bar, _arg_("bar_buf_A", datatypes::f32,
    //            {10})) {
    //        _bind_(bar_buf_A);
    //    }

    _function_(datatypes::void_t, foo, _arg_("buf_A", datatypes::f32, {10}),
            _arg_("buf_B", datatypes::f32, {10})) {
        _bind_(buf_A, buf_B);
        buf_A[3] = buf_B[5];
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);
    float A[10];
    float B[10];

    for (size_t i = 0; i < 10; ++i) {
        A[i] = float(i);
        B[i] = float(-1.0 * i);
    }

    generic_val generic_args[] = {(void *)(A), (void *)(B)};

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

        // TODO(xxx): In the cfake_jit, the generic wrapper has TWO
        // parameters if 'is_parallel' is true. (See
        // 'write_cpp_generic_wrapper(...)' in codegen/codegen_c.cpp) For now
        // we're assuming 'is_parallel' is false.
        jf->call_generic_default(generic_args);
    }
}

TEST(GCCore_CPU_jit_workload_for_debugging, TestOneReturnF32) {
    ir_builder_t builder;

    _function_(datatypes::f32, foo) { _return_(42.0f); }

    _function_(datatypes::void_t, bar, _arg_("pf32", datatypes::f32, {1})) {
        _bind_(pf32);
        pf32[0] = foo();
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo, bar}, 0);
    float x = 21;

    generic_val generic_args[] = {(void *)(&x)};

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

        shared_ptr<jit_function_t> j_bar = jm->get_function("bar");

        EXPECT_NE(j_bar, nullptr);
        if (!j_bar) { continue; }

        j_bar->call_generic_default(generic_args);
        EXPECT_FLOAT_EQ(x, 42.0f);
    }
}

TEST(GCCore_CPU_jit_workload_for_debugging, TestOneReturnU64) {
    ir_builder_t builder;

    _function_(datatypes::index, foo) { _return_(uint64_t(42)); }

    _function_(datatypes::void_t, bar, _arg_("p", datatypes::index, {1})) {
        _bind_(p);
        p[0] = foo();
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo, bar}, 0);
    uint64_t x = 123;

    generic_val generic_args[] = {(void *)(&x)};

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

        shared_ptr<jit_function_t> j_bar = jm->get_function("bar");

        EXPECT_NE(j_bar, nullptr);
        if (!j_bar) { continue; }

        j_bar->call_generic_default(generic_args);
        EXPECT_EQ(x, uint64_t(42));
    }
}

// Disabled until Xbyak JIT engine has the necessary support for u8.
TEST(GCCore_CPU_jit_workload_for_debugging, DISABLED_TestOneReturnU8) {
    ir_builder_t builder;

    _function_(datatypes::u8, foo) {
        _return_(make_expr<constant_node>(uint64_t(42), datatypes::u8));
    }

    _function_(datatypes::void_t, bar, _arg_("p", datatypes::u8, {1})) {
        _bind_(p);
        p[0] = foo();
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo, bar}, 0);
    uint8_t x = 21;

    generic_val generic_args[] = {(void *)(&x)};

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

        shared_ptr<jit_function_t> j_bar = jm->get_function("bar");

        EXPECT_NE(j_bar, nullptr);
        if (!j_bar) { continue; }

        j_bar->call_generic_default(generic_args);
        EXPECT_EQ(x, 42);
    }
}

TEST(GCCore_CPU_jit_workload_for_debugging, TestOneReturnS32) {
    ir_builder_t builder;

    _function_(datatypes::s32, foo) { _return_(-42); }

    _function_(datatypes::void_t, bar, _arg_("p", datatypes::s32, {1})) {
        _bind_(p);
        p[0] = foo();
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo, bar}, 0);
    int32_t x = 21;

    generic_val generic_args[] = {(void *)(&x)};

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

        shared_ptr<jit_function_t> j_bar = jm->get_function("bar");

        EXPECT_NE(j_bar, nullptr);
        if (!j_bar) { continue; }

        j_bar->call_generic_default(generic_args);
        EXPECT_EQ(x, -42);
    }
}

// Disabled until Xbyak-jit-engine supports the if/else branch used by the
// test function 'foo'.
TEST(GCCore_CPU_jit_workload_for_debugging, TestMultiReturnS32) {
    const uint64_t branch_selector_a = 0;
    const uint64_t branch_selector_b = 1;

    const int32_t return_val_a = 42;
    const int32_t return_val_b = -42;

    ir_builder_t builder;

    _function_(
            datatypes::s32, foo, _arg_("branch_selector", datatypes::index)) {
        _bind_(branch_selector);
        _if_(branch_selector == branch_selector_a) { _return_(return_val_a); }
        _else_ { _return_(return_val_b); }
    }

    _function_(datatypes::void_t, bar, _arg_("p", datatypes::s32, {1}),
            _arg_("branch_selector", datatypes::index)) {
        _bind_(p, branch_selector);
        p[0] = foo(branch_selector);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo, bar}, 1);

    int32_t x = 21;

    // branch A...
    {
        generic_val generic_args[] = {(void *)(&x), branch_selector_a};
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

            shared_ptr<jit_function_t> j_bar = jm->get_function("bar");

            EXPECT_NE(j_bar, nullptr);
            if (!j_bar) { continue; }

            j_bar->call_generic_default(generic_args);
            EXPECT_EQ(x, return_val_a);
        }
    }

    // branch B...
    {
        generic_val generic_args[] = {(void *)(&x), branch_selector_b};
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

            shared_ptr<jit_function_t> j_bar = jm->get_function("bar");

            EXPECT_NE(j_bar, nullptr);
            if (!j_bar) { continue; }

            j_bar->call_generic_default(generic_args);
            EXPECT_EQ(x, return_val_b);
        }
    }
}

// At a basic level, this unit test simply confirms that it's
// possible to JIT-compile a builder function that calls one of
// our `builtin::` functions.
//
// To confirm that the specified function (`print_int`) was
// actually called successfully, examine the gtest stdout for
// confirmation that the value '42' was actually printed.
TEST(GCCore_CPU_jit_workload_for_debugging, TestCallExternal) {
    ir_builder_t builder;

    _function_(datatypes::void_t, foo) {
        builtin::print_int(make_constant(int32_t(42)));
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    generic_val *generic_args = nullptr;

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

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        j_foo->call_generic_default(generic_args);
    }
}

TEST(GCCore_CPU_jit_workload_for_debugging, TestCallIntegerStackArgs) {
    ir_builder_t builder;

    _function_(datatypes::void_t, foo, _arg_("result", datatypes::s32, {1}),
            _arg_("p1", datatypes::s32), _arg_("p2", datatypes::s32),
            _arg_("p3", datatypes::s32), _arg_("p4", datatypes::s32),
            _arg_("p5", datatypes::s32), _arg_("p6", datatypes::s32),
            _arg_("p7", datatypes::s32), _arg_("p8", datatypes::s32)) {
        _bind_(result, p1, p2, p3, p4, p5, p6, p7, p8);
        result[0] = p8 * (p7 + p6 + p5 + p4 + p3 + p2 + p1);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo}, 0);

    int32_t host_p1 = 1;
    int32_t host_p2 = 2;
    int32_t host_p3 = 4;
    int32_t host_p4 = 8;
    int32_t host_p5 = 16;
    int32_t host_p6 = 32;
    int32_t host_p7 = 64;
    int32_t host_p8 = 128;

    int32_t actual_result = 0;
    int32_t expected_result = host_p8
            * (host_p7 + host_p6 + host_p5 + host_p4 + host_p3 + host_p2
                    + host_p1);

    generic_val generic_args[] = {&actual_result, host_p1, host_p2, host_p3,
            host_p4, host_p5, host_p6, host_p7, host_p8};

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

        shared_ptr<jit_function_t> j_foo = jm->get_function("foo");

        EXPECT_NE(j_foo, nullptr);
        if (!j_foo) { continue; }

        j_foo->call_generic_default(generic_args);

        EXPECT_EQ(expected_result, actual_result);
    }
}

TEST(GCCore_CPU_jit_workload_for_debugging, TestDeadFuncCallReturnValue) {
    ir_builder_t builder;

    // TODO(xxx): add case for dead return value but still need call happen
    _function_(datatypes::f32, tmp) {
        _var_(c, datatypes::f32);
        c = 1.0f;
        _return_(c);
    }
    _function_(datatypes::f32, foo, _arg_("idx", datatypes::f32)) {
        _bind_(idx);
        _var_(b, datatypes::f32);
        b = 0;
        _for_(i, 0, 200) {
            _var_(c, datatypes::f32);
            c = tmp();
            b = b + 1;
        }
        _return_(b);
    }

    ir_module_ptr ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), vector<func_t> {foo, tmp}, 0);

    generic_val generic_args[] = {1.0f};

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

        shared_ptr<jit_function_t> jf = jm->get_function("tmp");

        EXPECT_NE(jf, nullptr);
        if (!jf) { continue; }

        jf->call_generic_default(generic_args);
    }
}
