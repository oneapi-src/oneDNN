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

#include <sstream>
#include "test_utils.hpp"
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

static context_ptr get_ctx() {
    static context_ptr ret
            = std::make_shared<context_t>(*get_default_context());
    ret->flags_.dead_write_elimination_ = false;
    ret->flags_.index2var_ = false;
    ret->flags_.tensor2var_ = false;
    return ret;
}

TEST(GCCore_CPU_codegenc_cpp, TestCodegenC) {
    REQUIRE_PARALLEL();
    builder::ir_builder_t builder;
    const int shape1 = 128;
    for_loop li, lj, lk, lp;
    std::stringstream ss;
    std::stringstream offline_source;
    std::stringstream header_source;
    std::stringstream data_source;
    c_generator_optional_out_t opt_out {
            &offline_source, &header_source, &data_source};
    auto cgen = create_c_generator(ss, get_ctx(), true, &opt_out);

    _decl_func_(datatypes::void_t, bbb,
            _arg_("A", datatypes::f32, {shape1, shape1}),
            _arg_("len", datatypes::s32), _arg_("tsr", datatypes::pointer));

    // called twice in global var, check if is declared once
    _decl_func_(datatypes::f32, ginit);

    ir_module_ptr m = std::make_shared<ir_module_t>(get_ctx());
    _module_var_(m, val, datatypes::f32, 12.34f);
    _module_var_(m, val2, datatypes::f32, ginit());
    _global_var_(m, val3, datatypes::f32, ginit());
    _global_tensor_(m, gtsr, datatypes::f32, 100);
    _module_tensor_(m, stsr, datatypes::f32, 10000);
    _function_(datatypes::void_t, aaa,
            _arg_("A", datatypes::f32, {shape1, shape1}),
            _arg_("B", datatypes::f32, {shape1, shape1}),
            _arg_("C", datatypes::f32, {shape1, shape1}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, C, len);
        _tensor_(D, datatypes::f32, 2, 10);
        _tensor_(E, datatypes::f32, 100, 20);
        _tensor_(F, datatypes::f32, len);
        _tensor_(F_view, datatypes::s32, 10);
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(F, {3});
        builder.get_current_scope().body.back()->attr()["comments"]
                = std::vector<std::string> {"hello", "hi"};
        _named_for_(li, i, 0, shape1) {
            _named_for_(lj, j, 0, shape1) {
                _named_for_(lk, k, 0, shape1) {
                    C[{i, j}] = C[{i, j}] + A[{i, k}] * B[{k, j}];
                }
            }
        }
        F_view[0] = 1;
        gtsr[0] = 1.0f;
        val = 1.0f;
    }
    aaa->attr()["comments"]
            = std::vector<std::string> {"aaa", "@param 1", "@param 2"};
    _function_(datatypes::void_t, ddd, _arg_("len", datatypes::s32)) {}
    _function_(datatypes::s32, ccc,
            _arg_("A", datatypes::f32, {shape1, shape1}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, len);
        _for_(i, 0, shape1) {
            _for_(j, 0, shape1) { A[{i, j}] = 0; }
            builder.push_evaluate(
                    bbb(A, len, builder::tensor_ptr(A, {0, 100})));
        }
        _evaluate_call_(aaa, A, A, A, len);
        _evaluate_call_(ddd, 0);
        _var_(bbb, datatypes::pointer);
        bbb->attr()["prototype"] = ddd;
        builder.push_evaluate(
                make_expr<call_node>(bbb.get(), std::vector<expr> {2}));
        _return_(12);
    }
    aaa->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_NONE;
    aaa->attr()["allow_tensor_view"] = true;
    bbb->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_NONE;
    ccc->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_NONE;
    ddd->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_NONE;
    m->add_func({ccc});

    aaa->attr()[function_attrs::is_main] = true;
    cgen(m);
    std::string expected1 = R"(#include <runtime/kernel_include/cpu_include.hpp>

extern "C" void bbb(float* __restrict__ A, int32_t len, void* tsr) noexcept __attribute__((nonnull (1)));
/**
 * aaa
 * @param __stream the stream pointer, usually get_default_stream()
 * @param __module_data the module global data
 * @param 1
 * @param 2
*/
extern "C" void aaa(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int32_t len) noexcept __attribute__((nonnull (2,3,4,5)));
void* (*sc_aligned_malloc_fptr)(void* stream, uint64_t size) noexcept __attribute__((returns_nonnull))  /*__attribute__((malloc))*/;
void (*sc_aligned_free_fptr)(void* stream, void* ptr) noexcept;
extern "C" float ginit() noexcept;
extern "C" int32_t ccc(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, int32_t len) noexcept __attribute__((nonnull (2,3)));
extern "C" void __sc_init__(void* __stream, int8_t* __restrict__ __module_data) noexcept __attribute__((nonnull (2)));


extern "C" int32_t ccc(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, int32_t len) noexcept{
  for (uint64_t i = 0UL; i < 128UL; i += 1UL) {
    for (uint64_t j = 0UL; j < 128UL; j += 1UL) {
      A[((i * 128UL) + j)] = 0.f;
    }
    bbb(A, len, &A[100UL]);
  }
  aaa(__stream, __module_data, A, A, A, len);
  void* bbb;
  ((void(*)(void*, int8_t*, int32_t))bbb)(__stream, __module_data, 2);
  return 12;
}

extern "C" void aaa(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int32_t len) noexcept{
  float* gtsr = (float*)&__module_data[64UL];
  float& val = *(float*)(__module_data + 0);
  alignas(64) float D[20UL];
  float* E = (float*)sc_aligned_malloc_fptr(__stream, 8000UL);
  float* F = (float*)sc_aligned_malloc_fptr(__stream, ((uint64_t)len * 4UL));
  // hello
  // hi
  int32_t* F_view = (int32_t*)&F[3];
  for (uint64_t i = 0UL; i < 128UL; i += 1UL) {
    for (uint64_t j = 0UL; j < 128UL; j += 1UL) {
      for (uint64_t k = 0UL; k < 128UL; k += 1UL) {
        C[((i * 128UL) + j)] = (C[((i * 128UL) + j)] + (A[((i * 128UL) + k)] * B[(j + (k * 128UL))]));
      }
    }
  }
  F_view[0] = 1;
  gtsr[0] = 1.f;
  val = 1.f;
  sc_aligned_free_fptr(__stream, F);
  sc_aligned_free_fptr(__stream, E);
}

extern "C" void ddd(void* __stream, int8_t* __restrict__ __module_data, int32_t len) noexcept{
}

extern "C" void __sc_init__(void* __stream, int8_t* __restrict__ __module_data) noexcept{
  float& val = *(float*)(__module_data + 0);
  float& val2 = *(float*)(__module_data + 4);
  float& val3 = *(float*)(__module_data + 8);
  val = 12.3400002;
  val2 = ginit();
  val3 = ginit();
}

extern "C" void ccc_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
  ccc(__stream, __module_data, (float*)(args[0UL].v_ptr), args[1UL].v_int32_t);
}

extern "C" void aaa_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
  aaa(__stream, __module_data, (float*)(args[0UL].v_ptr), (float*)(args[1UL].v_ptr), (float*)(args[2UL].v_ptr), args[3UL].v_int32_t);
}

extern "C" void ddd_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
}

extern "C" void __sc_init___0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
  __sc_init__(__stream, __module_data);
}

)";
    EXPECT_EQ(ss.str(), expected1);

    std::string expected_offline
            = R"(#include <runtime/kernel_include/cpu_include.hpp>

#include <omp.h>
#define sc_get_thread_id omp_get_thread_num
#define sc_parallel_call_cpu_with_env sc_parallel_call_cpu_with_env_impl
static void bbb(float* __restrict__ A, int32_t len, void* tsr) noexcept __attribute__((nonnull (1)));
/**
 * aaa
 * @param __stream the stream pointer, usually get_default_stream()
 * @param __module_data the module global data
 * @param 1
 * @param 2
*/
static void aaa(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int32_t len) noexcept __attribute__((nonnull (2,3,4,5)));
extern "C" void* sc_aligned_malloc(void* stream, uint64_t size) noexcept __attribute__((returns_nonnull))  __attribute__((malloc));
extern "C" void sc_aligned_free(void* stream, void* ptr) noexcept;
static float ginit() noexcept;
static int32_t ccc(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, int32_t len) noexcept __attribute__((nonnull (2,3)));
extern "C" void __sc_init__(void* __stream, int8_t* __restrict__ __module_data) noexcept __attribute__((nonnull (2)));


static int32_t ccc(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, int32_t len) noexcept{
  for (uint64_t i = 0UL; i < 128UL; i += 1UL) {
    for (uint64_t j = 0UL; j < 128UL; j += 1UL) {
      A[((i * 128UL) + j)] = 0.f;
    }
    bbb(A, len, &A[100UL]);
  }
  aaa(__stream, __module_data, A, A, A, len);
  void* bbb;
  ((void(*)(void*, int8_t*, int32_t))bbb)(__stream, __module_data, 2);
  return 12;
}

static void aaa(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int32_t len) noexcept{
  float* gtsr = (float*)&__module_data[64UL];
  float& val = *(float*)(__module_data + 0);
  alignas(64) float D[20UL];
  float* E = (float*)sc_aligned_malloc(__stream, 8000UL);
  float* F = (float*)sc_aligned_malloc(__stream, ((uint64_t)len * 4UL));
  // hello
  // hi
  int32_t* F_view = (int32_t*)&F[3];
  for (uint64_t i = 0UL; i < 128UL; i += 1UL) {
    for (uint64_t j = 0UL; j < 128UL; j += 1UL) {
      for (uint64_t k = 0UL; k < 128UL; k += 1UL) {
        C[((i * 128UL) + j)] = (C[((i * 128UL) + j)] + (A[((i * 128UL) + k)] * B[(j + (k * 128UL))]));
      }
    }
  }
  F_view[0] = 1;
  gtsr[0] = 1.f;
  val = 1.f;
  sc_aligned_free(__stream, F);
  sc_aligned_free(__stream, E);
}

static void ddd(void* __stream, int8_t* __restrict__ __module_data, int32_t len) noexcept{
}

extern "C" void sc_init_aaa(void* __stream, int8_t* __restrict__ __module_data) noexcept{
  float& val = *(float*)(__module_data + 0);
  float& val2 = *(float*)(__module_data + 4);
  float& val3 = *(float*)(__module_data + 8);
  val = 12.3400002;
  val2 = ginit();
  val3 = ginit();
}

static void ccc_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
  ccc(__stream, __module_data, (float*)(args[0UL].v_ptr), args[1UL].v_int32_t);
}

extern "C" void aaa_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
  aaa(__stream, __module_data, (float*)(args[0UL].v_ptr), (float*)(args[1UL].v_ptr), (float*)(args[2UL].v_ptr), args[3UL].v_int32_t);
}

static void ddd_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
}

)";
    EXPECT_EQ(offline_source.str(), expected_offline);

    const char *expected_header = R"(#include <stdint.h>
#include <runtime/generic_val.hpp>
using generic_val = dnnl::impl::graph::gc::generic_val;

extern uint8_t aaa_data[40512];

/**
 * Initialize the aaa
 * @param __stream the stream pointer, usually get_default_stream()
 * @param __module_data the module global data
*/
extern "C" void sc_init_aaa(void* __stream, int8_t* __restrict__ __module_data) noexcept __attribute__((nonnull (2)));
/**
 * aaa
 * @param __stream the stream pointer, usually get_default_stream()
 * @param __module_data the module global data
 * @param args The array of arguments. It should contain the following:
 *   -param 1
 *   -param 2
*/
extern "C" void aaa_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept __attribute__((nonnull (2,3)));
)";

    EXPECT_EQ(header_source.str(), expected_header);
    const char *expected_data = R"(#include <stdint.h>

alignas(64) uint8_t aaa_data[40512] = {)";
    EXPECT_TRUE(utils::string_startswith(data_source.str(), expected_data));
}

TEST(GCCore_CPU_codegenc_cpp, TestCodegenCParallelFor) {
    REQUIRE_PARALLEL();
    builder::ir_builder_t builder;
    const int shape1 = 128;
    for_loop li, lj, lk, lp;
    std::stringstream ss;
    auto cgen = create_c_generator(ss, get_ctx(), true);
    auto m = std::make_shared<ir_module_t>(get_ctx());
    _global_var_(m, gv, datatypes::s32, expr());
    _function_(datatypes::void_t, aaa,
            _arg_("A", datatypes::f32, {shape1, shape1}),
            _arg_("B", datatypes::f32, {shape1, shape1}),
            _arg_("C", datatypes::f32, {shape1, shape1}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, C, len);
        _named_for_(li, i, 0, shape1, 1, for_type::PARALLEL) {
            gv = 1;
            _var_(v1, datatypes::f32);
            _tensor_(D, datatypes::f32, 2, 10);
            _tensor_(E, datatypes::f32, 10, 200);
            _named_for_(lj, j, 0, shape1) {
                _named_for_(lk, k, 0, shape1) {
                    C[{i, j}] = C[{i, j}] + D[{i, k}] + A[{i, k}] * B[{k, j}]
                            + len + v1;
                }
            }
        }
        _var_(t, datatypes::s32);
        t = t & 10;
        t = t | 10;
        t = t >> 10;
        t = t << t;
        t = len;
        _for_(i, 1, 100, 2, for_type::PARALLEL) { A[{i, i}] = i + t; }

        _tensor_(D, datatypes::s32, {8});
        _tensor_(E, datatypes::s32, {8});
        D[span_t({expr(0)}, 8)] = D[span_t({expr(0)}, 8)]
                << E[span_t({expr(0)}, 8)];
    }
    aaa->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_NONE;
    m->add_func({aaa});
    cgen(m);
    std::string expected1 = R"(#include <runtime/kernel_include/cpu_include.hpp>

void (*sc_parallel_call_cpu_with_env_fptr)(void* func, uint64_t flags, void* stream, int8_t* env, uint64_t begin, uint64_t end, uint64_t step, generic_val* args) noexcept;
static void aaa0_closure_0_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept __attribute__((nonnull (2,4)));
static void aaa0_closure_1_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept __attribute__((nonnull (2,4)));
extern "C" void aaa(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int32_t len) noexcept __attribute__((nonnull (2,3,4,5)));
void* (*sc_thread_aligned_malloc_fptr)(void* stream, uint64_t size) noexcept __attribute__((returns_nonnull))  /*__attribute__((malloc))*/;
void (*sc_thread_aligned_free_fptr)(void* stream, void* ptr) noexcept;
static void aaa0_closure_0(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, float* __restrict__ C, float* __restrict__ A, float* __restrict__ B, int32_t len) noexcept __attribute__((nonnull (2,4,5,6)));
static void aaa0_closure_1(void* __stream, int8_t* __restrict__ __module_data, uint64_t i_1, float* __restrict__ A, int32_t t) noexcept __attribute__((nonnull (2,4)));


extern "C" void aaa(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int32_t len) noexcept{
  generic_val __tempargs0[4UL];
  __tempargs0[0UL] = C;
  __tempargs0[1UL] = A;
  __tempargs0[2UL] = B;
  __tempargs0[3UL] = len;
  sc_parallel_call_cpu_with_env_fptr((void*)&aaa0_closure_0_0wrapper, 0UL, __stream, __module_data, 0UL, 128UL, 1UL, __tempargs0);
  int32_t t;
  t = (t & 10);
  t = (t | 10);
  t = (t >> 10);
  t = (t << t);
  t = len;
  generic_val __tempargs1[2UL];
  __tempargs1[0UL] = A;
  __tempargs1[1UL] = t;
  sc_parallel_call_cpu_with_env_fptr((void*)&aaa0_closure_1_0wrapper, 0UL, __stream, __module_data, 1UL, 100UL, 2UL, __tempargs1);
  int32_t D_2[8];
  int32_t E_3[8];
  vec_s32x8::store((vec_s32x8::load(&D_2[0]) << vec_s32x8::load(&E_3[0])), &D_2[0]);
}

extern "C" void aaa_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
  aaa(__stream, __module_data, (float*)(args[0UL].v_ptr), (float*)(args[1UL].v_ptr), (float*)(args[2UL].v_ptr), args[3UL].v_int32_t);
}

static void aaa0_closure_0(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, float* __restrict__ C, float* __restrict__ A, float* __restrict__ B, int32_t len) noexcept{
  int32_t& gv = *(int32_t*)(__module_data + 0);
  gv = 1;
  float v1;
  alignas(64) float D[20UL];
  float* E = (float*)sc_thread_aligned_malloc_fptr(__stream, 8000UL);
  for (uint64_t j = 0UL; j < 128UL; j += 1UL) {
    for (uint64_t k = 0UL; k < 128UL; k += 1UL) {
      C[((i * 128UL) + j)] = ((((C[((i * 128UL) + j)] + D[((i * 10UL) + k)]) + (A[((i * 128UL) + k)] * B[(j + (k * 128UL))])) + (float)len) + v1);
    }
  }
  sc_thread_aligned_free_fptr(__stream, E);
}

static void aaa0_closure_0_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept{
  aaa0_closure_0(__stream, __module_data, i, (float*)(args[0UL].v_ptr), (float*)(args[1UL].v_ptr), (float*)(args[2UL].v_ptr), args[3UL].v_int32_t);
}

static void aaa0_closure_1(void* __stream, int8_t* __restrict__ __module_data, uint64_t i_1, float* __restrict__ A, int32_t t) noexcept{
  A[(i_1 + (i_1 * 128UL))] = (float)((uint64_t)t + i_1);
}

static void aaa0_closure_1_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept{
  aaa0_closure_1(__stream, __module_data, i, (float*)(args[0UL].v_ptr), args[1UL].v_int32_t);
}

)";
    EXPECT_EQ(ss.str(), expected1);
}

TEST(GCCore_CPU_codegenc_cpp, TestCodegenCVector) {
    builder::ir_builder_t builder;
    std::stringstream ss;
    auto cgen = create_c_generator(ss, get_ctx(), true);

    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {512}),
            _arg_("B", datatypes::f32, {512}),
            _arg_("C", datatypes::f32, {512}),
            _arg_("D", datatypes::s32, {512})) {
        _bind_(A, B, C, D);
        _for_(i, 0, 512, 8) {
            D[span_t({i}, 8)] = make_expr<constant_node>(
                    std::vector<union_val> {INT64_C(1), INT64_C(2), INT64_C(3),
                            INT64_C(4), INT64_C(5), INT64_C(6), INT64_C(7),
                            INT64_C(8)},
                    sc_data_type_t::s32(8));
            C[span_t({i}, 8)] = make_expr<constant_node>(
                    std::vector<union_val> {
                            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                    sc_data_type_t::f32(8));
            D[span_t({i}, 8)] = make_expr<constant_node>(
                    INT64_C(1), sc_data_type_t::s32(8));
            C[span_t({i}, 8)]
                    = make_expr<constant_node>(2.0f, sc_data_type_t::f32(8));
            C[span_t({i}, 8)] = A[span_t({i}, 8)] + B[span_t({i}, 8)];
        }
    }

    cgen(ir_module_t::from_entry_func(get_ctx(), aaa));
    std::string expected1 = R"(#include <runtime/kernel_include/cpu_include.hpp>

extern "C" void aaa(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int32_t* __restrict__ D) noexcept __attribute__((nonnull (2,3,4,5,6)));


extern "C" void aaa(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int32_t* __restrict__ D) noexcept{
  for (uint64_t i = 0UL; i < 512UL; i += 8UL) {
    vec_s32x8::store(vec_s32x8(1, 2, 3, 4, 5, 6, 7, 8), &D[i]);
    vec_f32x8::store(vec_f32x8(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f), &C[i]);
    vec_s32x8::store(vec_s32x8(1), &D[i]);
    vec_f32x8::store(vec_f32x8(2.f), &C[i]);
    vec_f32x8::store((vec_f32x8::load(&A[i]) + vec_f32x8::load(&B[i])), &C[i]);
  }
}

extern "C" void aaa_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
  aaa(__stream, __module_data, (float*)(args[0UL].v_ptr), (float*)(args[1UL].v_ptr), (float*)(args[2UL].v_ptr), (int32_t*)(args[3UL].v_ptr));
}

)";
    EXPECT_EQ(ss.str(), expected1);
}

TEST(GCCore_CPU_codegenc_cpp, TestCodegenCGenericVal) {
    builder::ir_builder_t builder;
    std::stringstream ss;
    auto cgen = create_c_generator(ss, get_ctx(), false);

    _function_(datatypes::s32, aaa) {
        _var_(a, datatypes::generic);
        a = 100;
        a = 1.2f;
        _var_(b, datatypes::pointer);
        b = builder::make_cast(datatypes::pointer, a);
        b = builder::make_cast(
                sc_data_type_t::pointerof(sc_data_etype::F32), a);
        _return_(builder::make_cast(datatypes::s32, a));
    }

    cgen(aaa);
    std::string expected1 = R"(#include <runtime/kernel_include/cpu_include.hpp>

extern "C" int32_t aaa(void* __stream, int8_t* __restrict__ __module_data) noexcept{
  generic_val a;
  a = 100;
  a = 1.20000005;
  void* b;
  b = (void*)(a.v_ptr);
  b = (float*)(a.v_ptr);
  return a.v_int32_t;
}

)";
    EXPECT_EQ(ss.str(), expected1);
}

TEST(GCCore_CPU_codegenc_cpp, TestCodegenCCondition) {
    builder::ir_builder_t builder;
    std::stringstream ss;
    auto cgen = create_c_generator(ss, get_ctx(), true);

    _function_(datatypes::void_t, bbb, _arg_("A", datatypes::f32, {512}),
            _arg_("B", datatypes::f32, {512}),
            _arg_("C", datatypes::f32, {512}),
            _arg_("D", datatypes::f32, {512}),
            _arg_("E", datatypes::f32, {512})) {
        _bind_(A, B, C, D, E);
        _for_(i, 0, 512, 8) {
            E[span_t({i}, 8)] = builder::make_select(
                    C[span_t({i}, 8)] <= D[span_t({i}, 8)], A[span_t({i}, 8)],
                    B[span_t({i}, 8)]);
        }
    }

    cgen(ir_module_t::from_entry_func(get_ctx(), bbb));
    std::string expected1 = R"(#include <runtime/kernel_include/cpu_include.hpp>

extern "C" void bbb(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, float* __restrict__ D, float* __restrict__ E) noexcept __attribute__((nonnull (2,3,4,5,6,7)));


extern "C" void bbb(void* __stream, int8_t* __restrict__ __module_data, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, float* __restrict__ D, float* __restrict__ E) noexcept{
  for (uint64_t i = 0UL; i < 512UL; i += 8UL) {
    vec_f32x8::store(sc_select((vec_f32x8::load(&C[i]) <= vec_f32x8::load(&D[i])), vec_f32x8::load(&A[i]), vec_f32x8::load(&B[i])), &E[i]);
  }
}

extern "C" void bbb_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
  bbb(__stream, __module_data, (float*)(args[0UL].v_ptr), (float*)(args[1UL].v_ptr), (float*)(args[2UL].v_ptr), (float*)(args[3UL].v_ptr), (float*)(args[4UL].v_ptr));
}

)";
    EXPECT_EQ(ss.str(), expected1);
}

TEST(GCCore_CPU_codegenc_cpp, TestCodegenCGlobalTensor) {
    builder::ir_builder_t builder;
    std::stringstream ss;
    auto m = std::make_shared<ir_module_t>(get_ctx());
    _global_tensor_(m, gv, datatypes::s32, 4);
    int32_t values[] = {1, 2, 3, 4};
    gv.static_as<tensor>()->init_value_
            = std::make_shared<static_data_t>(values, 4 * sizeof(int32_t));

    auto cgen = create_c_generator(ss, get_ctx(), false);

    _function_(datatypes::s32, bbb) {
        gv[1] = 123;
        _return_(gv[0]);
    }
    m->add_func({bbb});
    auto ret = cgen(m);
    ret->attr_.get<std::shared_ptr<statics_table_t>>(
            ir_module_t::attr_key_t::MODULE_DATA_BUFFERS);
    std::stringstream expected;
    expected << R"(#include <runtime/kernel_include/cpu_include.hpp>



extern "C" int32_t bbb(void* __stream, int8_t* __restrict__ __module_data) noexcept{
  int32_t* gv = (int32_t*)&__module_data[0UL];
  gv[1] = 123;
  return gv[0];
}

)";
    EXPECT_EQ(ss.str(), expected.str());
}
