/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_OCL_DISPATCH_H
#define GPU_OCL_DISPATCH_H

#define ROUND_UP(a, b) (((a) + (b)-1) / (b))

#ifndef USE_CUSTOM_GWS_GET_ID
#define GWS_GET_THREAD_ID(index) get_global_id(index)
#endif

#ifdef USE_INT32_OFFSET
#define off_t int
#else
#define off_t long
#endif

#ifdef GWS_WITH_RUNTIME_PARAMS

// Shortcut accessors for special cases.
#define GWS_OP_ZERO(idx, stride, max, block) 0
#define GWS_OP_SOLO(idx, stride, max, block) (idx)
#define GWS_OP_FIRST(idx, stride, max, block) ((idx) / (stride))
#define GWS_OP_MOD(idx, stride, max, block) ((idx) / (stride) % (max))
#define GWS_OP_SOLO_BLOCK(idx, stride, max, block) (idx) * (block)
#define GWS_OP_FIRST_BLOCK(idx, stride, max, block) ((idx) / (stride)) * (block)
#define GWS_OP_MOD_BLOCK(idx, stride, max, block) \
    ((idx) / (stride) % (max)) * (block)

#define DEFAULT_DISPATCHER_SUFFIX DEFAULT

// In C++:
// Create a reusable_dispatch_config_t class, and:
// 1. register buffers with reusable_dispatch_config_t::register_buffer
// 2. label dimensions with reusable_dispatch_config_t::define_dim_index
//
// Then use reusable_dispatch_config_t::generate to create the frozen
// reusable_dispatch_t class. This can then be further split into compile
// and runtime structs with
//
// reusable_dispatch_t::get_compile_params
// reusable_dispatch_t::get_runtime_params
//
// When initializing the kernel, use the dispatch_compile_params::def_kernel_macros to
// 1. Set the dispatcher suffix (defaults to DEFAULT)
// 2. Define buffer access macros (GWS_GET_OFF_NAMED and GWS_GET_BUFFER_POS_NAMED)

// GWS_GET_NAMED(DIMNAME, SUFFIX, rt_params) expands to a sum of GWS<X>_GET_ID<Y>
// terms (defined below), which the dispatcher is responsible for preparing. For each
// term, the dispatcher must also define GWS<X>_OP<Y>, GWS<X>_IDX<Y>, and GWS<X>_RT_IDX<Y>
// to fully define the term's operation.

// TODO: implement vectorization
// clang-format off
#define GWS0_GET_ID0(rt_params) GWS0_OP0(GWS_GET_THREAD_ID(GWS0_IDX0), (rt_params).strides[GWS0_RT_IDX0], (rt_params).sizes[GWS0_RT_IDX0], (rt_params).blocks[GWS0_RT_IDX0])
#define GWS0_GET_ID1(rt_params) GWS0_OP1(GWS_GET_THREAD_ID(GWS0_IDX1), (rt_params).strides[GWS0_RT_IDX1], (rt_params).sizes[GWS0_RT_IDX1], (rt_params).blocks[GWS0_RT_IDX1])
#define GWS0_GET_ID2(rt_params) GWS0_OP2(GWS_GET_THREAD_ID(GWS0_IDX2), (rt_params).strides[GWS0_RT_IDX2], (rt_params).sizes[GWS0_RT_IDX2], (rt_params).blocks[GWS0_RT_IDX2])
#define GWS0_GET_ID3(rt_params) GWS0_OP3(GWS_GET_THREAD_ID(GWS0_IDX3), (rt_params).strides[GWS0_RT_IDX3], (rt_params).sizes[GWS0_RT_IDX3], (rt_params).blocks[GWS0_RT_IDX3])
#define GWS0_GET_ID4(rt_params) GWS0_OP4(GWS_GET_THREAD_ID(GWS0_IDX4), (rt_params).strides[GWS0_RT_IDX4], (rt_params).sizes[GWS0_RT_IDX4], (rt_params).blocks[GWS0_RT_IDX4])
#define GWS0_GET_ID5(rt_params) GWS0_OP5(GWS_GET_THREAD_ID(GWS0_IDX5), (rt_params).strides[GWS0_RT_IDX5], (rt_params).sizes[GWS0_RT_IDX5], (rt_params).blocks[GWS0_RT_IDX5])
#define GWS0_GET_ID6(rt_params) GWS0_OP6(GWS_GET_THREAD_ID(GWS0_IDX6), (rt_params).strides[GWS0_RT_IDX6], (rt_params).sizes[GWS0_RT_IDX6], (rt_params).blocks[GWS0_RT_IDX6])
#define GWS0_GET_ID7(rt_params) GWS0_OP7(GWS_GET_THREAD_ID(GWS0_IDX7), (rt_params).strides[GWS0_RT_IDX7], (rt_params).sizes[GWS0_RT_IDX7], (rt_params).blocks[GWS0_RT_IDX7])
#define GWS0_GET_ID8(rt_params) GWS0_OP8(GWS_GET_THREAD_ID(GWS0_IDX8), (rt_params).strides[GWS0_RT_IDX8], (rt_params).sizes[GWS0_RT_IDX8], (rt_params).blocks[GWS0_RT_IDX8])
#define GWS0_GET_ID9(rt_params) GWS0_OP9(GWS_GET_THREAD_ID(GWS0_IDX9), (rt_params).strides[GWS0_RT_IDX9], (rt_params).sizes[GWS0_RT_IDX9], (rt_params).blocks[GWS0_RT_IDX9])

#define GWS1_GET_ID0(rt_params) GWS1_OP0(GWS_GET_THREAD_ID(GWS1_IDX0), (rt_params).strides[GWS1_RT_IDX0], (rt_params).sizes[GWS1_RT_IDX0], (rt_params).blocks[GWS1_RT_IDX0])
#define GWS1_GET_ID1(rt_params) GWS1_OP1(GWS_GET_THREAD_ID(GWS1_IDX1), (rt_params).strides[GWS1_RT_IDX1], (rt_params).sizes[GWS1_RT_IDX1], (rt_params).blocks[GWS1_RT_IDX1])
#define GWS1_GET_ID2(rt_params) GWS1_OP2(GWS_GET_THREAD_ID(GWS1_IDX2), (rt_params).strides[GWS1_RT_IDX2], (rt_params).sizes[GWS1_RT_IDX2], (rt_params).blocks[GWS1_RT_IDX2])
#define GWS1_GET_ID3(rt_params) GWS1_OP3(GWS_GET_THREAD_ID(GWS1_IDX3), (rt_params).strides[GWS1_RT_IDX3], (rt_params).sizes[GWS1_RT_IDX3], (rt_params).blocks[GWS1_RT_IDX3])
#define GWS1_GET_ID4(rt_params) GWS1_OP4(GWS_GET_THREAD_ID(GWS1_IDX4), (rt_params).strides[GWS1_RT_IDX4], (rt_params).sizes[GWS1_RT_IDX4], (rt_params).blocks[GWS1_RT_IDX4])
#define GWS1_GET_ID5(rt_params) GWS1_OP5(GWS_GET_THREAD_ID(GWS1_IDX5), (rt_params).strides[GWS1_RT_IDX5], (rt_params).sizes[GWS1_RT_IDX5], (rt_params).blocks[GWS1_RT_IDX5])
#define GWS1_GET_ID6(rt_params) GWS1_OP6(GWS_GET_THREAD_ID(GWS1_IDX6), (rt_params).strides[GWS1_RT_IDX6], (rt_params).sizes[GWS1_RT_IDX6], (rt_params).blocks[GWS1_RT_IDX6])
#define GWS1_GET_ID7(rt_params) GWS1_OP7(GWS_GET_THREAD_ID(GWS1_IDX7), (rt_params).strides[GWS1_RT_IDX7], (rt_params).sizes[GWS1_RT_IDX7], (rt_params).blocks[GWS1_RT_IDX7])
#define GWS1_GET_ID8(rt_params) GWS1_OP8(GWS_GET_THREAD_ID(GWS1_IDX8), (rt_params).strides[GWS1_RT_IDX8], (rt_params).sizes[GWS1_RT_IDX8], (rt_params).blocks[GWS1_RT_IDX8])
#define GWS1_GET_ID9(rt_params) GWS1_OP9(GWS_GET_THREAD_ID(GWS1_IDX9), (rt_params).strides[GWS1_RT_IDX9], (rt_params).sizes[GWS1_RT_IDX9], (rt_params).blocks[GWS1_RT_IDX9])

#define GWS2_GET_ID0(rt_params) GWS2_OP0(GWS_GET_THREAD_ID(GWS2_IDX0), (rt_params).strides[GWS2_RT_IDX0], (rt_params).sizes[GWS2_RT_IDX0], (rt_params).blocks[GWS2_RT_IDX0])
#define GWS2_GET_ID1(rt_params) GWS2_OP1(GWS_GET_THREAD_ID(GWS2_IDX1), (rt_params).strides[GWS2_RT_IDX1], (rt_params).sizes[GWS2_RT_IDX1], (rt_params).blocks[GWS2_RT_IDX1])
#define GWS2_GET_ID2(rt_params) GWS2_OP2(GWS_GET_THREAD_ID(GWS2_IDX2), (rt_params).strides[GWS2_RT_IDX2], (rt_params).sizes[GWS2_RT_IDX2], (rt_params).blocks[GWS2_RT_IDX2])
#define GWS2_GET_ID3(rt_params) GWS2_OP3(GWS_GET_THREAD_ID(GWS2_IDX3), (rt_params).strides[GWS2_RT_IDX3], (rt_params).sizes[GWS2_RT_IDX3], (rt_params).blocks[GWS2_RT_IDX3])
#define GWS2_GET_ID4(rt_params) GWS2_OP4(GWS_GET_THREAD_ID(GWS2_IDX4), (rt_params).strides[GWS2_RT_IDX4], (rt_params).sizes[GWS2_RT_IDX4], (rt_params).blocks[GWS2_RT_IDX4])
#define GWS2_GET_ID5(rt_params) GWS2_OP5(GWS_GET_THREAD_ID(GWS2_IDX5), (rt_params).strides[GWS2_RT_IDX5], (rt_params).sizes[GWS2_RT_IDX5], (rt_params).blocks[GWS2_RT_IDX5])
#define GWS2_GET_ID6(rt_params) GWS2_OP6(GWS_GET_THREAD_ID(GWS2_IDX6), (rt_params).strides[GWS2_RT_IDX6], (rt_params).sizes[GWS2_RT_IDX6], (rt_params).blocks[GWS2_RT_IDX6])
#define GWS2_GET_ID7(rt_params) GWS2_OP7(GWS_GET_THREAD_ID(GWS2_IDX7), (rt_params).strides[GWS2_RT_IDX7], (rt_params).sizes[GWS2_RT_IDX7], (rt_params).blocks[GWS2_RT_IDX7])
#define GWS2_GET_ID8(rt_params) GWS2_OP8(GWS_GET_THREAD_ID(GWS2_IDX8), (rt_params).strides[GWS2_RT_IDX8], (rt_params).sizes[GWS2_RT_IDX8], (rt_params).blocks[GWS2_RT_IDX8])
#define GWS2_GET_ID9(rt_params) GWS2_OP9(GWS_GET_THREAD_ID(GWS2_IDX9), (rt_params).strides[GWS2_RT_IDX9], (rt_params).sizes[GWS2_RT_IDX9], (rt_params).blocks[GWS2_RT_IDX9])

#define GWS3_GET_ID0(rt_params) GWS3_OP0(GWS_GET_THREAD_ID(GWS3_IDX0), (rt_params).strides[GWS3_RT_IDX0], (rt_params).sizes[GWS3_RT_IDX0], (rt_params).blocks[GWS3_RT_IDX0])
#define GWS3_GET_ID1(rt_params) GWS3_OP1(GWS_GET_THREAD_ID(GWS3_IDX1), (rt_params).strides[GWS3_RT_IDX1], (rt_params).sizes[GWS3_RT_IDX1], (rt_params).blocks[GWS3_RT_IDX1])
#define GWS3_GET_ID2(rt_params) GWS3_OP2(GWS_GET_THREAD_ID(GWS3_IDX2), (rt_params).strides[GWS3_RT_IDX2], (rt_params).sizes[GWS3_RT_IDX2], (rt_params).blocks[GWS3_RT_IDX2])
#define GWS3_GET_ID3(rt_params) GWS3_OP3(GWS_GET_THREAD_ID(GWS3_IDX3), (rt_params).strides[GWS3_RT_IDX3], (rt_params).sizes[GWS3_RT_IDX3], (rt_params).blocks[GWS3_RT_IDX3])
#define GWS3_GET_ID4(rt_params) GWS3_OP4(GWS_GET_THREAD_ID(GWS3_IDX4), (rt_params).strides[GWS3_RT_IDX4], (rt_params).sizes[GWS3_RT_IDX4], (rt_params).blocks[GWS3_RT_IDX4])
#define GWS3_GET_ID5(rt_params) GWS3_OP5(GWS_GET_THREAD_ID(GWS3_IDX5), (rt_params).strides[GWS3_RT_IDX5], (rt_params).sizes[GWS3_RT_IDX5], (rt_params).blocks[GWS3_RT_IDX5])
#define GWS3_GET_ID6(rt_params) GWS3_OP6(GWS_GET_THREAD_ID(GWS3_IDX6), (rt_params).strides[GWS3_RT_IDX6], (rt_params).sizes[GWS3_RT_IDX6], (rt_params).blocks[GWS3_RT_IDX6])
#define GWS3_GET_ID7(rt_params) GWS3_OP7(GWS_GET_THREAD_ID(GWS3_IDX7), (rt_params).strides[GWS3_RT_IDX7], (rt_params).sizes[GWS3_RT_IDX7], (rt_params).blocks[GWS3_RT_IDX7])
#define GWS3_GET_ID8(rt_params) GWS3_OP8(GWS_GET_THREAD_ID(GWS3_IDX8), (rt_params).strides[GWS3_RT_IDX8], (rt_params).sizes[GWS3_RT_IDX8], (rt_params).blocks[GWS3_RT_IDX8])
#define GWS3_GET_ID9(rt_params) GWS3_OP9(GWS_GET_THREAD_ID(GWS3_IDX9), (rt_params).strides[GWS3_RT_IDX9], (rt_params).sizes[GWS3_RT_IDX9], (rt_params).blocks[GWS3_RT_IDX9])
// clang-format on

// GWS_<NAME>_<SUFFIX>
#define DISPATCH_BUFFER_ALIAS(NAME, SUFFIX) \
    CONCAT2(CONCAT2(GWS_, NAME), CONCAT2(_, SUFFIX))

// Ultimately resolves to GWS_<NAME>_<SUFFIX>_OFF, defined by dispatcher
// as sums of GWS<X>_GET_ID<Y> terms
#define GWS_GET_OFF_NAMED(NAME, SUFFIX, rt_params) \
    CONCAT2(DISPATCH_BUFFER_ALIAS(NAME, SUFFIX), _OFF)(rt_params)

#define GWS_GET_OFF(NAME, rt_params) \
    GWS_GET_OFF_NAMED(NAME, DEFAULT_DISPATCHER_SUFFIX, rt_params)

// Convert GWS_GET(dim, dispatch, rt_params) to GWS_GET_dim_dispatch(rt_params) (which is in turn defined by the dispatcher)
#define GWS_GET_NAMED(dim_name, dispatcher_name, rt_params) \
    CONCAT2(CONCAT2(GWS_GET_, dim_name), CONCAT2(_, dispatcher_name))(rt_params)
#define GWS_GET(dim_name, rt_params) \
    GWS_GET_NAMED(dim_name, DEFAULT_DISPATCHER_SUFFIX, rt_params)

// Macros for getting a buffer pointer at the correct location,
// so that indexing is completely opaque to the implementation
#define GWS_GET_BUFFER_POS_NAMED(NAME, SUFFIX, rt_params, buffer) \
    &buffer[GWS_GET_OFF_NAMED(NAME, SUFFIX, rt_params)]

#define GWS_GET_BUFFER_POS(NAME, rt_params, buffer) \
    GWS_GET_BUFFER_POS_NAMED(NAME, DEFAULT_DISPATCHER_SUFFIX, rt_params, buffer)

#else

// Shortcut accessors for special cases.
#define GWS_OP_ZERO(idx, stride, max) 0
#define GWS_OP_SOLO(idx, stride, max) (idx)
#define GWS_OP_FIRST(idx, stride, max) ((idx) / (stride))
#define GWS_OP_MOD(idx, stride, max) ((idx) / (stride) % (max))

// clang-format off
#define GWS0_GET_ID0() GWS0_OP0(GWS_GET_THREAD_ID(GWS0_IDX0), GWS0_STRIDE0, ROUND_UP(GWS0_DIM0, GWS0_BLOCK0)) / GWS0_VEC_SIZE0 * GWS0_VEC_SIZE0 * GWS0_BLOCK0
#define GWS0_GET_ID1() GWS0_OP1(GWS_GET_THREAD_ID(GWS0_IDX1), GWS0_STRIDE1, ROUND_UP(GWS0_DIM1, GWS0_BLOCK1)) / GWS0_VEC_SIZE1 * GWS0_VEC_SIZE1 * GWS0_BLOCK1
#define GWS0_GET_ID2() GWS0_OP2(GWS_GET_THREAD_ID(GWS0_IDX2), GWS0_STRIDE2, ROUND_UP(GWS0_DIM2, GWS0_BLOCK2)) / GWS0_VEC_SIZE2 * GWS0_VEC_SIZE2 * GWS0_BLOCK2
#define GWS0_GET_ID3() GWS0_OP3(GWS_GET_THREAD_ID(GWS0_IDX3), GWS0_STRIDE3, ROUND_UP(GWS0_DIM3, GWS0_BLOCK3)) / GWS0_VEC_SIZE3 * GWS0_VEC_SIZE3 * GWS0_BLOCK3
#define GWS0_GET_ID4() GWS0_OP4(GWS_GET_THREAD_ID(GWS0_IDX4), GWS0_STRIDE4, ROUND_UP(GWS0_DIM4, GWS0_BLOCK4)) / GWS0_VEC_SIZE4 * GWS0_VEC_SIZE4 * GWS0_BLOCK4
#define GWS0_GET_ID5() GWS0_OP5(GWS_GET_THREAD_ID(GWS0_IDX5), GWS0_STRIDE5, ROUND_UP(GWS0_DIM5, GWS0_BLOCK5)) / GWS0_VEC_SIZE5 * GWS0_VEC_SIZE5 * GWS0_BLOCK5

#define GWS1_GET_ID0() GWS1_OP0(GWS_GET_THREAD_ID(GWS1_IDX0), GWS1_STRIDE0, ROUND_UP(GWS1_DIM0, GWS1_BLOCK0)) / GWS1_VEC_SIZE0 * GWS1_VEC_SIZE0 * GWS1_BLOCK0
#define GWS1_GET_ID1() GWS1_OP1(GWS_GET_THREAD_ID(GWS1_IDX1), GWS1_STRIDE1, ROUND_UP(GWS1_DIM1, GWS1_BLOCK1)) / GWS1_VEC_SIZE1 * GWS1_VEC_SIZE1 * GWS1_BLOCK1
#define GWS1_GET_ID2() GWS1_OP2(GWS_GET_THREAD_ID(GWS1_IDX2), GWS1_STRIDE2, ROUND_UP(GWS1_DIM2, GWS1_BLOCK2)) / GWS1_VEC_SIZE2 * GWS1_VEC_SIZE2 * GWS1_BLOCK2
#define GWS1_GET_ID3() GWS1_OP3(GWS_GET_THREAD_ID(GWS1_IDX3), GWS1_STRIDE3, ROUND_UP(GWS1_DIM3, GWS1_BLOCK3)) / GWS1_VEC_SIZE3 * GWS1_VEC_SIZE3 * GWS1_BLOCK3
#define GWS1_GET_ID4() GWS1_OP4(GWS_GET_THREAD_ID(GWS1_IDX4), GWS1_STRIDE4, ROUND_UP(GWS1_DIM4, GWS1_BLOCK4)) / GWS1_VEC_SIZE4 * GWS1_VEC_SIZE4 * GWS1_BLOCK4
#define GWS1_GET_ID5() GWS1_OP5(GWS_GET_THREAD_ID(GWS1_IDX5), GWS1_STRIDE5, ROUND_UP(GWS1_DIM5, GWS1_BLOCK5)) / GWS1_VEC_SIZE5 * GWS1_VEC_SIZE5 * GWS1_BLOCK5


#define GWS2_GET_ID0() GWS2_OP0(GWS_GET_THREAD_ID(GWS2_IDX0), GWS2_STRIDE0, ROUND_UP(GWS2_DIM0, GWS2_BLOCK0)) / GWS2_VEC_SIZE0 * GWS2_VEC_SIZE0 * GWS2_BLOCK0
#define GWS2_GET_ID1() GWS2_OP1(GWS_GET_THREAD_ID(GWS2_IDX1), GWS2_STRIDE1, ROUND_UP(GWS2_DIM1, GWS2_BLOCK1)) / GWS2_VEC_SIZE1 * GWS2_VEC_SIZE1 * GWS2_BLOCK1
#define GWS2_GET_ID2() GWS2_OP2(GWS_GET_THREAD_ID(GWS2_IDX2), GWS2_STRIDE2, ROUND_UP(GWS2_DIM2, GWS2_BLOCK2)) / GWS2_VEC_SIZE2 * GWS2_VEC_SIZE2 * GWS2_BLOCK2
#define GWS2_GET_ID3() GWS2_OP3(GWS_GET_THREAD_ID(GWS2_IDX3), GWS2_STRIDE3, ROUND_UP(GWS2_DIM3, GWS2_BLOCK3)) / GWS2_VEC_SIZE3 * GWS2_VEC_SIZE3 * GWS2_BLOCK3
#define GWS2_GET_ID4() GWS2_OP4(GWS_GET_THREAD_ID(GWS2_IDX4), GWS2_STRIDE4, ROUND_UP(GWS2_DIM4, GWS2_BLOCK4)) / GWS2_VEC_SIZE4 * GWS2_VEC_SIZE4 * GWS2_BLOCK4
#define GWS2_GET_ID5() GWS2_OP5(GWS_GET_THREAD_ID(GWS2_IDX5), GWS2_STRIDE5, ROUND_UP(GWS2_DIM5, GWS2_BLOCK5)) / GWS2_VEC_SIZE5 * GWS2_VEC_SIZE5 * GWS2_BLOCK5


#define GWS3_GET_ID0() GWS3_OP0(GWS_GET_THREAD_ID(GWS3_IDX0), GWS3_STRIDE0, ROUND_UP(GWS3_DIM0, GWS3_BLOCK0)) / GWS3_VEC_SIZE0 * GWS3_VEC_SIZE0 * GWS3_BLOCK0
#define GWS3_GET_ID1() GWS3_OP1(GWS_GET_THREAD_ID(GWS3_IDX1), GWS3_STRIDE1, ROUND_UP(GWS3_DIM1, GWS3_BLOCK1)) / GWS3_VEC_SIZE1 * GWS3_VEC_SIZE1 * GWS3_BLOCK1
#define GWS3_GET_ID2() GWS3_OP2(GWS_GET_THREAD_ID(GWS3_IDX2), GWS3_STRIDE2, ROUND_UP(GWS3_DIM2, GWS3_BLOCK2)) / GWS3_VEC_SIZE2 * GWS3_VEC_SIZE2 * GWS3_BLOCK2
#define GWS3_GET_ID3() GWS3_OP3(GWS_GET_THREAD_ID(GWS3_IDX3), GWS3_STRIDE3, ROUND_UP(GWS3_DIM3, GWS3_BLOCK3)) / GWS3_VEC_SIZE3 * GWS3_VEC_SIZE3 * GWS3_BLOCK3
#define GWS3_GET_ID4() GWS3_OP4(GWS_GET_THREAD_ID(GWS3_IDX4), GWS3_STRIDE4, ROUND_UP(GWS3_DIM4, GWS3_BLOCK4)) / GWS3_VEC_SIZE4 * GWS3_VEC_SIZE4 * GWS3_BLOCK4
#define GWS3_GET_ID5() GWS3_OP5(GWS_GET_THREAD_ID(GWS3_IDX5), GWS3_STRIDE5, ROUND_UP(GWS3_DIM5, GWS3_BLOCK5)) / GWS3_VEC_SIZE5 * GWS3_VEC_SIZE5 * GWS3_BLOCK5
// clang-format on
#endif

#define GWS0_GET_BLOCK0() GWS0_BLOCK0
#define GWS0_GET_BLOCK1() GWS0_BLOCK1
#define GWS0_GET_BLOCK2() GWS0_BLOCK2
#define GWS0_GET_BLOCK3() GWS0_BLOCK3
#define GWS0_GET_BLOCK4() GWS0_BLOCK4
#define GWS0_GET_BLOCK5() GWS0_BLOCK5

#define GWS1_GET_BLOCK0() GWS1_BLOCK0
#define GWS1_GET_BLOCK1() GWS1_BLOCK1
#define GWS1_GET_BLOCK2() GWS1_BLOCK2
#define GWS1_GET_BLOCK3() GWS1_BLOCK3
#define GWS1_GET_BLOCK4() GWS1_BLOCK4
#define GWS1_GET_BLOCK5() GWS1_BLOCK5

#define GWS2_GET_BLOCK0() GWS2_BLOCK0
#define GWS2_GET_BLOCK1() GWS2_BLOCK1
#define GWS2_GET_BLOCK2() GWS2_BLOCK2
#define GWS2_GET_BLOCK3() GWS2_BLOCK3
#define GWS2_GET_BLOCK4() GWS2_BLOCK4
#define GWS2_GET_BLOCK5() GWS2_BLOCK5

#define GWS3_GET_BLOCK0() GWS3_BLOCK0
#define GWS3_GET_BLOCK1() GWS3_BLOCK1
#define GWS3_GET_BLOCK2() GWS3_BLOCK2
#define GWS3_GET_BLOCK3() GWS3_BLOCK3
#define GWS3_GET_BLOCK4() GWS3_BLOCK4
#define GWS3_GET_BLOCK5() GWS3_BLOCK5

// Named kernel attributes - when source contains multiple kernels.
#ifdef GWS_WITH_RUNTIME_PARAMS
// Don't define LWS sizes since they can change at runtime
#define NAMED_KERNEL_ATTR_SG0(name)
#define NAMED_KERNEL_ATTR_SG1(name) \
    __attribute__((intel_reqd_sub_group_size(CONCAT2(GWS_SGS_, name))))
#else
#define NAMED_KERNEL_ATTR_SG0(name) \
    __attribute__((reqd_work_group_size(CONCAT2(GWS_LWS0_, name), \
            CONCAT2(GWS_LWS1_, name), CONCAT2(GWS_LWS2_, name))))

#define NAMED_KERNEL_ATTR_SG1(name) \
    NAMED_KERNEL_ATTR_SG0(name) \
    __attribute__((intel_reqd_sub_group_size(CONCAT2(GWS_SGS_, name))))

#endif

#define NAMED_KERNEL_ATTR(name) \
    CONCAT2(NAMED_KERNEL_ATTR_SG, CONCAT2(GWS_WITH_SG_, name))(name)
#define KERNEL_ATTR NAMED_KERNEL_ATTR(DEFAULT)

// Macro to emulate behavior of non-uniform work-groups. It is expected to be
// called at the beginning of the kernel.
// NOTE: The kernel cannot use synchronization within work-group (barrier,
// etc).
#define MAYBE_SKIP_NON_UNIFORM_WG() \
    do { \
        if ((GWS_0 != GWS_ORIG_0) && (GWS_ORIG_0 % LWS_0 != 0) \
                && (GWS_GET_THREAD_ID(0) >= GWS_ORIG_0)) \
            return; \
        if ((GWS_1 != GWS_ORIG_1) && (GWS_ORIG_1 % LWS_1 != 0) \
                && (GWS_GET_THREAD_ID(1) >= GWS_ORIG_1)) \
            return; \
        if ((GWS_2 != GWS_ORIG_2) && (GWS_ORIG_2 % LWS_2 != 0) \
                && (GWS_GET_THREAD_ID(2) >= GWS_ORIG_2)) \
            return; \
    } while (0)

#endif
