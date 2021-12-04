/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

/// @example convolution.cpp
/// > Annotated version: @ref convolution_example_cpp
///
/// @page convolution_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Convolution](@ref dev_guide_convolution) primitive in forward propagation
/// mode in two configurations - with and without groups.
///
/// Key optimizations included in this example:
/// - Creation of optimized memory format from the primitive descriptor;
/// - Primitive attributes with fused post-ops.
///
/// @page convolution_example_cpp Convolution Primitive Example
/// @copydetails convolution_example_cpp_short
///
/// @include convolution.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void convolution_example(dnnl::engine::kind engine_kind) {
    // Setup convolution op descriptor
    dnnl::engine engine(engine_kind, 0);
    dnnl::stream engine_stream(engine);

    memory::dim IC1 = 32, OC1 = IC1;
    memory::dim IC2 = 32, OC2 = IC2;

    memory::dim N = 1, IH = 1, IW = 1, KH = 3, KW = 3, PH_L = 1, PH_R = 1,
                PW_L = 1, PW_R = 1, SH = 1, SW = 1,
                OH = (IH - KH + PH_L + PH_R) / SH + 1,
                OW = (IW - KW + PW_L + PW_R) / SW + 1;

    memory::dims src1_dims = {N, IC1, IH, IW};
    memory::dims weights1_dims = {OC1, IC1, KH, KW};
    memory::dims bias1_dims = {OC1};
    memory::dims dst1_dims = {N, OC1, OH, OW};

    memory::dims src2_dims = {N, IC2, IH, IW};
    memory::dims weights2_dims = {OC2, IC2, KH, KW};
    memory::dims bias2_dims = {OC2};
    memory::dims dst2_dims = {N, OC2, OH, OW};

    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};

    std::vector<dt> dts {dt::u8, dt::s8, dt::f32, dt::u8};
    //std::vector<dt> dts {dt::f16, dt::f16, dt::f32, dt::f16};

    // ========================================================================
    // Prepare user data for convolution 1 (for validation)
    // ========================================================================
    std::vector<float> src1_data(product(src1_dims));
    std::vector<float> weights1_data(product(weights1_dims));
    std::vector<float> bias1_data(product(bias1_dims));

    std::generate(src1_data.begin(), src1_data.end(), []() {
        static int i = 0;
        return 3.f * std::cos(i++ / 10.f);
    });
    std::generate(weights1_data.begin(), weights1_data.end(), []() {
        static int i = 0;
        return 3.f * std::sin(i++ * 2.f);
    });
    std::generate(bias1_data.begin(), bias1_data.end(), []() {
        static int i = 0;
        return std::tanh(i++);
    });

    auto user_src1_mem
            = ocl_interop::make_memory({src1_dims, dt::f32, tag::abcd}, engine,
                    ocl_interop::memory_kind::usm);
    auto user_weights1_mem
            = ocl_interop::make_memory({weights1_dims, dt::f32, tag::abcd},
                    engine, ocl_interop::memory_kind::usm);
    auto user_bias1_mem
            = ocl_interop::make_memory({bias1_dims, dt::f32, tag::a}, engine,
                    ocl_interop::memory_kind::usm);

    write_to_dnnl_memory(src1_data.data(), user_src1_mem);
    write_to_dnnl_memory(weights1_data.data(), user_weights1_mem);
    write_to_dnnl_memory(bias1_data.data(), user_bias1_mem);

    // ========================================================================
    // Prepare user data for convolution 2 (for validation)
    // ========================================================================
    std::vector<float> src2_data(product(src2_dims));
    std::vector<float> weights2_data(product(weights2_dims));
    std::vector<float> bias2_data(product(bias2_dims));

    std::generate(src2_data.begin(), src2_data.end(), []() {
        static int i = 0;
        return 2.f * std::cos(i++ / 9.f);
    });
    std::generate(weights2_data.begin(), weights2_data.end(), []() {
        static int i = 0;
        return 2.f * std::sin(i++ * 3.f);
    });
    std::generate(bias2_data.begin(), bias2_data.end(), []() {
        static int i = 0;
        return std::tanh(i++ * 4.5f);
    });

    auto user_src2_mem
            = ocl_interop::make_memory({src2_dims, dt::f32, tag::abcd}, engine,
                    ocl_interop::memory_kind::usm);
    auto user_weights2_mem
            = ocl_interop::make_memory({weights2_dims, dt::f32, tag::abcd},
                    engine, ocl_interop::memory_kind::usm);
    auto user_bias2_mem
            = ocl_interop::make_memory({bias2_dims, dt::f32, tag::a}, engine,
                    ocl_interop::memory_kind::usm);

    write_to_dnnl_memory(src2_data.data(), user_src2_mem);
    write_to_dnnl_memory(weights2_data.data(), user_weights2_mem);
    write_to_dnnl_memory(bias2_data.data(), user_bias2_mem);

    // ========================================================================
    // Create primitive descriptor for conv 1
    // ========================================================================
    auto conv1_src_md = memory::desc(src1_dims, dts[0], tag::any);
    auto conv1_weights_md = memory::desc(weights1_dims, dts[1], tag::any);
    auto conv1_bias_md = memory::desc(bias1_dims, dts[2], tag::a);
    auto conv1_dst_md = memory::desc(dst1_dims, dts[3], tag::any);

    auto conv1_desc = convolution_forward::desc(prop_kind::forward_training,
            algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
            conv1_bias_md, conv1_dst_md, strides_dims, padding_dims_l,
            padding_dims_r);

    auto conv1_pd = convolution_forward::primitive_desc(conv1_desc, engine);

    auto conv_src1_mem = ocl_interop::make_memory(
            conv1_pd.src_desc(), engine, ocl_interop::memory_kind::usm);
    auto conv_weights1_mem = ocl_interop::make_memory(
            conv1_pd.weights_desc(), engine, ocl_interop::memory_kind::usm);
    auto conv_bias1_mem = ocl_interop::make_memory(
            conv1_pd.bias_desc(), engine, ocl_interop::memory_kind::usm);
    auto conv_dst1_mem = ocl_interop::make_memory(
            conv1_pd.dst_desc(), engine, ocl_interop::memory_kind::usm);

    // Note: do reorder regardless to pass user data.
    reorder(user_src1_mem, conv_src1_mem)
            .execute(engine_stream, user_src1_mem, conv_src1_mem);
    reorder(user_weights1_mem, conv_weights1_mem)
            .execute(engine_stream, user_weights1_mem, conv_weights1_mem);
    reorder(user_bias1_mem, conv_bias1_mem)
            .execute(engine_stream, user_bias1_mem, conv_bias1_mem);

    auto conv1_prim = convolution_forward(conv1_pd);
    std::unordered_map<int, memory> conv1_args;
    conv1_args.insert({DNNL_ARG_SRC, conv_src1_mem});
    conv1_args.insert({DNNL_ARG_WEIGHTS, conv_weights1_mem});
    conv1_args.insert({DNNL_ARG_BIAS, user_bias1_mem});
    conv1_args.insert({DNNL_ARG_DST, conv_dst1_mem});
    conv1_prim.execute(engine_stream, conv1_args);

    // ========================================================================
    // Create primitive descriptor for conv 2
    // ========================================================================
    auto conv2_src_md = memory::desc(src2_dims, dts[0], tag::any);
    auto conv2_weights_md = memory::desc(weights2_dims, dts[1], tag::any);
    auto conv2_bias_md = memory::desc(bias2_dims, dts[2], tag::a);
    auto conv2_dst_md = memory::desc(dst2_dims, dts[3], tag::any);

    auto conv2_desc = convolution_forward::desc(prop_kind::forward_training,
            algorithm::convolution_direct, conv2_src_md, conv2_weights_md,
            conv2_bias_md, conv2_dst_md, strides_dims, padding_dims_l,
            padding_dims_r);

    auto conv2_pd = convolution_forward::primitive_desc(conv2_desc, engine);

    auto conv_src2_mem = ocl_interop::make_memory(
            conv2_pd.src_desc(), engine, ocl_interop::memory_kind::usm);
    auto conv_weights2_mem = ocl_interop::make_memory(
            conv2_pd.weights_desc(), engine, ocl_interop::memory_kind::usm);
    auto conv_bias2_mem = ocl_interop::make_memory(
            conv2_pd.bias_desc(), engine, ocl_interop::memory_kind::usm);
    auto conv_dst2_mem = ocl_interop::make_memory(
            conv2_pd.dst_desc(), engine, ocl_interop::memory_kind::usm);

    reorder(user_src2_mem, conv_src2_mem)
            .execute(engine_stream, user_src2_mem, conv_src2_mem);
    reorder(user_weights2_mem, conv_weights2_mem)
            .execute(engine_stream, user_weights2_mem, conv_weights2_mem);
    reorder(user_bias2_mem, conv_bias2_mem)
            .execute(engine_stream, user_bias2_mem, conv_bias2_mem);

    auto conv2_prim = convolution_forward(conv2_pd);
    std::unordered_map<int, memory> conv2_args;
    conv2_args.insert({DNNL_ARG_SRC, conv_src2_mem});
    conv2_args.insert({DNNL_ARG_WEIGHTS, conv_weights2_mem});
    conv2_args.insert({DNNL_ARG_BIAS, user_bias2_mem});
    conv2_args.insert({DNNL_ARG_DST, conv_dst2_mem});
    conv2_prim.execute(engine_stream, conv2_args);

    // ========================================================================
    // Create primitive descriptor for concat
    // ========================================================================

    // Create concat sources from convolution destinations
    std::vector<memory::desc> concat_src_mds {
            conv1_pd.dst_desc(), conv2_pd.dst_desc()};
    // Let concat deduce destination for us
    auto concat_pd = concat::primitive_desc(/*axis=*/1, concat_src_mds, engine);

    auto concat_dst_md = concat_pd.dst_desc();
    auto concat_dst_dims = concat_dst_md.dims();

    // ========================================================================
    // Validate concat destination
    // ========================================================================

    // It may happen that concat would return plain layout for destination
    // instead of blocked, same for sources, in case descriptors are not
    // compatible for given blocking. E.g. first input is padded.
    //
    // To check that, we create a memory descriptor with concat dimensions and
    // plain `abx` layout. If they match, validation failed, explicit concat
    // should be used.
    auto verif_md = memory::desc(
            concat_dst_dims, concat_dst_md.data_type(), tag::abcd);

    // This is a second part of validation - that explicit and implicit concats
    // provide the same answer.

    // Execute concat for results validation
    auto concat_dst_mem = ocl_interop::make_memory(
            concat_dst_md, engine, ocl_interop::memory_kind::usm);

    auto concat_prim = concat(concat_pd);
    std::unordered_map<int, memory> concat_args;
    concat_args.insert({DNNL_ARG_MULTIPLE_SRC + 0, conv_dst1_mem});
    concat_args.insert({DNNL_ARG_MULTIPLE_SRC + 1, conv_dst2_mem});
    concat_args.insert({DNNL_ARG_DST, concat_dst_mem});
    concat_prim.execute(engine_stream, concat_args);

    // Put check here to see the output from concat primitive.
    if (verif_md == concat_dst_md)
        throw std::runtime_error("implicit concat cannot be created!");

    // Create a memory for implicit concat where convolutions will write their
    // destination outputs.
    auto impl_concat_mem = ocl_interop::make_memory(
            concat_dst_md, engine, ocl_interop::memory_kind::usm);

    // ========================================================================
    // Re-create primitive descriptor for conv1 with updated destination
    // ========================================================================

    // Get updated strides with submemory API. It preserves blocking edsc but
    // makes strides bigger between blocks of channels.
    memory::desc conv1_upd_dst_md = concat_dst_md.submemory_desc(
            dst1_dims, /*offsets=*/ {0, 0, 0, 0});

    // Pass descriptors from initially submitted convolution. This would
    // preserve blocked layouts for activations and weights. Pass updated
    // destination descriptor to utilize bigger strides in destination.
    conv1_desc = convolution_forward::desc(prop_kind::forward_training,
            algorithm::convolution_direct, conv1_pd.src_desc(),
            conv1_pd.weights_desc(), conv1_pd.bias_desc(), conv1_upd_dst_md,
            strides_dims, padding_dims_l, padding_dims_r);

    conv1_pd = convolution_forward::primitive_desc(conv1_desc, engine);

    // Create a new memory object containing a handle to existing memory which
    // was already created. Re-use rest memory objects with data as is.
    conv_dst1_mem = ocl_interop::make_memory(conv1_pd.dst_desc(), engine,
            ocl_interop::memory_kind::usm, impl_concat_mem.get_data_handle());

    conv1_prim = convolution_forward(conv1_pd);
    std::unordered_map<int, memory> conv1_upd_args;
    conv1_upd_args.insert({DNNL_ARG_SRC, conv_src1_mem});
    conv1_upd_args.insert({DNNL_ARG_WEIGHTS, conv_weights1_mem});
    conv1_upd_args.insert({DNNL_ARG_BIAS, user_bias1_mem});
    conv1_upd_args.insert({DNNL_ARG_DST, conv_dst1_mem});
    conv1_prim.execute(engine_stream, conv1_upd_args);

    // ========================================================================
    // Re-create primitive descriptor for conv2 with updated destination
    // ========================================================================

    memory::desc conv2_upd_dst_md = concat_dst_md.submemory_desc(
            dst2_dims, /*offsets=*/ {0, OC1, 0, 0});

    conv2_desc = convolution_forward::desc(prop_kind::forward_training,
            algorithm::convolution_direct, conv2_pd.src_desc(),
            conv2_pd.weights_desc(), conv2_pd.bias_desc(), conv2_upd_dst_md,
            strides_dims, padding_dims_l, padding_dims_r);

    conv2_pd = convolution_forward::primitive_desc(conv2_desc, engine);

    conv_dst2_mem = ocl_interop::make_memory(conv2_pd.dst_desc(), engine,
            ocl_interop::memory_kind::usm, impl_concat_mem.get_data_handle());

    conv2_prim = convolution_forward(conv2_pd);
    std::unordered_map<int, memory> conv2_upd_args;
    conv2_upd_args.insert({DNNL_ARG_SRC, conv_src2_mem});
    conv2_upd_args.insert({DNNL_ARG_WEIGHTS, conv_weights2_mem});
    conv2_upd_args.insert({DNNL_ARG_BIAS, user_bias2_mem});
    conv2_upd_args.insert({DNNL_ARG_DST, conv_dst2_mem});
    conv2_prim.execute(engine_stream, conv2_upd_args);

    engine_stream.wait();

    // ========================================================================
    // Validation of outputs
    // ========================================================================

    // Convert both memories back to f32 plain formats and compare them
    // pointwise.
    auto user_dst_mem
            = ocl_interop::make_memory({concat_dst_dims, dt::f32, tag::abcd},
                    engine, ocl_interop::memory_kind::usm);
    reorder(concat_dst_mem, user_dst_mem)
            .execute(engine_stream, concat_dst_mem, user_dst_mem);

    auto user_impl_concat_mem
            = ocl_interop::make_memory({concat_dst_dims, dt::f32, tag::abcd},
                    engine, ocl_interop::memory_kind::usm);
    reorder(impl_concat_mem, user_impl_concat_mem)
            .execute(engine_stream, impl_concat_mem, user_impl_concat_mem);

    engine_stream.wait();

    std::vector<float> dst_data(product(concat_dst_dims));
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);

    std::vector<float> dst_upd_data(product(concat_dst_dims));
    read_from_dnnl_memory(dst_upd_data.data(), user_impl_concat_mem);

    bool ok = true;
    for (memory::dim i = 0; i < product(concat_dst_dims); i++) {
        if (dst_data[i] != dst_upd_data[i]) {
            printf("[%4ld]:  ref:%lf  got:%lf\n", i, dst_data[i],
                    dst_upd_data[i]);
            ok = false;
        }
    }
    if (!ok) throw std::runtime_error("incorrect result!");
}

int main(int argc, char **argv) {
    auto exit_code = handle_example_errors(
            convolution_example, parse_engine_kind(argc, argv));
    return exit_code;
}
