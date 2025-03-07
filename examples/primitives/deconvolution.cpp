/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

/// @example deconvolution.cpp
/// > Annotated version: @ref deconvolution_example_cpp
///
/// @page deconvolution_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Deconvolution](@ref dev_guide_convolution) primitive in forward propagation
/// mode.
///
/// Key optimizations included in this example:
/// - Creation of optimized memory format from the primitive descriptor;
/// - Primitive attributes with fused post-ops.
///
/// @page deconvolution_example_cpp Deconvolution Primitive Example
/// @copydetails deconvolution_example_cpp_short
///
/// @include deconvolution.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

void deconvolution_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 32, // input channels
            IH = 13, // input height
            IW = 13, // input width
            OC = 64, // output channels
            KH = 3, // weights height
            KW = 3, // weights width
            PH_L = 1, // height padding: left
            PH_R = 1, // height padding: right
            PW_L = 1, // width padding: left
            PW_R = 1, // width padding: right
            SH = 4, // height-wise stride
            SW = 4, // width-wise stride
            // In a convolution operation, the output height and
            // width are computed as:
            // OH = (IH - KH + PH_L + PH_R) / SH + 1
            // OW = (IW - KW + PW_L + PW_R) / SW + 1
            // However, in a deconvolution operation, the computation
            // is reversed:
            OH = (IH - 1) * SH - PH_L - PH_R + KH, // output height
            OW = (IW - 1) * SW - PW_L - PW_R + KW; // output width

    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims weights_dims = {OC, IC, KH, KW};
    memory::dims bias_dims = {OC};
    memory::dims dst_dims = {N, OC, OH, OW};

    // Strides, padding dimensions.
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> bias_data(OC);
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights, and dst tensors.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory objects for tensor data (src, weights, dst). In this
    // example, NCHW layout is assumed for src and dst, and OIHW for weights.
    auto user_src_mem = memory(
            {src_dims, memory::data_type::f32, memory::format_tag::nchw},
            engine);
    auto user_weights_mem = memory(
            {weights_dims, memory::data_type::f32, memory::format_tag::oihw},
            engine);
    auto user_dst_mem = memory(
            {dst_dims, memory::data_type::f32, memory::format_tag::nchw},
            engine);

    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the deconvolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto deconv_src_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::any);
    auto deconv_weights_md = memory::desc(
            weights_dims, memory::data_type::f32, memory::format_tag::any);
    auto deconv_dst_md = memory::desc(
            dst_dims, memory::data_type::f32, memory::format_tag::any);

    // Create memory descriptor and memory object for input bias.
    auto user_bias_md = memory::desc(
            bias_dims, memory::data_type::f32, memory::format_tag::a);
    auto user_bias_mem = memory(user_bias_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), user_src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
    write_to_dnnl_memory(bias_data.data(), user_bias_mem);

    // Create primitive post-ops (ReLU).
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops deconv_ops;
    deconv_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    primitive_attr deconv_attr;
    deconv_attr.set_post_ops(deconv_ops);

    // Create primitive descriptor.
    // Here we use deconvolution which is a transposed convolution.
    // The way the weights are applied is the key difference between convolution
    // and deconvolution. In a convolution, the weights are used to reduce
    // the input data, while in a deconvolution, they are used to expand
    // the input data.
    auto deconv_pd = deconvolution_forward::primitive_desc(engine,
            prop_kind::forward_training, algorithm::deconvolution_direct,
            deconv_src_md, deconv_weights_md, user_bias_md, deconv_dst_md,
            strides_dims, padding_dims_l, padding_dims_r, deconv_attr);

    // For now, assume that the src, weights, and dst memory layouts generated
    // by the primitive and the ones provided by the user are identical.
    auto deconv_src_mem = user_src_mem;
    auto deconv_weights_mem = user_weights_mem;
    auto deconv_dst_mem = user_dst_mem;

    // Reorder the data in case the src and weights memory layouts generated by
    // the primitive and the ones provided by the user are different. In this
    // case, we create additional memory objects with internal buffers that will
    // contain the reordered data. The data in dst will be reordered after the
    // deconvolution computation has finalized.
    if (deconv_pd.src_desc() != user_src_mem.get_desc()) {
        deconv_src_mem = memory(deconv_pd.src_desc(), engine);
        reorder(user_src_mem, deconv_src_mem)
                .execute(engine_stream, user_src_mem, deconv_src_mem);
    }

    if (deconv_pd.weights_desc() != user_weights_mem.get_desc()) {
        deconv_weights_mem = memory(deconv_pd.weights_desc(), engine);
        reorder(user_weights_mem, deconv_weights_mem)
                .execute(engine_stream, user_weights_mem, deconv_weights_mem);
    }

    if (deconv_pd.dst_desc() != user_dst_mem.get_desc()) {
        deconv_dst_mem = memory(deconv_pd.dst_desc(), engine);
    }

    // Create the primitive.
    auto deconv_prim = deconvolution_forward(deconv_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> deconv_args;
    deconv_args.insert({DNNL_ARG_SRC, deconv_src_mem});
    deconv_args.insert({DNNL_ARG_WEIGHTS, deconv_weights_mem});
    deconv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    deconv_args.insert({DNNL_ARG_DST, deconv_dst_mem});

    // Primitive execution: deconvolution with ReLU.
    deconv_prim.execute(engine_stream, deconv_args);

    // Reorder the data in case the dst memory descriptor generated by the
    // primitive and the one provided by the user are different.
    if (deconv_pd.dst_desc() != user_dst_mem.get_desc()) {
        reorder(deconv_dst_mem, user_dst_mem)
                .execute(engine_stream, deconv_dst_mem, user_dst_mem);
    } else
        user_dst_mem = deconv_dst_mem;

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            deconvolution_example, parse_engine_kind(argc, argv));
}
