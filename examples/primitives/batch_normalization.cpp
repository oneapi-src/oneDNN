/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

/// @example batch_normalization.cpp
/// > Annotated version: @ref batch_normalization_example_cpp
///
/// @page batch_normalization_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Batch Normalization](@ref dev_guide_batch_normalization) primitive in
/// forward training propagation mode.
///
/// Key optimizations included in this example:
/// - In-place primitive execution;
/// - Source memory format for an optimized primitive implementation;
/// - Fused post-ops via operation descriptor flags;
///
/// @page batch_normalization_example_cpp Batch Normalization Primitive Example
/// @copydetails batch_normalization_example_cpp_short
///
/// @include batch_normalization.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

void batch_normalization_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 3, // channels
            IH = 227, // tensor height
            IW = 227; // tensor width

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC, IH, IW};

    // Scale/shift tensor dimensions.
    memory::dims scaleshift_dims = {IC};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> scale_data(product(scaleshift_dims));
    std::vector<float> shift_data(product(scaleshift_dims));

    // Initialize src.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Initialize scale.
    std::generate(scale_data.begin(), scale_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });

    // Initialize shift.
    std::generate(shift_data.begin(), shift_data.end(), []() {
        static int i = 0;
        return std::tan(float(i++));
    });

    // Create src and scale/shift memory descriptors and memory objects.
    auto src_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto scaleshift_md = memory::desc(
            scaleshift_dims, memory::data_type::f32, memory::format_tag::x);

    auto src_mem = memory(src_md, engine);
    auto scale_mem = memory(scaleshift_md, engine);
    auto shift_mem = memory(scaleshift_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(scale_data.data(), scale_mem);
    write_to_dnnl_memory(shift_data.data(), shift_mem);

    // Create primitive descriptor.
    auto bnorm_pd = batch_normalization_forward::primitive_desc(engine,
            prop_kind::forward_training, src_md, dst_md, 1.e-10f,
            normalization_flags::use_scale | normalization_flags::use_shift
                    | normalization_flags::fuse_norm_relu);

    // Create memory objects using memory descriptors created by the primitive
    // descriptor: mean, variance, workspace.
    // NOTE: Here, the ReLU post-ops require a workspace for later usage in
    // backward propagation mode.
    auto mean_mem = memory(bnorm_pd.mean_desc(), engine);
    auto variance_mem = memory(bnorm_pd.variance_desc(), engine);
    auto workspace_mem = memory(bnorm_pd.workspace_desc(), engine);

    // Create the primitive.
    auto bnorm_prim = batch_normalization_forward(bnorm_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> bnorm_args;
    bnorm_args.insert({DNNL_ARG_SRC, src_mem});
    bnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
    bnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
    bnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
    bnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
    bnorm_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});
    bnorm_args.insert({DNNL_ARG_DST, src_mem});

    // Primitive execution: batch normalization with ReLU.
    bnorm_prim.execute(engine_stream, bnorm_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(src_data.data(), src_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            batch_normalization_example, parse_engine_kind(argc, argv));
}
