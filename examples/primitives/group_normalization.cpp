/*******************************************************************************
* Copyright 2023 Intel Corporation
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

/// @example group_normalization.cpp
/// > Annotated version: @ref group_normalization_example_cpp
///
/// @page group_normalization_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Group Normalization](@ref dev_guide_group_normalization) primitive in
/// forward training propagation mode.
///
/// Key optimizations included in this example:
/// - In-place primitive execution;
/// - Source memory format for an optimized primitive implementation;
///
/// @page group_normalization_example_cpp Group Normalization Primitive Example
/// @copydetails group_normalization_example_cpp_short
///
/// @include group_normalization.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void group_normalization_example(engine::kind engine_kind) {
    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 6, // batch size
            IC = 256, // channels
            ID = 20, // tensor depth
            IH = 28, // tensor height
            IW = 28; // tensor width

    // Normalization groups
    memory::dim groups = IC; // Instance normalization

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC, ID, IH, IW};

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
    auto src_md = memory::desc(src_dims, dt::f32, tag::ncdhw);
    auto dst_md = memory::desc(src_dims, dt::f32, tag::ncdhw);
    auto scaleshift_md = memory::desc(scaleshift_dims, dt::f32, tag::x);

    auto src_mem = memory(src_md, engine);
    auto scale_mem = memory(scaleshift_md, engine);
    auto shift_mem = memory(scaleshift_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(scale_data.data(), scale_mem);
    write_to_dnnl_memory(shift_data.data(), shift_mem);

    // Create primitive descriptor.
    auto gnorm_pd = group_normalization_forward::primitive_desc(engine,
            prop_kind::forward_training, src_md, dst_md, groups, 1.e-10f,
            normalization_flags::use_scale | normalization_flags::use_shift);

    // Create memory objects using memory descriptors created by the primitive
    // descriptor: mean, variance.

    auto mean_mem = memory(gnorm_pd.mean_desc(), engine);
    auto variance_mem = memory(gnorm_pd.variance_desc(), engine);

    // Create the primitive.
    auto gnorm_prim = group_normalization_forward(gnorm_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> gnorm_args;
    gnorm_args.insert({DNNL_ARG_SRC, src_mem});
    gnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
    gnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
    gnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
    gnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
    gnorm_args.insert({DNNL_ARG_DST, src_mem});

    // Primitive execution: group normalization.
    gnorm_prim.execute(engine_stream, gnorm_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(src_data.data(), src_mem);
}

int main(int argc, char **argv) {
    auto engine_kind = parse_engine_kind(argc, argv);
    // GPU is not supported
    if (engine_kind != engine::kind::cpu) return 0;
    return handle_example_errors(group_normalization_example, engine_kind);
}
