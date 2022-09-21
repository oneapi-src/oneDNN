/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

/// @example layer_normalization.cpp
/// > Annotated version: @ref layer_normalization_example_cpp
///
/// @page layer_normalization_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Layer normalization](@ref dev_guide_layer_normalization) primitive in
/// forward propagation mode.
///
/// Key optimizations included in this example:
/// - In-place primitive execution;
/// - Creation of memory objects using the primitive descriptor.
///
/// @page layer_normalization_example_cpp Layer Normalization Primitive Example
/// @copydetails layer_normalization_example_cpp_short
///
/// @include layer_normalization.cpp

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

void layer_normalization_example(dnnl::engine::kind engine_kind) {

    /// Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim T = 12, // time steps
            N = 3, // batch
            C = 227; // channels

    // Source (src) and destination (dst) tensors dimensions.
    const memory::dims src_dims = {T, N, C};

    // Scale/shift tensor dimensions.
    memory::dims scaleshift_dims = {C};

    // Allocate buffer.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> scale_data(product(scaleshift_dims));
    std::vector<float> shift_data(product(scaleshift_dims));

    // Initialize src tensor.
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

    // Create src memory descriptor and memory objects.
    auto src_md = memory::desc(src_dims, dt::f32, tag::tnc);
    auto dst_md = memory::desc(src_dims, dt::f32, tag::tnc);
    auto scaleshift_md = memory::desc(scaleshift_dims, dt::f32, tag::x);

    auto src_mem = memory(src_md, engine);
    auto scale_mem = memory(scaleshift_md, engine);
    auto shift_mem = memory(scaleshift_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(scale_data.data(), scale_mem);
    write_to_dnnl_memory(shift_data.data(), shift_mem);

    // Create primitive descriptor.
    const float epsilon = 1.e-10f;
    auto lnorm_pd = layer_normalization_forward::primitive_desc(engine,
            prop_kind::forward_training, src_md, dst_md, epsilon,
            normalization_flags::use_scale | normalization_flags::use_shift);

    // Use the memory descriptors from the primitive to create memory objects
    // required for the primitive: mean, variance, scale/shift.
    auto mean_mem = memory(lnorm_pd.mean_desc(), engine);
    auto variance_mem = memory(lnorm_pd.variance_desc(), engine);

    // Create the primitive.
    auto lnorm_prim = layer_normalization_forward(lnorm_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> lnorm_args;
    lnorm_args.insert({DNNL_ARG_SRC, src_mem});
    lnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
    lnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
    lnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
    lnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
    lnorm_args.insert({DNNL_ARG_DST, src_mem});

    // Primitive execution: layer normalization.
    lnorm_prim.execute(engine_stream, lnorm_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.s
    read_from_dnnl_memory(src_data.data(), src_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            layer_normalization_example, parse_engine_kind(argc, argv));
}
