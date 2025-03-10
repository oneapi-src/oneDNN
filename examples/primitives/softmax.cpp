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

/// @example softmax.cpp
/// > Annotated version: @ref softmax_example_cpp
///
/// @page softmax_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Softmax](@ref dev_guide_softmax) primitive in forward training propagation
/// mode.
///
/// Key optimizations included in this example:
/// - In-place primitive execution;
/// - Softmax along axis 1 (C) for 2D tensors.
///
/// @page softmax_example_cpp Softmax Primitive Example
/// @copydetails softmax_example_cpp_short
///
/// @include softmax.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

void softmax_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 1000; // channels

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC};

    // Allocate buffer.
    std::vector<float> src_data(product(src_dims));

    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Create src memory descriptor and memory object.
    auto src_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::nc);
    auto dst_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::nc);
    auto src_mem = memory(src_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Softmax axis.
    const int axis = 1;

    // Create primitive descriptor.
    auto softmax_pd = softmax_forward::primitive_desc(engine,
            prop_kind::forward_training, algorithm::softmax_accurate, src_md,
            dst_md, axis);

    // Create the primitive.
    auto softmax_prim = softmax_forward(softmax_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, src_mem});

    // Primitive execution.
    softmax_prim.execute(engine_stream, softmax_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(src_data.data(), src_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            softmax_example, parse_engine_kind(argc, argv));
}
