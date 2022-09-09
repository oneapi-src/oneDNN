/*******************************************************************************
* Copyright 2022 Intel Corporation
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

/// @example augru.cpp
/// > Annotated version: @ref augru_example_cpp
///
/// @page augru_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute an
/// [AUGRU RNN](@ref dev_guide_rnn) primitive in forward training propagation
/// mode.
///
/// Key optimizations included in this example:
/// - Creation of optimized memory format from the primitive descriptor.
///
/// @page augru_example_cpp AUGRU RNN Primitive Example
/// @copydetails augru_example_cpp_short
///
/// @include augru.cpp

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

void augru_example(dnnl::engine::kind engine_kind) {

    if (engine_kind == engine::kind::gpu)
        throw example_allows_unimplemented {
                "No AUGRU implementation is available for GPU.\n"};

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 26, // batch size
            T = 6, // time steps
            C = 12, // channels
            G = 3, // gates
            L = 1, // layers
            D = 1; // directions

    // Source (src), weights, bias, attention, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {T, N, C};
    memory::dims attention_dims = {T, N, 1};
    memory::dims weights_dims = {L, D, C, G, C};
    memory::dims bias_dims = {L, D, G, C};
    memory::dims dst_dims = {T, N, C};

    // Allocate buffers.
    std::vector<float> src_layer_data(product(src_dims));
    std::vector<float> attention_data(product(attention_dims));
    std::vector<float> weights_layer_data(product(weights_dims));
    std::vector<float> weights_iter_data(product(weights_dims));
    std::vector<float> bias_data(product(bias_dims));
    std::vector<float> dst_layer_data(product(dst_dims));

    // Initialize src, weights, and bias tensors.
    std::generate(src_layer_data.begin(), src_layer_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(attention_data.begin(), attention_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(weights_layer_data.begin(), weights_layer_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory descriptors and memory objects for src, bias, and dst.
    auto src_layer_md = memory::desc(src_dims, dt::f32, tag::tnc);
    auto attention_md = memory::desc(attention_dims, dt::f32, tag::tnc);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::ldgo);
    auto dst_layer_md = memory::desc(dst_dims, dt::f32, tag::tnc);

    auto src_layer_mem = memory(src_layer_md, engine);
    auto attention_mem = memory(attention_md, engine);
    auto bias_mem = memory(bias_md, engine);
    auto dst_layer_mem = memory(dst_layer_md, engine);

    // Create memory objects for weights using user's memory layout. In this
    // example, LDIGO is assumed.
    auto user_weights_layer_mem
            = memory({weights_dims, dt::f32, tag::ldigo}, engine);
    auto user_weights_iter_mem
            = memory({weights_dims, dt::f32, tag::ldigo}, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_layer_data.data(), src_layer_mem);
    write_to_dnnl_memory(attention_data.data(), attention_mem);
    write_to_dnnl_memory(bias_data.data(), bias_mem);
    write_to_dnnl_memory(weights_layer_data.data(), user_weights_layer_mem);
    write_to_dnnl_memory(weights_iter_data.data(), user_weights_iter_mem);

    // Create memory descriptors for weights with format_tag::any. This enables
    // the AUGRU primitive to choose the optimized memory layout.
    auto augru_weights_layer_md = memory::desc(weights_dims, dt::f32, tag::any);
    auto augru_weights_iter_md = memory::desc(weights_dims, dt::f32, tag::any);

    // Optional memory descriptors for recurrent data.
    auto src_iter_md = memory::desc();
    auto dst_iter_md = memory::desc();

    // Create primitive descriptor.
    auto augru_pd
            = augru_forward::primitive_desc(engine, prop_kind::forward_training,
                    rnn_direction::unidirectional_left2right, src_layer_md,
                    src_iter_md, attention_md, augru_weights_layer_md,
                    augru_weights_iter_md, bias_md, dst_layer_md, dst_iter_md);

    // For now, assume that the weights memory layout generated by the primitive
    // and the ones provided by the user are identical.
    auto augru_weights_layer_mem = user_weights_layer_mem;
    auto augru_weights_iter_mem = user_weights_iter_mem;

    // Reorder the data in case the weights memory layout generated by the
    // primitive and the one provided by the user are different. In this case,
    // we create additional memory objects with internal buffers that will
    // contain the reordered data.
    if (augru_pd.weights_desc() != user_weights_layer_mem.get_desc()) {
        augru_weights_layer_mem = memory(augru_pd.weights_desc(), engine);
        reorder(user_weights_layer_mem, augru_weights_layer_mem)
                .execute(engine_stream, user_weights_layer_mem,
                        augru_weights_layer_mem);
    }

    if (augru_pd.weights_iter_desc() != user_weights_iter_mem.get_desc()) {
        augru_weights_iter_mem = memory(augru_pd.weights_iter_desc(), engine);
        reorder(user_weights_iter_mem, augru_weights_iter_mem)
                .execute(engine_stream, user_weights_iter_mem,
                        augru_weights_iter_mem);
    }

    // Create the memory objects from the primitive descriptor. A workspace is
    // also required for AUGRU.
    // NOTE: Here, the workspace is required for later usage in backward
    // propagation mode.
    auto src_iter_mem = memory(augru_pd.src_iter_desc(), engine);
    auto weights_iter_mem = memory(augru_pd.weights_iter_desc(), engine);
    auto dst_iter_mem = memory(augru_pd.dst_iter_desc(), engine);
    auto workspace_mem = memory(augru_pd.workspace_desc(), engine);

    // Create the primitive.
    auto augru_prim = augru_forward(augru_pd);

    // Primitive arguments
    std::unordered_map<int, memory> augru_args;
    augru_args.insert({DNNL_ARG_SRC_LAYER, src_layer_mem});
    augru_args.insert({DNNL_ARG_AUGRU_ATTENTION, attention_mem});
    augru_args.insert({DNNL_ARG_WEIGHTS_LAYER, augru_weights_layer_mem});
    augru_args.insert({DNNL_ARG_WEIGHTS_ITER, augru_weights_iter_mem});
    augru_args.insert({DNNL_ARG_BIAS, bias_mem});
    augru_args.insert({DNNL_ARG_DST_LAYER, dst_layer_mem});
    augru_args.insert({DNNL_ARG_SRC_ITER, src_iter_mem});
    augru_args.insert({DNNL_ARG_DST_ITER, dst_iter_mem});
    augru_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

    // Primitive execution: AUGRU.
    augru_prim.execute(engine_stream, augru_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_layer_data.data(), dst_layer_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(augru_example, parse_engine_kind(argc, argv));
}
