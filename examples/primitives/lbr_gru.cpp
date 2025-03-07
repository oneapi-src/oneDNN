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

/// @example lbr_gru.cpp
/// > Annotated version: @ref lbr_gru_example_cpp
///
/// @page lbr_gru_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Linear-Before-Reset GRU RNN](@ref dev_guide_rnn) primitive in forward
/// training propagation mode.
///
/// Key optimizations included in this example:
/// - Creation of optimized memory format from the primitive descriptor.
///
/// @page lbr_gru_example_cpp Linear-Before-Reset GRU RNN Primitive Example
/// @copydetails lbr_gru_example_cpp_short
///
/// @include lbr_gru.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

void lbr_gru_example(dnnl::engine::kind engine_kind) {
    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 2, // batch size
            T = 3, // time steps
            IC = 2, // src channels
            OC = 3, // dst channels
            G = 3, // gates
            L = 1, // layers
            D = 1, // directions
            E = 1; // extra Bias number. Extra Bias for u' gate

    // Source (src), weights, bias, attention, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {T, N, IC};
    memory::dims weights_layer_dims = {L, D, IC, G, OC};
    memory::dims weights_iter_dims = {L, D, OC, G, OC};
    memory::dims bias_dims = {L, D, G + E, OC};
    memory::dims dst_layer_dims = {T, N, OC};
    memory::dims dst_iter_dims = {L, D, N, OC};

    // Allocate buffers.
    std::vector<float> src_layer_data(product(src_dims));
    std::vector<float> weights_layer_data(product(weights_layer_dims));
    std::vector<float> weights_iter_data(product(weights_iter_dims));
    std::vector<float> bias_data(product(bias_dims));
    std::vector<float> dst_layer_data(product(dst_layer_dims));
    std::vector<float> dst_iter_data(product(dst_iter_dims));

    // Initialize src, weights, and bias tensors.
    std::generate(src_layer_data.begin(), src_layer_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_layer_data.begin(), weights_layer_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(weights_iter_data.begin(), weights_iter_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory descriptors and memory objects for src, bias, and dst.
    auto src_layer_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::tnc);
    auto bias_md = memory::desc(
            bias_dims, memory::data_type::f32, memory::format_tag::ldgo);
    auto dst_layer_md = memory::desc(
            dst_layer_dims, memory::data_type::f32, memory::format_tag::tnc);

    auto src_layer_mem = memory(src_layer_md, engine);
    auto bias_mem = memory(bias_md, engine);
    auto dst_layer_mem = memory(dst_layer_md, engine);

    // Create memory objects for weights using user's memory layout. In this
    // example, LDIGO (num_layers, num_directions, input_channels, num_gates,
    // output_channels) is assumed.
    auto user_weights_layer_mem
            = memory({weights_layer_dims, memory::data_type::f32,
                             memory::format_tag::ldigo},
                    engine);
    auto user_weights_iter_mem
            = memory({weights_iter_dims, memory::data_type::f32,
                             memory::format_tag::ldigo},
                    engine);

    // Write data to memory object's handle.
    // For GRU cells, the gates order is update, reset and output
    // gate except the bias. For the bias tensor, the gates order is
    // u, r, o and u' gate.
    write_to_dnnl_memory(src_layer_data.data(), src_layer_mem);
    write_to_dnnl_memory(bias_data.data(), bias_mem);
    write_to_dnnl_memory(weights_layer_data.data(), user_weights_layer_mem);
    write_to_dnnl_memory(weights_iter_data.data(), user_weights_iter_mem);

    // Create memory descriptors for weights with format_tag::any. This enables
    // the lbr_gru primitive to choose the optimized memory layout.
    auto weights_layer_md = memory::desc(weights_layer_dims,
            memory::data_type::f32, memory::format_tag::any);
    auto weights_iter_md = memory::desc(
            weights_iter_dims, memory::data_type::f32, memory::format_tag::any);

    // Optional memory descriptors for recurrent data.
    // Default memory descriptor for initial hidden states of the GRU cells
    auto src_iter_md = memory::desc();
    auto dst_iter_md = memory::desc();

    // Create primitive descriptor.
    auto lbr_gru_pd = lbr_gru_forward::primitive_desc(engine,
            prop_kind::forward_training,
            rnn_direction::unidirectional_left2right, src_layer_md, src_iter_md,
            weights_layer_md, weights_iter_md, bias_md, dst_layer_md,
            dst_iter_md);

    // For now, assume that the weights memory layout generated by the primitive
    // and the ones provided by the user are identical.
    auto weights_layer_mem = user_weights_layer_mem;
    auto weights_iter_mem = user_weights_iter_mem;

    // Reorder the data in case the weights memory layout generated by the
    // primitive and the one provided by the user are different. In this case,
    // we create additional memory objects with internal buffers that will
    // contain the reordered data.
    if (lbr_gru_pd.weights_desc() != user_weights_layer_mem.get_desc()) {
        weights_layer_mem = memory(lbr_gru_pd.weights_desc(), engine);
        reorder(user_weights_layer_mem, weights_layer_mem)
                .execute(engine_stream, user_weights_layer_mem,
                        weights_layer_mem);
    }

    if (lbr_gru_pd.weights_iter_desc() != user_weights_iter_mem.get_desc()) {
        weights_iter_mem = memory(lbr_gru_pd.weights_iter_desc(), engine);
        reorder(user_weights_iter_mem, weights_iter_mem)
                .execute(
                        engine_stream, user_weights_iter_mem, weights_iter_mem);
    }

    // Create the memory objects from the primitive descriptor. A workspace is
    // also required for Linear-Before-Reset GRU RNN.
    // NOTE: Here, the workspace is required for later usage in backward
    // propagation mode.
    auto src_iter_mem = memory(lbr_gru_pd.src_iter_desc(), engine);
    auto dst_iter_mem = memory(lbr_gru_pd.dst_iter_desc(), engine);
    auto workspace_mem = memory(lbr_gru_pd.workspace_desc(), engine);

    // Create the primitive.
    auto lbr_gru_prim = lbr_gru_forward(lbr_gru_pd);

    // Primitive arguments
    std::unordered_map<int, memory> lbr_gru_args;
    lbr_gru_args.insert({DNNL_ARG_SRC_LAYER, src_layer_mem});
    lbr_gru_args.insert({DNNL_ARG_WEIGHTS_LAYER, weights_layer_mem});
    lbr_gru_args.insert({DNNL_ARG_WEIGHTS_ITER, weights_iter_mem});
    lbr_gru_args.insert({DNNL_ARG_BIAS, bias_mem});
    lbr_gru_args.insert({DNNL_ARG_DST_LAYER, dst_layer_mem});
    lbr_gru_args.insert({DNNL_ARG_SRC_ITER, src_iter_mem});
    lbr_gru_args.insert({DNNL_ARG_DST_ITER, dst_iter_mem});
    lbr_gru_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

    // Primitive execution: lbr_gru.
    lbr_gru_prim.execute(engine_stream, lbr_gru_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_layer_data.data(), dst_layer_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            lbr_gru_example, parse_engine_kind(argc, argv));
}
