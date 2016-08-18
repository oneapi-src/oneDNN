/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include <iostream>
#include "mkldnn.hpp"

using namespace mkldnn;

void simple_net(){
    auto cpu_engine = engine(engine::cpu, 0);

    const uint32_t batch = 256;

    std::vector<float> net_src(batch * 3 * 227 * 227);
    std::vector<float> net_dst(batch * 96 * 27 * 27);

    /* AlexNet: conv
     * {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
     * strides: {4, 4}
     */
    tensor::dims conv_src_tz = {batch, 3, 227, 227};
    tensor::dims conv_weights_tz = {96, 3, 11, 11};
    tensor::dims conv_bias_tz = {96};
    tensor::dims conv_dst_tz = {batch, 96, 55, 55};
    tensor::dims conv_strides = {4, 4};
    tensor::nd_offset conv_padding = {0, 0};

    std::vector<float> conv_weights(std::accumulate(conv_weights_tz.begin(),
        conv_weights_tz.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> conv_bias(std::accumulate(conv_bias_tz.begin(),
        conv_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto conv_user_src_memory = memory({{{conv_src_tz}, memory::precision::f32,
        memory::format::nchw}, cpu_engine}, net_src.data());
    auto conv_user_weights_memory = memory({{{conv_weights_tz},
        memory::precision::f32, memory::format::oihw}, cpu_engine},
        conv_weights.data());
    auto conv_user_bias_memory = memory({{{conv_bias_tz},
        memory::precision::f32, memory::format::x}, cpu_engine},
        conv_bias.data());

    /* create memory descriptors for convolution data w/ no specified format */
    auto conv_src_md = memory::desc({conv_src_tz}, memory::precision::f32,
        memory::format::any);
    auto conv_bias_md = memory::desc({conv_bias_tz}, memory::precision::f32,
        memory::format::any);
    auto conv_weights_md = memory::desc({conv_weights_tz},
        memory::precision::f32, memory::format::any);
    auto conv_dst_md = memory::desc({conv_dst_tz}, memory::precision::f32,
        memory::format::any);

    /* create a convolution */
    auto conv_desc = convolution::desc(prop_kind::forward, convolution::direct,
        conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
        conv_padding, padding_kind::zero);
    auto conv_pd = convolution::primitive_desc(conv_desc, cpu_engine);

    std::vector<primitive> net;

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    auto conv_src_memory = conv_user_src_memory;
    if (memory::primitive_desc(conv_pd.data.src_primitive_desc) !=
        conv_user_src_memory.get_primitive_desc()) {
        conv_src_memory = memory(conv_pd.data.src_primitive_desc);
        net.push_back(reorder(conv_user_src_memory, conv_src_memory));
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (memory::primitive_desc(conv_pd.data.weights_primitive_desc) !=
        conv_user_weights_memory.get_primitive_desc()) {
        conv_weights_memory = memory(conv_pd.data.src_primitive_desc);
        net.push_back(reorder(conv_user_weights_memory, conv_weights_memory));
    }

    auto conv_dst_memory = memory(conv_pd.data.dst_primitive_desc);

    /* create convolution primitive and add it to net */
    net.push_back(convolution(conv_pd, conv_src_memory, conv_weights_memory,
        conv_user_bias_memory, conv_dst_memory));

    /* AlexNet: relu
     * {batch, 96, 55, 55} -> {batch, 96, 55, 55}
     */
    const double negative_slope = 1.0;

    auto relu_dst_memory = memory(conv_pd.data.dst_primitive_desc);

    /* create relu primitive and add it to net */
    net.push_back(relu(prop_kind::forward, negative_slope, conv_dst_memory,
        relu_dst_memory));

    /* AlexNet: lrn
     * {batch, 96, 55, 55} -> {batch, 96, 55, 55}
     * local size: 5
     * alpha: 0.0001
     * beta: 0.75
     */
    const uint32_t local_size = 5;
    const double alpha = 0.0001;
    const double beta = 0.75;

    auto lrn_dst_memory = memory(relu_dst_memory.get_primitive_desc());

    /* create lrn scratch memory from lrn src */
    auto lrn_scratch_memory = memory(lrn_dst_memory.get_primitive_desc());

    /* create lrn primitive and add it to net */
    net.push_back(lrn(prop_kind::forward, lrn::across_channels,
        relu_dst_memory, lrn_scratch_memory, lrn_dst_memory,
        alpha, beta, local_size));

    /* AlexNet: pool
     * {batch, 96, 55, 55} -> {batch, 96, 27, 27}
     * kernel: {3, 3}
     * strides: {2, 2}
     */
    tensor::dims pool_dst_tz = {batch, 96, 27, 27};
    tensor::dims pool_kernel = {3, 3};
    tensor::dims pool_strides = {2, 2};
    tensor::nd_offset pool_padding = {0, 0};

    auto pool_user_dst_memory = memory({{{pool_dst_tz}, memory::precision::f32,
        memory::format::nchw}, cpu_engine}, net_dst.data());

    auto pool_dst_md = memory::desc({pool_dst_tz}, memory::precision::f32,
        memory::format::any);

    /* create a pooling */
    auto pool_desc = pooling::desc(prop_kind::forward, pooling::max,
        lrn_dst_memory.get_primitive_desc().desc(), pool_dst_md, pool_strides,
        pool_kernel, pool_padding, padding_kind::zero);
    auto pool_pd = pooling::primitive_desc(pool_desc, cpu_engine);

    auto pool_dst_memory = pool_user_dst_memory;
    if (memory::primitive_desc(pool_pd.data.dst_primitive_desc) !=
        pool_user_dst_memory.get_primitive_desc()) {
        pool_dst_memory = memory(pool_pd.data.dst_primitive_desc);
    }

    /* create pooling indices memory from pooling dst */
    auto pool_indices_memory = memory(pool_dst_memory.get_primitive_desc());

    /* create pooling primitive an add it to net */
    net.push_back(pooling(pool_pd, lrn_dst_memory, pool_indices_memory,
        pool_dst_memory));

    /* create reorder between internal and user data if it is needed and
     *  add it to net after pooling */
    if (pool_dst_memory != pool_user_dst_memory) {
        net.push_back(reorder(pool_dst_memory, pool_user_dst_memory));
    }

    stream().submit(net).wait();
}

int main(int argc, char **argv) {
    try {
        simple_net();
    }
    catch(error& e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
