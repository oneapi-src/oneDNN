/***************************************************************************
 *  Copyright 2020 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 **************************************************************************/

#include <assert.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <vector>
#include "dnnl.hpp"
#include "example_utils.hpp"

#include <cassert>
#include <random>
#include <string>
#include <unordered_map>

#include <stdio.h>

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

memory::dim BATCH = 1;

void hash_and_print(std::vector<float> const &tensor, bool print, int limit,
        int hash_limit = -1) {
    double accum = 0;
    hash_limit = hash_limit == -1 ? tensor.size() : hash_limit;
    for (std::size_t i = 0; i < hash_limit; ++i) {
        if (i % 2 == 0) {
            accum += tensor[i];
        } else {
            accum -= tensor[i];
        }
    }
    for (std::size_t i = 0; i < limit && print; ++i) {
        std::cout << tensor[i] << "\t";
    }
    std::cout << "\nHash: "
              << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << accum << "\n"
              << std::endl;
}

void read_binary_data(void *tensor, int size, std::string const &name) {
    FILE *file = fopen(name.c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error("Failed to open file " + name);
    }

    int bytes_read = fread(tensor, sizeof(char), size * sizeof(float), file);
}

void load_weights_conv(int block_number, int index, void *weights_data,
        int weights_size, void *biases_data, int bias_size) {
    std::string base_filename = "vgg16_transposed_param_files/conv_"
            + std::to_string(index) + "_" + std::to_string(block_number) + "_";

    read_binary_data(biases_data, bias_size, base_filename + "biases.bin");
    read_binary_data(weights_data, weights_size, base_filename + "weights.bin");
}

void load_weights_fc(int index, void *weights_data, int weights_size,
        void *biases_data, int bias_size) {
    std::string base_filename
            = "vgg16_transposed_param_files/fc" + std::to_string(index) + "_";

    read_binary_data(biases_data, bias_size, base_filename + "biases.bin");
    read_binary_data(weights_data, weights_size, base_filename + "weights.bin");
}

// NOTE: src_memory should have its data set *before* it is passed to this
// function
void do_conv(engine &eng, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> &net_args,
        void *weights_data, void *bias_data, memory::dims &weights_dims,
        memory::dims &bias_dims, memory::dims &strides, memory::dims &padding,
        memory &src_memory, memory &dst_memory) {
    auto weights_memory = memory({{weights_dims}, dt::f32, tag::ohwi}, eng);
    auto bias_memory = memory({{bias_dims}, dt::f32, tag::x}, eng);

    write_to_dnnl_memory(weights_data, weights_memory);
    write_to_dnnl_memory(bias_data, bias_memory);

    auto src_md = src_memory.get_desc();
    auto bias_md = bias_memory.get_desc();
    auto weights_md = weights_memory.get_desc();
    auto dst_md = dst_memory.get_desc();

    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md,
            strides, padding, padding);
    auto prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);

    net.push_back(convolution_forward(prim_desc));
    net_args.push_back(
            {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_WEIGHTS, weights_memory},
                    {DNNL_ARG_BIAS, bias_memory}, {DNNL_ARG_DST, dst_memory}});
}

void do_relu(engine &eng, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> &net_args,
        memory &src_dst_mem) {
    const float slope = 0.0f;
    auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, src_dst_mem.get_desc(), slope);
    auto relu1_prim_desc = eltwise_forward::primitive_desc(relu1_desc, eng);
    net.push_back(eltwise_forward(relu1_prim_desc));
    net_args.push_back(
            {{DNNL_ARG_SRC, src_dst_mem}, {DNNL_ARG_DST, src_dst_mem}});
}

void do_pool(engine &eng, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> &net_args,
        memory::dims &strides_dims, memory::dims &kernel_dims,
        memory::dims &pad_dims_l, memory::dims &pad_dims_r, memory &src_mem,
        memory &dst_mem) {

    auto src_md = src_mem.get_desc();
    auto dst_md = dst_mem.get_desc();

    auto pooling_d = pooling_forward::desc(prop_kind::forward_inference,
            algorithm::pooling_max, src_md, dst_md, strides_dims, kernel_dims,
            pad_dims_l, pad_dims_r);

    auto pooling_pd = pooling_forward::primitive_desc(pooling_d, eng);
    auto workspace_mem = memory(pooling_pd.workspace_desc(), eng);
    auto pooling_prim = pooling_forward(pooling_pd);

    std::unordered_map<int, memory> pooling_args;
    pooling_args.insert({DNNL_ARG_SRC, src_mem});
    pooling_args.insert({DNNL_ARG_DST, dst_mem});
    pooling_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

    net.push_back(pooling_prim);
    net_args.push_back(pooling_args);
}

void do_ip(engine &eng, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> &net_args,
        memory &weights_mem, memory &bias_mem, memory &src_mem, memory &dst_mem,
        bool relu_post_op = false) {

    auto inner_product_d = inner_product_forward::desc(
            prop_kind::forward_inference, src_mem.get_desc(),
            weights_mem.get_desc(), bias_mem.get_desc(), dst_mem.get_desc());

    inner_product_forward::primitive_desc inner_product_pd;
    if (relu_post_op) {
        const float scale = 1.0f;
        const float alpha = 0.f;
        const float beta = 0.f;
        post_ops inner_product_ops;
        inner_product_ops.append_eltwise(
                scale, algorithm::eltwise_relu, alpha, beta);
        primitive_attr inner_product_attr;
        inner_product_attr.set_post_ops(inner_product_ops);
        inner_product_pd = inner_product_forward::primitive_desc(
                inner_product_d, inner_product_attr, eng);
    } else {
        inner_product_pd
                = inner_product_forward::primitive_desc(inner_product_d, eng);
    }

    auto inner_product_prim = inner_product_forward(inner_product_pd);
    net.push_back(inner_product_prim);
    net_args.push_back(
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weights_mem},
                    {DNNL_ARG_BIAS, bias_mem}, {DNNL_ARG_DST, dst_mem}});
}

void set_up_vgg(engine &eng, stream &s, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> &net_args,
        std::vector<char> &input_image) {
#define STAMP_OUT_WEIGHTS_CONV(BLOCK_NUMBER, INDEX, WEI_DIMS, BIA_DIMS) \
    std::vector<char> weights_##BLOCK_NUMBER##_##INDEX( \
            product(WEI_DIMS) * sizeof(float)); \
    std::vector<char> biases_##BLOCK_NUMBER##_##INDEX( \
            product(BIA_DIMS) * sizeof(float)); \
    load_weights_conv(BLOCK_NUMBER, INDEX, \
            weights_##BLOCK_NUMBER##_##INDEX.data(), product(WEI_DIMS), \
            biases_##BLOCK_NUMBER##_##INDEX.data(), product(BIA_DIMS));

    /* Convolution dims */
    memory::dim IC = 3, OC = 64, WIDTH = 224, HEIGHT = 224;

    memory::dims conv1_strides = {1, 1};
    memory::dims conv1_padding = {1, 1};

    memory::dims conv1_src_dims = {BATCH, IC, WIDTH, HEIGHT};
    memory::dims conv1_dst_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims conv1_weights_dims = {OC, IC, 3, 3};
    memory::dims conv1_bias_dims = {OC};

    int src_size = IC * WIDTH * HEIGHT;
    std::vector<char> src_data(src_size * sizeof(float));
    std::vector<char> src_data_copied;
    src_data_copied.reserve(src_data.size() * BATCH);
    std::string image_filename = "kangaroo.jpg.bin";
    read_binary_data(src_data.data(), src_size, image_filename);
    for (size_t i = 0; i < BATCH; i++) {
        std::copy(src_data.begin(), src_data.end(),
                std::back_inserter(src_data_copied));
    }

    auto tensor_tag = tag::nhwc;
    auto conv1_src_md_nchw = memory::desc(conv1_src_dims, dt::f32, tag::nchw);
    auto conv1_src_md = memory::desc(conv1_src_dims, dt::f32, tensor_tag);
    auto conv1_src_memory_nchw = memory(conv1_src_md_nchw, eng);
    auto conv1_src_memory = memory(conv1_src_md, eng);
    write_to_dnnl_memory(src_data_copied.data(), conv1_src_memory_nchw);
    reorder::primitive_desc r_pd(
            eng, conv1_src_md_nchw, eng, conv1_src_md, primitive_attr());

    auto r = reorder(r_pd);
    r.execute(s, conv1_src_memory_nchw, conv1_src_memory);
    s.wait();
    std::vector<float> reorder_check(src_data_copied.size());
    read_from_dnnl_memory(reorder_check.data(), conv1_src_memory);

    /* -------------- conv -> relu -> conv -> relu -> pool -------------- */

    auto conv1_dst_md = memory::desc(conv1_dst_dims, dt::f32, tensor_tag);
    auto conv1_dst_memory = memory(conv1_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(1, 1, conv1_weights_dims, conv1_bias_dims)
    do_conv(eng, net, net_args, weights_1_1.data(), biases_1_1.data(),
            conv1_weights_dims, conv1_bias_dims, conv1_strides, conv1_padding,
            conv1_src_memory, conv1_dst_memory);
    do_relu(eng, net, net_args, conv1_dst_memory);

    IC = 64;
    OC = 64;
    memory::dims conv1_2_weights_dims = {OC, IC, 3, 3};
    memory::dims conv1_2_bias_dims = {OC};

    auto conv1_2_dst_md = conv1_dst_memory.get_desc();
    auto conv1_2_dst_memory = memory(conv1_2_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(1, 2, conv1_2_weights_dims, conv1_2_bias_dims)
    do_conv(eng, net, net_args, weights_1_2.data(), biases_1_2.data(),
            conv1_2_weights_dims, conv1_2_bias_dims, conv1_strides,
            conv1_padding, conv1_dst_memory, conv1_2_dst_memory);
    do_relu(eng, net, net_args, conv1_2_dst_memory);

    /* Pooling dims */
    const memory::dim KH = 2, // kernel height
            KW = 2, // kernel width
            PH_L = 0, // height padding: left
            PH_R = 0, // height padding: right
            PW_L = 0, // width padding: left
            PW_R = 0, // width padding: right
            SH = 2, // height-wise stride
            SW = 2; // width-wise stride

    memory::dims kernel_dims = {KH, KW};
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};
    memory::dims pool1_src_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims pool1_dst_dims = {BATCH, OC, WIDTH / 2, HEIGHT / 2};
    auto pool1_dst_md = memory::desc(pool1_dst_dims, dt::f32, tensor_tag);
    auto pool1_dst_memory = memory(pool1_dst_md, eng);
    do_pool(eng, net, net_args, strides_dims, kernel_dims, padding_dims_l,
            padding_dims_r, conv1_2_dst_memory, pool1_dst_memory);
    /* ---------------------------------- */

    /* -------------- conv -> relu -> conv -> relu -> pool --------------
     */
    IC = 64;
    OC = 128;
    WIDTH /= 2;
    HEIGHT /= 2;

    memory::dims conv2_dst_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims conv2_weights_dims = {OC, IC, 3, 3};
    memory::dims conv2_bias_dims = {OC};
    auto conv2_dst_md = memory::desc(conv2_dst_dims, dt::f32, tensor_tag);
    auto conv2_dst_memory = memory(conv2_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(2, 1, conv2_weights_dims, conv2_bias_dims)
    do_conv(eng, net, net_args, weights_2_1.data(), biases_2_1.data(),
            conv2_weights_dims, conv2_bias_dims, conv1_strides, conv1_padding,
            pool1_dst_memory, conv2_dst_memory);
    do_relu(eng, net, net_args, conv2_dst_memory);

    IC = 128;
    OC = 128;
    memory::dims conv2_2_weights_dims = {OC, IC, 3, 3};
    memory::dims conv2_2_bias_dims = {OC};
    auto conv2_2_dst_md = conv2_dst_memory.get_desc();
    auto conv2_2_dst_memory = memory(conv2_2_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(2, 2, conv2_2_weights_dims, conv2_2_bias_dims)
    do_conv(eng, net, net_args, weights_2_2.data(), biases_2_2.data(),
            conv2_2_weights_dims, conv2_2_bias_dims, conv1_strides,
            conv1_padding, conv2_dst_memory, conv2_2_dst_memory);
    do_relu(eng, net, net_args, conv2_2_dst_memory);

    memory::dims pool2_src_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims pool2_dst_dims = {BATCH, OC, WIDTH / 2, HEIGHT / 2};
    auto pool2_dst_md = memory::desc(pool2_dst_dims, dt::f32, tensor_tag);
    auto pool2_dst_memory = memory(pool2_dst_md, eng);
    do_pool(eng, net, net_args, strides_dims, kernel_dims, padding_dims_l,
            padding_dims_r, conv2_2_dst_memory, pool2_dst_memory);
    // /* ---------------------------------- */

    // /* --- conv -> relu -> conv -> relu -> conv -> relu -> pool --- */
    IC = 128;
    OC = 256;
    WIDTH /= 2;
    HEIGHT /= 2;

    memory::dims conv3_dst_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims conv3_weights_dims = {OC, IC, 3, 3};
    memory::dims conv3_bias_dims = {OC};
    auto conv3_dst_md = memory::desc(conv3_dst_dims, dt::f32, tensor_tag);
    auto conv3_dst_memory = memory(conv3_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(3, 1, conv3_weights_dims, conv3_bias_dims)
    do_conv(eng, net, net_args, weights_3_1.data(), biases_3_1.data(),
            conv3_weights_dims, conv3_bias_dims, conv1_strides, conv1_padding,
            pool2_dst_memory, conv3_dst_memory);
    do_relu(eng, net, net_args, conv3_dst_memory);

    IC = 256;
    OC = 256;
    memory::dims conv3_2_dst_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims conv3_2_weights_dims = {OC, IC, 3, 3};
    memory::dims conv3_2_bias_dims = {OC};
    auto conv3_2_dst_md = memory::desc(conv3_2_dst_dims, dt::f32, tensor_tag);
    auto conv3_2_dst_memory = memory(conv3_2_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(3, 2, conv3_2_weights_dims, conv3_2_bias_dims)
    do_conv(eng, net, net_args, weights_3_2.data(), biases_3_2.data(),
            conv3_2_weights_dims, conv3_2_bias_dims, conv1_strides,
            conv1_padding, conv3_dst_memory, conv3_2_dst_memory);
    do_relu(eng, net, net_args, conv3_2_dst_memory);

    auto conv3_3_dst_memory = memory(conv3_2_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(3, 3, conv3_2_weights_dims, conv3_2_bias_dims)
    do_conv(eng, net, net_args, weights_3_3.data(), biases_3_3.data(),
            conv3_2_weights_dims, conv3_2_bias_dims, conv1_strides,
            conv1_padding, conv3_2_dst_memory, conv3_3_dst_memory);
    do_relu(eng, net, net_args, conv3_3_dst_memory);

    memory::dims pool3_src_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims pool3_dst_dims = {BATCH, OC, WIDTH / 2, HEIGHT / 2};
    auto pool3_dst_md = memory::desc(pool3_dst_dims, dt::f32, tensor_tag);
    auto pool3_dst_memory = memory(pool3_dst_md, eng);
    do_pool(eng, net, net_args, strides_dims, kernel_dims, padding_dims_l,
            padding_dims_r, conv3_3_dst_memory, pool3_dst_memory);
    // /* ---------------------------------- */

    /* --- conv -> relu -> conv -> relu -> conv -> relu -> pool --- */
    IC = 256;
    OC = 512;
    WIDTH /= 2;
    HEIGHT /= 2;

    memory::dims conv4_dst_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims conv4_weights_dims = {OC, IC, 3, 3};
    memory::dims conv4_bias_dims = {OC};
    auto conv4_dst_md = memory::desc(conv4_dst_dims, dt::f32, tensor_tag);
    auto conv4_dst_memory = memory(conv4_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(4, 1, conv4_weights_dims, conv4_bias_dims)
    do_conv(eng, net, net_args, weights_4_1.data(), biases_4_1.data(),
            conv4_weights_dims, conv4_bias_dims, conv1_strides, conv1_padding,
            pool3_dst_memory, conv4_dst_memory);
    do_relu(eng, net, net_args, conv4_dst_memory);

    IC = 512;
    OC = 512;
    memory::dims conv4_2_dst_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims conv4_2_weights_dims = {OC, IC, 3, 3};
    memory::dims conv4_2_bias_dims = {OC};
    auto conv4_2_dst_md = memory::desc(conv4_2_dst_dims, dt::f32, tensor_tag);
    auto conv4_2_dst_memory = memory(conv4_2_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(4, 2, conv4_2_weights_dims, conv4_2_bias_dims)
    do_conv(eng, net, net_args, weights_4_2.data(), biases_4_2.data(),
            conv4_2_weights_dims, conv4_2_bias_dims, conv1_strides,
            conv1_padding, conv4_dst_memory, conv4_2_dst_memory);
    do_relu(eng, net, net_args, conv4_2_dst_memory);

    auto conv4_3_dst_memory = memory(conv4_2_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(4, 3, conv4_2_weights_dims, conv4_2_bias_dims)
    do_conv(eng, net, net_args, weights_4_3.data(), biases_4_3.data(),
            conv4_2_weights_dims, conv4_2_bias_dims, conv1_strides,
            conv1_padding, conv4_2_dst_memory, conv4_3_dst_memory);
    do_relu(eng, net, net_args, conv4_3_dst_memory);

    memory::dims pool4_src_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims pool4_dst_dims = {BATCH, OC, WIDTH / 2, HEIGHT / 2};
    auto pool4_dst_md = memory::desc(pool4_dst_dims, dt::f32, tensor_tag);
    auto pool4_dst_memory = memory(pool4_dst_md, eng);
    do_pool(eng, net, net_args, strides_dims, kernel_dims, padding_dims_l,
            padding_dims_r, conv4_3_dst_memory, pool4_dst_memory);
    /* ---------------------------------- */

    /* --- conv -> relu -> conv -> relu -> conv -> relu -> pool --- */
    IC = 512;
    OC = 512;
    HEIGHT /= 2;
    WIDTH /= 2;

    memory::dims conv5_dst_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims conv5_weights_dims = {OC, IC, 3, 3};
    memory::dims conv5_bias_dims = {OC};
    auto conv5_dst_md = memory::desc(conv5_dst_dims, dt::f32, tensor_tag);
    auto conv5_dst_memory = memory(conv5_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(5, 1, conv5_weights_dims, conv5_bias_dims)
    do_conv(eng, net, net_args, weights_5_1.data(), biases_5_1.data(),
            conv5_weights_dims, conv5_bias_dims, conv1_strides, conv1_padding,
            pool4_dst_memory, conv5_dst_memory);
    do_relu(eng, net, net_args, conv5_dst_memory);

    auto conv5_2_dst_memory = memory(conv5_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(5, 2, conv5_weights_dims, conv5_bias_dims)
    do_conv(eng, net, net_args, weights_5_2.data(), biases_5_2.data(),
            conv5_weights_dims, conv5_bias_dims, conv1_strides, conv1_padding,
            conv5_dst_memory, conv5_2_dst_memory);
    do_relu(eng, net, net_args, conv5_2_dst_memory);

    auto conv5_3_dst_memory = memory(conv5_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(5, 3, conv5_weights_dims, conv5_bias_dims)
    do_conv(eng, net, net_args, weights_5_3.data(), biases_5_3.data(),
            conv5_weights_dims, conv5_bias_dims, conv1_strides, conv1_padding,
            conv5_2_dst_memory, conv5_3_dst_memory);
    do_relu(eng, net, net_args, conv5_3_dst_memory);

    memory::dims pool5_src_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims pool5_dst_dims = {BATCH, OC, WIDTH / 2, HEIGHT / 2};
    auto pool5_dst_md = memory::desc(pool5_dst_dims, dt::f32, tensor_tag);
    auto pool5_dst_memory = memory(pool5_dst_md, eng);
    do_pool(eng, net, net_args, strides_dims, kernel_dims, padding_dims_l,
            padding_dims_r, conv5_3_dst_memory, pool5_dst_memory);
    /* ------------------------------------------ */

    /* --------- inner product -> ReLU --------- */
    WIDTH /= 2;
    HEIGHT /= 2;

    memory::dims ip_src_dims = pool5_dst_md.dims();
    memory::dims ip_weights_dims = {4096, IC, HEIGHT, WIDTH};
    memory::dims ip_bias_dims = {4096};
    memory::dims ip_dst_dims = {BATCH, 4096};

    std::vector<char> ip_weights_data(product(ip_weights_dims) * sizeof(float));
    std::vector<char> ip_bias_data(product(ip_bias_dims) * sizeof(float));

    auto ip_bias_md = memory::desc(ip_bias_dims, dt::f32, tag::a);
    auto ip_bias_mem = memory(ip_bias_md, eng);
    auto ip_weights_mem = memory({ip_weights_dims, dt::f32, tag::hwio}, eng);
    load_weights_fc(1, ip_weights_data.data(), product(ip_weights_dims),
            ip_bias_data.data(), product(ip_bias_dims));
    write_to_dnnl_memory(ip_bias_data.data(), ip_bias_mem);
    write_to_dnnl_memory(ip_weights_data.data(), ip_weights_mem);

    auto ip_dst_md = memory::desc(ip_dst_dims, dt::f32, tag::nc);
    auto ip_dst_mem = memory(ip_dst_md, eng);
    do_ip(eng, net, net_args, ip_weights_mem, ip_bias_mem, pool5_dst_memory,
            ip_dst_mem, true);
    /* ------------------------------------------ */

    /* --------- inner product -> ReLU --------- */
    memory::dims ip2_src_dims = ip_dst_mem.get_desc().dims();
    memory::dims ip2_weights_dims = {4096, 4096};
    memory::dims ip2_bias_dims = {4096};
    memory::dims ip2_dst_dims = {BATCH, 4096};

    std::vector<char> ip2_weights_data(
            product(ip2_weights_dims) * sizeof(float));
    std::vector<char> ip2_bias_data(product(ip2_bias_dims) * sizeof(float));

    auto ip2_bias_md = memory::desc(ip2_bias_dims, dt::f32, tag::a);
    auto ip2_bias_mem = memory(ip2_bias_md, eng);
    auto ip2_weights_mem = memory({ip2_weights_dims, dt::f32, tag::io}, eng);
    load_weights_fc(2, ip2_weights_data.data(), product(ip2_weights_dims),
            ip2_bias_data.data(), product(ip2_bias_dims));
    write_to_dnnl_memory(ip2_bias_data.data(), ip2_bias_mem);
    write_to_dnnl_memory(ip2_weights_data.data(), ip2_weights_mem);

    auto ip2_dst_md = memory::desc(ip2_dst_dims, dt::f32, tag::nc);
    auto ip2_dst_mem = memory(ip2_dst_md, eng);

    do_ip(eng, net, net_args, ip2_weights_mem, ip2_bias_mem, ip_dst_mem,
            ip2_dst_mem, true);
    /* ------------------------------------------ */

    /* --------- inner product --------- */
    IC = 4096;
    OC = 1000;
    memory::dims ip3_src_dims = ip2_dst_mem.get_desc().dims();
    memory::dims ip3_weights_dims = {OC, IC};
    memory::dims ip3_bias_dims = {OC};
    memory::dims ip3_dst_dims = {BATCH, OC};

    std::vector<char> ip3_weights_data(
            product(ip3_weights_dims) * sizeof(float));
    std::vector<char> ip3_bias_data(product(ip3_bias_dims) * sizeof(float));

    auto ip3_bias_md = memory::desc(ip3_bias_dims, dt::f32, tag::a);
    auto ip3_bias_mem = memory(ip3_bias_md, eng);
    auto ip3_weights_mem = memory({ip3_weights_dims, dt::f32, tag::io}, eng);
    load_weights_fc(3, ip3_weights_data.data(), product(ip3_weights_dims),
            ip3_bias_data.data(), product(ip3_bias_dims));
    write_to_dnnl_memory(ip3_bias_data.data(), ip3_bias_mem);
    write_to_dnnl_memory(ip3_weights_data.data(), ip3_weights_mem);

    auto ip3_dst_md = memory::desc(ip3_dst_dims, dt::f32, tag::nc);
    auto ip3_dst_mem = memory(ip3_dst_md, eng);
    do_ip(eng, net, net_args, ip3_weights_mem, ip3_bias_mem, ip2_dst_mem,
            ip3_dst_mem, false);
    /* ------------------------------------------ */

    /* --------- softmax --------- */
    const int axis = 1;
    auto softmax_d = softmax_forward::desc(
            prop_kind::forward_inference, ip3_dst_mem.get_desc(), axis);
    auto softmax_pd = softmax_forward::primitive_desc(softmax_d, eng);
    auto softmax_prim = softmax_forward(softmax_pd);

    net.push_back(softmax_prim);
    net_args.push_back(
            {{DNNL_ARG_SRC, ip3_dst_mem}, {DNNL_ARG_DST, ip3_dst_mem}});
    /* ------------------------------------------ */
    // Execute model

    for (size_t i = 0; i < net.size(); ++i) {
        net[i].execute(s, net_args[i]);
        // auto dst_mem = net_args.at(i).at(DNNL_ARG_DST);
        // std::vector<float> temp(dst_mem.get_desc().get_size() /
        // sizeof(float)); read_from_dnnl_memory(temp.data(), dst_mem); auto
        // kind = net.at(i).get_kind(); std::string kind_str; switch (kind) {
        //     case dnnl::primitive::kind::convolution:
        //         kind_str = "Convolution: ";
        //         break;
        //     case dnnl::primitive::kind::eltwise: kind_str = "Eltwise: ";
        //     break; case dnnl::primitive::kind::pooling: kind_str = "Pooling:
        //     "; break; case dnnl::primitive::kind::batch_normalization:
        //         kind_str = "Bnorm: ";
        //         break;
        //     case dnnl::primitive::kind::inner_product:
        //         kind_str = "Inner Product: ";
        //         break;
        // }
        // std::cout << kind_str << std::endl;
        // hash_and_print(temp, true, 10);
    }

    // Wait for completion.
    s.wait();
    std::vector<float> result(product(ip3_dst_mem.get_desc().dims()));
    read_from_dnnl_memory(result.data(), ip3_dst_mem);
    for (size_t i = 0; i < BATCH; i++) {
        auto index = std::max_element(
                result.begin() + i * 1000, result.begin() + (i + 1) * 1000);
        //std::cout << "classed as "
        //          << std::distance(result.begin() + i * 1000, index)
        //          << ", value " << (index != std::end(result) ? *index : 0.f)
        //          << std::endl;
    }

#undef STAMP_OUT_WEIGHTS_CONV
}

void run_vgg(engine &eng, stream &s, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> net_args,
        std::vector<char> &input_image, int times) {
    // Execute model
    for (int j = 0; j < times; ++j) {
        for (size_t i = 0; i < net.size(); ++i) {
            net[i].execute(s, net_args[i]);
        }
    }
    // Wait for completion.
    s.wait();
}

void check_vgg(engine &eng, stream &s, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> net_args,
        std::vector<char> &input_image) {
    run_vgg(eng, s, net, net_args, input_image, 1);
}

void do_it(engine::kind engine_kind) {
    engine eng(engine_kind, 0);
    stream s(eng);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    std::vector<char> input_image;

    std::cout << "Setting up... for BATCH SIZE = " << BATCH << "\n";
    set_up_vgg(eng, s, net, net_args, input_image);
    //     int times = 1;
    //     std::cout << "Warming up...\n";
    //     run_vgg(eng, s, net, net_args, input_image, 0);

    std::cout << "Timing...\n";
    auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    // FIXME: revert to 100
    int times = 100;
    run_vgg(eng, s, net, net_args, input_image, times);
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
                       .count();
    std::cout << "Use time: " << (end - begin) / (times + 0.0)
              << " ms per iteration." << std::endl;
}

int main(int argc, char **argv) {
    for (int i = 1; i <= 64; i *= 2) {
        BATCH = i;
        do_it(parse_engine_kind(argc, argv));
    }
}
