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

void dump_to_file(std::vector<float> &data, std::string fname) {
    // Check correctness
    std::stringstream ss;
    ss.precision(9);
    for (size_t i = 0; i < data.size(); i++) {
        ss << data[i] << "\n";
    }

    std::ofstream ofs;
    ofs.open(fname.c_str());
    ofs << ss.str();
    ofs.close();
}
void check_batch(std::vector<float> const &tensor, int batch_size) {
    double last_accum = 0;
    for (size_t j = 0; j < batch_size; j++) {
        double accum = 0;
        auto hwc = tensor.size() / batch_size;
        for (std::size_t i = 0; i < hwc; ++i) {
            if (i % 2 == 0) {
                accum += tensor[i + j * hwc];
            } else {
                accum -= tensor[i + j * hwc];
            }
        }
        if (j == 0)
            last_accum = accum;
        else if (accum != last_accum) {
            throw std::runtime_error("");
        }
    }
}
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
    fclose(file);
}

void load_weights_conv(int layer_index, void *weights_data, int weights_size,
        void *biases_data, int bias_size) {
    std::string base_filename = "resnet50_transposed_param_files/conv"
            + std::to_string(layer_index) + "_";

    read_binary_data(biases_data, bias_size, base_filename + "biases.bin");
    read_binary_data(weights_data, weights_size, base_filename + "weights.bin");
}

void load_weights_conv_branched(int layer_index, int block_index,
        int branch_index, int sub_branch_index, void *weights_data,
        int weights_size, void *biases_data, int bias_size) {
    std::string block_index_str
            = (block_index > 0 ? "_" + std::to_string(block_index) : "");
    std::string sub_branch_index_str
            = (sub_branch_index > 0 ? "_" + std::to_string(sub_branch_index)
                                    : "");
    std::string base_filename = "resnet50_transposed_param_files/res"
            + std::to_string(layer_index) + block_index_str + "_branch"
            + std::to_string(branch_index) + sub_branch_index_str + "_";

    read_binary_data(biases_data, bias_size, base_filename + "biases.bin");
    read_binary_data(weights_data, weights_size, base_filename + "weights.bin");
}

void load_weights_fc(int index, void *weights_data, int weights_size,
        void *biases_data, int bias_size) {
    std::string base_filename = "resnet50_transposed_param_files/fc"
            + std::to_string(index) + "_";

    read_binary_data(biases_data, bias_size, base_filename + "biases.bin");
    read_binary_data(weights_data, weights_size, base_filename + "weights.bin");
}

void load_bnorm_branched(int layer_index, int block_index, int branch_index,
        int sub_branch_index, void *mean_data, int mean_size, void *var_data,
        int var_size, void *scale_data, int scale_size, void *shift_data,
        int shift_size) {

    std::string block_index_str
            = (block_index > 0 ? "_" + std::to_string(block_index) : "");
    std::string sub_branch_index_str
            = (sub_branch_index > 0 ? "_" + std::to_string(sub_branch_index)
                                    : "");
    std::string base_filename = "resnet50_transposed_param_files/bn"
            + std::to_string(layer_index) + block_index_str + "_branch"
            + std::to_string(branch_index) + sub_branch_index_str + "_";
    read_binary_data(mean_data, mean_size, base_filename + "mean.bin");
    read_binary_data(var_data, var_size, base_filename + "var.bin");
    read_binary_data(scale_data, scale_size, base_filename + "scale.bin");
    read_binary_data(shift_data, shift_size, base_filename + "shift.bin");
}

void load_bnorm(int layer_index, void *mean_data, int mean_size, void *var_data,
        int var_size, void *scale_data, int scale_size, void *shift_data,
        int shift_size) {
    std::string base_filename = "resnet50_transposed_param_files/bn_conv"
            + std::to_string(layer_index) + "_";
    read_binary_data(mean_data, mean_size, base_filename + "mean.bin");
    read_binary_data(var_data, var_size, base_filename + "var.bin");
    read_binary_data(scale_data, scale_size, base_filename + "scale.bin");
    read_binary_data(shift_data, shift_size, base_filename + "shift.bin");
}

#define STAMP_OUT_WEIGHTS_CONV(INDEX, WEI_DIMS, BIA_DIMS) \
    std::vector<float> weights_##INDEX(product(WEI_DIMS)); \
    std::vector<float> biases_##INDEX(product(BIA_DIMS)); \
    load_weights_conv(INDEX, weights_##INDEX.data(), product(WEI_DIMS), \
            biases_##INDEX.data(), product(BIA_DIMS))

#define STAMP_OUT_WEIGHTS_CONV_BRANCHED(LAYER_INDEX, BLOCK_INDEX, \
        BRANCH_INDEX, SUB_BRANCH_INDEX, WEI_DIMS, BIA_DIMS) \
    std::vector<float> \
            weights_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX( \
                    product(WEI_DIMS)); \
    std::vector<float> \
            biases_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX( \
                    product(BIA_DIMS)); \
    load_weights_conv_branched(LAYER_INDEX, BLOCK_INDEX, BRANCH_INDEX, \
            SUB_BRANCH_INDEX, \
            weights_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX \
                    .data(), \
            product(WEI_DIMS), \
            biases_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX \
                    .data(), \
            product(BIA_DIMS))

#define STAMP_OUT_BNORM_BRANCHED( \
        LAYER_INDEX, BLOCK_INDEX, BRANCH_INDEX, SUB_BRANCH_INDEX, C) \
    std::vector<float> \
            mean_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX( \
                    C); \
    std::vector<float> \
            var_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX( \
                    C); \
    std::vector<float> \
            scale_shift_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX( \
                    2 * C); \
    load_bnorm_branched(LAYER_INDEX, BLOCK_INDEX, BRANCH_INDEX, \
            SUB_BRANCH_INDEX, \
            mean_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX \
                    .data(), \
            C, \
            var_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX \
                    .data(), \
            C, \
            scale_shift_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX \
                    .data(), \
            C, \
            scale_shift_##LAYER_INDEX##_##BLOCK_INDEX##_##BRANCH_INDEX##_##SUB_BRANCH_INDEX \
                            .data() \
                    + C, \
            2 * C)

#define STAMP_OUT_BNORM(LAYER_INDEX, C) \
    std::vector<float> mean_##LAYER_INDEX(C); \
    std::vector<float> var_##LAYER_INDEX(C); \
    std::vector<float> scale_shift_##LAYER_INDEX(2 * C); \
    load_bnorm(LAYER_INDEX, mean_##LAYER_INDEX.data(), C, \
            var_##LAYER_INDEX.data(), C, scale_shift_##LAYER_INDEX.data(), C, \
            scale_shift_##LAYER_INDEX.data() + C, C)
void do_reorder(engine &eng, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> &net_args, memory &src,
        memory &dst) {
    // Create primitive descriptor.
    auto src_md = src.get_desc();
    auto dst_md = dst.get_desc();
    auto reorder_pd = reorder::primitive_desc(
            eng, src_md, eng, dst_md, primitive_attr());

    net.push_back(reorder(reorder_pd));
    net_args.push_back({{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
}
void do_bnorm(engine &eng, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> &net_args, void *mean_data,
        void *var_data, void *scale_shift_data, int oc, memory &data_mem) {
    memory::dims mean_var_dims = {oc};
    memory::dims scale_shift_dims = {2, oc};

    auto mean_memory = memory({{mean_var_dims}, dt::f32, tag::x}, eng);
    auto var_memory = memory({{mean_var_dims}, dt::f32, tag::x}, eng);
    auto scale_shift_memory
            = memory({{scale_shift_dims}, dt::f32, tag::ab}, eng);

    write_to_dnnl_memory(mean_data, mean_memory);
    write_to_dnnl_memory(var_data, var_memory);
    write_to_dnnl_memory(scale_shift_data, scale_shift_memory);

    auto data_md = data_mem.get_desc();
    auto bnorm_desc = batch_normalization_forward::desc(
            prop_kind::forward_inference, data_md, 1.001e-5,
            normalization_flags::use_global_stats
                    | normalization_flags::use_scale_shift);
    auto prim_desc
            = batch_normalization_forward::primitive_desc(bnorm_desc, eng);

    net.push_back(batch_normalization_forward(prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, data_mem}, {DNNL_ARG_MEAN, mean_memory},
            {DNNL_ARG_VARIANCE, var_memory},
            {DNNL_ARG_SCALE_SHIFT, scale_shift_memory},
            {DNNL_ARG_DST, data_mem}});
}
// NOTE: src_memory should have its data set *before* it is passed to this
// function
void do_conv(engine &eng, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> &net_args,
        void *weights_data, void *bias_data, memory::dims &weights_dims,
        memory::dims &bias_dims, memory::dims &strides, memory::dims &padding,
        memory &src_memory, memory &dst_memory, bool with_bias = true) {
    auto weights_memory = memory({{weights_dims}, dt::f32, tag::ohwi}, eng);
    //     auto weights_memory_nchw
    //             = memory({{weights_dims}, dt::f32, tag::oihw}, eng);
    memory bias_memory;
    if (with_bias) bias_memory = memory({{bias_dims}, dt::f32, tag::x}, eng);

    write_to_dnnl_memory(weights_data, weights_memory);
    //     do_reorder(eng, net, net_args, weights_memory, weights_memory_nchw);
    if (with_bias) write_to_dnnl_memory(bias_data, bias_memory);

    auto src_md = src_memory.get_desc();
    memory::desc bias_md = {};
    if (with_bias) bias_md = bias_memory.get_desc();
    auto weights_md = weights_memory.get_desc();
    //     auto weights_md_nchw = memory::desc(weights_md.dims(), dt::f32,
    //     tag::nchw);
    auto dst_md = dst_memory.get_desc();

    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md,
            strides, padding, padding);
    auto prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);

    net.push_back(convolution_forward(prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, src_memory},
            {DNNL_ARG_WEIGHTS, weights_memory}, {DNNL_ARG_DST, dst_memory}});
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

memory::desc do_input_padding(memory &input, std::vector<float> &new_data,
        memory::dims &padding_l, memory::dims &padding_r) {
    auto md = input.get_desc();
    auto new_dims = md.dims();
    new_dims[2] += (padding_l[0] + padding_r[0]);
    new_dims[3] += (padding_l[1] + padding_r[1]);
    std::vector<float> old_data(product(md.dims()));
    read_from_dnnl_memory(old_data.data(), input);
    new_data.resize(product(new_dims));
    const size_t old_h = md.dims()[2];
    const size_t old_w = md.dims()[3];
    const size_t new_h = new_dims[2];
    const size_t new_w = new_dims[3];
    size_t N = md.dims()[0], C = md.dims()[1], H = md.dims()[2],
           W = md.dims()[3];
    size_t new_N = new_dims[0], new_C = new_dims[1], new_H = new_dims[2],
           new_W = new_dims[3];
    for (size_t n = 0; n < N; n++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                for (size_t c = 0; c < C; c++) {

                    size_t old_ofs = n * H * W * C + h * W * C + w * C + c;
                    size_t new_ofs = n * new_C * new_H * new_W
                            + (h + padding_l[0]) * new_W * new_C
                            + (w + padding_l[1]) * new_C + c;

                    new_data[new_ofs] = old_data[old_ofs];
                }
            }
        }
    }

    // Check correctness

    //     for (size_t n = 0; n < new_N; n++) {
    //         for (size_t c = 0; c < new_C; c++) {
    //             for (size_t h = 0; h < new_H; h++) {
    //                 for (size_t w = 0; w < new_W; w++) {
    //                     size_t new_ofs = n * new_C * new_H * new_W
    //                             + c * new_H * new_W
    //                             + (h /* + 2 * padding_l[0] */) * new_W
    //                             + (w /* + 2 * padding_l[1] */);
    //                     if ((w >= new_W - 2 || h >= new_H - 2)
    //                             && new_data[new_ofs] != 0) {
    //                         printf("\n\n\n\n\nFAILURE!!!!\n\n\n\n\n");
    //                     }
    //                 }
    //             }
    //         }
    //     }

    return memory::desc(new_dims, dt::f32, tag::nhwc);
}
void do_pool(engine &eng, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> &net_args,
        memory::dims &strides_dims, memory::dims &kernel_dims,
        memory::dims &pad_dims_l, memory::dims &pad_dims_r, memory &src_mem,
        memory &dst_mem, dnnl::algorithm alg_kind = algorithm::pooling_max,
        bool pad_input = false) {

    auto src_md = src_mem.get_desc();
    if (pad_input) {

        std::vector<float> new_data;
        src_md = do_input_padding(src_mem, new_data, pad_dims_l, pad_dims_r);
        src_mem = memory(src_md, eng);
        write_to_dnnl_memory(new_data.data(), src_mem);
        pad_dims_l = {0, 0};
        pad_dims_r = {0, 0};
    }

    auto dst_md = dst_mem.get_desc();

    auto pooling_d = pooling_forward::desc(prop_kind::forward_inference,
            alg_kind, src_md, dst_md, strides_dims, kernel_dims, pad_dims_l,
            pad_dims_r);

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

void do_binary(engine &eng, std::vector<primitive> &net,
        std::vector<std::unordered_map<int, memory>> &net_args,
        memory &src0_mem, memory &src1_mem, memory &dst_mem) {
    auto src0_md = src0_mem.get_desc();
    auto src1_md = src1_mem.get_desc();
    auto dst_md = dst_mem.get_desc();
    auto binary_desc
            = binary::desc(algorithm::binary_add, src0_md, src1_md, dst_md);
    auto binary_prim_desc = binary::primitive_desc(binary_desc, eng);
    auto binary_prim = binary(binary_prim_desc);
    net.push_back(binary_prim);
    net_args.push_back({{DNNL_ARG_SRC_0, src0_mem}, {DNNL_ARG_SRC_1, src1_mem},
            {DNNL_ARG_DST, dst_mem}});
}

struct block_params {

    block_params(int layer_id, int block_id, int branch_id, int branch_block_id,
            bool with_relu, bool with_bias, memory::dim batch, memory::dim ic,
            memory::dim oc, memory::dim oh, memory::dim ow, memory::dim kh,
            memory::dim kw, memory::dims padding, memory::dims strides)
        : layer_id(layer_id)
        , block_id(block_id)
        , branch_id(branch_id)
        , branch_block_id(branch_block_id)
        , with_relu(with_relu)
        , with_bias(with_bias)
        , batch(batch)
        , ic(ic)
        , oc(oc)
        , oh(oh)
        , ow(ow)
        , kh(kh)
        , kw(kw)
        , padding(padding)
        , strides(strides) {}
    int layer_id;
    int block_id;
    int branch_id;
    int branch_block_id;
    bool with_relu;
    bool with_bias;
    memory::dim batch;
    memory::dim ic;
    memory::dim oc;
    memory::dim oh, ow;
    memory::dim kh, kw;
    memory::dims padding;
    memory::dims strides;
};
class res_net_block {
public:
    res_net_block(engine &eng, std::vector<primitive> &net,
            std::vector<std::unordered_map<int, memory>> &net_args,
            memory &src_mem, block_params params)
        : eng_(eng)
        , net_(net)
        , net_args_(net_args)
        , src_mem_(src_mem)
        , params_(params) {

        dst_dims_ = {params.batch, params.oc, params.oh, params.ow};
        weights_dims_ = {params.oc, params.ic, params.kh, params.kw};
        if (params.with_bias) bias_dims_ = {params.oc};
        // auto src_md_nchw
        //         = memory::desc {src_mem_.get_desc().dims(), dt::f32,
        //         tag::nchw};
        // src_mem_nchw_ = memory(src_md_nchw, eng);
        // do_reorder(eng, net, net_args, src_mem_, src_mem_nchw_);
        auto dst_md = memory::desc(dst_dims_, dt::f32, tag::nhwc);
        dst_mem_ = memory(dst_md, eng);
        // auto dst_md_nchw = memory::desc(dst_dims_, dt::f32, tag::nchw);
        // dst_mem_nchw_ = memory(dst_md_nchw, eng);

        // Load conv data
        weights_data_.resize(product(weights_dims_));
        bias_data_.resize(product(bias_dims_));
        load_weights_conv_branched(params.layer_id, params.block_id,
                params.branch_id, params.branch_block_id, weights_data_.data(),
                product(weights_dims_), bias_data_.data(), product(bias_dims_));

        // hash_and_print(weights_data_, true, 10);
        // load bnorm data
        mean_data_.resize(params.oc);
        var_data_.resize(params.oc);
        scale_shift_data_.resize(2 * params.oc);
        load_bnorm_branched(params.layer_id, params.block_id, params.branch_id,
                params.branch_block_id, mean_data_.data(), params.oc,
                var_data_.data(), params.oc, scale_shift_data_.data(),
                params.oc, scale_shift_data_.data() + params.oc, params.oc);

        do_conv(eng, net, net_args, weights_data_.data(),
                params.with_bias ? bias_data_.data() : nullptr, weights_dims_,
                bias_dims_, params.strides, params.padding, src_mem_, dst_mem_,
                false);
        do_bnorm(eng, net, net_args, mean_data_.data(), var_data_.data(),
                scale_shift_data_.data(), params.oc, dst_mem_);
        // do_reorder(eng, net, net_args, dst_mem_nchw_, dst_mem_);
        if (params.with_relu) { do_relu(eng_, net_, net_args_, dst_mem_); }
    }

    memory &get_dst_mem() { return dst_mem_; }

private:
    engine &eng_;
    std::vector<primitive> &net_;
    std::vector<std::unordered_map<int, memory>> &net_args_;
    memory &src_mem_;
    //     memory src_mem_nchw_;
    //     memory dst_mem_nchw_;
    block_params params_;

    // Weights data
    std::vector<float> weights_data_;
    std::vector<float> bias_data_;
    std::vector<float> mean_data_;
    std::vector<float> var_data_;
    std::vector<float> scale_shift_data_;

    // Memory
    memory dst_mem_;

    // Dims
    memory::dims weights_dims_;
    memory::dims bias_dims_;
    memory::dims dst_dims_;
};

void res_net(engine::kind engine_kind, int times = 100) {

    engine eng(engine_kind, 0);
    stream s(eng);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    std::vector<memory::dim> ic_vec = {3, 64, 128, 256, 512};
    std::vector<memory::dim> oc_vec = {64, 128, 256, 512, 512};

    /* Convolution dims */
    memory::dim IC = 3, OC = 64, WIDTH = 224, HEIGHT = 224;

    memory::dims conv1_strides = {2, 2};
    memory::dims conv1_padding = {3, 3};

    memory::dims conv1_src_dims = {BATCH, IC, WIDTH, HEIGHT};
    memory::dims conv1_dst_dims = {BATCH, OC, WIDTH / 2, HEIGHT / 2};
    memory::dims conv1_weights_dims = {OC, IC, 7, 7};
    memory::dims conv1_bias_dims = {0};

    int src_size = product(conv1_src_dims) / BATCH;
    std::vector<char> src_data(src_size * sizeof(float));

    tag mem_format = tag::nhwc;
    std::string image_filename = "grace_hopper_nhwc.bin";
    read_binary_data(src_data.data(), src_size, image_filename);
    std::vector<char> src_data_copied;
    src_data_copied.reserve(src_data.size() * BATCH);
    for (size_t i = 0; i < BATCH; i++) {
        std::copy(src_data.begin(), src_data.end(),
                std::back_inserter(src_data_copied));
    }
    /* -------------- conv -> relu -> conv -> relu -> pool -------------- */
    auto conv1_src_md = memory::desc(conv1_src_dims, dt::f32, mem_format);
    auto conv1_src_memory = memory(conv1_src_md, eng);
    write_to_dnnl_memory(src_data_copied.data(), conv1_src_memory);
    //     std::vector<float> test_data;
    //     test_data.resize(product(conv1_src_dims));
    //     read_from_dnnl_memory(test_data.data(), conv1_src_memory);
    //     std::vector<float> padded_data;
    //     memory::dims pad_r {3, 3};
    //     auto conv1_src_md_padded = do_input_padding(
    //             conv1_src_memory, padded_data, conv1_padding, pad_r);
    //     auto conv1_src_mem_padded = memory(conv1_src_md_padded, eng);
    //     conv1_padding = {0, 0};
    //     write_to_dnnl_memory(padded_data.data(), conv1_src_mem_padded);
    auto conv1_dst_md = memory::desc(conv1_dst_dims, dt::f32, mem_format);
    auto conv1_dst_memory = memory(conv1_dst_md, eng);
    STAMP_OUT_WEIGHTS_CONV(1, conv1_weights_dims, conv1_bias_dims);
    STAMP_OUT_BNORM(1, OC);
    do_conv(eng, net, net_args, weights_1.data(), biases_1.data(),
            conv1_weights_dims, conv1_bias_dims, conv1_strides, conv1_padding,
            conv1_src_memory, conv1_dst_memory, false);
    do_bnorm(eng, net, net_args, mean_1.data(), var_1.data(),
            scale_shift_1.data(), OC, conv1_dst_memory);
    do_relu(eng, net, net_args, conv1_dst_memory);

    /* Pooling dims */
    const memory::dim KH = 3, // kernel height
            KW = 3, // kernel width
            PH_L = 1, // height padding: left
            PH_R = 1, // height padding: right
            PW_L = 1, // width padding: left
            PW_R = 1, // width padding: right
            SH = 2, // height-wise stride
            SW = 2; // width-wise stride
    WIDTH = WIDTH / 2, HEIGHT = HEIGHT / 2;
    memory::dims kernel_dims = {KH, KW};
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};
    memory::dims pool1_src_dims = {BATCH, OC, WIDTH, HEIGHT};
    memory::dims pool1_dst_dims = {BATCH, OC, WIDTH / 2, HEIGHT / 2};
    auto pool1_dst_md = memory::desc(pool1_dst_dims, dt::f32, mem_format);
    auto pool1_dst_memory = memory(pool1_dst_md, eng);
    do_pool(eng, net, net_args, strides_dims, kernel_dims, padding_dims_l,
            padding_dims_r, conv1_dst_memory, pool1_dst_memory,
            algorithm::pooling_max, false);
    /* ---------------------------------- */

    /* -----------------res2a_branch1----------------- */
    /* -----------------conv > bnorm > scale----------------- */
    block_params block_2a1_params {2, 1, 1, 0, false, false, BATCH, 64, 256, 56,
            56, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_2a1(
            eng, net, net_args, pool1_dst_memory, block_2a1_params);
    //     WIDTH = WIDTH / 2, HEIGHT = HEIGHT / 2;
    //     IC = 64, OC = 256;
    //     memory::dims res2a_strides = {1, 1};
    //     memory::dims res2a_padding = {0, 0};
    //     memory::dims res2a_src_dims = {BATCH, IC, WIDTH, HEIGHT};
    //     memory::dims res2a_dst_dims = {BATCH, OC, WIDTH, HEIGHT};
    //     memory::dims res2a_weights_dims = {OC, IC, 1, 1};
    //     memory::dims res2a_bias_dims = {};

    //     auto res2a_dst_md = memory::desc(res2a_dst_dims, dt::f32,
    //     mem_format); auto res2a_dst_memory = memory(res2a_dst_md, eng);
    //     STAMP_OUT_WEIGHTS_CONV_BRANCHED(
    //             2, 1, 1, 0, res2a_weights_dims, res2a_bias_dims);
    //     STAMP_OUT_BNORM_BRANCHED(2, 1, 1, 0, OC);
    //     do_conv(eng, net, net_args, weights_2_1_1_0.data(), nullptr,
    //             res2a_weights_dims, res2a_bias_dims, res2a_strides,
    //             res2a_padding, pool1_dst_memory, res2a_dst_memory);
    //     do_bnorm(eng, net, net_args, mean_2_1_1_0.data(), var_2_1_1_0.data(),
    //             scale_shift_2_1_1_0.data(), OC, res2a_dst_memory);

    /* -----------------res2a_branch2----------------- */
    /* -----------------conv > bnorm > scale > relu----------------- */
    block_params block_2a2a_params {2, 1, 2, 1, true, false, BATCH, 64, 64, 56,
            56, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_2a2a(
            eng, net, net_args, pool1_dst_memory, block_2a2a_params);

    /* -----------------conv > bnorm > scale > relu----------------- */
    block_params block_2a2b_params {2, 1, 2, 2, true, false, BATCH, 64, 64, 56,
            56, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_2a2b(
            eng, net, net_args, block_2a2a.get_dst_mem(), block_2a2b_params);

    /* -----------------conv > bnorm > scale----------------- */
    block_params block_2a2c_params {2, 1, 2, 3, false, false, BATCH, 64, 256,
            56, 56, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_2a2c(
            eng, net, net_args, block_2a2b.get_dst_mem(), block_2a2c_params);

    /* -----------------add > relu----------------- */
    OC = 256;
    WIDTH = 56, HEIGHT = 56;
    memory::dims res2a_add_dst_dims = {BATCH, OC, WIDTH, HEIGHT};
    auto res2a_add_dst_md
            = memory::desc(res2a_add_dst_dims, dt::f32, mem_format);
    auto res2a_add_dst_memory = memory(res2a_add_dst_md, eng);

    do_binary(eng, net, net_args, block_2a1.get_dst_mem(),
            block_2a2c.get_dst_mem(), res2a_add_dst_memory);
    do_relu(eng, net, net_args, res2a_add_dst_memory);

    /* -----------------res2b----------------- */
    /* -----------------branch2----------------- */
    block_params block_2b2a_params {2, 2, 2, 1, true, false, BATCH, 256, 64, 56,
            56, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_2b2a(
            eng, net, net_args, res2a_add_dst_memory, block_2b2a_params);

    block_params block_2b2b_params {2, 2, 2, 2, true, false, BATCH, 64, 64, 56,
            56, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_2b2b(
            eng, net, net_args, block_2b2a.get_dst_mem(), block_2b2b_params);

    block_params block_2b2c_params {2, 2, 2, 3, false, false, BATCH, 64, 256,
            56, 56, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_2b2c(
            eng, net, net_args, block_2b2b.get_dst_mem(), block_2b2c_params);

    /* -----------------add > relu----------------- */
    IC = 256, OC = 256;
    memory::dims res2b_add_dst_dims = {BATCH, 256, 56, 56};
    auto res2b_add_dst_md
            = memory::desc(res2b_add_dst_dims, dt::f32, mem_format);
    auto res2b_add_dst_memory = memory(res2b_add_dst_md, eng);

    do_binary(eng, net, net_args, res2a_add_dst_memory,
            block_2b2c.get_dst_mem(), res2b_add_dst_memory);
    do_relu(eng, net, net_args, res2b_add_dst_memory);

    /* -----------------res2c----------------- */
    /* -----------------branch2----------------- */
    block_params block_2c2a_params {2, 3, 2, 1, true, false, BATCH, 256, 64, 56,
            56, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_2c2a(
            eng, net, net_args, res2b_add_dst_memory, block_2c2a_params);

    block_params block_2c2b_params {2, 3, 2, 2, true, false, BATCH, 64, 64, 56,
            56, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_2c2b(
            eng, net, net_args, block_2c2a.get_dst_mem(), block_2c2b_params);

    block_params block_2c2c_params {2, 3, 2, 3, false, false, BATCH, 64, 256,
            56, 56, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_2c2c(
            eng, net, net_args, block_2c2b.get_dst_mem(), block_2c2c_params);

    /* -----------------add > relu----------------- */
    IC = 256, OC = 256;
    memory::dims res2c_add_dst_dims = {BATCH, 256, 56, 56};
    auto res2c_add_dst_md
            = memory::desc(res2c_add_dst_dims, dt::f32, mem_format);
    auto res2c_add_dst_memory = memory(res2c_add_dst_md, eng);

    do_binary(eng, net, net_args, res2b_add_dst_memory,
            block_2c2c.get_dst_mem(), res2c_add_dst_memory);
    do_relu(eng, net, net_args, res2c_add_dst_memory);

    /* -----------------res3a_branch1----------------- */
    /* -----------------conv > bnorm > scale----------------- */
    block_params block_3a1_params {3, 1, 1, 0, false, false, BATCH, 256, 512,
            28, 28, 1, 1, {0, 0}, {2, 2}};
    res_net_block block_3a1(
            eng, net, net_args, res2c_add_dst_memory, block_3a1_params);

    /* -----------------res2a_branch2----------------- */
    /* -----------------conv > bnorm > scale > relu----------------- */
    block_params block_3a2a_params {3, 1, 2, 1, true, false, BATCH, 256, 128,
            28, 28, 1, 1, {0, 0}, {2, 2}};
    res_net_block block_3a2a(
            eng, net, net_args, res2c_add_dst_memory, block_3a2a_params);

    /* -----------------conv > bnorm > scale > relu----------------- */
    block_params block_3a2b_params {3, 1, 2, 2, true, false, BATCH, 128, 128,
            28, 28, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_3a2b(
            eng, net, net_args, block_3a2a.get_dst_mem(), block_3a2b_params);

    /* -----------------conv > bnorm > scale----------------- */
    block_params block_3a2c_params {3, 1, 2, 3, false, false, BATCH, 128, 512,
            28, 28, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_3a2c(
            eng, net, net_args, block_3a2b.get_dst_mem(), block_3a2c_params);

    /* -----------------add > relu----------------- */
    OC = 512;
    WIDTH = 28, HEIGHT = 28;
    memory::dims res3a_add_dst_dims = {BATCH, OC, WIDTH, HEIGHT};
    auto res3a_add_dst_md
            = memory::desc(res3a_add_dst_dims, dt::f32, mem_format);
    auto res3a_add_dst_memory = memory(res3a_add_dst_md, eng);

    do_binary(eng, net, net_args, block_3a1.get_dst_mem(),
            block_3a2c.get_dst_mem(), res3a_add_dst_memory);
    do_relu(eng, net, net_args, res3a_add_dst_memory);

    /* -----------------res3b----------------- */
    /* -----------------branch2----------------- */
    WIDTH = 28, HEIGHT = 28;
    block_params block_3b2a_params {3, 2, 2, 1, true, false, BATCH, 512, 128,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_3b2a(
            eng, net, net_args, res3a_add_dst_memory, block_3b2a_params);

    block_params block_3b2b_params {3, 2, 2, 2, true, false, BATCH, 128, 128,
            WIDTH, HEIGHT, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_3b2b(
            eng, net, net_args, block_3b2a.get_dst_mem(), block_3b2b_params);

    block_params block_3b2c_params {3, 2, 2, 3, false, false, BATCH, 128, 512,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_3b2c(
            eng, net, net_args, block_3b2b.get_dst_mem(), block_3b2c_params);

    /* -----------------add > relu----------------- */
    IC = 512, OC = 512;
    memory::dims res3b_add_dst_dims = {BATCH, OC, 28, 28};
    auto res3b_add_dst_md
            = memory::desc(res3b_add_dst_dims, dt::f32, mem_format);
    auto res3b_add_dst_memory = memory(res3b_add_dst_md, eng);

    do_binary(eng, net, net_args, res3a_add_dst_memory,
            block_3b2c.get_dst_mem(), res3b_add_dst_memory);
    do_relu(eng, net, net_args, res3b_add_dst_memory);

    /* -----------------res3c----------------- */
    /* -----------------branch2----------------- */
    WIDTH = 28, HEIGHT = 28;
    block_params block_3c2a_params {3, 3, 2, 1, true, false, BATCH, 512, 128,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_3c2a(
            eng, net, net_args, res3b_add_dst_memory, block_3c2a_params);

    block_params block_3c2b_params {3, 3, 2, 2, true, false, BATCH, 128, 128,
            WIDTH, HEIGHT, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_3c2b(
            eng, net, net_args, block_3c2a.get_dst_mem(), block_3c2b_params);

    block_params block_3c2c_params {3, 3, 2, 3, false, false, BATCH, 128, 512,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_3c2c(
            eng, net, net_args, block_3c2b.get_dst_mem(), block_3c2c_params);

    /* -----------------add > relu----------------- */
    IC = 512, OC = 512;
    memory::dims res3c_add_dst_dims = {BATCH, OC, 28, 28};
    auto res3c_add_dst_md
            = memory::desc(res3c_add_dst_dims, dt::f32, mem_format);
    auto res3c_add_dst_memory = memory(res3c_add_dst_md, eng);

    do_binary(eng, net, net_args, res3b_add_dst_memory,
            block_3c2c.get_dst_mem(), res3c_add_dst_memory);
    do_relu(eng, net, net_args, res3c_add_dst_memory);

    /* -----------------res3d----------------- */
    /* -----------------branch2----------------- */
    WIDTH = 28, HEIGHT = 28;
    block_params block_3d2a_params {3, 4, 2, 1, true, false, BATCH, 512, 128,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_3d2a(
            eng, net, net_args, res3c_add_dst_memory, block_3d2a_params);

    block_params block_3d2b_params {3, 4, 2, 2, true, false, BATCH, 128, 128,
            WIDTH, HEIGHT, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_3d2b(
            eng, net, net_args, block_3d2a.get_dst_mem(), block_3d2b_params);

    block_params block_3d2c_params {3, 4, 2, 3, false, false, BATCH, 128, 512,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_3d2c(
            eng, net, net_args, block_3d2b.get_dst_mem(), block_3d2c_params);

    /* -----------------add > relu----------------- */
    IC = 512, OC = 512;
    memory::dims res3d_add_dst_dims = {BATCH, OC, 28, 28};
    auto res3d_add_dst_md
            = memory::desc(res3d_add_dst_dims, dt::f32, mem_format);
    auto res3d_add_dst_memory = memory(res3d_add_dst_md, eng);

    do_binary(eng, net, net_args, res3c_add_dst_memory,
            block_3d2c.get_dst_mem(), res3d_add_dst_memory);
    do_relu(eng, net, net_args, res3d_add_dst_memory);

    /* -----------------res4a_branch1----------------- */
    /* -----------------conv > bnorm > scale----------------- */
    IC = 512, OC = 1024;
    block_params block_4a1_params {4, 1, 1, 0, false, false, BATCH, IC, OC, 14,
            14, 1, 1, {0, 0}, {2, 2}};
    res_net_block block_4a1(
            eng, net, net_args, res3d_add_dst_memory, block_4a1_params);

    /* -----------------res4a_branch2----------------- */
    /* -----------------conv > bnorm > scale > relu----------------- */
    WIDTH = 14, HEIGHT = 14;
    IC = 512, OC = 256;
    block_params block_4a2a_params {4, 1, 2, 1, true, false, BATCH, IC, OC,
            HEIGHT, WIDTH, 1, 1, {0, 0}, {2, 2}};
    res_net_block block_4a2a(
            eng, net, net_args, res3d_add_dst_memory, block_4a2a_params);

    /* -----------------conv > bnorm > scale > relu----------------- */
    IC = 256, OC = 256;
    block_params block_4a2b_params {4, 1, 2, 2, true, false, BATCH, IC, OC,
            HEIGHT, WIDTH, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_4a2b(
            eng, net, net_args, block_4a2a.get_dst_mem(), block_4a2b_params);

    /* -----------------conv > bnorm > scale----------------- */
    IC = 256, OC = 1024;
    block_params block_4a2c_params {4, 1, 2, 3, false, false, BATCH, IC, OC,
            HEIGHT, WIDTH, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4a2c(
            eng, net, net_args, block_4a2b.get_dst_mem(), block_4a2c_params);

    /* -----------------add > relu----------------- */

    memory::dims res4a_add_dst_dims = {BATCH, OC, HEIGHT, WIDTH};
    auto res4a_add_dst_md
            = memory::desc(res4a_add_dst_dims, dt::f32, mem_format);
    auto res4a_add_dst_memory = memory(res4a_add_dst_md, eng);

    do_binary(eng, net, net_args, block_4a1.get_dst_mem(),
            block_4a2c.get_dst_mem(), res4a_add_dst_memory);
    do_relu(eng, net, net_args, res4a_add_dst_memory);

    /* -----------------res4b----------------- */
    /* -----------------branch2----------------- */
    IC = 1024, OC = 256;
    WIDTH = 14, HEIGHT = 14;
    block_params block_4b2a_params {4, 2, 2, 1, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4b2a(
            eng, net, net_args, res4a_add_dst_memory, block_4b2a_params);
    IC = 256, OC = 256;
    block_params block_4b2b_params {4, 2, 2, 2, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_4b2b(
            eng, net, net_args, block_4b2a.get_dst_mem(), block_4b2b_params);
    IC = 256, OC = 1024;
    block_params block_4b2c_params {4, 2, 2, 3, false, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4b2c(
            eng, net, net_args, block_4b2b.get_dst_mem(), block_4b2c_params);

    /* -----------------add > relu----------------- */
    IC = 1024, OC = 1024;
    memory::dims res4b_add_dst_dims = {BATCH, OC, HEIGHT, WIDTH};
    auto res4b_add_dst_md
            = memory::desc(res4b_add_dst_dims, dt::f32, mem_format);
    auto res4b_add_dst_memory = memory(res4b_add_dst_md, eng);

    do_binary(eng, net, net_args, res4a_add_dst_memory,
            block_4b2c.get_dst_mem(), res4b_add_dst_memory);
    do_relu(eng, net, net_args, res4b_add_dst_memory);

    /* -----------------res4c----------------- */
    /* -----------------branch2----------------- */
    IC = 1024, OC = 256;
    WIDTH = 14, HEIGHT = 14;
    block_params block_4c2a_params {4, 3, 2, 1, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4c2a(
            eng, net, net_args, res4b_add_dst_memory, block_4c2a_params);
    IC = 256, OC = 256;
    block_params block_4c2b_params {4, 3, 2, 2, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_4c2b(
            eng, net, net_args, block_4c2a.get_dst_mem(), block_4c2b_params);
    IC = 256, OC = 1024;
    block_params block_4c2c_params {4, 3, 2, 3, false, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4c2c(
            eng, net, net_args, block_4c2b.get_dst_mem(), block_4c2c_params);

    /* -----------------add > relu----------------- */
    IC = 1024, OC = 1024;
    memory::dims res4c_add_dst_dims = {BATCH, OC, HEIGHT, WIDTH};
    auto res4c_add_dst_md
            = memory::desc(res4c_add_dst_dims, dt::f32, mem_format);
    auto res4c_add_dst_memory = memory(res4c_add_dst_md, eng);

    do_binary(eng, net, net_args, res4b_add_dst_memory,
            block_4c2c.get_dst_mem(), res4c_add_dst_memory);
    do_relu(eng, net, net_args, res4c_add_dst_memory);

    /* -----------------res4d----------------- */
    /* -----------------branch2----------------- */
    IC = 1024, OC = 256;
    WIDTH = 14, HEIGHT = 14;
    block_params block_4d2a_params {4, 4, 2, 1, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4d2a(
            eng, net, net_args, res4c_add_dst_memory, block_4d2a_params);
    IC = 256, OC = 256;
    block_params block_4d2b_params {4, 4, 2, 2, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_4d2b(
            eng, net, net_args, block_4d2a.get_dst_mem(), block_4d2b_params);
    IC = 256, OC = 1024;
    block_params block_4d2c_params {4, 4, 2, 3, false, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4d2c(
            eng, net, net_args, block_4d2b.get_dst_mem(), block_4d2c_params);

    /* -----------------add > relu----------------- */
    IC = 1024, OC = 1024;
    memory::dims res4d_add_dst_dims = {BATCH, OC, HEIGHT, WIDTH};
    auto res4d_add_dst_md
            = memory::desc(res4d_add_dst_dims, dt::f32, mem_format);
    auto res4d_add_dst_memory = memory(res4d_add_dst_md, eng);

    do_binary(eng, net, net_args, res4c_add_dst_memory,
            block_4d2c.get_dst_mem(), res4d_add_dst_memory);
    do_relu(eng, net, net_args, res4d_add_dst_memory);

    /* -----------------res4e----------------- */
    /* -----------------branch2----------------- */
    IC = 1024, OC = 256;
    WIDTH = 14, HEIGHT = 14;
    block_params block_4e2a_params {4, 5, 2, 1, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4e2a(
            eng, net, net_args, res4d_add_dst_memory, block_4e2a_params);
    IC = 256, OC = 256;
    block_params block_4e2b_params {4, 5, 2, 2, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_4e2b(
            eng, net, net_args, block_4e2a.get_dst_mem(), block_4e2b_params);
    IC = 256, OC = 1024;
    block_params block_4e2c_params {4, 5, 2, 3, false, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4e2c(
            eng, net, net_args, block_4e2b.get_dst_mem(), block_4e2c_params);

    /* -----------------add > relu----------------- */
    IC = 1024, OC = 1024;
    memory::dims res4e_add_dst_dims = {BATCH, OC, HEIGHT, WIDTH};
    auto res4e_add_dst_md
            = memory::desc(res4e_add_dst_dims, dt::f32, mem_format);
    auto res4e_add_dst_memory = memory(res4e_add_dst_md, eng);

    do_binary(eng, net, net_args, res4d_add_dst_memory,
            block_4e2c.get_dst_mem(), res4e_add_dst_memory);
    do_relu(eng, net, net_args, res4e_add_dst_memory);

    /* -----------------res4f----------------- */
    /* -----------------branch2----------------- */
    IC = 1024, OC = 256;
    WIDTH = 14, HEIGHT = 14;
    block_params block_4f2a_params {4, 6, 2, 1, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4f2a(
            eng, net, net_args, res4e_add_dst_memory, block_4f2a_params);
    IC = 256, OC = 256;
    block_params block_4f2b_params {4, 6, 2, 2, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_4f2b(
            eng, net, net_args, block_4f2a.get_dst_mem(), block_4f2b_params);
    IC = 256, OC = 1024;
    block_params block_4f2c_params {4, 6, 2, 3, false, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_4f2c(
            eng, net, net_args, block_4f2b.get_dst_mem(), block_4f2c_params);

    /* -----------------add > relu----------------- */
    IC = 1024, OC = 1024;
    memory::dims res4f_add_dst_dims = {BATCH, OC, HEIGHT, WIDTH};
    auto res4f_add_dst_md
            = memory::desc(res4f_add_dst_dims, dt::f32, mem_format);
    auto res4f_add_dst_memory = memory(res4f_add_dst_md, eng);

    do_binary(eng, net, net_args, res4e_add_dst_memory,
            block_4f2c.get_dst_mem(), res4f_add_dst_memory);
    do_relu(eng, net, net_args, res4f_add_dst_memory);

    /* -----------------res5a_branch1----------------- */
    /* -----------------conv > bnorm > scale----------------- */
    IC = 1024, OC = 2048;
    HEIGHT = 7, WIDTH = 7;
    block_params block_5a1_params {5, 1, 1, 0, false, false, BATCH, IC, OC,
            HEIGHT, WIDTH, 1, 1, {0, 0}, {2, 2}};
    res_net_block block_5a1(
            eng, net, net_args, res4f_add_dst_memory, block_5a1_params);

    /* -----------------res5a_branch2----------------- */
    /* -----------------conv > bnorm > scale > relu----------------- */
    WIDTH = 7, HEIGHT = 7;
    IC = 1024, OC = 512;
    block_params block_5a2a_params {5, 1, 2, 1, true, false, BATCH, IC, OC,
            HEIGHT, WIDTH, 1, 1, {0, 0}, {2, 2}};
    res_net_block block_5a2a(
            eng, net, net_args, res4f_add_dst_memory, block_5a2a_params);

    /* -----------------conv > bnorm > scale > relu----------------- */
    IC = 512, OC = 512;
    block_params block_5a2b_params {5, 1, 2, 2, true, false, BATCH, IC, OC,
            HEIGHT, WIDTH, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_5a2b(
            eng, net, net_args, block_5a2a.get_dst_mem(), block_5a2b_params);

    /* -----------------conv > bnorm > scale----------------- */
    IC = 512, OC = 2048;
    block_params block_5a2c_params {5, 1, 2, 3, false, false, BATCH, IC, OC,
            HEIGHT, WIDTH, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_5a2c(
            eng, net, net_args, block_5a2b.get_dst_mem(), block_5a2c_params);

    /* -----------------add > relu----------------- */

    memory::dims res5a_add_dst_dims = {BATCH, OC, HEIGHT, WIDTH};
    auto res5a_add_dst_md
            = memory::desc(res5a_add_dst_dims, dt::f32, mem_format);
    auto res5a_add_dst_memory = memory(res5a_add_dst_md, eng);

    do_binary(eng, net, net_args, block_5a1.get_dst_mem(),
            block_5a2c.get_dst_mem(), res5a_add_dst_memory);
    do_relu(eng, net, net_args, res5a_add_dst_memory);

    /* -----------------res5b----------------- */
    /* -----------------branch2----------------- */
    IC = 2048, OC = 512;
    WIDTH = 7, HEIGHT = 7;
    block_params block_5b2a_params {5, 2, 2, 1, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_5b2a(
            eng, net, net_args, res5a_add_dst_memory, block_5b2a_params);
    IC = 512, OC = 512;
    block_params block_5b2b_params {5, 2, 2, 2, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_5b2b(
            eng, net, net_args, block_5b2a.get_dst_mem(), block_5b2b_params);
    IC = 512, OC = 2048;
    block_params block_5b2c_params {5, 2, 2, 3, false, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_5b2c(
            eng, net, net_args, block_5b2b.get_dst_mem(), block_5b2c_params);

    /* -----------------add > relu----------------- */
    IC = 2048, OC = 2048;
    memory::dims res5b_add_dst_dims = {BATCH, OC, HEIGHT, WIDTH};
    auto res5b_add_dst_md
            = memory::desc(res5b_add_dst_dims, dt::f32, mem_format);
    auto res5b_add_dst_memory = memory(res5b_add_dst_md, eng);

    do_binary(eng, net, net_args, res5a_add_dst_memory,
            block_5b2c.get_dst_mem(), res5b_add_dst_memory);
    do_relu(eng, net, net_args, res5b_add_dst_memory);

    /* -----------------res5c----------------- */
    /* -----------------branch2----------------- */
    IC = 2048, OC = 512;
    WIDTH = 7, HEIGHT = 7;
    block_params block_5c2a_params {5, 3, 2, 1, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_5c2a(
            eng, net, net_args, res5b_add_dst_memory, block_5c2a_params);
    IC = 512, OC = 512;
    block_params block_5c2b_params {5, 3, 2, 2, true, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 3, 3, {1, 1}, {1, 1}};
    res_net_block block_5c2b(
            eng, net, net_args, block_5c2a.get_dst_mem(), block_5c2b_params);
    IC = 512, OC = 2048;
    block_params block_5c2c_params {5, 3, 2, 3, false, false, BATCH, IC, OC,
            WIDTH, HEIGHT, 1, 1, {0, 0}, {1, 1}};
    res_net_block block_5c2c(
            eng, net, net_args, block_5c2b.get_dst_mem(), block_5c2c_params);

    /* -----------------add > relu----------------- */
    IC = 2048, OC = 2048;
    memory::dims res5c_add_dst_dims = {BATCH, OC, HEIGHT, WIDTH};
    auto res5c_add_dst_md
            = memory::desc(res5c_add_dst_dims, dt::f32, mem_format);
    auto res5c_add_dst_memory = memory(res5c_add_dst_md, eng);

    do_binary(eng, net, net_args, res5b_add_dst_memory,
            block_5c2c.get_dst_mem(), res5c_add_dst_memory);
    do_relu(eng, net, net_args, res5c_add_dst_memory);

    /* -----------------pool5----------------- */
    IC = 2048, OC = 2048;
    HEIGHT = 7, WIDTH = 7;
    memory::dims pool5_kernel_dims = {7, 7};
    memory::dims pool5_strides_dims = {1, 1};
    memory::dims pool5_padding_dims_l = {0, 0};
    memory::dims pool5_padding_dims_r = {0, 0};
    memory::dims pool5_dst_dims = {BATCH, OC, 1, 1};
    auto pool5_dst_md = memory::desc(pool5_dst_dims, dt::f32, mem_format);
    auto pool5_dst_memory = memory(pool5_dst_md, eng);
    do_pool(eng, net, net_args, pool5_strides_dims, pool5_kernel_dims,
            pool5_padding_dims_l, pool5_padding_dims_r, res5c_add_dst_memory,
            pool5_dst_memory, algorithm::pooling_avg_exclude_padding);

    /* -----------------inner product----------------- */
    IC = 2048, OC = 1000;
    memory::dims ip_src_dims = pool5_dst_md.dims();
    memory::dims ip_weights_dims = {OC, IC, 1, 1};
    memory::dims ip_bias_dims = {OC};
    memory::dims ip_dst_dims = {BATCH, OC};

    std::vector<char> ip_weights_data(product(ip_weights_dims) * sizeof(float));
    std::vector<char> ip_bias_data(product(ip_bias_dims) * sizeof(float));

    auto ip_bias_md = memory::desc(ip_bias_dims, dt::f32, tag::a);
    auto ip_bias_mem = memory(ip_bias_md, eng);
    auto ip_weights_mem = memory({ip_weights_dims, dt::f32, tag::hwio}, eng);
    load_weights_fc(1000, ip_weights_data.data(), product(ip_weights_dims),
            ip_bias_data.data(), product(ip_bias_dims));
    write_to_dnnl_memory(ip_bias_data.data(), ip_bias_mem);
    write_to_dnnl_memory(ip_weights_data.data(), ip_weights_mem);

    auto ip_dst_md = memory::desc(ip_dst_dims, dt::f32, tag::nc);
    auto ip_dst_mem = memory(ip_dst_md, eng);
    do_ip(eng, net, net_args, ip_weights_mem, ip_bias_mem, pool5_dst_memory,
            ip_dst_mem, true);
    /* --------- softmax --------- */
    const int axis = 1;
    auto softmax_d = softmax_forward::desc(
            prop_kind::forward_inference, ip_dst_mem.get_desc(), axis);
    auto softmax_pd = softmax_forward::primitive_desc(softmax_d, eng);
    auto softmax_prim = softmax_forward(softmax_pd);

    net.push_back(softmax_prim);
    net_args.push_back(
            {{DNNL_ARG_SRC, ip_dst_mem}, {DNNL_ARG_DST, ip_dst_mem}});
    /* ------------------------------------------ */
    std::cout << "Warming up... BATCH SIZE = " << BATCH << "\n";

    assert(net.size() == net_args.size() && "something is missing");
    for (size_t j = 0; j < 5; j++) {

        for (size_t i = 0; i < net.size(); ++i) {

            net.at(i).execute(s, net_args.at(i));
            // auto dst_mem = net_args.at(i).at(DNNL_ARG_DST);
            // std::vector<float> dst_data(product(dst_mem.get_desc().dims()));
            // read_from_dnnl_memory(dst_data.data(), dst_mem);
            // check_batch(dst_data, BATCH);
        }
    }
    s.wait();
    std::cout << "Benchmarking now...\n";
    // Execute model
    //std::cout << "\n###########################################################"
    //             "#############################################################"
    //             "#############################################################"
    //             "#########\n";

    auto begin = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < times; ++j) {
        // assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i) {

            net.at(i).execute(s, net_args.at(i));
        }
        // s.get_sycl_queue().wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Use time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(
                         end - begin)
                            .count()
                    / (1e6 * static_cast<double>(times))
              << " ms per iteration." << std::endl;
    s.wait();
    // Execute model
    std::vector<float> result(product(ip_dst_mem.get_desc().dims()));
    read_from_dnnl_memory(result.data(), ip_dst_mem);
    for (size_t i = 0; i < BATCH; i++) {
        auto index = std::max_element(
                result.begin() + i * 1000, result.begin() + (i + 1) * 1000);
        //std::cout << "classed as "
        //          << std::distance(result.begin() + i * 1000, index)
        //          << ", value " << (index != std::end(result) ? *index : 0.f)
        //          << std::endl;
    }
}

void do_it(engine::kind engine_kind) {

    // FIXME: revert to 100
    int times = 1000;
    res_net(engine_kind, times);
}

int main(int argc, char **argv) {
    for (int i = 1; i <= 64; i *= 2) {
        BATCH = i;
        /* return handle_example_errors( */ do_it(
                parse_engine_kind(argc, argv));
    }
}
