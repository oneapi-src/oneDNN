/*******************************************************************************
* Copyright Codeplay Software Ltd.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#include <array>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>

#include <oneapi/dnnl/dnnl.hpp>
#include <sycl/sycl.hpp>

#include "example_utils.hpp"

#define CHECK_PERF 0

using DType = float;
static std::array<std::string, 10> object_map = {"bathtub", "bed", "chair",
        "desk", "dresser", "monitor", "night stand", "sofa", "table", "toilet"};

namespace helpers {

template <typename T>
void copy_to_device(
        const std::vector<char> &inputs, T *dev_ptr, sycl::queue q) {
    q.submit([&](sycl::handler &cgh) {
         cgh.copy(inputs.data(), reinterpret_cast<char *>(dev_ptr),
                 inputs.size());
     }).wait_and_throw();
}

// Helper function that reads binary data into a vector
std::vector<char> read_binary_data(std::string const &name) {
    std::ifstream file(name, std::ios_base::binary | std::ios_base::in);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + name);
    }
    std::vector<char> output {std::istreambuf_iterator<char> {file}, {}};
    return output;
}
} // namespace helpers

template <typename T>
struct Layer {
    explicit Layer(dnnl::engine &engine, dnnl::stream &stream)
        : engine_(engine), stream_(stream), out_mem_(), out_desc_() {}

    Layer(dnnl::engine &engine, dnnl::stream &stream,
            const dnnl::memory &out_mem, const dnnl::memory::desc &out_desc)
        : engine_(engine)
        , stream_(stream)
        , out_mem_(out_mem)
        , out_desc_(out_desc) {}

    virtual ~Layer() {}

    virtual void execute(dnnl::memory &in_mem) = 0;

    dnnl::memory &get_output_mem() { return out_mem_; }

    dnnl::engine &engine_;
    dnnl::stream &stream_;

protected:
    dnnl::memory out_mem_;
    dnnl::memory::desc out_desc_;
};

template <typename T>
struct ConvBiasLayer : public Layer<T> {
    ConvBiasLayer(dnnl::engine &engine, dnnl::stream &stream,
            std::string const &filter_file, std::string const &bias_file,
            const int in_n, const int in_c, const int in_h, const int in_w,
            const int filt_f, const int filt_c, const int filt_h,
            const int filt_w,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream) {

        dnnl::memory::dim oh, ow;

        //complete formula from documentation -> oh= (ih - kh + ph_l + ph_r)/sh + 1;
        oh = in_h - filt_h + 1;
        //complete formula from documentation -> ow= (iw - kw + pw_l + pw_r)/sw + 1;
        ow = in_w - filt_w + 1;

        // Read weights from binary file
        auto weights = helpers::read_binary_data(filter_file);
        auto bias_value = helpers::read_binary_data(bias_file);

        dnnl::memory::dims src_dims = {in_n, in_c, in_h, in_w};
        dnnl::memory::dims weights_dims = {filt_f, in_c, filt_h, filt_w};
        dnnl::memory::dims dst_dims = {in_n, filt_f, oh, ow};
        dnnl::memory::dims bias_dims = {filt_f};
        dnnl::memory::dims strides_dims = {1, 1};
        dnnl::memory::dims padding_dims_l = {0, 0};
        dnnl::memory::dims padding_dims_r = {0, 0};
        dnnl::memory::dims dilates = {0, 0};

        const auto sycl_queue = dnnl::sycl_interop::get_queue(this->stream_);

        // Create memory descriptors
        conv_src_md = dnnl::memory::desc(
                src_dims, data_type, dnnl::memory::format_tag::nhwc);
        conv_weights_md = dnnl::memory::desc(
                weights_dims, data_type, dnnl::memory::format_tag::iohw);
        this->out_desc_ = dnnl::memory::desc(
                dst_dims, data_type, dnnl::memory::format_tag::nhwc);

        conv_bias_md = dnnl::memory::desc(
                bias_dims, data_type, dnnl::memory::format_tag::a);

        // Create memory
        conv_weights_mem = dnnl::memory(
                {weights_dims, data_type, dnnl::memory::format_tag::iohw},
                this->engine_);
        this->out_mem_ = dnnl::memory(
                {dst_dims, data_type, dnnl::memory::format_tag::nchw},
                this->engine_);

        conv_bias_mem = dnnl::memory(conv_bias_md, this->engine_);

        write_to_dnnl_memory(weights.data(), conv_weights_mem);
        write_to_dnnl_memory(bias_value.data(), conv_bias_mem);

        // Create primitive descriptor for Convolution
        conv_pd_ = dnnl::convolution_forward::primitive_desc(this->engine_,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_auto, conv_src_md, conv_weights_md,
                conv_bias_md, this->out_desc_, strides_dims, dilates,
                padding_dims_l, padding_dims_r);
    }

    void execute(dnnl::memory &in_mem) override {

        // Create the primitive.
        auto conv_prim = dnnl::convolution_forward(conv_pd_);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> conv_args;
        conv_args.insert({DNNL_ARG_SRC, in_mem});
        conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
        conv_args.insert({DNNL_ARG_BIAS, conv_bias_mem});
        conv_args.insert({DNNL_ARG_DST, this->out_mem_});

        conv_prim.execute(this->stream_, conv_args);
    }

    ~ConvBiasLayer() {}

private:
    size_t ws_size_;
    dnnl::memory conv_weights_mem;
    dnnl::memory conv_bias_mem;
    dnnl::memory::desc conv_src_md;
    dnnl::memory::desc conv_weights_md;
    dnnl::memory::desc conv_bias_md;

    dnnl::convolution_forward::primitive_desc conv_pd_;
};

template <typename T>
struct BatchNormLayer : public Layer<T> {
    BatchNormLayer(dnnl::engine &engine, dnnl::stream &stream,
            std::string const &scale_file, std::string const &bias_file,
            std::string const &mean_file, std::string const &var_file,
            const int batch, const int channels, const int rows, const int cols,
            const bool add_relu = true,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream), _relu(add_relu) {

        // Configuring dimensions
        dnnl::memory::dims src_dims = {batch, channels, rows, cols};
        dnnl::memory::dims scaleshift_dims = {channels};
        dnnl::memory::dims mean_dims = {channels};
        dnnl::memory::dims var_dims = {channels};

        // Reading input data from file
        auto bias_value = helpers::read_binary_data(bias_file);
        auto scale_value = helpers::read_binary_data(scale_file);
        auto mean_value = helpers::read_binary_data(mean_file);
        auto var_value = helpers::read_binary_data(var_file);

        // Create memory descriptors
        src_md = dnnl::memory::desc(src_dims, data_type, format);
        this->out_desc_ = dnnl::memory::desc(src_dims, data_type, format);
        scaleshift_md = dnnl::memory::desc(
                scaleshift_dims, data_type, dnnl::memory::format_tag::a);
        mean_md = dnnl::memory::desc(
                mean_dims, data_type, dnnl::memory::format_tag::x);
        variance_md = dnnl::memory::desc(
                var_dims, data_type, dnnl::memory::format_tag::x);

        // Create memory
        this->out_mem_ = dnnl::memory(this->out_desc_, this->engine_);
        scale_mem = dnnl::memory(scaleshift_md, this->engine_);
        shift_mem = dnnl::memory(scaleshift_md, this->engine_);
        mean_mem = dnnl::memory(mean_md, this->engine_);
        variance_mem = dnnl::memory(variance_md, this->engine_);

        write_to_dnnl_memory(mean_value.data(), mean_mem);
        write_to_dnnl_memory(var_value.data(), variance_mem);
        write_to_dnnl_memory(scale_value.data(), scale_mem);
        write_to_dnnl_memory(bias_value.data(), shift_mem);

        // Set flags for bnorm
        dnnl::normalization_flags flags = (dnnl::normalization_flags::use_scale
                | dnnl::normalization_flags::use_shift
                | dnnl::normalization_flags::use_global_stats);

        if (_relu) flags |= dnnl::normalization_flags::fuse_norm_relu;

        bnorm_pd = dnnl::batch_normalization_forward::primitive_desc(
                this->engine_, dnnl::prop_kind::forward_inference, src_md,
                this->out_desc_, eps_, flags);
    }

    void execute(dnnl::memory &in_mem) override {

        auto bnorm_prim = dnnl::batch_normalization_forward(bnorm_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> bnorm_args;
        bnorm_args.insert({DNNL_ARG_SRC, in_mem});
        bnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
        bnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
        bnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
        bnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
        bnorm_args.insert({DNNL_ARG_DST, this->out_mem_});

        bnorm_prim.execute(this->stream_, bnorm_args);
    }

    ~BatchNormLayer() {}

private:
    dnnl::memory scale_mem;
    dnnl::memory shift_mem;
    dnnl::memory mean_mem;
    dnnl::memory variance_mem;
    dnnl::memory::desc src_md;
    dnnl::memory::desc scaleshift_md;
    dnnl::memory::desc mean_md;
    dnnl::memory::desc variance_md;
    bool _relu {true};

    dnnl::batch_normalization_forward::primitive_desc bnorm_pd;

    float eps_ = 1.0e-5;
};

template <typename T>
struct GlobalMaxPoolLayer : public Layer<T> {
    GlobalMaxPoolLayer(dnnl::engine &engine, dnnl::stream &stream,
            const int batch, const int channels, const int rows, const int cols,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream) {

        dnnl::memory::dims src_dims = {batch, channels, rows, cols};

        dnnl::memory::dims dst_dims = {batch, channels, 1, 1};
        dnnl::memory::dims kernel_dims = {rows, cols};
        dnnl::memory::dims strides_dims = {rows, cols};
        dnnl::memory::dims padding_dims_l = {0, 0};
        dnnl::memory::dims padding_dims_r = {0, 0};
        dnnl::memory::dims dilation_dims = {1, 1};

        auto src_md = dnnl::memory::desc(src_dims, data_type, format);

        this->out_desc_ = dnnl::memory::desc(dst_dims, data_type, format);
        this->out_mem_ = dnnl::memory(this->out_desc_, this->engine_);

        pooling_pd = dnnl::pooling_forward::primitive_desc(this->engine_,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::pooling_max, src_md, this->out_desc_,
                strides_dims, kernel_dims, dilation_dims, padding_dims_l,
                padding_dims_r);
    }

    void execute(dnnl::memory &in_mem) override {
        auto pooling_prim = dnnl::pooling_forward(pooling_pd);

        // Primitive arguments. Set up in-place execution by assigning src as DST.
        std::unordered_map<int, dnnl::memory> pooling_args;
        pooling_args.insert({DNNL_ARG_SRC, in_mem});
        pooling_args.insert({DNNL_ARG_DST, this->out_mem_});

        // Primitive execution: pooling.
        pooling_prim.execute(this->stream_, pooling_args);
    }
    ~GlobalMaxPoolLayer() = default;

private:
    dnnl::pooling_forward::primitive_desc pooling_pd;
};

template <typename T>
struct FCLayer : public Layer<T> {
    FCLayer(dnnl::engine &engine, dnnl::stream &stream,
            const std::string &weights_file, const std::string &bias_file,
            const int batch, const int in_channels, const int out_channels,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream) {

        dnnl::memory::dims src_dims, dst_dims, weights_dims, bias_dims;

        src_dims = {1, batch, in_channels};
        weights_dims = {1, in_channels, out_channels};
        bias_dims = {1, 1, out_channels};
        dst_dims = {1, batch, out_channels};

        src_md = dnnl::memory::desc(
                src_dims, data_type, dnnl::memory::format_tag::abc);
        this->out_desc_ = dnnl::memory::desc(
                dst_dims, data_type, dnnl::memory::format_tag::abc);
        weights_md = dnnl::memory::desc(
                weights_dims, data_type, dnnl::memory::format_tag::abc);
        bias_md = dnnl::memory::desc(
                bias_dims, data_type, dnnl::memory::format_tag::abc);

        auto weights = helpers::read_binary_data(weights_file);
        auto bias = helpers::read_binary_data(bias_file);

        weights_mem = dnnl::memory(weights_md, this->engine_);
        bias_mem = dnnl::memory(bias_md, this->engine_);
        this->out_mem_ = dnnl::memory(this->out_desc_, this->engine_);

        write_to_dnnl_memory(weights.data(), weights_mem);
        write_to_dnnl_memory(bias.data(), bias_mem);
    }

    void execute(dnnl::memory &in_mem) override {

        matmul_pd = dnnl::matmul::primitive_desc(
                this->engine_, src_md, weights_md, bias_md, this->out_desc_);

        // Create the primitive.
        auto matmul_prim = dnnl::matmul(matmul_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, in_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
        matmul_args.insert({DNNL_ARG_DST, this->out_mem_});

        // Primitive execution.
        matmul_prim.execute(this->stream_, matmul_args);
    }

    ~FCLayer() {}

private:
    int m_, k_, n_;
    dnnl::memory bias_mem;
    dnnl::memory weights_mem;
    dnnl::memory::desc src_md;
    dnnl::memory::desc weights_md;
    dnnl::memory::desc bias_md;

    dnnl::matmul::primitive_desc matmul_pd;
};

template <typename T>
struct MMLayer : public Layer<T> {
    MMLayer(dnnl::engine &engine, dnnl::stream &stream, dnnl::memory &lhs_ptr,
            const int batch, const int m, const int k, const int n,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream), src_mem(lhs_ptr) {

        dnnl::memory::dims src_dims = {batch, m, k};
        dnnl::memory::dims weights_dims = {batch, k, n};
        dnnl::memory::dims dst_dims = {batch, m, n};

        src_desc = dnnl::memory::desc(
                src_dims, data_type, dnnl::memory::format_tag::abc);
        weights_desc = dnnl::memory::desc(
                weights_dims, data_type, dnnl::memory::format_tag::abc);
        this->out_desc_ = dnnl::memory::desc(
                dst_dims, data_type, dnnl::memory::format_tag::abc);

        this->out_mem_ = dnnl::memory(this->out_desc_, this->engine_);

        matmul_pd = dnnl::matmul::primitive_desc(
                this->engine_, src_desc, weights_desc, this->out_desc_);
    }

    void execute(dnnl::memory &in_mem) override {
        auto matmul = dnnl::matmul(matmul_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, src_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, in_mem});
        matmul_args.insert({DNNL_ARG_DST, this->out_mem_});

        // Primitive execution
        matmul.execute(this->stream_, matmul_args);
    }

    ~MMLayer() = default;

private:
    dnnl::memory::desc src_desc;
    dnnl::memory::desc weights_desc;
    dnnl::memory src_mem;
    dnnl::matmul::primitive_desc matmul_pd;
};

template <typename T>
struct SumLayer : public Layer<T> {
    SumLayer(dnnl::engine &engine, dnnl::stream &stream,
            std::string const &bias_file, const int batch, const int channels,
            const int rows, const int cols,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream) {

        dnnl::memory::dims src_dims = {batch, channels, rows, cols};
        dnnl::memory::dims scale_dims = {batch, channels, rows, cols};

        src_desc_ = dnnl::memory::desc(src_dims, data_type, format);
        bias_desc_ = dnnl::memory::desc(scale_dims, data_type, format);
        this->out_desc_ = dnnl::memory::desc(src_dims, data_type, format);

        auto scale_chars = helpers::read_binary_data(bias_file);

        bias_mem = dnnl::memory(bias_desc_, this->engine_);
        this->out_mem_ = dnnl::memory(this->out_desc_, this->engine_);

        write_to_dnnl_memory(scale_chars.data(), bias_mem);

        sum_pd = dnnl::binary::primitive_desc(this->engine_,
                dnnl::algorithm::binary_add, src_desc_, bias_desc_,
                this->out_desc_);
    }

    void execute(dnnl::memory &in_mem) override {

        auto sum_prim = dnnl::binary(sum_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> binary_args;
        binary_args.insert({DNNL_ARG_SRC_0, in_mem});
        binary_args.insert({DNNL_ARG_SRC_1, bias_mem});
        binary_args.insert({DNNL_ARG_DST, this->out_mem_});

        // Primitive execution
        sum_prim.execute(this->stream_, binary_args);
    }
    ~SumLayer() = default;

private:
    dnnl::memory::desc src_desc_;
    dnnl::memory::desc bias_desc_;
    dnnl::memory bias_mem;
    dnnl::binary::primitive_desc sum_pd;
};
template <typename T>
struct LogSoftMaxLayer : public Layer<T> {
    LogSoftMaxLayer(dnnl::engine &engine, dnnl::stream &stream, const int batch,
            const int channels, const int rows, const int cols,
            dnnl::algorithm algo = dnnl::algorithm::softmax_log,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream) {

        dnnl::memory::dims src_dst_dims = {batch, channels, rows, cols};
        src_md = dnnl::memory::desc(src_dst_dims, data_type, format);
        this->out_desc_ = dnnl::memory::desc(src_dst_dims, data_type, format);

        this->out_mem_ = dnnl::memory(this->out_desc_, this->engine_);
        constexpr int axis = 1;

        softmax_pd = dnnl::softmax_forward::primitive_desc(this->engine_,
                dnnl::prop_kind::forward_training, algo, src_md,
                this->out_desc_, axis);
    }

    void execute(dnnl::memory &in_mem) override {

        // Create the primitive.
        auto softmax_prim = dnnl::softmax_forward(softmax_pd);

        // Primitive arguments. Set up in-place execution by assigning src as DST.
        std::unordered_map<int, dnnl::memory> softmax_args;
        softmax_args.insert({DNNL_ARG_SRC, in_mem});
        softmax_args.insert({DNNL_ARG_DST, this->out_mem_});

        // Primitive execution.
        softmax_prim.execute(this->stream_, softmax_args);
    }
    ~LogSoftMaxLayer() = default;

private:
    dnnl::memory::desc src_md;
    dnnl::memory src_mem;
    dnnl::softmax_forward::primitive_desc softmax_pd;
};

template <typename T>
struct Network {
    void add_layer(std::unique_ptr<Layer<T>> layer) {
        layers.emplace_back(std::move(layer));
    }

    void execute(dnnl::memory &in_mem) {
        for (auto &layer : layers) {
            layer->execute(in_mem);
            in_mem = layer->get_output_mem();
        }
    }

    dnnl::memory &get_output_mem() { return layers.back()->get_output_mem(); }

    std::vector<T> get_output_as_host_vec() {
        auto &last_layer = layers.back();
        auto &output_mem = last_layer->get_output_mem();
        auto tmp = output_mem.get_desc().get_dims();
        int output_dim {1};
        for (const auto &e : tmp) {
            output_dim *= e;
        }
        std::vector<T> output(output_dim);

        read_from_dnnl_memory(output.data(), output_mem);

        return output;
    }

    std::vector<std::unique_ptr<Layer<T>>> layers;
};

template <typename T>
inline void add_conv_bias_layer(Network<T> &net, dnnl::engine &handle,
        dnnl::stream &stream, std::string const &filter_file,
        std::string const &bias_file, const int in_n, const int in_c,
        const int in_h, const int in_w, const int filt_f, const int filt_c,
        const int filt_h, const int filt_w) {
    net.add_layer(std::make_unique<ConvBiasLayer<T>>(handle, stream,
            filter_file, bias_file, in_n, in_c, in_h, in_w, filt_f, filt_c,
            filt_h, filt_w));
}

template <typename T>
inline void add_batchnorm_layer(Network<T> &net, dnnl::engine &handle,
        dnnl::stream &stream, std::string const &scale_file,
        std::string const &bias_file, std::string const &mean_file,
        std::string const &var_file, const int n, const int c, const int h,
        const int w, const bool add_relu = true) {
    net.add_layer(std::make_unique<BatchNormLayer<T>>(handle, stream,
            scale_file, bias_file, mean_file, var_file, n, c, h, w, add_relu));
}

template <typename T>
inline void add_global_max_pool_layer(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, const int n, const int c, const int h,
        const int w) {
    net.add_layer(std::make_unique<GlobalMaxPoolLayer<T>>(
            engine, stream, n, c, h, w));
}

template <typename T>
inline void add_fc_layer(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, const std::string &weights_file,
        const std::string &bias_file, const int batch, const int in_c,
        const int out_c) {
    net.add_layer(std::make_unique<FCLayer<T>>(
            engine, stream, weights_file, bias_file, batch, in_c, out_c));
}

template <typename T>
inline void add_mm_layer(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, dnnl::memory lhs_ptr, const int batch,
        const int m, const int k, const int n) {
    net.add_layer(std::make_unique<MMLayer<T>>(
            engine, stream, lhs_ptr, batch, m, k, n));
}

template <typename T>
inline void add_logsoftmax_layer(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, const int n, const int c, const int h,
        const int w) {
    net.add_layer(
            std::make_unique<LogSoftMaxLayer<T>>(engine, stream, n, c, h, w));
}

template <typename T>
inline void add_sum_layer(Network<T> &net, dnnl::engine &handle,
        dnnl::stream &stream, std::string const &bias_file, const int n,
        const int c, const int h, const int w) {
    net.add_layer(std::make_unique<SumLayer<T>>(
            handle, stream, bias_file, n, c, h, w));
}

template <typename T>
inline void add_conv_bias_bnorm_relu_block(Network<T> &net,
        dnnl::engine &engine, dnnl::stream &stream,
        std::string const &file_directory, std::string const &conv_filter_file,
        std::string const &conv_bias_file, std::string const &bn_scale_file,
        std::string const &bn_bias_file, std::string const &bn_mean_file,
        std::string const &bn_var_file, const int in_n, const int in_c,
        const int in_h, const int in_w, const int out_c, const int filt_h,
        const int filt_w, bool add_relu = true) {
    add_conv_bias_layer(net, engine, stream, file_directory + conv_filter_file,
            file_directory + conv_bias_file, in_n, in_c, in_h, in_w, out_c,
            in_c, filt_h, filt_w);
    add_batchnorm_layer(net, engine, stream, file_directory + bn_scale_file,
            file_directory + bn_bias_file, file_directory + bn_mean_file,
            file_directory + bn_var_file, in_n, out_c, in_h, in_w, add_relu);
}

template <typename T>
inline void add_fc_bias_bnorm_relu_block(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, std::string const &file_directory,
        std::string const &fc_filter_file, std::string const &fc_bias_file,
        std::string const &bn_scale_file, std::string const &bn_bias_file,
        std::string const &bn_mean_file, std::string const &bn_var_file,
        const int batch, const int in_c, const int out_c) {
    add_fc_layer(net, engine, stream, file_directory + fc_filter_file,
            file_directory + fc_bias_file, batch, in_c, out_c);
    add_batchnorm_layer(net, engine, stream, file_directory + bn_scale_file,
            file_directory + bn_bias_file, file_directory + bn_mean_file,
            file_directory + bn_var_file, batch, out_c, 1, 1);
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        std::cout << "USAGE: " << argv[0] << " <directory> <image>"
                  << std::endl;
        return 1;
    }

    dnnl::engine eng(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(eng);
    auto sycl_queue = dnnl::sycl_interop::get_queue(stream);

    std::string data_dir {argv[1]};
    data_dir += "/";

    auto input = helpers::read_binary_data(argv[2]);

    auto in_mem = dnnl::memory({{32, 3, 1024, 1}, dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::nchw},
            eng);
    write_to_dnnl_memory(input.data(), in_mem);

    Network<DType> input_transform_block;
    Network<DType> base_transform_block;
    Network<DType> feature_transform_block;

    // Construct input transformation block of network
    add_conv_bias_bnorm_relu_block(input_transform_block, eng, stream, data_dir,
            "transform.input_transform.conv1.weight.bin",
            "transform.input_transform.conv1.bias.bin",
            "transform.input_transform.bn1.weight.bin",
            "transform.input_transform.bn1.bias.bin",
            "transform.input_transform.bn1.running_mean.bin",
            "transform.input_transform.bn1.running_var.bin", 32, 3, 1024, 1, 64,
            1, 1);

    add_conv_bias_bnorm_relu_block(input_transform_block, eng, stream, data_dir,
            "transform.input_transform.conv2.weight.bin",
            "transform.input_transform.conv2.bias.bin",
            "transform.input_transform.bn2.weight.bin",
            "transform.input_transform.bn2.bias.bin",
            "transform.input_transform.bn2.running_mean.bin",
            "transform.input_transform.bn2.running_var.bin", 32, 64, 1024, 1,
            128, 1, 1);

    add_conv_bias_bnorm_relu_block(input_transform_block, eng, stream, data_dir,
            "transform.input_transform.conv3.weight.bin",
            "transform.input_transform.conv3.bias.bin",
            "transform.input_transform.bn3.weight.bin",
            "transform.input_transform.bn3.bias.bin",
            "transform.input_transform.bn3.running_mean.bin",
            "transform.input_transform.bn3.running_var.bin", 32, 128, 1024, 1,
            1024, 1, 1);

    add_global_max_pool_layer(
            input_transform_block, eng, stream, 32, 1024, 1024, 1);

    add_fc_bias_bnorm_relu_block(input_transform_block, eng, stream, data_dir,
            "transform.input_transform.fc1.weight.bin",
            "transform.input_transform.fc1.bias.bin",
            "transform.input_transform.bn4.weight.bin",
            "transform.input_transform.bn4.bias.bin",
            "transform.input_transform.bn4.running_mean.bin",
            "transform.input_transform.bn4.running_var.bin", 32, 1024, 512);

    add_fc_bias_bnorm_relu_block(input_transform_block, eng, stream, data_dir,
            "transform.input_transform.fc2.weight.bin",
            "transform.input_transform.fc2.bias.bin",
            "transform.input_transform.bn5.weight.bin",
            "transform.input_transform.bn5.bias.bin",
            "transform.input_transform.bn5.running_mean.bin",
            "transform.input_transform.bn5.running_var.bin", 32, 512, 256);

    add_fc_layer(input_transform_block, eng, stream,
            data_dir + "transform.input_transform.fc3.weight.bin",
            data_dir + "transform.input_transform.fc3.bias.bin", 32, 256, 9);

    add_sum_layer(input_transform_block, eng, stream,
            data_dir + "transform.input_transform.id.bin", 1, 32 * 9, 1, 1);

    // Transform input
    add_mm_layer(input_transform_block, eng, stream, in_mem, 32, 1024, 3, 3);

    // Construct base transformation block
    add_conv_bias_bnorm_relu_block(base_transform_block, eng, stream, data_dir,
            "transform.conv1.weight.bin", "transform.conv1.bias.bin",
            "transform.bn1.weight.bin", "transform.bn1.bias.bin",
            "transform.bn1.running_mean.bin", "transform.bn1.running_var.bin",
            32, 3, 1024, 1, 64, 1, 1);

    // Construct feature transformation block
    add_conv_bias_bnorm_relu_block(feature_transform_block, eng, stream,
            data_dir, "transform.feature_transform.conv1.weight.bin",
            "transform.feature_transform.conv1.bias.bin",
            "transform.feature_transform.bn1.weight.bin",
            "transform.feature_transform.bn1.bias.bin",
            "transform.feature_transform.bn1.running_mean.bin",
            "transform.feature_transform.bn1.running_var.bin", 32, 64, 1024, 1,
            64, 1, 1);
    add_conv_bias_bnorm_relu_block(feature_transform_block, eng, stream,
            data_dir, "transform.feature_transform.conv2.weight.bin",
            "transform.feature_transform.conv2.bias.bin",
            "transform.feature_transform.bn2.weight.bin",
            "transform.feature_transform.bn2.bias.bin",
            "transform.feature_transform.bn2.running_mean.bin",
            "transform.feature_transform.bn2.running_var.bin", 32, 64, 1024, 1,
            128, 1, 1);

    add_conv_bias_bnorm_relu_block(feature_transform_block, eng, stream,
            data_dir, "transform.feature_transform.conv3.weight.bin",
            "transform.feature_transform.conv3.bias.bin",
            "transform.feature_transform.bn3.weight.bin",
            "transform.feature_transform.bn3.bias.bin",
            "transform.feature_transform.bn3.running_mean.bin",
            "transform.feature_transform.bn3.running_var.bin", 32, 128, 1024, 1,
            1024, 1, 1);

    add_global_max_pool_layer(
            feature_transform_block, eng, stream, 32, 1024, 1024, 1);

    add_fc_bias_bnorm_relu_block(feature_transform_block, eng, stream, data_dir,
            "transform.feature_transform.fc1.weight.bin",
            "transform.feature_transform.fc1.bias.bin",
            "transform.feature_transform.bn4.weight.bin",
            "transform.feature_transform.bn4.bias.bin",
            "transform.feature_transform.bn4.running_mean.bin",
            "transform.feature_transform.bn4.running_var.bin", 32, 1024, 512);

    add_fc_bias_bnorm_relu_block(feature_transform_block, eng, stream, data_dir,
            "transform.feature_transform.fc2.weight.bin",
            "transform.feature_transform.fc2.bias.bin",
            "transform.feature_transform.bn5.weight.bin",
            "transform.feature_transform.bn5.bias.bin",
            "transform.feature_transform.bn5.running_mean.bin",
            "transform.feature_transform.bn5.running_var.bin", 32, 512, 256);

    add_fc_layer(feature_transform_block, eng, stream,
            data_dir + "transform.feature_transform.fc3.weight.bin",
            data_dir + "transform.feature_transform.fc3.bias.bin", 32, 256,
            4096);

    add_sum_layer(feature_transform_block, eng, stream,
            data_dir + "transform.feature_transform.id.bin", 1, 32 * 4096, 1,
            1);
    add_mm_layer(feature_transform_block, eng, stream,
            base_transform_block.get_output_mem(), 32, 1024, 64, 64);

    add_conv_bias_bnorm_relu_block(feature_transform_block, eng, stream,
            data_dir, "transform.conv2.weight.bin", "transform.conv2.bias.bin",
            "transform.bn2.weight.bin", "transform.bn2.bias.bin",
            "transform.bn2.running_mean.bin", "transform.bn2.running_var.bin",
            32, 64, 1024, 1, 128, 1, 1);

    add_conv_bias_bnorm_relu_block(feature_transform_block, eng, stream,
            data_dir, "transform.conv3.weight.bin", "transform.conv3.bias.bin",
            "transform.bn3.weight.bin", "transform.bn3.bias.bin",
            "transform.bn3.running_mean.bin", "transform.bn3.running_var.bin",
            32, 128, 1024, 1, 1024, 1, 1, false);

    add_global_max_pool_layer(
            feature_transform_block, eng, stream, 32, 1024, 1, 1024);

    add_fc_bias_bnorm_relu_block(feature_transform_block, eng, stream, data_dir,
            "fc1.weight.bin", "fc1.bias.bin", "bn1.weight.bin", "bn1.bias.bin",
            "bn1.running_mean.bin", "bn1.running_var.bin", 32, 1024, 512);

    add_fc_bias_bnorm_relu_block(feature_transform_block, eng, stream, data_dir,
            "fc2.weight.bin", "fc2.bias.bin", "bn2.weight.bin", "bn2.bias.bin",
            "bn2.running_mean.bin", "bn2.running_var.bin", 32, 512, 256);

    add_fc_layer(feature_transform_block, eng, stream,
            data_dir + "fc3.weight.bin", data_dir + "fc3.bias.bin", 32, 256,
            10);

    add_logsoftmax_layer(feature_transform_block, eng, stream, 32, 10, 1, 1);

    input_transform_block.execute(in_mem);
    base_transform_block.execute(input_transform_block.get_output_mem());
    feature_transform_block.execute(base_transform_block.get_output_mem());

    auto output = feature_transform_block.get_output_as_host_vec();

    // Find index of max value in each row of output, then calculate mode of
    // results to get final classification
    std::vector<int> predicted(32);
    for (int i = 0; i < 32; i++) {
        auto maxVal = std::max_element(
                output.begin() + (i * 10), output.begin() + (i * 10) + 10);
        predicted[i] = std::distance(output.begin() + (i * 10), maxVal);
    }
    std::sort(predicted.begin(), predicted.end());

    int prev = predicted[0];
    int count = 1;
    int mode = 0;
    int mode_count = 0;
    for (size_t i = 1; i < predicted.size(); ++i) {
        if (predicted[i] == prev) {
            count++;
        } else {
            if (count > mode_count) {
                mode = prev;
                mode_count = count;
            }
            count = 1;
        }
        prev = predicted[i];
    }
    if (count > mode_count) {
        mode = prev;
        mode_count = count;
    }
    std::cout << "classed as " << mode << " (i.e., " << object_map[mode] << ")"
              << std::endl;

#if CHECK_PERF
    int loops = 8;
    do {
        auto start = std::chrono::high_resolution_clock::now();
        input_transform_block.execute(in_ptr);
        base_transform_block.execute(
                input_transform_block.get_last_output_ptr());
        feature_transform_block.execute(
                base_transform_block.get_last_output_ptr());
        sycl_queue.wait_and_throw();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << (end - start).count() << " ns" << std::endl;
    } while (--loops);
#endif

    return 0;
}
