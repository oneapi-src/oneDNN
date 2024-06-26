/*******************************************************************************
* Copyright 2024 Intel Corporation
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

/// @example cache_primitive_desc.cpp
/// > Annotated version: @ref cache_primitive_desc_cpp
///
/// @page cache_primitive_desc_cpp_short
///
/// This C++ API example demonstrates how to implement a cache mechanism for
/// primitive descriptors. The purpose of this cache is to prevent the repeated
/// construction of the same primitive descriptor for identical shapes. This is
/// particularly beneficial when creating and executing a large number of convolution
/// primitives, as it can significantly reduce the overhead of construction of the
/// primitive descriptors.
///
/// Key optimizations included in this example:
/// - Utilizing hash values of keys for faster unordered_map queries.
/// - A Least Recently Used (LRU) list to maintain key order and ensure frequently
/// used primitive descriptors remain within the MAX_CACHE_SIZE limit.
///
/// @page cache_primitive_desc_cpp Cache Primitive Descriptor Example
/// @copydetails cache_primitive_desc_cpp_short
///
/// @include cache_primitive_desc.cpp

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <list>
#include <unordered_map>

#include "dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

// Define the maximum size of the cache
const std::size_t MAX_CACHE_SIZE = 128;

// Define the key for the cache
struct conv_cache_key_t {
    memory::dims src_dims;
    memory::dims weights_dims;
    memory::dims bias_dims;
    memory::dims dst_dims;
    memory::dims strides_dims;
    memory::dims padding_dims_l;
    memory::dims padding_dims_r;
    mutable std::size_t hash; // A field to store the hash value

    conv_cache_key_t(const memory::dims &src, const memory::dims &weights,
            const memory::dims &bias, const memory::dims &dst,
            const memory::dims &strides, const memory::dims &padding_l,
            const memory::dims &padding_r)
        : src_dims(src)
        , weights_dims(weights)
        , bias_dims(bias)
        , dst_dims(dst)
        , strides_dims(strides)
        , padding_dims_l(padding_l)
        , padding_dims_r(padding_r) {
        // Compute the hash value when the key is constructed
        hash = compute_hash();
    }

private:
    std::size_t compute_hash() const {
        std::size_t seed = 0;
        for (const auto &dim : src_dims) {
            // 0x9e3779b9 is derived from the golden ratio and is used to
            // help distribute the hash values evenly across the hash table.
            seed ^= std::hash<int> {}(dim) + 0x9e3779b9 + (seed << 6)
                    + (seed >> 2);
        }
        for (const auto &dim : weights_dims) {
            seed ^= std::hash<int> {}(dim) + 0x9e3779b9 + (seed << 6)
                    + (seed >> 2);
        }
        for (const auto &dim : bias_dims) {
            seed ^= std::hash<int> {}(dim) + 0x9e3779b9 + (seed << 6)
                    + (seed >> 2);
        }
        for (const auto &dim : dst_dims) {
            seed ^= std::hash<int> {}(dim) + 0x9e3779b9 + (seed << 6)
                    + (seed >> 2);
        }
        for (const auto &dim : strides_dims) {
            seed ^= std::hash<int> {}(dim) + 0x9e3779b9 + (seed << 6)
                    + (seed >> 2);
        }
        for (const auto &dim : padding_dims_l) {
            seed ^= std::hash<int> {}(dim) + 0x9e3779b9 + (seed << 6)
                    + (seed >> 2);
        }
        for (const auto &dim : padding_dims_r) {
            seed ^= std::hash<int> {}(dim) + 0x9e3779b9 + (seed << 6)
                    + (seed >> 2);
        }
        return seed;
    }
};

// Define the computation function of hash value for the key
struct conv_cache_key_hash {
    std::size_t operator()(const conv_cache_key_t &k) const { return k.hash; }
};

// Define the comparison function for the key
struct conv_cache_key_equal {
    bool operator()(
            const conv_cache_key_t &a, const conv_cache_key_t &b) const {
        return a.src_dims == b.src_dims && a.weights_dims == b.weights_dims
                && a.bias_dims == b.bias_dims && a.dst_dims == b.dst_dims
                && a.strides_dims == b.strides_dims
                && a.padding_dims_l == b.padding_dims_l
                && a.padding_dims_r == b.padding_dims_r;
    }
};

// Define a list to keep track of the order of the keys
std::list<conv_cache_key_t> lru_list;
// Define the cache
std::unordered_map<conv_cache_key_t,
        std::pair<convolution_forward::primitive_desc,
                std::list<conv_cache_key_t>::iterator>,
        conv_cache_key_hash, conv_cache_key_equal>
        conv_cache;

// Reserve memory for the unordered_map
void initialize_conv_cache() {
    conv_cache.reserve(MAX_CACHE_SIZE);
}

// Function to get a conv primitive_desc from the cache or create a new one
convolution_forward::primitive_desc get_or_add_conv_primitive_desc(
        const conv_cache_key_t &key, engine engine) {
    // Initialize the cache size only at first call
    static bool first_call = true;
    if (first_call) {
        initialize_conv_cache();
        first_call = false;
    }

    auto it = conv_cache.find(key);

    if (it != conv_cache.end()) {
        // Move the key to the front of the LRU list only if it is not already at the front
        if (it->second.second != lru_list.begin()) {
            lru_list.splice(lru_list.begin(), lru_list, it->second.second);
        }

        // Display the cache hit message
        std::cout << "Conv primitive_desc cache hit - ";
    } else {
        // If the cache is full, remove the least recently used item
        if (conv_cache.size() == MAX_CACHE_SIZE) {
            conv_cache.erase(lru_list.back());
            lru_list.pop_back();
        }
        // Add the new key to the front of the LRU list
        lru_list.emplace_front(key);

        // Define the required data types
        dt conv_dtype = dt::f32;

        // Create memory descriptors with format_tag::any for the primitive. This
        // enables the convolution primitive to choose memory layouts for an
        // optimized primitive implementation, and these layouts may differ from the
        // ones provided by the user.
        auto conv_src_md = memory::desc(key.src_dims, conv_dtype, tag::any);
        auto conv_weights_md
                = memory::desc(key.weights_dims, conv_dtype, tag::any);
        auto conv_bias_md = memory::desc(key.bias_dims, conv_dtype, tag::any);
        auto conv_dst_md = memory::desc(key.dst_dims, conv_dtype, tag::any);

        // Create primitive post-ops (ReLU).
        // The post-ops can also be added in the conv_cache_key_t to differentiate between
        // different post-ops configurations.
        const float alpha = 0.f;
        const float beta = 0.f;
        post_ops conv_ops;
        conv_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
        primitive_attr conv_attr;
        conv_attr.set_post_ops(conv_ops);

        // Create primitive descriptor.
        auto conv_pd = convolution_forward::primitive_desc(engine,
                prop_kind::forward_inference, algorithm::convolution_direct,
                conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md,
                key.strides_dims, key.padding_dims_l, key.padding_dims_r,
                conv_attr);

        bool inserted;
        std::tie(it, inserted) = conv_cache.emplace(
                key, std::make_pair(std::move(conv_pd), lru_list.begin()));

        // Display the cache miss message
        std::cout << "Conv primitive_desc newly constructed - ";
    }

    return it->second.first;
}

void run_case(engine engine, stream engine_stream, memory::dim ic_value) {
    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = ic_value, // input channels
            IH = 13, // input height
            IW = 13, // input width
            OC = 64, // output channels
            KH = 3, // weights height
            KW = 3, // weights width
            PH_L = 1, // height padding: left
            PH_R = 1, // height padding: right
            PW_L = 1, // width padding: left
            PW_R = 1, // width padding: right
            SH = 4, // height-wise stride
            SW = 4, // width-wise stride
            OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
            OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims weights_dims = {OC, IC, KH, KW};
    memory::dims bias_dims = {OC};
    memory::dims dst_dims = {N, OC, OH, OW};

    // Strides, padding dimensions.
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> bias_data(OC);
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights, and dst tensors.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory objects for tensor data (src, weights, dst). In this
    // example, NCHW layout is assumed for src and dst, and OIHW for weights.
    auto user_src_mem = memory({src_dims, dt::f32, tag::nchw}, engine);
    auto user_weights_mem = memory({weights_dims, dt::f32, tag::oihw}, engine);
    auto user_dst_mem = memory({dst_dims, dt::f32, tag::nchw}, engine);

    // Create memory object for input bias.
    auto user_bias_mem = memory({bias_dims, dt::f32, tag::a}, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), user_src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
    write_to_dnnl_memory(bias_data.data(), user_bias_mem);

    // Create a key for the cache
    conv_cache_key_t key = {src_dims, weights_dims, bias_dims, dst_dims,
            strides_dims, padding_dims_l, padding_dims_r};

    // Timing runs.
    auto start = std::chrono::steady_clock::now();

    // Get the conv primitive_desc based on the key
    auto conv_pd = get_or_add_conv_primitive_desc(key, engine);

    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> duration = end - start;

    // Display input channel size and the time for conv primitive_desc
    std::cout << "IC: " << ic_value
              << ", Time for conv primitive_desc: " << duration.count() * 1e6
              << " us" << std::endl;

    // For now, assume that the src, weights, bias and dst memory layouts generated
    // by the primitive and the ones provided by the user are identical.
    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_bias_mem = user_bias_mem;
    auto conv_dst_mem = user_dst_mem;

    // Reorder the data in case the src, weights and bias memory layouts generated by
    // the primitive and the ones provided by the user are different. In this
    // case, we create additional memory objects with internal buffers that will
    // contain the reordered data. The data in dst will be reordered after the
    // convolution computation has finalized.
    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = memory(conv_pd.src_desc(), engine);
        reorder(user_src_mem, conv_src_mem)
                .execute(engine_stream, user_src_mem, conv_src_mem);
    }

    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), engine);
        reorder(user_weights_mem, conv_weights_mem)
                .execute(engine_stream, user_weights_mem, conv_weights_mem);
    }

    if (conv_pd.bias_desc() != user_bias_mem.get_desc()) {
        conv_bias_mem = memory(conv_pd.bias_desc(), engine);
        reorder(user_bias_mem, conv_bias_mem)
                .execute(engine_stream, user_bias_mem, conv_bias_mem);
    }

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = memory(conv_pd.dst_desc(), engine);
    }

    // Create the primitive.
    auto conv_prim = convolution_forward(conv_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, conv_bias_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

    // Primitive execution: convolution with ReLU.
    conv_prim.execute(engine_stream, conv_args);

    // Reorder the data in case the dst memory descriptor generated by the
    // primitive and the one provided by the user are different.
    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        reorder(conv_dst_mem, user_dst_mem)
                .execute(engine_stream, conv_dst_mem, user_dst_mem);
    } else
        user_dst_mem = conv_dst_mem;

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
}

void cache_primitive_desc_example(engine::kind engine_kind) {
    // Create execution dnnl::engine.
    engine engine(engine_kind, 0);

    // Create dnnl::stream.
    stream engine_stream(engine);

    // Different problem shapes are from the dimension differnences in inputs,
    // weights, outputs, padding, strides, etc. and also the configurations
    // in primitive attribute, which will result in different primitive_descs.
    // This example sets different input channel values to show how the cache for
    // primitive_desc works.
    // For the same ic_value input, the time for primitive_desc is much shorter
    // when the cache hits than when the conv primitive_desc is newly constructed.
    std::vector<memory::dim> ic_values = {32, 64, 128, 32, 64, 128, 256};
    for (memory::dim ic_value : ic_values) {
        run_case(engine, engine_stream, ic_value);
    }
}

int main(int argc, char **argv) {
    return handle_example_errors(
            cache_primitive_desc_example, parse_engine_kind(argc, argv));
}
