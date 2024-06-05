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

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "graph_example_utils.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

struct sdpa_dims_t {
    dim mb;
    dim seq_len;
    dim head_num;
    dim head_size;
};

static const int min_runs = 4;

// this is changed from the fill_random() function in matmul_perf.cpp.
void fill_random(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

// initialize the mask with first 3/4 elements with 0s and the last 1/4 elements
// with -inf.
void fill_mask(std::vector<float> &mask, size_t seq_len) {
    const size_t pos = seq_len * 3 / 4;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (i % seq_len < pos)
            mask[i] = 0.f;
        else
            mask[i] = -1 * std::numeric_limits<float>::infinity();
    }
}

const char *get_type_string(data_type dt) {
    const char *type_string = "unknown";

#define TYPE_CASE(T) \
    if (dt == data_type::T) type_string = #T;
    TYPE_CASE(f16);
    TYPE_CASE(f32);
    TYPE_CASE(bf16);
#undef TYPE_CASE

    return type_string;
}

void print_test_case(data_type dt, const sdpa_dims_t &p) {
    std::cout << '[' << std::setw(4) << get_type_string(dt);
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", head_num = " << p.head_num
              << ", head_size = " << p.head_size;
    std::cout << "] " << std::flush;
}

void bench_sdpa(engine::kind ekind, data_type dt, const sdpa_dims_t &p,
        double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);
    if (!quick_test) print_test_case(dt, p);

    // Create execution dnnl::engine.
    dnnl::engine eng(ekind, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Prepare input and output shapes to construct the sdpa graph.
    const dims qkv_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const dims score_sz = {p.mb, p.head_num, p.seq_len, p.seq_len};
    const dims scale_sz = {1};
    const dims mask_sz = {p.mb, 1, 1, p.seq_len};

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    // score = query x key.T
    auto query = logical_tensor(id++, dt, qkv_sz, layout_type::strided);
    auto key = logical_tensor(id++, dt, qkv_sz, layout_type::strided);
    auto score = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key});
    bmm1.add_outputs({score});

    // scaled_score = score / scale
    auto scale = logical_tensor(id++, dt, scale_sz, layout_type::strided);
    auto scaled_score
            = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto scale_div = op(id++, op::kind::Divide, "scale_div");
    scale_div.add_inputs({score, scale});
    scale_div.add_outputs({scaled_score});

    // masked_score = scaled_score + mask
    auto mask = logical_tensor(id++, dt, mask_sz, layout_type::strided);
    auto masked_score
            = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto mask_add = op(id++, op::kind::Add, "mask_add");
    mask_add.add_inputs({scaled_score, mask});
    mask_add.add_outputs({masked_score});

    // attention_probs = softmax(masked_score)
    auto probs = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto softmax = op(id++, op::kind::SoftMax, "softmax");
    softmax.set_attr<int64_t>(op::attr::axis, -1);
    softmax.add_inputs({masked_score});
    softmax.add_outputs({probs});

    // attention_output = attention_probs x value
    auto value = logical_tensor(id++, dt, qkv_sz, layout_type::strided);
    auto output = logical_tensor(id++, dt, qkv_sz, layout_type::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value});
    bmm2.add_outputs({output});

    // Construct a sdpa graph with engine kind and operations.
    graph sdpa(ekind);
    sdpa.add_op(bmm1);
    sdpa.add_op(scale_div);
    sdpa.add_op(mask_add);
    sdpa.add_op(softmax);
    sdpa.add_op(bmm2);
    sdpa.finalize();

    // Get partitions from the sdpa graph.
    std::vector<partition> partitions = sdpa.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "unsupported sdpa" << std::endl;
        return;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp = partitions[0].compile(
            {query, key, scale, mask, value}, {output}, eng);

    // Create tensor objects
    auto ts_query = tensor(query, eng);
    auto ts_key = tensor(key, eng);
    auto ts_scale = tensor(scale, eng);
    auto ts_mask = tensor(mask, eng);
    auto ts_value = tensor(value, eng);
    auto ts_output = tensor(output, eng);

    // Allocate user data.
    std::vector<float> query_data(product(qkv_sz));
    std::vector<float> key_data(product(qkv_sz));
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(qkv_sz));
    std::vector<float> output_data(product(qkv_sz));

    fill_random(query_data);
    fill_random(key_data);
    fill_random(value_data);
    fill_mask(mask_data, static_cast<size_t>(p.seq_len));

    // Write data to tensor object's handle.
    write_to_dnnl_tensor(query_data.data(), ts_query);
    write_to_dnnl_tensor(key_data.data(), ts_key);
    write_to_dnnl_tensor(scale_data.data(), ts_scale);
    write_to_dnnl_tensor(mask_data.data(), ts_mask);
    write_to_dnnl_tensor(value_data.data(), ts_value);

    // Warmup run.
    // Execute the compiled partition of sdpa.
    cp.execute(
            strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value}, {ts_output});

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    cp.execute(
            strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value}, {ts_output});
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    const int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++)
        cp.execute(strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value},
                {ts_output});
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "runs: " << runs + 1 << " ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;

    return;
}

void bad_args() {
    std::cerr << "Usage: graph-sdpa-cpp [cpu|gpu]\n"
                 "       graph-sdpa-cpp [cpu|gpu] <mb> <seq_len> <head_num> "
                 "<head_size>\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

void sdpa_perf(engine::kind ekind, int argc, char **argv) {
    // default testing parameters
    sdpa_dims_t params = {1, 128, 16, 64};

    if (argc > 2) {
        if (argc == 6) {
            params.mb = std::atoi(argv[2]);
            params.seq_len = std::atoi(argv[3]);
            params.head_num = std::atoi(argv[4]);
            params.head_size = std::atoi(argv[5]);
        } else {
            bad_args();
        }

        if (params.mb <= 0 || params.seq_len <= 0 || params.head_num <= 0
                || params.head_size <= 0) {
            bad_args();
        }
    }

    bench_sdpa(ekind, data_type::f32, params, 4000.0 /*ms*/);
    bench_sdpa(ekind, data_type::bf16, params, 4000.0 /*ms*/);
    bench_sdpa(ekind, data_type::f16, params, 4000.0 /*ms*/);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            sdpa_perf, parse_engine_kind(argc, argv, 4), argc, argv);
}
