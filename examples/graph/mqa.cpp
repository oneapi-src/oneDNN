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

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "graph_example_utils.hpp"

using namespace dnnl;

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

struct mqa_dims_t {
    dim mb;
    dim seq_len;
    dim q_head_num;
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

const char *get_type_string(logical_tensor::data_type dt) {
    const char *type_string = "unknown";

#define TYPE_CASE(T) \
    if (dt == logical_tensor::data_type::T) type_string = #T;
    TYPE_CASE(f16);
    TYPE_CASE(f32);
    TYPE_CASE(bf16);
#undef TYPE_CASE

    return type_string;
}

void print_test_case(logical_tensor::data_type dt, const mqa_dims_t &p) {
    std::cout << '[' << std::setw(4) << get_type_string(dt);
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", q_head_num = " << p.q_head_num
              << ", head_size = " << p.head_size;
    std::cout << "] " << std::flush;
}

void bench_mqa(engine::kind ekind, logical_tensor::data_type dt,
        const mqa_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    allocator alloc = create_allocator(ekind);

    // Create execution dnnl::engine.
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Prepare input and output shapes to construct the mqa graph.
    const dims q_sz = {p.mb, p.q_head_num, p.seq_len, p.head_size};
    // For mqa, key tensor and value tensor always have single head.
    const dims kv_sz = {p.mb, 1, p.seq_len, p.head_size};
    const dims score_sz = {p.mb, p.q_head_num, p.seq_len, p.seq_len};
    const dims scale_sz = {1};
    const dims mask_sz = {p.mb, 1, 1, p.seq_len};

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    // score = query x key.T
    auto query = logical_tensor(id++, dt, q_sz, layout_type::strided);
    auto key = logical_tensor(id++, dt, kv_sz, layout_type::strided);
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
    auto value = logical_tensor(id++, dt, kv_sz, layout_type::strided);
    auto output = logical_tensor(id++, dt, q_sz, layout_type::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value});
    bmm2.add_outputs({output});

    // Construct a mqa graph with engine kind and operations. It has the same
    // graph structure as the sdpa defined in the sdpa.cpp.
    dnnl::graph::graph mqa(ekind);
    mqa.add_op(bmm1);
    mqa.add_op(scale_div);
    mqa.add_op(mask_add);
    mqa.add_op(softmax);
    mqa.add_op(bmm2);
    mqa.finalize();

    // Get partitions from the mqa graph.
    std::vector<partition> partitions = mqa.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "unsupported mqa" << std::endl;
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
    std::vector<float> query_data(product(q_sz));
    std::vector<float> key_data(product(kv_sz));
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(kv_sz));
    std::vector<float> output_data(product(q_sz));

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
    // Execute the compiled partition of mqa.
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
    std::cout << "graph runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;
}

void bad_args() {
    std::cerr << "Usage: graph-mqa-cpp [cpu|gpu]\n"
                 "       graph-mqa-cpp [cpu|gpu] <mb> <seq_len> <q_head_num> "
                 "<head_size>\n\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

void bench(engine::kind ekind, dnnl_data_type_t dt, const mqa_dims_t &p,
        double time_limit = 0.) {
    try {
        bench_mqa(ekind, static_cast<logical_tensor::data_type>(dt), p,
                time_limit);
        get_mem_pool().clear();
    } catch (dnnl::error &e) {
        // Catch and report unimplemented cases.
        if (e.status == dnnl_unimplemented) {
            std::cout << "unsupported mqa" << std::endl;
        } else
            throw;
    }
}

void mqa_perf(engine::kind ekind, int argc, char **argv) {
    // default testing parameters
    mqa_dims_t params = {32, 384, 16, 64};

    if (argc > 2) {
        if (argc == 6) {
            params.mb = std::atoi(argv[2]);
            params.seq_len = std::atoi(argv[3]);
            params.q_head_num = std::atoi(argv[4]);
            params.head_size = std::atoi(argv[5]);
        } else {
            bad_args();
        }

        if (params.mb <= 0 || params.seq_len <= 0 || params.q_head_num <= 0
                || params.head_size <= 0) {
            bad_args();
        }
    }

    bench(ekind, dnnl_f32, params, 2000.0 /*ms*/);
    bench(ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    bench(ekind, dnnl_f16, params, 2000.0 /*ms*/);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            mqa_perf, parse_engine_kind(argc, argv, 4), argc, argv);
}
