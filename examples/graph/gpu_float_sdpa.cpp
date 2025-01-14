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

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "graph_example_utils.hpp"

using namespace dnnl;
using tag = memory::format_tag;

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

struct timers_t {
    void reset() {
        ms_start_ = 0;
        sum_ = 0;
        ticks_ = 0;
    }
    void start() { ms_start_ = ms_now(); }
    void stop() {
        double cur_time = ms_now() - ms_start_;
        sum_ += cur_time;
        ticks_++;
    }
    double avg() { return sum_ / ticks_; }
    double ms_now() {
        auto timePointTmp
                = std::chrono::high_resolution_clock::now().time_since_epoch();
        return std::chrono::duration<double, std::milli>(timePointTmp).count();
    }

    double ms_start_ = 0;
    double sum_ = 0;
    size_t ticks_ = 0;
};

struct sdpa_dims_t {
    dim mb;
    dim seq_len;
    dim head_num;
    dim head_size;
    dim query_num;
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

void print_test_case(logical_tensor::data_type dt, const sdpa_dims_t &p) {
    std::cout << '[' << std::setw(4) << get_type_string(dt);
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", head_num = " << p.head_num
              << ", head_size = " << p.head_size
              << ", query_num = " << p.query_num;
    std::cout << "] " << std::flush;
}

engine::kind ekind = engine::kind::gpu;
allocator alloc = create_allocator(ekind);
// Create execution dnnl::engine.
dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
// Create dnnl::stream.
dnnl::stream strm(eng);
void bench_sdpa(engine::kind ekind, logical_tensor::data_type dt,
        const sdpa_dims_t &p, timers_t &graph_finilization_timer,
        timers_t &graph_construction_timer, timers_t &get_partition_timer,
        timers_t &compilation_timer, timers_t &execution_timer) {
    //print_test_case(dt, p);

    // allocator alloc = create_allocator(ekind);

    // // Create execution dnnl::engine.
    // dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // // Create dnnl::stream.
    // dnnl::stream strm(eng);

    graph_construction_timer.start();
    // Prepare input and output shapes to construct the sdpa graph.
    const dims qv_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const dims k_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
    const dims scale_sz = {1};
    const dims mask_sz = {p.mb, 1, p.query_num, p.seq_len};

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    // score = query x key.T
    auto query = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto key = logical_tensor(id++, dt, k_sz, layout_type::strided);
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
    auto value = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto output = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value});
    bmm2.add_outputs({output});

    // Construct a sdpa graph with engine kind and operations.
    dnnl::graph::graph sdpa(ekind);
    sdpa.add_op(bmm1);
    sdpa.add_op(scale_div);
    sdpa.add_op(mask_add);
    sdpa.add_op(softmax);
    sdpa.add_op(bmm2);
    graph_construction_timer.stop();

    graph_finilization_timer.start();
    sdpa.finalize();
    graph_finilization_timer.stop();

    get_partition_timer.start();
    // Get partitions from the sdpa graph.
    std::vector<partition> partitions = sdpa.get_partitions();
    get_partition_timer.stop();

    compilation_timer.start();
    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp = partitions[0].compile(
            {query, key, scale, mask, value}, {output}, eng);
    compilation_timer.stop();

    // Create tensor objects
    auto ts_query = tensor(query, eng);
    auto ts_key = tensor(key, eng);
    auto ts_scale = tensor(scale, eng);
    auto ts_mask = tensor(mask, eng);
    auto ts_value = tensor(value, eng);
    auto ts_output = tensor(output, eng);

    // Allocate user data.
    std::vector<float> query_data(product(qv_sz));
    std::vector<float> key_data(product(k_sz));
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(k_sz));
    std::vector<float> output_data(product(qv_sz));

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

    execution_timer.start();
    cp.execute(
            strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value}, {ts_output});
    strm.wait();
    execution_timer.stop();
}

void bad_args() {
    std::cerr << "Usage: graph-sdpa-cpp [cpu|gpu]\n"
                 "       graph-sdpa-cpp [cpu|gpu] <mb> <seq_len> "
                 "<head_num> <head_size> [<query_num>]\n\n"
                 "On CPU, it's recommended to test with numactl and memory "
                 "allocation tools like jemalloc or tcmalloc.\n\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

enum class api_kind {
    primitive,
    graph,
};

void bench(api_kind api, engine::kind ekind, dnnl_data_type_t dt,
        const sdpa_dims_t &p, double time_limit = 0.) {

    timers_t graph_finilization_timer, graph_construction_timer,
            get_partition_timer, compilation_timer, execution_timer;

    // api == api_kind::graph
    size_t fixed_run_times = 500;
    size_t warmup_run_times = 5;
    for (size_t iter = 0; iter < warmup_run_times + fixed_run_times; ++iter) {
        if (iter == warmup_run_times) {
            graph_finilization_timer.reset();
            graph_construction_timer.reset();
            get_partition_timer.reset();
            compilation_timer.reset();
            execution_timer.reset();
        }
        bench_sdpa(ekind, static_cast<logical_tensor::data_type>(dt), p,
                graph_finilization_timer, graph_construction_timer,
                get_partition_timer, compilation_timer, execution_timer);
    }

    get_mem_pool().clear();

    std::cout << "perf summary:" << std::endl;
    double total_time = graph_finilization_timer.avg()
            + graph_construction_timer.avg() + get_partition_timer.avg()
            + compilation_timer.avg() + execution_timer.avg();
    std::cout << "graph_finilization_timer time:"
              << graph_finilization_timer.avg()
              << " ms, percentage of total time: "
              << graph_finilization_timer.avg() / total_time << std::endl;
    std::cout << "graph_construction_timer time:"
              << graph_construction_timer.avg()
              << " ms, percentage of total time: "
              << graph_construction_timer.avg() / total_time << std::endl;
    std::cout << "get partition time:" << get_partition_timer.avg()
              << " ms, percentage of total time: "
              << get_partition_timer.avg() / total_time << std::endl;
    std::cout << "graph compilation time:" << compilation_timer.avg()
              << " ms, percentage of total time: "
              << compilation_timer.avg() / total_time << std::endl;
    std::cout << "graph execution time:" << execution_timer.avg()
              << " ms, percentage of total time: "
              << execution_timer.avg() / total_time << std::endl;
}

void sdpa_perf(engine::kind ekind, int argc, char **argv) {
    // default testing parameters
    sdpa_dims_t params = {1, 384, 16, 64, 384};

    if (argc > 2) {
        if (argc == 6) {
            params.mb = std::atoi(argv[2]);
            params.seq_len = std::atoi(argv[3]);
            params.query_num = std::atoi(argv[3]);
            params.head_num = std::atoi(argv[4]);
            params.head_size = std::atoi(argv[5]);
        } else if (argc == 7) {
            params.mb = std::atoi(argv[2]);
            params.seq_len = std::atoi(argv[3]);
            params.head_num = std::atoi(argv[4]);
            params.head_size = std::atoi(argv[5]);
            params.query_num = std::atoi(argv[6]);
        } else {
            bad_args();
        }

        if (params.mb <= 0 || params.seq_len <= 0 || params.head_num <= 0
                || params.head_size <= 0) {
            bad_args();
        }
    }

    bench(api_kind::graph, ekind, dnnl_f16, params);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            sdpa_perf, parse_engine_kind(argc, argv, 5), argc, argv);
}
