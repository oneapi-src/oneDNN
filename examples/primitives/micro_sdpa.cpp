/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "tensor_utils.hpp"

#define FORT "/home/pryorgal/pryorgal-fort/"

void single_prompt_problem() {
    const int batch_size_in_sequences = 2;

    const int num_queries = 5;
    const int num_blocks = 3, block_size = 4;
    const int max_num_blocks = 3; // max(diff(block_indices_begins))

    tensor prompt_lens = cast(tensor({5}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins = cast(tensor({0, 5}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(tensor({0, 4}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices_begins
            = cast(tensor({0, 2}, {1, 1, 1, 2}), dt::s32);

    const int num_heads = 1, head_size = 4, num_kv_heads = 1;

    tensor query = read(FORT"tensors/query.txt");
    tensor key_cache = read(FORT"tensors/key_cache.txt");
    tensor value_cache = read(FORT"tensors/value_cache.txt");
    tensor output = zeros({1, 1, num_queries, 4});
    dnnl::sdpa_micro::primitive_desc sdpa_pd
            = sdpa_micro::primitive_desc(global_engine, max_num_blocks,
                    query.md_, key_cache.md_, value_cache.md_, output.md_,
                    tensor().md_, prompt_lens.md_, subsequence_begins.md_,
                    block_indices.md_, block_indices_begins.md_);
    auto sdpa_prim = sdpa_micro(sdpa_pd);

    std::unordered_map<int, memory> sdpa_args;
    sdpa_args.insert({DNNL_ARG_QUERIES, query.mem_});
    sdpa_args.insert({DNNL_ARG_KEYS, key_cache.mem_});
    sdpa_args.insert({DNNL_ARG_VALUES, value_cache.mem_});
    sdpa_args.insert({DNNL_ARG_DST, output.mem_});
    sdpa_args.insert({DNNL_ARG_PROMPT_LENS, prompt_lens.mem_});
    sdpa_args.insert({DNNL_ARG_SUBSEQUENCE_BEGINS, subsequence_begins.mem_});
    sdpa_args.insert({DNNL_ARG_BLOCK_INDICES, block_indices.mem_});
    sdpa_args.insert(
            {DNNL_ARG_BLOCK_INDICES_BEGINS, block_indices_begins.mem_});
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();

    show(output);
}

void single_prompt_single_page_problem() {
    const int batch_size_in_sequences = 2;

    const int num_queries = 5;
    const int block_size = 4;
    const int max_num_blocks = 1;

    tensor prompt_lens = cast(tensor({5}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins = cast(tensor({0, 5}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(tensor({0}, {1, 1, 1, 1}), dt::s32);
    tensor block_indices_begins
            = cast(tensor({0, 1}, {1, 1, 1, 2}), dt::s32);

    const int num_heads = 1, head_size = 4, num_kv_heads = 1;

    tensor query = read(FORT"tensors/query.txt");
    tensor key_cache = read(FORT"tensors/key_cache.txt");
    tensor value_cache = read(FORT"tensors/value_cache.txt");
    tensor output = zeros({1, 1, num_queries, 4});
    dnnl::sdpa_micro::primitive_desc sdpa_pd
            = sdpa_micro::primitive_desc(global_engine, max_num_blocks,
                    query.md_, key_cache.md_, value_cache.md_, output.md_,
                    tensor().md_, prompt_lens.md_, subsequence_begins.md_,
                    block_indices.md_, block_indices_begins.md_);
    auto sdpa_prim = sdpa_micro(sdpa_pd);

    std::unordered_map<int, memory> sdpa_args;
    sdpa_args.insert({DNNL_ARG_QUERIES, query.mem_});
    sdpa_args.insert({DNNL_ARG_KEYS, key_cache.mem_});
    sdpa_args.insert({DNNL_ARG_VALUES, value_cache.mem_});
    sdpa_args.insert({DNNL_ARG_DST, output.mem_});
    sdpa_args.insert({DNNL_ARG_PROMPT_LENS, prompt_lens.mem_});
    sdpa_args.insert({DNNL_ARG_SUBSEQUENCE_BEGINS, subsequence_begins.mem_});
    sdpa_args.insert({DNNL_ARG_BLOCK_INDICES, block_indices.mem_});
    sdpa_args.insert(
            {DNNL_ARG_BLOCK_INDICES_BEGINS, block_indices_begins.mem_});
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();

    show(output);
}

void single_prompt_two_page_problem() {
    const int num_queries = 5, max_num_blocks = 2;

    tensor prompt_lens = cast(tensor({5}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins = cast(tensor({0, 5}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(tensor({0}, {1, 1, 1, 1}), dt::s32);
    tensor block_indices_begins
      = cast(tensor({0, 1}, {1, 1, 1, 2}), dt::s32);

    const int num_heads = 1, head_size = 4, num_kv_heads = 1;

    tensor query = read(FORT"tensors/query.txt");
    tensor key_cache = read(FORT"tensors/key_cache.txt");
    tensor value_cache = read(FORT"tensors/value_cache.txt");
    tensor output = zeros({1, 1, num_queries, head_size});
    dnnl::sdpa_micro::primitive_desc sdpa_pd
            = sdpa_micro::primitive_desc(global_engine, max_num_blocks,
                    query.md_, key_cache.md_, value_cache.md_, output.md_,
                    tensor().md_, prompt_lens.md_, subsequence_begins.md_,
                    block_indices.md_, block_indices_begins.md_);
    auto sdpa_prim = sdpa_micro(sdpa_pd);

    std::unordered_map<int, memory> sdpa_args;
    sdpa_args.insert({DNNL_ARG_QUERIES, query.mem_});
    sdpa_args.insert({DNNL_ARG_KEYS, key_cache.mem_});
    sdpa_args.insert({DNNL_ARG_VALUES, value_cache.mem_});
    sdpa_args.insert({DNNL_ARG_DST, output.mem_});
    sdpa_args.insert({DNNL_ARG_PROMPT_LENS, prompt_lens.mem_});
    sdpa_args.insert({DNNL_ARG_SUBSEQUENCE_BEGINS, subsequence_begins.mem_});
    sdpa_args.insert({DNNL_ARG_BLOCK_INDICES, block_indices.mem_});
    sdpa_args.insert(
            {DNNL_ARG_BLOCK_INDICES_BEGINS, block_indices_begins.mem_});
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();

    show(output);
}

void two_prompt_two_page_problem() {
    const int num_queries = 5, max_num_blocks = 2;

    tensor prompt_lens = cast(tensor({3, 2}, {1, 1, 1, 2}), dt::s32);
    tensor subsequence_begins = cast(tensor({0, 3, 5}, {1, 1, 1, 3}), dt::s32);
    tensor block_indices = cast(tensor({0, 0}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices_begins
      = cast(tensor({0, 1, 2}, {1, 1, 1, 3}), dt::s32);

    const int num_heads = 1, head_size = 4, num_kv_heads = 1;

    tensor query = read(FORT"tensors/query.txt");
    tensor key_cache = read(FORT"tensors/key_cache.txt");
    tensor value_cache = read(FORT"tensors/value_cache.txt");
    tensor output = zeros({1, 1, num_queries, head_size});
    dnnl::sdpa_micro::primitive_desc sdpa_pd
            = sdpa_micro::primitive_desc(global_engine, max_num_blocks,
                    query.md_, key_cache.md_, value_cache.md_, output.md_,
                    tensor().md_, prompt_lens.md_, subsequence_begins.md_,
                    block_indices.md_, block_indices_begins.md_);
    auto sdpa_prim = sdpa_micro(sdpa_pd);

    std::unordered_map<int, memory> sdpa_args;
    sdpa_args.insert({DNNL_ARG_QUERIES, query.mem_});
    sdpa_args.insert({DNNL_ARG_KEYS, key_cache.mem_});
    sdpa_args.insert({DNNL_ARG_VALUES, value_cache.mem_});
    sdpa_args.insert({DNNL_ARG_DST, output.mem_});
    sdpa_args.insert({DNNL_ARG_PROMPT_LENS, prompt_lens.mem_});
    sdpa_args.insert({DNNL_ARG_SUBSEQUENCE_BEGINS, subsequence_begins.mem_});
    sdpa_args.insert({DNNL_ARG_BLOCK_INDICES, block_indices.mem_});
    sdpa_args.insert(
            {DNNL_ARG_BLOCK_INDICES_BEGINS, block_indices_begins.mem_});
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();

    show(output);
}

int main(int argc, char **argv) {
    global_engine = dnnl::engine(dnnl::engine::kind::gpu, 0);
    global_engine_stream = dnnl::stream(global_engine);

    two_prompt_two_page_problem();
    // single_prompt_single_page_problem();
      return 0;

    // const int batch_size_in_sequences = 2;

    // const int num_queries = 3 + 2;
    // const int num_blocks = 3, block_size = 4;
    // const int max_num_blocks = 3; // max(diff(block_indices_begins))

    // tensor prompt_lens = cast(tensor({3, 2}, {1, 1, 1, 2}), dt::s32);
    // tensor subsequence_begins = cast(tensor({0, 3, 5}, {1, 1, 1, 3}), dt::s32);
    // tensor block_indices = cast(tensor({0, 4, 1}, {1, 1, 1, 3}), dt::s32);
    // tensor block_indices_begins
    //         = cast(tensor({0, 2, 3}, {1, 1, 1, 3}), dt::s32);

    // const int num_heads = 1, head_size = 4, num_kv_heads = 1;

    // tensor query = read("tensors/query.txt");
    // tensor key_cache = read("tensors/key_cache.txt");
    // tensor value_cache = read("tensors/value_cache.txt");
    // tensor output = zeros({1, 1, num_queries, 4});
    // // tensor output = zeros({block_indices.size(), num_queries, head_size});
    // dnnl::sdpa_micro::primitive_desc sdpa_pd
    //         = sdpa_micro::primitive_desc(global_engine, max_num_blocks,
    //                 query.md_, key_cache.md_, value_cache.md_, output.md_,
    //                 tensor().md_, prompt_lens.md_, subsequence_begins.md_,
    //                 block_indices.md_, block_indices_begins.md_);
    // auto sdpa_prim = sdpa_micro(sdpa_pd);

    // std::unordered_map<int, memory> sdpa_args;
    // sdpa_args.insert({DNNL_ARG_QUERIES, query.mem_});
    // sdpa_args.insert({DNNL_ARG_KEYS, key_cache.mem_});
    // sdpa_args.insert({DNNL_ARG_VALUES, value_cache.mem_});
    // sdpa_args.insert({DNNL_ARG_DST, output.mem_});
    // sdpa_args.insert({DNNL_ARG_PROMPT_LENS, prompt_lens.mem_});
    // sdpa_args.insert({DNNL_ARG_SUBSEQUENCE_BEGINS, subsequence_begins.mem_});
    // sdpa_args.insert({DNNL_ARG_BLOCK_INDICES, block_indices.mem_});
    // sdpa_args.insert(
    //         {DNNL_ARG_BLOCK_INDICES_BEGINS, block_indices_begins.mem_});
    // sdpa_prim.execute(global_engine_stream, sdpa_args);
    // global_engine_stream.wait();

    // show(output);

    // return 0;
}
