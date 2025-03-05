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

#define div_up(x, y) (((x) + (y) - 1) / (y))
#define FORT "/export/users/pryorgal-fort/"

void single_prompt_problem() {

    tensor prompt_lens = cast(tensor({5}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins = cast(tensor({0, 5}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(tensor({0, 4}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices_begins = cast(tensor({0, 2}, {1, 1, 1, 2}), dt::s32);

    tensor query = read(FORT "tensors/query.txt");
    tensor key_cache = read(FORT "tensors/key_cache.txt");
    tensor value_cache = read(FORT "tensors/value_cache.txt");
    tensor output = zeros({1, 1, 5, 4}); // 5 queries with head size 4
    dnnl::sdpa_micro::primitive_desc sdpa_pd = sdpa_micro::primitive_desc(
            global_engine, 5 * 4, // num pages * page size
            query.md_, key_cache.md_, value_cache.md_, output.md_, tensor().md_,
            prompt_lens.md_, subsequence_begins.md_, block_indices.md_,
            block_indices_begins.md_);
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

    tensor prompt_lens = cast(tensor({5}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins = cast(tensor({0, 5}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(tensor({0}, {1, 1, 1, 1}), dt::s32);
    tensor block_indices_begins = cast(tensor({0, 1}, {1, 1, 1, 2}), dt::s32);

    tensor query = read(FORT "tensors/query.txt");
    tensor key_cache = read(FORT "tensors/key_cache.txt");
    tensor value_cache = read(FORT "tensors/value_cache.txt");
    tensor output = zeros({1, 1, 5, 4}); // 5 queries
    dnnl::sdpa_micro::primitive_desc sdpa_pd = sdpa_micro::primitive_desc(
            global_engine, 3, query.md_, key_cache.md_, value_cache.md_,
            output.md_, tensor().md_, prompt_lens.md_, subsequence_begins.md_,
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
    const int num_queries = 5;

    tensor prompt_lens = cast(tensor({5}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins = cast(tensor({0, 5}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(tensor({0}, {1, 1, 1, 1}), dt::s32);
    tensor block_indices_begins = cast(tensor({0, 1}, {1, 1, 1, 2}), dt::s32);

    const int head_size = 4;

    tensor query = read(FORT "tensors/query.txt");
    tensor key_cache = read(FORT "tensors/key_cache.txt");
    tensor value_cache = read(FORT "tensors/value_cache.txt");
    tensor output = zeros({1, 1, num_queries, head_size});
    dnnl::sdpa_micro::primitive_desc sdpa_pd = sdpa_micro::primitive_desc(
            global_engine, 3, query.md_, key_cache.md_, value_cache.md_,
            output.md_, tensor().md_, prompt_lens.md_, subsequence_begins.md_,
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

    const int head_size = 4;

    tensor query = read(FORT "tensors/query.txt");
    tensor key_cache = read(FORT "tensors/key_cache.txt");
    tensor value_cache = read(FORT "tensors/value_cache.txt");
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

void mask_check() {
    printf("doing mask check\n");
    /* need to check a few conditions:
       - tile_m < page_size
       - page_size < tile_m
     */
    const int seq_len = 384, page_size = 96, head_size = 4;

    assert(seq_len % page_size == 0);
    const int pages_num = seq_len / page_size;
    const int heads_num = 1;

    tensor query = zeros({1, 1, seq_len, head_size * heads_num});
    tensor key_cache = zeros({pages_num, heads_num, head_size, page_size});
    tensor value_cache = zeros({pages_num, heads_num, page_size, head_size});
    tensor output = zeros({1, 1, seq_len, head_size});

    tensor prompt_lens = cast(tensor({(float)seq_len}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins
            = cast(tensor({0, (float)seq_len}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(rand(pages_num), dt::s32);
    tensor block_indices_begins
            = cast(tensor({0, (float)pages_num}, {1, 1, 1, 2}), dt::s32);

    dnnl::sdpa_micro::primitive_desc sdpa_pd = sdpa_micro::primitive_desc(
            global_engine, seq_len, query.md_, key_cache.md_, value_cache.md_,
            output.md_, tensor().md_, prompt_lens.md_, subsequence_begins.md_,
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
}

void prefill(int seq_len, int head_size, int page_size) {
    const int pages_num = div_up(seq_len, page_size);
    const int heads_num = 1;

    tensor query = zeros({1, 1, seq_len, head_size * heads_num});
    tensor key_cache = zeros({pages_num, heads_num, head_size, page_size});
    tensor value_cache = zeros({pages_num, heads_num, page_size, head_size});
    tensor output = zeros({1, 1, seq_len, head_size});

    tensor prompt_lens = cast(tensor({(float)seq_len}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins
            = cast(tensor({0, (float)seq_len}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(rand(pages_num), dt::s32);
    tensor block_indices_begins
            = cast(tensor({0, (float)pages_num}, {1, 1, 1, 2}), dt::s32);

    dnnl::sdpa_micro::primitive_desc sdpa_pd = sdpa_micro::primitive_desc(
            global_engine, seq_len, query.md_, key_cache.md_, value_cache.md_,
            output.md_, tensor().md_, prompt_lens.md_, subsequence_begins.md_,
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

    tic();
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();
    float seconds = toc();
    printf("prefill, seq_len %d, head_size %d, page_size %d, milliseconds %f\n",
            seq_len, head_size, page_size, seconds * 1e3);
}

void generate(int seq_len, int head_size, int page_size) {
    const int pages_num = seq_len / page_size;
    const int heads_num = 1;

    tensor query = zeros({1, 1, 1, head_size * heads_num});
    tensor key_cache = zeros({pages_num, heads_num, head_size, page_size});
    tensor value_cache = zeros({pages_num, heads_num, page_size, head_size});
    tensor output = zeros({1, 1, 1, head_size});

    tensor prompt_lens = cast(tensor({(float)1}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins
            = cast(tensor({0, (float)1}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(rand(pages_num), dt::s32);
    tensor block_indices_begins
            = cast(tensor({0, (float)pages_num}, {1, 1, 1, 2}), dt::s32);

    dnnl::sdpa_micro::primitive_desc sdpa_pd = sdpa_micro::primitive_desc(
            global_engine, seq_len, query.md_, key_cache.md_, value_cache.md_,
            output.md_, tensor().md_, prompt_lens.md_, subsequence_begins.md_,
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

    tic();
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();
    float seconds = toc();
    printf("generate seq_len %d, head_size %d, page_size %d, milliseconds %f\n",
            seq_len, head_size, page_size, seconds * 1e3);
}

void generate_non_paged(int seq_len, int head_size, int page_size) {
    const int pages_num = seq_len / page_size;
    const int heads_num = 1;

    tensor query = zeros({1, 1, 1, head_size * heads_num});
    tensor key_cache = zeros({pages_num, heads_num, head_size, seq_len});
    tensor value_cache = zeros({pages_num, heads_num, seq_len, head_size});
    tensor output = zeros({1, 1, 1, head_size});

    tensor prompt_lens = cast(tensor({(float)1}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins
            = cast(tensor({0, (float)1}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(rand(pages_num), dt::s32);
    tensor block_indices_begins
            = cast(tensor({0, (float)pages_num}, {1, 1, 1, 2}), dt::s32);

    dnnl::sdpa_micro::primitive_desc sdpa_pd = sdpa_micro::primitive_desc(
            global_engine, seq_len, query.md_, key_cache.md_, value_cache.md_,
            output.md_, tensor().md_, prompt_lens.md_, subsequence_begins.md_,
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

    tic();
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();
    float seconds = toc();
    printf("generate_non_paged seq_len %d, head_size %d, page_size %d, "
           "milliseconds %f\n",
            seq_len, head_size, page_size, seconds * 1e3);
}

void combined(int context_len, int head_size, int page_size) {
    const int pages_num = context_len / page_size;
    const int heads_num = 1;

    tensor query = zeros({1, 1, 129, head_size * heads_num});
    tensor key_cache = zeros({pages_num, heads_num, head_size, page_size});
    tensor value_cache = zeros({pages_num, heads_num, page_size, head_size});
    tensor output = zeros({1, 1, 129, head_size});

    tensor prompt_lens = cast(tensor({128, 1}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins = cast(tensor({0, 128}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices
            = cast(::concat(rand(pages_num), rand(pages_num)), dt::s32);
    tensor block_indices_begins = cast(
            tensor({0, (float)pages_num, (float)(2 * pages_num)}, {1, 1, 1, 3}),
            dt::s32);

    dnnl::sdpa_micro::primitive_desc sdpa_pd
            = sdpa_micro::primitive_desc(global_engine, context_len, query.md_,
                    key_cache.md_, value_cache.md_, output.md_, tensor().md_,
                    prompt_lens.md_, subsequence_begins.md_, block_indices.md_,
                    block_indices_begins.md_);
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

    tic();
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();
    float seconds = toc();
    printf("combined context_len %d, head_size %d, page_size %d, milliseconds "
           "%f\n",
            context_len, head_size, page_size, seconds * 1e3);
}

void prefill_non_paged(int seq_len, int head_size, int page_size) {
    // assert(seq_len % page_size == 0);
    const int pages_num = seq_len / page_size;
    const int heads_num = 1;

    tensor query = zeros({1, 1, seq_len, head_size * heads_num});
    tensor key_cache = zeros({1, heads_num, head_size, seq_len});
    tensor value_cache = zeros({1, heads_num, seq_len, head_size});
    tensor output = zeros({1, 1, seq_len, head_size});

    tensor prompt_lens = cast(tensor({(float)seq_len}, {1, 1, 1, 1}), dt::s32);
    tensor subsequence_begins
            = cast(tensor({0, (float)seq_len}, {1, 1, 1, 2}), dt::s32);
    tensor block_indices = cast(rand(pages_num), dt::s32);
    tensor block_indices_begins
            = cast(tensor({0, (float)pages_num}, {1, 1, 1, 2}), dt::s32);

    dnnl::sdpa_micro::primitive_desc sdpa_pd = sdpa_micro::primitive_desc(
            global_engine, seq_len, query.md_, key_cache.md_, value_cache.md_,
            output.md_, tensor().md_, prompt_lens.md_, subsequence_begins.md_,
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

    tic();
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();
    float seconds = toc();
    printf("prefill_non_paged, seq_len %d, head_size %d, milliseconds %f\n",
            seq_len, head_size, seconds * 1e3);
}

int main(int argc, char **argv) {
    global_engine = dnnl::engine(dnnl::engine::kind::gpu, 0);
    global_engine_stream = dnnl::stream(global_engine);

    // prefill
    for (int head_size : {128}) {
        for (int seq_len : {384, 512, 1024, 2048, 4096}) {
            prefill(seq_len, head_size, 64);
        }
    }

    // generate
    for (int head_size : {128}) {
        for (int seq_len : {385, 513, 1025, 2049, 4097}) {
            generate(seq_len, head_size, 64);
        }
    }

    // combined
    for (int head_size : {128}) {
        for (int context_len : {384, 1024, 2048, 4096}) {
            combined(context_len, head_size, 64); // query len 129
        }
    }

    return 0;
}
