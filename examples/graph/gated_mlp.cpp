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
#include <type_traits>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "oneapi/dnnl/experimental/dnnl_experimental.hpp" ///TMP FOR TESTING INTERNAL

#include "../half.hpp"
#include "graph_example_utils.hpp"

using namespace dnnl;
using tag = memory::format_tag;

using half_float::half;
using half_float::half_cast;

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

struct mlp_dims_t {
    dim mb;
    dim ic;
    dim oc;
};

static const int min_runs = 4;

// this is changed from the fill_random() function in matmul_perf.cpp.
template<typename T>
void fill_random(std::vector<T> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;
    const unsigned seed = 2;

    if (random_data_f.empty()) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

   size_t chunk = std::min(nrand, out.size());
   for(int i=0; i<out.size(); ++i) {
        out[i] = static_cast<T>(random_data_f[i%chunk]);
   }
   //for (size_t i = 0; i < out.size(); i += nrand) {
   //    size_t chunk = std::min(nrand, out.size() - i);
   //    std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
   //}
}

template<>
void fill_random(std::vector<half> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;
    const unsigned seed = 2;

    if (random_data_f.empty()) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

   size_t chunk = std::min(nrand, out.size());
   for(int i=0; i<out.size(); ++i) {
        out[i] = half_cast<half>(random_data_f[i%chunk]);
        //out[i] = half_cast<half>(i); //TMP matmul only  <---------test

        // out[i] = half_cast<half>(0); //TMP matmul only
                                     //
        // out[i] = half_cast<half>( (i/32) + (i%32) ); //TMP matmul only

        // out[i] = half_cast<half>( (i/64) + (i%64) ); //TMP matmul only
        // out[i] = half_cast<half>( (i/33) + (i%33) ); //TMP matmul only
    }

    //out[0] = half_cast<half>(1); //TMP matmul only
    //out[65] = half_cast<half>(1); //TMP matmul only

    //for (size_t i = 0; i < out.size(); i += nrand) {
    //    size_t chunk = std::min(nrand, out.size() - i);
    //    std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    //}
}

void fill_const(std::vector<half> &out, const float c) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;
    const unsigned seed = 2;

    if (random_data_f.empty()) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

   size_t chunk = std::min(nrand, out.size());
   for(int i=0; i<out.size(); ++i) {
        out[i] = half_cast<half>(c); //TMP matmul only
   }
   //for (size_t i = 0; i < out.size(); i += nrand) {
   //    size_t chunk = std::min(nrand, out.size() - i);
   //    std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
   //}
}

void fill_hceye(std::vector<half> &out, int ldi=32) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

   for(int i=0; i<out.size(); ++i) {
        //out[i] = half_cast<half>( (i/33) == (i%33) ? 1.f : 0.f); //TMP matmul only
        //
        //out[i] = half_cast<half>( (i/32)%32 == (i%32) ? 1.f : 0.f); //TMP matmul only

        out[i] = half_cast<half>((( (i/ldi)%32 == (i%32))) ? 1.f : 0.f); //TMP matmul only
        //out[i] = half_cast<half>((((i/ldi)%32  == ((i+2)%32)) || ( (i/ldi) == (i%32))) ? 1.f : 0.f); //TMP matmul only
        //out[i] = half_cast<half>((((i/ldi)  == ((i+2)%ldi)) || ( (i/ldi) == (i%ldi))) ? 1.f : 0.f); //TMP matmul only
        //out[i] = half_cast<half>((( (i/ldi) == (i%ldi))) ? 1.f : 0.f); //TMP matmul only

        /*
        if((i/ldi == i % ldi) && i/ldi == 1) {
            out[i] = half_cast<half>(-1); //TMP matmul only
        }
        */

        //
        //out[i] = half_cast<half>( (i/64) == (i%64) ? 1.f : 0.f); //TMP matmul only
        //out[i] = 1.f;
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

void print_test_case(memory::data_type dt, const mlp_dims_t &p) {
    std::cout << '[' << std::setw(4) << dnnl_dt2str(memory::convert_to_c(dt));
    std::cout << " mb = " << p.mb << ", ic = " << p.ic << ", oc = " << p.oc;
    std::cout << "] " << std::flush;
}

template<typename T>
void bench_gated_mlp_primitives(std::vector<T> &res, engine::kind ekind, memory::data_type dt,
        const mlp_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    // Create execution dnnl::engine.
    dnnl::engine eng(ekind, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Prepare input and output shapes to construct the swiglu graph.
    const memory::dims O_proj_sz = {p.mb, p.ic};
    //const memory::dims W_gate_sz  = {p.oc, p.ic};
    const memory::dims W_gate_sz
            = {p.ic, p.oc}; // .T() transposed? does this match OV?
    //const memory::dims W_up_sz    = {p.oc, p.ic}; // .T()
    const memory::dims W_up_sz = {p.ic, p.oc};
    //const memory::dims W_down_sz  = {p.ic, p.oc};
    const memory::dims W_down_sz = {p.oc, p.ic}; // .T()
    const memory::dims FC_gate_sz = {p.mb, p.oc};
    const memory::dims FC_up_sz = {p.mb, p.oc};
    const memory::dims FC_down_sz = {p.mb, p.ic};
    const memory::dims scale_sz = {1, 1};

    // All combined in a single matmul primitive.
    auto O_proj_md
            = memory::desc(O_proj_sz, dt, tag::ab); //TODO: layout? ba ab ?
    auto W_gate_md = memory::desc(W_gate_sz, dt, tag::ab);
    auto W_up_md = memory::desc(W_up_sz, dt, tag::ab);
    auto W_down_md = memory::desc(W_down_sz, dt, tag::ab);
    auto FC_gate_md = memory::desc(FC_gate_sz, dt, tag::ab);
    auto FC_up_md = memory::desc(FC_up_sz, dt, tag::ab);
    auto FC_down_md = memory::desc(FC_down_sz, dt, tag::ab);
    auto scale_md = memory::desc(scale_sz, dt, tag::ab);

    // fc_up
    primitive_attr bmm0_attr;
    //bmm0_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto bmm0_pd = matmul::primitive_desc(
            eng, O_proj_md, W_up_md, FC_up_md, bmm0_attr);
    auto bmm0_prim = matmul(bmm0_pd);

    // fc_gate -> swish -> mul
    primitive_attr bmm1_attr;
    //bmm1_attr.set_scratchpad_mode(scratchpad_mode::user); // TODO: needed? no threading in this example...
    post_ops bmm1_po;
    bmm1_po.append_eltwise(algorithm::eltwise_swish, 1.f, 1.f);
    bmm1_po.append_binary(algorithm::binary_mul, FC_up_md);
    bmm1_attr.set_post_ops(bmm1_po);

    auto bmm1_pd = matmul::primitive_desc(
            eng, O_proj_md, W_gate_md, FC_gate_md, bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    primitive_attr bmm2_attr;
    //bmm2_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto bmm2_pd = matmul::primitive_desc(
            eng, FC_gate_md, W_down_md, FC_down_md, bmm2_attr);
    auto bmm2_prim = matmul(bmm2_pd);

    // Create memory objects
    auto m_O_proj = memory(O_proj_md, eng);
    auto m_W_gate = memory(W_gate_md, eng);
    auto m_W_up = memory(W_up_md, eng);
    auto m_W_down = memory(W_down_md, eng);
    auto m_FC_gate = memory(FC_gate_md, eng);
    auto m_FC_up = memory(FC_up_md, eng);
    auto m_FC_down = memory(FC_down_md, eng);
    auto m_scale = memory(scale_md, eng);

    // Allocate user data.
    std::vector<T> O_proj_data(product(O_proj_sz));
    std::vector<T> W_gate_data(product(W_gate_sz));
    std::vector<T> W_up_data(product(W_up_sz));
    std::vector<T> W_down_data(product(W_down_sz));
    std::vector<T> FC_gate_data(product(FC_gate_sz));
    std::vector<T> FC_up_data(product(FC_up_sz));
    std::vector<T> FC_down_data(product(FC_down_sz));
    std::vector<T> scale_data(product(scale_sz), 1); //?? sz 1?

    fill_random(O_proj_data);
    //fill_const(O_proj_data, 0.01f); //TMP, matmul test only
    //fill_hceye(O_proj_data, p.ic); //TMP, matmul test only

    //
    fill_random(W_gate_data);
    //fill_hceye(W_gate_data, p.oc);
    //fill_hceye(W_gate_data);

    fill_const(W_up_data, 0.01f);
    //fill_random(W_down_data);
    fill_const(W_down_data, 0.1f);
    //fill_hceye(W_down_data);

    // Write data to tensor object's handle.
    write_to_dnnl_memory(O_proj_data.data(), m_O_proj);
    write_to_dnnl_memory(W_gate_data.data(), m_W_gate);
    write_to_dnnl_memory(W_up_data.data(), m_W_up);
    write_to_dnnl_memory(W_down_data.data(), m_W_down);
    write_to_dnnl_memory(scale_data.data(), m_scale);

    /*
    size_t max_scratchpad_size = 0;
    auto bmm1_scratchpad = bmm1_pd.scratchpad_desc().get_size();
    auto softmax_scratchpad = softmax_pd.scratchpad_desc().get_size();
    auto bmm2_scratchpad = bmm2_pd.scratchpad_desc().get_size();
    for (auto &sz : {bmm1_scratchpad, softmax_scratchpad, bmm2_scratchpad}) {
        if (max_scratchpad_size < sz) max_scratchpad_size = sz;
    }
    auto scratchpad_md
            = memory::desc({static_cast<memory::dim>(max_scratchpad_size)},
                    memory::data_type::u8, tag::a);

    // allocate intermediate memory
    auto m_score = memory(score_md, eng);
    auto m_scratchpad = memory(scratchpad_md, eng);
    */

    ////////////////TMP transpose for test
    //const memory::dims FC_gate_sz_t = {p.oc, p.mb};
    const memory::dims FC_gate_sz_t = {p.mb, p.oc};
    auto FC_gate_md_t = memory::desc(FC_gate_sz_t, dt, tag::ba);
    auto m_FC_gate_t = memory(FC_gate_md_t, eng);

    primitive_attr reorder_attr;
    auto reorder_pd = reorder::primitive_desc(
        eng, FC_gate_md, eng, FC_gate_md_t, reorder_attr);

    auto reorder_prim = reorder(reorder_pd);

    std::unordered_map<int, memory> reorder_args;
    reorder_args.insert({DNNL_ARG_SRC, m_FC_gate});
    reorder_args.insert({DNNL_ARG_DST, m_FC_gate_t});
    ///////////////////


    const auto loop = [&]() {
        ///////////////////
        //TMP!!!! 
// test first MM only
#if 0
        bmm0_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_gate},
                        {DNNL_ARG_DST, m_FC_gate}});

        reorder_prim.execute(strm, reorder_args);

//// TMP!!! mm + swish + elwise_mul
#elif 1
        bmm0_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_up},
                        {DNNL_ARG_DST, m_FC_up}});

        // each primitive will use all threads
        bmm1_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_gate},
                        {DNNL_ARG_DST, m_FC_gate},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                m_FC_up}});
        reorder_prim.execute(strm, reorder_args);

#else //TEST full gated mlp
        bmm0_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_up},
                        {DNNL_ARG_DST, m_FC_up}});

        // each primitive will use all threads
        bmm1_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_gate},
                        {DNNL_ARG_DST, m_FC_gate},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                m_FC_up}});

        bmm2_prim.execute(strm,
                {{DNNL_ARG_SRC, m_FC_gate}, {DNNL_ARG_WEIGHTS, m_W_down},
                        {DNNL_ARG_DST, m_FC_down}});
#endif
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop();

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    loop();
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    runs = 5;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
        loop();
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "primitive runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;

    //TODO ON MONDAY:/ y is output wrong??? repeats? copy?

    std::vector<T> a, b;
    a.resize(product(O_proj_sz));
    b.resize(product(W_gate_sz));
    read_from_dnnl_memory(a.data(), m_O_proj);
    read_from_dnnl_memory(b.data(), m_W_gate);

    res.resize(product(FC_down_sz));
    //std::vector<float> res(product(FC_down_sz));
    read_from_dnnl_memory(res.data(), m_FC_down);


    {
        //TMPPPPP
        res.resize(product(FC_gate_sz));
        //read_from_dnnl_memory(res.data(), m_FC_gate);
        read_from_dnnl_memory(res.data(), m_FC_gate_t); //transpose test
        //read_from_dnnl_memory(res.data(), m_FC_gate); //untranspose test
    }

    //if (product(FC_down_sz) < 1025) {
    if (product(FC_down_sz) < (64*64)+1) {
        const memory::dims FC_down_sz = {p.mb, p.ic};

        printf("\ninpA----------[%d %d]\n", p.mb, p.ic);
        //for (int y = 0; y < 33; ++y) {
            //for (int x = 0; x < 33; ++x) {
        for (int y = 0; y < p.mb; ++y) {
            for (int x = 0; x < p.ic; ++x) {
                if constexpr(std::is_same<half, T>::value) {
                    printf("%5.1f ", half_cast<float>(a[y * p.ic + x]));
                } else {
                    printf("%f ", a[y * p.ic + x]);
                }
            }
            printf("\n");
        }
        printf("inpB----------[%d %d]\n", p.ic, p.oc);
        for (int y = 0; y < p.ic; ++y) {
            for (int x = 0; x < p.oc; ++x) {
                if constexpr(std::is_same<half, T>::value) {
                    printf("%5.1f ", half_cast<float>(b[y * p.oc + x]));
                } else {
                    printf("%f ", b[y * p.oc + x]);
                }
            }
            printf("\n");
        }
        printf("----------\n");


        printf("resprim----------[%d %d]\n", p.mb, p.ic);
        printf("TMPres----------[%d %d]\n", p.mb, p.oc);
        //for (int y = 0; y < p.mb; ++y) {
            //for (int x = 0; x < p.ic; ++x) {
        for (int y = 0; y < p.oc; ++y) {
            for (int x = 0; x < p.mb; ++x) {
        //for (int y = 0; y < p.mb; ++y) {
            //for (int x = 0; x < p.oc; ++x) {
                if constexpr(std::is_same<half, T>::value) {
                    //printf("%5.1f ", half_cast<float>(res[y * p.oc + x]));
                    printf("%5.1f ", half_cast<float>(res[y * p.mb + x]));
                } else {
                    printf("%f ", res[y * p.oc + x]);
                }
            }
            printf("\n");
        }
        printf("----------\n");
    }
}

template<typename T>
void bench_gated_mlp_internal(std::vector<T> &res, engine::kind ekind, memory::data_type dt,
        const mlp_dims_t &p, double time_limit = 0.) {
    printf("eng?\n");
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    printf("eng?\n");
    // Create execution dnnl::engine.
    dnnl::engine eng(ekind, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);
    printf("enginit?\n");

    // Prepare input and output shapes to construct the swiglu graph.
    //const memory::dims O_proj_sz = {p.mb, p.ic};
    const memory::dims O_proj_sz = {p.mb, p.ic};
    //const memory::dims W_gate_sz  = {p.oc, p.ic};
    const memory::dims W_gate_sz
            = {p.ic, p.oc}; // .T() transposed? does this match OV?
    //const memory::dims W_up_sz    = {p.oc, p.ic}; // .T()
    const memory::dims W_up_sz = {p.ic, p.oc};
    //const memory::dims W_down_sz  = {p.ic, p.oc};
    const memory::dims W_down_sz = {p.oc, p.ic}; // .T()
    //const memory::dims FC_gate_sz = {p.mb, p.oc};
    const memory::dims FC_gate_sz = {p.oc, p.mb};
    const memory::dims FC_up_sz = {p.mb, p.oc};
    const memory::dims FC_down_sz = {p.mb, p.ic};
    const memory::dims scale_sz = {1, 1};

    // All combined in a single matmul primitive.
    auto O_proj_md
            = memory::desc(O_proj_sz, dt, tag::ab); //TODO: layout? ba ab ?
    auto W_gate_md = memory::desc(W_gate_sz, dt, tag::ab);
    auto W_up_md = memory::desc(W_up_sz, dt, tag::ab);
    auto W_down_md = memory::desc(W_down_sz, dt, tag::ab);
    auto FC_gate_md = memory::desc(FC_gate_sz, dt, tag::ab);
    auto FC_up_md = memory::desc(FC_up_sz, dt, tag::ab);
    auto FC_down_md = memory::desc(FC_down_sz, dt, tag::ab);
    auto scale_md = memory::desc(scale_sz, dt, tag::ab);


    // Create memory objects
    auto m_O_proj = memory(O_proj_md, eng);
    auto m_W_gate = memory(W_gate_md, eng);
    auto m_W_up = memory(W_up_md, eng);
    auto m_W_down = memory(W_down_md, eng);
    auto m_FC_gate = memory(FC_gate_md, eng);
    auto m_FC_up = memory(FC_up_md, eng);
    auto m_FC_down = memory(FC_down_md, eng);
    auto m_scale = memory(scale_md, eng);

    // Allocate user data.
    std::vector<T> O_proj_data(product(O_proj_sz));
    std::vector<T> W_gate_data(product(W_gate_sz));
    std::vector<T> W_up_data(product(W_up_sz));
    std::vector<T> W_down_data(product(W_down_sz));
    std::vector<T> FC_gate_data(product(FC_gate_sz));
    std::vector<T> FC_up_data(product(FC_up_sz));
    std::vector<T> FC_down_data(product(FC_down_sz));
    std::vector<T> scale_data(product(scale_sz), 1); //?? sz 1?

    fill_random(O_proj_data);
    //fill_const(O_proj_data, 0.01f); //TMP, matmul test only
    //fill_hceye(O_proj_data, p.ic); //TMP, matmul test only

    fill_random(W_gate_data);
    //fill_hceye(W_gate_data);
    //fill_hceye(W_gate_data, p.oc);

    fill_const(W_up_data, 0.01f);
    // fill_random(W_down_data);
    //fill_hceye(W_down_data);
    fill_const(W_down_data, 0.1f);
    std::fill(FC_down_data.begin(), FC_down_data.end(), 0.f);

    // Write data to tensor object's handle.
    write_to_dnnl_memory(O_proj_data.data(), m_O_proj);
    write_to_dnnl_memory(W_gate_data.data(), m_W_gate);
    write_to_dnnl_memory(W_up_data.data(), m_W_up);
    write_to_dnnl_memory(W_down_data.data(), m_W_down);
    write_to_dnnl_memory(scale_data.data(), m_scale);
    write_to_dnnl_memory(FC_down_data.data(), m_FC_down);

    //primitive_attr bmm0_attr;
    //bmm0_attr.set_scratchpad_mode(scratchpad_mode::user);
    //auto bmm0_pd = matmul::primitive_desc(
            //eng, O_proj_md, W_up_md, FC_up_md, bmm0_attr);
    //auto prim_fused_internal = matmul(bmm0_pd);
    using namespace dnnl::experimental;
    //auto gmlp_pd = gmlp::primitive_desc(eng, O_proj_md, W_gate_md, W_up_md,
            //W_down_md, FC_down_md);
    auto gmlp_pd = gmlp::primitive_desc(eng, O_proj_md, W_gate_md, W_up_md,
            W_down_md, FC_gate_md); //TMP for mm test
    auto prim_fused_internal = gmlp(gmlp_pd);

    const auto loop = [&]() {
        prim_fused_internal.execute(strm,
                {{DNNL_ARG_SRC_0, m_O_proj},
                 {DNNL_ARG_SRC_1, m_W_gate},
                 {DNNL_ARG_SRC_2, m_W_up},
                 {DNNL_ARG_SRC_3, m_W_down},
                 //{DNNL_ARG_DST, m_FC_down}}); //CORRECT ARG
                 {DNNL_ARG_DST, m_FC_gate}});   // TMP ARG for mm test
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop();

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    //loop();
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    runs = 10;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
       loop();
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "internal gmlp primitive runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;

    std::vector<T> a, b;
    a.resize(product(O_proj_sz));
    b.resize(product(W_gate_sz));
    read_from_dnnl_memory(a.data(), m_O_proj);
    read_from_dnnl_memory(b.data(), m_W_gate);

    res.resize(product(FC_down_sz));
    read_from_dnnl_memory(res.data(), m_FC_down);

    {
        //TMPPPPP
        res.resize(product(FC_gate_sz));
        read_from_dnnl_memory(res.data(), m_FC_gate);
    }

    //if (product(FC_down_sz) < 1025) {
    if (product(FC_down_sz) < (64*64)+1) {

        printf("\ninpA----------[%d %d]\n", p.mb, p.ic);
        //for (int y = 0; y < 33; ++y) {
            //for (int x = 0; x < 33; ++x) {
        for (int y = 0; y < p.mb; ++y) {
            for (int x = 0; x < p.ic; ++x) {
                if constexpr(std::is_same<half, T>::value) {
                    printf("%5.1f ", half_cast<float>(a[y * p.ic + x]));
                } else {
                    printf("%f ", a[y * p.ic + x]);
                }
            }
            printf("\n");
        }
        printf("inpB----------[%d %d]\n", p.ic, p.oc);
        for (int y = 0; y < p.ic; ++y) {
            for (int x = 0; x < p.oc; ++x) {
                if constexpr(std::is_same<half, T>::value) {
                    printf("%5.1f ", half_cast<float>(b[y * p.oc + x]));
                } else {
                    printf("%f ", b[y * p.oc + x]);
                }
            }
            printf("\n");
        }
        printf("----------\n");



        //std::vector<float> res(product(FC_down_sz));

        printf("resint----------[%d %d]\n", p.mb, p.ic);
        printf("TMPres----------[%d %d]\n", p.mb, p.oc);

        const memory::dims FC_down_sz = {p.mb, p.ic};
        printf("\n");
        for (int y = 0; y < p.oc; ++y) {
            for (int x = 0; x < p.mb; ++x) {

        //for (int y = 0; y < p.mb; ++y) {
            //for (int x = 0; x < p.oc; ++x) {

        //for (int y = 0; y < p.mb; ++y) {
            //for (int x = 0; x < p.ic; ++x) {
                if constexpr(std::is_same<half, T>::value) {
                    //printf("%5.1f ", half_cast<float>(res[y * p.oc + x]));
                    printf("%5.1f ", half_cast<float>(res[y * p.mb + x]));
                }else{
                    //printf("%f ", res[y * p.mb + x]);
                    printf("%f ", res[y * p.mb + x]);
                }
            }
            printf("\n");
        }
        printf("\n");
/*
*/
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

void print_test_case(logical_tensor::data_type dt, const mlp_dims_t &p) {
    std::cout << '[' << std::setw(4) << get_type_string(dt);
    std::cout << " mb = " << p.mb << ", ic = " << p.ic << ", oc = " << p.oc;
    std::cout << "] " << std::flush;
}

void bench_gated_mlp_graph(engine::kind ekind, logical_tensor::data_type dt,
        const mlp_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    allocator alloc = create_allocator(ekind);

    // Create execution dnnl::engine.
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // input shape
    const dims src_sz = {p.mb, p.ic};
    // weight0/weight1 shape: fc_gate and fc_up
    const dims wei0_sz = {p.ic, p.oc};
    // hidden shape
    const dims hd_sz = {p.mb, p.oc};
    // weight2 shape: fc_down
    const dims wei2_sz = {p.oc, p.ic};
    // output shape
    const dims out_sz = {p.mb, p.ic};

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    // fc_gate
    auto src = logical_tensor(id++, dt, src_sz, layout_type::strided);
    auto wei0 = logical_tensor(id++, dt, wei0_sz, layout_type::strided);
    auto out0 = logical_tensor(id++, dt, hd_sz, layout_type::strided);
    auto fc_gate = op(id++, op::kind::MatMul, "fc_gate");
    fc_gate.add_inputs({src, wei0});
    fc_gate.add_outputs({out0});

    // fc_up
    auto wei1 = logical_tensor(id++, dt, wei0_sz, layout_type::strided);
    auto out1 = logical_tensor(id++, dt, hd_sz, layout_type::strided);
    auto fc_up = op(id++, op::kind::MatMul, "fc_up");
    fc_up.add_inputs({src, wei1});
    fc_up.add_outputs({out1});

    // activation: swish
    auto out2 = logical_tensor(id++, dt, hd_sz, layout_type::strided);
    auto act = op(id++, op::kind::HardSwish, "swish");
    act.add_inputs({out0});
    act.add_outputs({out2});

    // multiplication
    auto out3 = logical_tensor(id++, dt, hd_sz, layout_type::strided);
    auto mul = op(id++, op::kind::Multiply, "mul");
    mul.add_inputs({out2, out1});
    mul.add_outputs({out3});

    // fc_down
    auto wei2 = logical_tensor(id++, dt, wei2_sz, layout_type::strided);
    auto dst = logical_tensor(id++, dt, out_sz, layout_type::strided);
    auto fc_down = op(id++, op::kind::MatMul, "fc_down");
    fc_down.add_inputs({out3, wei2});
    fc_down.add_outputs({dst});

    // Construct a gated mlp graph with engine kind and operations.
    dnnl::graph::graph mlp(ekind);
    mlp.add_op(fc_gate);
    mlp.add_op(fc_up);
    mlp.add_op(act);
    mlp.add_op(mul);
    mlp.add_op(fc_down);
    mlp.finalize();

    // Get partitions from the mlp graph.
    std::vector<partition> partitions = mlp.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "unsupported mlp" << std::endl;
        return;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp
            = partitions[0].compile({src, wei0, wei1, wei2}, {dst}, eng);

    // Create tensor objects
    auto ts_src = tensor(src, eng);
    auto ts_wei0 = tensor(wei0, eng);
    auto ts_wei1 = tensor(wei1, eng);
    auto ts_wei2 = tensor(wei2, eng);
    auto ts_dst = tensor(dst, eng);

    // Allocate user data.
    std::vector<float> src_data(product(src_sz));
    std::vector<float> wei0_data(product(wei0_sz));
    std::vector<float> wei1_data(product(wei0_sz));
    std::vector<float> wei2_data(product(wei2_sz));

    fill_random(src_data);
    fill_random(wei0_data);
    fill_random(wei1_data);
    fill_random(wei2_data);

    // Write data to tensor object's handle.
    write_to_dnnl_tensor(src_data.data(), ts_src);
    write_to_dnnl_tensor(wei0_data.data(), ts_wei0);
    write_to_dnnl_tensor(wei1_data.data(), ts_wei1);
    write_to_dnnl_tensor(wei2_data.data(), ts_wei2);

    // Warmup run.
    // Execute the compiled partition of mqa.
    cp.execute(strm, {ts_src, ts_wei0, ts_wei1, ts_wei2}, {ts_dst});

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    cp.execute(strm, {ts_src, ts_wei0, ts_wei1, ts_wei2}, {ts_dst});
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    runs = 10;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
        cp.execute(strm, {ts_src, ts_wei0, ts_wei1, ts_wei2}, {ts_dst});
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "graph runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;

    /*
    if (product(out_sz) < 128) {
        std::vector<float> res(product(out_sz));
        read_from_dnnl_tensor(res.data(), ts_dst);

        for (int y = 0; y < p.mb; ++y) {
            for (int x = 0; x < p.ic; ++x) {
                printf("%f ", res[y * p.ic + x]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */
}

void bad_args() {
    std::cerr << "Usage: graph-gated-mlp-cpp [cpu|gpu]\n"
                 "       graph-gated-mlp-cpp [cpu|gpu] <mb> <ic> <oc>\n\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

enum class api_kind {
    primitive,
    graph,
    internal_hack
};

template<typename T>
void bench(std::vector<T> &res, api_kind api, engine::kind ekind, dnnl_data_type_t dt,
        const mlp_dims_t &p, double time_limit = 0.) {
    try {
        if (api == api_kind::primitive) {
            bench_gated_mlp_primitives(res,
                    ekind, static_cast<memory::data_type>(dt), p, time_limit);
        } else if (api == api_kind::graph) {
            //bench_gated_mlp_graph(ekind, static_cast<logical_tensor::data_type>(dt),
                    //p, time_limit);
            //get_mem_pool().clear();
        } else {
            bench_gated_mlp_internal(res,
                    ekind, static_cast<memory::data_type>(dt), p, time_limit);
        }
    } catch (dnnl::error &e) {
        // Catch and report unimplemented cases.
        if (e.status == dnnl_unimplemented) {
            std::cout << "unsupported mlp" << std::endl;
        } else
            throw;
    }
}

void mlp_perf(engine::kind ekind, int argc, char **argv) {
    // default testing parameters
    mlp_dims_t params = {8, 4096, 14336};

    if (argc > 2) {
        if (argc == 5) { // 6? which ones? TODO: asktao
            params.mb = std::atoi(argv[2]);
            params.ic = std::atoi(argv[3]);
            params.oc = std::atoi(argv[4]);
        } else {
            bad_args();
        }

        if (params.mb <= 0 || params.ic <= 0 || params.oc <= 0) { bad_args(); }
    }

    //bench(ekind, dnnl_f32, params, 2000.0 /*ms*/);
    //bench(ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    //bench(ekind, dnnl_f16, params, 2000.0 /*ms*/);

    //TODO: merge w/existing graph PR
    //bench(api_kind::graph, ekind, dnnl_f32, params, 2000.0 /*ms*/);
    //bench(api_kind::graph, ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    //bench(api_kind::graph, ekind, dnnl_f16, params, 2000.0 /*ms*/);
    //bench(api_kind::graph, ekind, dnnl_f32, params, 2000.0 /*ms*/);

    //printf("GRAPH\n");
    //bench(api_kind::graph, ekind, dnnl_f32, params, 2000.0 /*ms*/);
    std::vector<float> resp, resi;
    std::vector<half> resph, resih;

    printf("PRIMITIVE\n");
//   //bench(resp, api_kind::primitive, ekind, dnnl_f32, params, 2000.0 /*ms*/);
    bench(resph, api_kind::primitive, ekind, dnnl_f16, params, 2000.0 /*ms*/);

    printf("INTERNAL\n");
    //bench(resi, api_kind::internal_hack, ekind, dnnl_f32, params, 2000.0 /*ms*/);
    bench(resih, api_kind::internal_hack, ekind, dnnl_f16, params, 2000.0 /*ms*/);

    /*
    if(resi.size() == 0) printf("[WARNING] Empty output! internal kernel fail X_X\n");
    for(int i=0; i<resi.size(); ++i) {
        if(std::abs((resi[i] - resp[i]) / resi[i]) > 1e-2) printf("mismatch @ %d, %f != %f\n", i, resi[i], resp[i]); //TODO: improve
    }
    */

    if(resih.size() == 0) printf("[WARNING] Empty output! internal kernel fail X_X\n");
    int n_mismatches=0, n_matches=0;
    printf("resih.size() %zu\n", resih.size());
    for(int i=0; i<resih.size(); ++i) {
        if(std::abs((resih[i] - resph[i]) / resih[i]) > 5e-2) {
            n_mismatches++;
            if(n_mismatches < 10)
                printf("mismatch @ %d, %f != %f\n", i, half_cast<float>(resih[i]), half_cast<float>(resph[i])); //TODO: improve
        } else {
            if(std::abs(half_cast<float>(resih[i])) > 5e-2) {
                n_matches++;
                if(n_matches < 10)
                    printf("vs @ %d, %f == %f\n", i, half_cast<float>(resih[i]), half_cast<float>(resph[i])); //TODO: improve
            }
        }
    }
    printf("total mismatches: %d \n", n_mismatches);
    //bench(api_kind::primitive, ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    //bench(api_kind::primitive, ekind, dnnl_f16, params, 2000.0 /*ms*/);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            mlp_perf, parse_engine_kind(argc, argv, 4), argc, argv);
}
