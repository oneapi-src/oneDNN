/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void instance_normalization_example(
        dnnl::engine::kind engine_kind, int n, int c, int d, int h, int w) {

    int warmup_iter = 10;
    int n_iter = 100;

    /// Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Source (src) and destination (dst) tensors dimensions.
    const memory::dims src_dims = {n, c, d, h, w};

    // Scale/shift tensor dimensions.
    memory::dims scaleshift_dims = {d * h * w};

    // Scale/shift tensor dimensions.
    memory::dims meanvariance_dims = {n * c};

    // Allocate buffer.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> scale_data(product(scaleshift_dims));
    std::vector<float> shift_data(product(scaleshift_dims));

    // Initialize src tensor.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Initialize scale.
    std::generate(scale_data.begin(), scale_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });

    // Initialize shift.
    std::generate(shift_data.begin(), shift_data.end(), []() {
        static int i = 0;
        return std::tan(float(i++));
    });

    // Create src memory descriptor and memory objects.
    auto inorm_src_md = memory::desc(src_dims, dt::f32, tag::ncdhw);
    auto inorm_dst_md = memory::desc(src_dims, dt::f32, tag::ncdhw);
    auto inorm_meanvariance_md
            = memory::desc(meanvariance_dims, dt::f32, tag::x);

    auto inorm_src_mem = memory(inorm_src_md, engine);
    auto inorm_mean_mem = memory(inorm_meanvariance_md, engine);
    auto inorm_variance_mem = memory(inorm_meanvariance_md, engine);
    auto inorm_dst_mem = memory(inorm_dst_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), inorm_src_mem);

    // Create primitive descriptor.
    const float epsilon = 1.e-10f;

    double avg_time_lnorm {0.0f}, avg_time_bnorm_loop {0.0f},
            avg_time_bnorm_join {0.0f};

    // layer normalization
    {
        auto lnorm_src_md = inorm_src_md.reshape({n, c, d * h * w});
        auto lnorm_dst_md = inorm_dst_md.reshape({n, c, d * h * w});
        auto lnorm_pd = layer_normalization_forward::primitive_desc(engine,
                prop_kind::forward_training, lnorm_src_md, lnorm_dst_md,
                epsilon, normalization_flags::none);
        auto lnorm_src_mem
                = memory(lnorm_src_md, engine, inorm_src_mem.get_data_handle());
        auto lnorm_dst_mem
                = memory(lnorm_dst_md, engine, inorm_dst_mem.get_data_handle());

        auto lnorm_prim = layer_normalization_forward(lnorm_pd);

        std::unordered_map<int, memory> lnorm_args;
        lnorm_args.insert({DNNL_ARG_SRC, lnorm_src_mem});
        lnorm_args.insert({DNNL_ARG_MEAN, inorm_mean_mem});
        lnorm_args.insert({DNNL_ARG_VARIANCE, inorm_variance_mem});
        lnorm_args.insert({DNNL_ARG_DST, lnorm_dst_mem});

        for (int i = 0; i < warmup_iter; ++i)
            lnorm_prim.execute(engine_stream, lnorm_args);
        // Wait for the computation to finalize.
        engine_stream.wait();

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < n_iter; ++i)
            lnorm_prim.execute(engine_stream, lnorm_args);
        // Wait for the computation to finalize.
        engine_stream.wait();

        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration = end - start;
        avg_time_lnorm = duration.count() / n_iter;

        // Read data from memory object's handle.s
        read_from_dnnl_memory(src_data.data(), inorm_src_mem);
    }

    // batch normalization
    {
        memory::desc bnorm_src_md
                = inorm_src_md.submemory_desc({1, c, d, h, w}, {0, 0, 0, 0, 0});
        memory::desc bnorm_dst_md
                = inorm_dst_md.submemory_desc({1, c, d, h, w}, {0, 0, 0, 0, 0});
        auto bnorm_pd = batch_normalization_forward::primitive_desc(engine,
                prop_kind::forward_training, bnorm_src_md, bnorm_dst_md,
                epsilon, normalization_flags::none);
        auto bnorm_prim = batch_normalization_forward(bnorm_pd);

        auto bnorm_src_mem
                = memory(bnorm_src_md, engine, inorm_src_mem.get_data_handle());
        auto bnorm_dst_mem
                = memory(bnorm_dst_md, engine, inorm_dst_mem.get_data_handle());
        auto bnorm_mean_mem = memory(inorm_meanvariance_md, engine,
                inorm_mean_mem.get_data_handle());
        auto bnorm_variance_mem = memory(inorm_meanvariance_md, engine,
                inorm_variance_mem.get_data_handle());

        std::unordered_map<int, memory> bnorm_args;
        bnorm_args.insert({DNNL_ARG_SRC, bnorm_src_mem});
        bnorm_args.insert({DNNL_ARG_MEAN, inorm_mean_mem});
        bnorm_args.insert({DNNL_ARG_VARIANCE, inorm_variance_mem});
        bnorm_args.insert({DNNL_ARG_DST, bnorm_dst_mem});

        const size_t elems_per_batch = c * d * h * w;

        for (int i = 0; i < warmup_iter; ++i) {
            for (int ni = 0; ni < n; ++ni) {
                bnorm_prim.execute(engine_stream, bnorm_args);
                bnorm_src_mem.set_data_handle(static_cast<void *>(
                        static_cast<float *>(bnorm_src_mem.get_data_handle())
                        + elems_per_batch));
                bnorm_dst_mem.set_data_handle(static_cast<void *>(
                        static_cast<float *>(bnorm_dst_mem.get_data_handle())
                        + elems_per_batch));
                bnorm_mean_mem.set_data_handle(static_cast<void *>(
                        static_cast<float *>(bnorm_mean_mem.get_data_handle())
                        + c));
                bnorm_variance_mem.set_data_handle(static_cast<void *>(
                        static_cast<float *>(
                                bnorm_variance_mem.get_data_handle())
                        + c));
            }

            bnorm_src_mem.set_data_handle(inorm_src_mem.get_data_handle());
            bnorm_dst_mem.set_data_handle(inorm_dst_mem.get_data_handle());
            bnorm_mean_mem.set_data_handle(inorm_mean_mem.get_data_handle());
            bnorm_variance_mem.set_data_handle(
                    inorm_variance_mem.get_data_handle());
        }

        // Wait for the computation to finalize.
        engine_stream.wait();

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < n_iter; ++i) {
            for (int ni = 0; ni < n; ++ni) {
                bnorm_prim.execute(engine_stream, bnorm_args);
                bnorm_src_mem.set_data_handle(static_cast<void *>(
                        static_cast<float *>(bnorm_src_mem.get_data_handle())
                        + elems_per_batch));
                bnorm_dst_mem.set_data_handle(static_cast<void *>(
                        static_cast<float *>(bnorm_dst_mem.get_data_handle())
                        + elems_per_batch));
                bnorm_mean_mem.set_data_handle(static_cast<void *>(
                        static_cast<float *>(bnorm_mean_mem.get_data_handle())
                        + c));
                bnorm_variance_mem.set_data_handle(static_cast<void *>(
                        static_cast<float *>(
                                bnorm_variance_mem.get_data_handle())
                        + c));
            }

            bnorm_src_mem.set_data_handle(inorm_src_mem.get_data_handle());
            bnorm_dst_mem.set_data_handle(inorm_dst_mem.get_data_handle());
            bnorm_mean_mem.set_data_handle(inorm_mean_mem.get_data_handle());
            bnorm_variance_mem.set_data_handle(
                    inorm_variance_mem.get_data_handle());
        }

        // Wait for the computation to finalize.
        engine_stream.wait();

        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration = end - start;
        avg_time_bnorm_loop = duration.count() / n_iter;

        // Read data from memory object's handle.s
        read_from_dnnl_memory(src_data.data(), inorm_src_mem);
    }

    // batch normalization v2
    {
        memory::desc bnorm_src_md = inorm_src_md.reshape({1, n * c, d, h, w});
        memory::desc bnorm_dst_md = inorm_dst_md.reshape({1, n * c, d, h, w});
        auto bnorm_pd = batch_normalization_forward::primitive_desc(engine,
                prop_kind::forward_training, bnorm_src_md, bnorm_dst_md,
                epsilon, normalization_flags::none);
        auto bnorm_prim = batch_normalization_forward(bnorm_pd);

        auto bnorm_src_mem
                = memory(bnorm_src_md, engine, inorm_src_mem.get_data_handle());
        auto bnorm_dst_mem
                = memory(bnorm_dst_md, engine, inorm_dst_mem.get_data_handle());

        std::unordered_map<int, memory> bnorm_args;
        bnorm_args.insert({DNNL_ARG_SRC, bnorm_src_mem});
        bnorm_args.insert({DNNL_ARG_MEAN, inorm_mean_mem});
        bnorm_args.insert({DNNL_ARG_VARIANCE, inorm_variance_mem});
        bnorm_args.insert({DNNL_ARG_DST, bnorm_dst_mem});

        // warm up
        for (int i = 0; i < warmup_iter; ++i) {
            bnorm_prim.execute(engine_stream, bnorm_args);
        }

        // Wait for the computation to finalize.
        engine_stream.wait();

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < n_iter; ++i) {
            bnorm_prim.execute(engine_stream, bnorm_args);
        }

        // Wait for the computation to finalize.
        engine_stream.wait();

        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration = end - start;
        avg_time_bnorm_join = duration.count() / n_iter;

        // Read data from memory object's handle.s
        read_from_dnnl_memory(src_data.data(), inorm_src_mem);
    }

    printf("%dx%dx%dx%dx%d time: lnorm: %f bnorm_loop: %f bnorm_join: %f\n", n,
            c, d, h, w, avg_time_lnorm, avg_time_bnorm_loop,
            avg_time_bnorm_join);
}

int main(int argc, char **argv) {
    const auto engine_kind = parse_engine_kind(argc, argv);
    instance_normalization_example(engine_kind, 6, 32, 160, 224, 224);
    instance_normalization_example(engine_kind, 6, 256, 20, 28, 28);
    instance_normalization_example(engine_kind, 6, 320, 10, 14, 14);
    instance_normalization_example(engine_kind, 6, 128, 6, 7, 7);
    return 0;
}
