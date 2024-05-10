/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#define TEST_DNNL_DPCPP_BUFFER

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include "xpu/sycl/compat.hpp"

#include <algorithm>
#include <memory>
#include <vector>

using namespace sycl;

class neg_sign_kernel;
class init_kernel;

namespace dnnl {

class sycl_memory_buffer_test : public ::testing::TestWithParam<engine::kind> {
};

TEST_P(sycl_memory_buffer_test, BasicInteropCtor) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dims tz = {4, 4, 4, 4};

    size_t sz = size_t(tz[0]) * tz[1] * tz[2] * tz[3];

    buffer<float, 1> buf {range<1>(sz)};

    memory::desc mem_d(tz, memory::data_type::f32, memory::format_tag::nchw);

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (eng_kind == engine::kind::cpu) {
        auto mem = test::make_memory(mem_d, eng);
        EXPECT_ANY_THROW(sycl_interop::make_memory(mem_d, eng, buf));
        EXPECT_ANY_THROW(sycl_interop::get_buffer<float>(mem));
        return;
    }
#endif

    auto mem = sycl_interop::make_memory(mem_d, eng, buf);
    auto buf_from_mem = sycl_interop::get_buffer<float>(mem);

    {
        auto a = buf.get_host_access();
        for (size_t i = 0; i < sz; ++i)
            a[i] = float(i);
    }

    {
        auto a = buf_from_mem.get_host_access();
        for (size_t i = 0; i < sz; ++i)
            ASSERT_EQ(a[i], float(i));
    }
}

TEST_P(sycl_memory_buffer_test, ConstructorNone) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::desc mem_d({0}, memory::data_type::f32, memory::format_tag::x);

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (eng_kind == engine::kind::cpu) {
        auto mem = test::make_memory(mem_d, eng);
        EXPECT_ANY_THROW(
                mem = sycl_interop::make_memory(mem_d, eng,
                        sycl_interop::memory_kind::buffer, DNNL_MEMORY_NONE));
        EXPECT_ANY_THROW(sycl_interop::get_buffer<float>(mem));
        return;
    }
#endif
    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::buffer, DNNL_MEMORY_NONE);

    auto buf = sycl_interop::get_buffer<float>(mem);
    (void)buf;
}

TEST_P(sycl_memory_buffer_test, ConstructorAllocate) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (eng_kind == engine::kind::cpu) {
        auto mem = test::make_memory(mem_d, eng);
        EXPECT_ANY_THROW(mem = sycl_interop::make_memory(mem_d, eng,
                                 sycl_interop::memory_kind::buffer,
                                 DNNL_MEMORY_ALLOCATE));
        EXPECT_ANY_THROW(sycl_interop::get_buffer<float>(mem));
        return;
    }
#endif
    auto mem = sycl_interop::make_memory(mem_d, eng,
            sycl_interop::memory_kind::buffer, DNNL_MEMORY_ALLOCATE);

    auto buf = sycl_interop::get_buffer<float>(mem);

    {
        auto acc = buf.get_host_access();
        for (int i = 0; i < n; i++) {
            acc[i] = float(i);
        }
    }

    float *mapped_ptr = mem.map_data<float>();
    GTEST_EXPECT_NE(mapped_ptr, nullptr);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(mapped_ptr[i], float(i));
    }
    mem.unmap_data(mapped_ptr);
}

TEST_P(sycl_memory_buffer_test, BasicInteropGetSet) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dims tz = {4, 4, 4, 4};

    size_t sz = size_t(tz[0]) * tz[1] * tz[2] * tz[3];
    memory::desc mem_d(tz, memory::data_type::f32, memory::format_tag::nchw);

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (eng_kind == engine::kind::cpu) {
        auto mem = test::make_memory(mem_d, eng);
        EXPECT_ANY_THROW(mem = sycl_interop::make_memory(mem_d, eng,
                                 sycl_interop::memory_kind::buffer));
        EXPECT_ANY_THROW(sycl_interop::get_buffer<float>(mem));
        return;
    }
#endif
    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::buffer);

    buffer<float, 1> interop_sycl_buf {range<1>(sz)};
    sycl_interop::set_buffer(mem, interop_sycl_buf);

    auto sycl_buf = sycl_interop::get_buffer<float>(mem);

    {
        auto a = interop_sycl_buf.get_host_access();
        for (size_t i = 0; i < sz; ++i)
            a[i] = float(i);
    }

    {
        auto a = sycl_buf.get_host_access();
        for (size_t i = 0; i < sz; ++i)
            ASSERT_EQ(a[i], float(i));
    }
}

TEST_P(sycl_memory_buffer_test, InteropReorder) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");
#ifdef DNNL_SYCL_HIP
    SKIP_IF(true,
            "Simple/sycl_memory_buffer_test.InteropReorder/gpu is skipped for "
            "HIP because of unimplemented Reorder");
#endif

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(eng_kind == engine::kind::cpu,
            "Skip this test for classic CPU runtime");
#endif

    const size_t N = 2;
    const size_t C = 3;
    const size_t H = 4;
    const size_t W = 5;

    size_t nelems = N * C * H * W;

    std::vector<float> src_host(nelems);
    std::vector<float> dst_host(nelems);
    std::iota(src_host.begin(), src_host.end(), 101.0f);

    {
        buffer<float, 1> src_buf(&src_host[0], range<1>(nelems));
        buffer<float, 1> dst_buf(&dst_host[0], range<1>(nelems));

        engine eng(eng_kind, 0);

        memory::dims tz = {int(N), int(C), int(H), int(W)};

        memory::desc src_mem_d(
                tz, memory::data_type::f32, memory::format_tag::nchw);
        auto src_mem = sycl_interop::make_memory(
                src_mem_d, eng, sycl_interop::memory_kind::buffer);

        memory::desc dst_mem_d(
                tz, memory::data_type::f32, memory::format_tag::nhwc);

        stream s(eng);
        auto dst_mem = sycl_interop::make_memory(
                dst_mem_d, eng, sycl_interop::memory_kind::buffer);

        sycl_interop::set_buffer(src_mem, src_buf);
        sycl_interop::set_buffer(dst_mem, dst_buf);

        reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
        s.wait();
    }

    // Assume that buffer destructors makes SYCL to wait on completion
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t nchw_index
                            = (n * C * H * W) + (c * H * W) + (h * W) + w;
                    size_t nhwc_index
                            = (n * H * W * C) + (h * W * C) + (w * C) + c;
                    EXPECT_EQ(src_host[nchw_index], dst_host[nhwc_index]);
                }
            }
        }
    }
}

TEST_P(sycl_memory_buffer_test, InteropReorderAndUserKernel) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

#ifdef DNNL_SYCL_HIP
    SKIP_IF(true,
            "Simple/sycl_memory_buffer_test.InteropReorderAndUserKernel/gpu is "
            "skipped for HIP because of unimplemented Reorder");
#endif

#ifdef DNNL_SYCL_CUDA
    SKIP_IF(eng_kind == engine::kind::gpu,
            "OpenCL features are not supported on CUDA backend");
#endif

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(eng_kind == engine::kind::cpu,
            "Skip this test for classic CPU runtime");
#endif

    const size_t N = 2;
    const size_t C = 3;
    const size_t H = 4;
    const size_t W = 5;

    size_t nelems = N * C * H * W;

    std::vector<float> data(nelems);
    std::iota(data.begin(), data.end(), 101.0f);

    std::vector<float> buf_host(data.begin(), data.end());
    std::vector<float> tmp_host(nelems);

    {
        buffer<float, 1> buf(&buf_host[0], range<1>(nelems));
        buffer<float, 1> tmp_buf(&tmp_host[0], range<1>(nelems));

        engine eng(eng_kind, 0);

        memory::dims tz = {int(N), int(C), int(H), int(W)};

        memory::desc mem_d(
                tz, memory::data_type::f32, memory::format_tag::nchw);
        auto mem = sycl_interop::make_memory(
                mem_d, eng, sycl_interop::memory_kind::buffer);

        memory::desc tmp_mem_d(
                tz, memory::data_type::f32, memory::format_tag::nhwc);

        stream s(eng);
        auto tmp_mem = sycl_interop::make_memory(
                tmp_mem_d, eng, sycl_interop::memory_kind::buffer);

        sycl_interop::set_buffer(mem, buf);
        sycl_interop::set_buffer(tmp_mem, tmp_buf);

        // Direct reorder mem -> tmp_mem
        reorder(mem, tmp_mem).execute(s, mem, tmp_mem);
        s.wait();

        // Invert the signs of the tmp elements
        auto q = sycl_interop::get_queue(s);
        q.submit([&](handler &cgh) {
            auto acc = buf.get_access<access::mode::write>(cgh);
            auto tmp_acc = tmp_buf.get_access<access::mode::read_write>(cgh);
            cgh.parallel_for<neg_sign_kernel>(range<1>(nelems), [=](item<1> i) {
                acc[i] = 0;
                tmp_acc[i] *= -1;
            });
        });

        // Back-reorder tmp -> mem
        reorder(tmp_mem, mem).execute(s, tmp_mem, mem);
        s.wait();
    }

    // Assume that buffer destructors makes SYCL to wait on completion
    for (size_t i = 0; i < nelems; i++) {
        EXPECT_EQ(buf_host[i], -data[i]);
    }
}

TEST_P(sycl_memory_buffer_test, EltwiseWithUserKernel) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

#ifdef DNNL_SYCL_HIP
    SKIP_IF(eng_kind == engine::kind::gpu,
            "OpenCL features are not supported on HIP backend");
#endif

#ifdef DNNL_SYCL_CUDA
    SKIP_IF(eng_kind == engine::kind::gpu,
            "OpenCL features are not supported on CUDA backend");
#endif

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(eng_kind == engine::kind::cpu,
            "Skip this test for classic CPU runtime");
#endif

    memory::dims tz = {2, 3, 4, 5};
    const int N = std::accumulate(tz.begin(), tz.end(), (memory::dim)1,
            std::multiplies<memory::dim>());

    memory::desc mem_d(tz, memory::data_type::f32, memory::format_tag::nchw);

    engine eng(eng_kind, 0);
    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::buffer);

    auto sycl_buf = sycl_interop::get_buffer<float>(mem);

    std::unique_ptr<queue> q;
    if (eng_kind == engine::kind::cpu) {
        q.reset(new queue(dnnl::impl::xpu::sycl::compat::cpu_selector_v));
    } else {
        q.reset(new queue(dnnl::impl::xpu::sycl::compat::gpu_selector_v));
    }

    q->submit([&](handler &cgh) {
        auto a = sycl_buf.get_access<access::mode::write>(cgh);
        cgh.parallel_for<init_kernel>(
                range<1>(N), [=](id<1> i) { a[i] = (int)i.get(0) - N / 2; });
    });

    auto eltwise_pd = eltwise_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::eltwise_relu, mem_d, mem_d, 0.0f);
    auto eltwise = eltwise_forward(eltwise_pd);

    stream s(eng);
    eltwise.execute(s, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
    s.wait();

    auto host_acc = sycl_buf.get_host_access();

    for (int i = 0; i < N; i++) {
        float exp_value = (i - N / 2) <= 0 ? 0 : (i - N / 2);
        EXPECT_EQ(host_acc[i], float(exp_value));
    }
}

namespace {
struct PrintToStringParamName {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        return to_string(static_cast<dnnl_engine_kind_t>(info.param));
    }
};
} // namespace

INSTANTIATE_TEST_SUITE_P(Simple, sycl_memory_buffer_test,
        ::testing::Values(engine::kind::cpu, engine::kind::gpu),
        PrintToStringParamName());

} // namespace dnnl
