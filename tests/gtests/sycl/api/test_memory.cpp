/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "dnnl.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <CL/sycl.hpp>

using namespace cl::sycl;

class neg_sign_kernel;
class init_kernel;

namespace dnnl {

class sycl_memory_test : public ::testing::TestWithParam<engine::kind> {};

TEST_P(sycl_memory_test, BasicInteropCtor) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dims tz = {4, 4, 4, 4};

    size_t sz = size_t(tz[0]) * tz[1] * tz[2] * tz[3];

    buffer<float, 1> buf {range<1>(sz)};

    memory::desc mem_d(tz, memory::data_type::f32, memory::format_tag::nchw);
    memory mem(mem_d, eng, buf);

    auto buf_from_mem = mem.get_sycl_buffer<float>();

    {
        auto a = buf.get_access<access::mode::write>();
        for (size_t i = 0; i < sz; ++i)
            a[i] = float(i);
    }

    {
        auto a = buf_from_mem.get_access<access::mode::read>();
        for (size_t i = 0; i < sz; ++i)
            ASSERT_EQ(a[i], float(i));
    }
}

TEST_P(sycl_memory_test, ConstructorNone) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::desc mem_d({0}, memory::data_type::f32, memory::format_tag::x);

    memory mem(mem_d, eng, DNNL_MEMORY_NONE);

    auto buf = mem.get_sycl_buffer<float>();
    (void)buf;
}

TEST_P(sycl_memory_test, ConstructorAllocate) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    memory mem(mem_d, eng, DNNL_MEMORY_ALLOCATE);

    auto buf = mem.get_sycl_buffer<float>();

    {
        auto acc = buf.get_access<access::mode::write>();
        for (int i = 0; i < n; i++) {
            acc[i] = float(i);
        }
    }

    float *mapped_ptr = mem.map_data<float>();
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(mapped_ptr[i], float(i));
    }
    mem.unmap_data(mapped_ptr);
}

TEST_P(sycl_memory_test, BasicInteropGetSet) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dims tz = {4, 4, 4, 4};

    size_t sz = size_t(tz[0]) * tz[1] * tz[2] * tz[3];

    memory::desc mem_d(tz, memory::data_type::f32, memory::format_tag::nchw);
    memory mem(mem_d, eng);

    buffer<float, 1> interop_sycl_buf {range<1>(sz)};
    mem.set_sycl_buffer(interop_sycl_buf);

    auto sycl_buf = mem.get_sycl_buffer<float>();

    {
        auto a = interop_sycl_buf.get_access<access::mode::write>();
        for (size_t i = 0; i < sz; ++i)
            a[i] = float(i);
    }

    {
        auto a = sycl_buf.get_access<access::mode::read>();
        for (size_t i = 0; i < sz; ++i)
            ASSERT_EQ(a[i], float(i));
    }
}

TEST_P(sycl_memory_test, InteropReorder) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

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
        memory src_mem(src_mem_d, eng);

        memory::desc dst_mem_d(
                tz, memory::data_type::f32, memory::format_tag::nhwc);

        stream s(eng);
        memory dst_mem(dst_mem_d, eng);

        src_mem.set_sycl_buffer(src_buf);
        dst_mem.set_sycl_buffer(dst_buf);

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

// FIXME: Intel(R) oneAPI DPC++ Compiler does not support compilation with host
// compiler and
// Intel(R) oneAPI DPC++ Compiler does not support mixing fat binaries and host
// binaries. So the test is enabled for ComputeCpp SYCL only
#ifdef DNNL_SYCL_COMPUTECPP
TEST_P(sycl_memory_test, InteropReorderAndUserKernel) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

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
        memory mem(mem_d, eng);

        memory::desc tmp_mem_d(
                tz, memory::data_type::f32, memory::format_tag::nhwc);

        stream s(eng);
        memory tmp_mem(tmp_mem_d, eng);

        mem.set_sycl_buffer(buf);
        tmp_mem.set_sycl_buffer(tmp_buf);

        // Direct reorder mem -> tmp_mem
        reorder(mem, tmp_mem).execute(s, mem, tmp_mem);
        s.wait();

        // Invert the signs of the tmp elements
        auto q = s.get_sycl_queue();
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

#endif

TEST_P(sycl_memory_test, EltwiseWithUserKernel) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    memory::dims tz = {2, 3, 4, 5};
    const size_t N = tz.size();

    memory::desc mem_d(tz, memory::data_type::f32, memory::format_tag::nchw);

    engine eng(eng_kind, 0);
    memory mem({mem_d, eng});

    auto sycl_buf = mem.get_sycl_buffer<float>();

    queue q;

    q.submit([&](handler &cgh) {
        auto a = sycl_buf.get_access<access::mode::write>(cgh);
        cgh.parallel_for<init_kernel>(
                range<1>(N), [=](id<1> i) { a[i] = i.get(0) - N / 2; });
    });

    auto eltwise_d = eltwise_forward::desc(
            prop_kind::forward, algorithm::eltwise_relu, mem_d, 0.0f);
    auto eltwise = eltwise_forward({eltwise_d, eng});

    stream s(eng);
    eltwise.execute(s, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
    s.wait();

    auto host_acc = sycl_buf.get_access<access::mode::read>();

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

INSTANTIATE_TEST_SUITE_P(Simple, sycl_memory_test,
        ::testing::Values(engine::kind::cpu, engine::kind::gpu),
        PrintToStringParamName());

} // namespace dnnl
