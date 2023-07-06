/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <memory>

using namespace sycl;

namespace dnnl {
class sycl_stream_test : public ::testing::TestWithParam<engine::kind> {
protected:
    virtual void SetUp() {
        if (engine::get_count(engine::kind::cpu) > 0) {
            cpu_eng = engine(engine::kind::cpu, 0);
        }
        if (engine::get_count(engine::kind::gpu) > 0) {
            gpu_eng = engine(engine::kind::gpu, 0);
        }
    }

    bool has(engine::kind eng_kind) const {
        switch (eng_kind) {
            case engine::kind::cpu: return bool(cpu_eng);
            case engine::kind::gpu: return bool(gpu_eng);
            default: assert(!"Not expected");
        }
        return false;
    }

    engine get_engine(engine::kind eng_kind) const {
        switch (eng_kind) {
            case engine::kind::cpu: return cpu_eng;
            case engine::kind::gpu: return gpu_eng;
            default: assert(!"Not expected");
        }
        return {};
    }

    device get_device(engine::kind eng_kind) const {
        switch (eng_kind) {
            case engine::kind::cpu: return sycl_interop::get_device(cpu_eng);
            case engine::kind::gpu: return sycl_interop::get_device(gpu_eng);
            default: assert(!"Not expected");
        }
        return {};
    }

    context get_context(engine::kind eng_kind) const {
        switch (eng_kind) {
            case engine::kind::cpu: return sycl_interop::get_context(cpu_eng);
            case engine::kind::gpu: return sycl_interop::get_context(gpu_eng);
            default: assert(!"Not expected");
        }
        return context();
    }

    engine cpu_eng;
    engine gpu_eng;
};

TEST_P(sycl_stream_test, Create) {
    engine::kind kind = GetParam();
    SKIP_IF(!has(kind), "Device not found.");

    stream s(get_engine(kind));

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (kind == engine::kind::cpu) {
        EXPECT_ANY_THROW(sycl_interop::get_queue(s));
        return;
    }
#endif
    queue sycl_queue = sycl_interop::get_queue(s);

    auto queue_dev = sycl_queue.get_device();
    auto queue_ctx = sycl_queue.get_context();

    EXPECT_EQ(get_device(kind), queue_dev);
    EXPECT_EQ(get_context(kind), queue_ctx);
}

TEST_P(sycl_stream_test, BasicInterop) {
    engine::kind kind = GetParam();
    SKIP_IF(!has(kind), "Device not found.");

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (kind == engine::kind::cpu) {
        ::sycl::queue dummy;
        EXPECT_ANY_THROW(sycl_interop::make_stream(get_engine(kind), dummy));
        return;
    }
#endif
    queue interop_queue(get_context(kind), get_device(kind));
    stream s = sycl_interop::make_stream(get_engine(kind), interop_queue);

    EXPECT_EQ(interop_queue, sycl_interop::get_queue(s));
}

TEST_P(sycl_stream_test, InteropIncompatibleQueue) {
    engine::kind kind = GetParam();
    SKIP_IF(!has(engine::kind::cpu) || !has(engine::kind::gpu),
            "CPU or GPU device not found.");

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(true, "Skip this test for classic CPU runtime");
#endif

    auto other_kind = (kind == engine::kind::gpu) ? engine::kind::cpu
                                                  : engine::kind::gpu;
    queue interop_queue(get_context(other_kind), get_device(other_kind));

    catch_expected_failures(
            [&] { sycl_interop::make_stream(get_engine(kind), interop_queue); },
            true, dnnl_invalid_arguments);
}

#ifndef DNNL_EXPERIMENTAL_PROFILING
extern "C" dnnl_status_t dnnl_reset_profiling(dnnl_stream_t stream);
#endif

TEST_P(sycl_stream_test, TestProfilingAPIDisabledAndEnabled) {
    engine::kind kind = GetParam();
    SKIP_IF(!has(kind), "Device not found.");
    SKIP_IF(kind == engine::kind::cpu, "Test is GPU specific.");

    auto sycl_queue = sycl::queue(get_context(kind), get_device(kind),
            sycl::property_list {sycl::property::queue::in_order {},
                    sycl::property::queue::enable_profiling {}});
    stream strm = sycl_interop::make_stream(get_engine(kind), sycl_queue);

    memory::dims tz_dims = {2, 3, 4, 5};
    const size_t N = std::accumulate(tz_dims.begin(), tz_dims.end(), (size_t)1,
            std::multiplies<size_t>());
    auto usm_buffer = (float *)malloc_shared(
            N * sizeof(float), get_device(kind), get_context(kind));

    memory::desc mem_d(
            tz_dims, memory::data_type::f32, memory::format_tag::nchw);

    memory mem = sycl_interop::make_memory(mem_d, get_engine(kind),
            sycl_interop::memory_kind::usm, usm_buffer);

    auto relu_pd = eltwise_forward::primitive_desc(get_engine(kind),
            prop_kind::forward, algorithm::eltwise_relu, mem_d, mem_d, 0.0f);
    auto relu = eltwise_forward(relu_pd);
    relu.execute(strm, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
    strm.wait();

    auto st = dnnl_reset_profiling(strm.get());

// If the experimental profiling API is not enabled then the library should not
// enable profiling regardless of the queue's properties.
#ifdef DNNL_EXPERIMENTAL_PROFILING
    EXPECT_EQ(st, dnnl_success);
#else
    EXPECT_EQ(st, dnnl_invalid_arguments);
#endif
}

// TODO: Enable the test below after sycl_stream_t is fixed to not reuse the
// service stream. Now it ignores the input stream flags and reuses the service
// stream which is constructed without any flags.
#if 0
TEST_P(sycl_stream_test, Flags) {
    engine::kind kind = GetParam();
    SKIP_IF(!has(kind), "Device not found.");

    stream in_order_stream(get_engine(kind), stream::flags::in_order);
    auto in_order_queue = sycl_interop::get_queue(in_order_stream);
    EXPECT_TRUE(in_order_queue.is_in_order());

    stream out_of_order_stream(get_engine(kind), stream::flags::out_of_order);
    auto out_of_order_queue = sycl_interop::get_queue(out_of_order_stream);
    EXPECT_TRUE(!out_of_order_queue.is_in_order());
}
#endif

namespace {
struct PrintToStringParamName {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        switch (info.param) {
            case engine::kind::cpu: return "cpu";
            case engine::kind::gpu: return "gpu";
            default: assert(!"Not expected");
        }
        return {};
    }
};
} // namespace

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, sycl_stream_test,
        ::testing::Values(engine::kind::cpu, engine::kind::gpu),
        PrintToStringParamName());

} // namespace dnnl
