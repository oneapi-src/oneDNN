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

#include <memory>
#include <CL/cl.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace dnnl {
class sycl_stream_test : public ::testing::Test {
protected:
    virtual void SetUp() {
        if (!find_ocl_device(CL_DEVICE_TYPE_GPU)) { return; }

        eng.reset(new engine(engine::kind::gpu, 0));
        dev.reset(new device(eng->get_sycl_device()));
        ctx.reset(new context(eng->get_sycl_context()));
    }

    std::unique_ptr<engine> eng;
    std::unique_ptr<device> dev;
    std::unique_ptr<context> ctx;
};

TEST_F(sycl_stream_test, Create) {
    SKIP_IF(!eng, "GPU device not found.");

    stream s(*eng);
    queue sycl_queue = s.get_sycl_queue();

    auto eng_dev = make_ocl_wrapper(dev->get());
    auto eng_ctx = make_ocl_wrapper(ctx->get());
    auto queue_dev = make_ocl_wrapper(sycl_queue.get_device().get());
    auto queue_ctx = make_ocl_wrapper(sycl_queue.get_context().get());

    EXPECT_EQ(eng_dev, queue_dev);
    EXPECT_EQ(eng_ctx, queue_ctx);
}

TEST_F(sycl_stream_test, BasicInterop) {
    SKIP_IF(!eng, "GPU devices not found.");

    auto ocl_dev = make_ocl_wrapper(eng->get_sycl_device().get());
    auto ocl_ctx = make_ocl_wrapper(eng->get_sycl_context().get());

    ::cl_int err;
    cl_command_queue interop_ocl_queue = clCreateCommandQueueWithProperties(
            ocl_ctx, ocl_dev, nullptr, &err);
    OCL_CHECK(err);

    queue interop_sycl_queue(interop_ocl_queue, eng->get_sycl_context());
    clReleaseCommandQueue(interop_ocl_queue);

    {
        stream s(*eng, interop_sycl_queue);

        // TODO: enable the following check when Intel(R) oneAPI DPC++ Compiler
        // adds support for it
#if 0
        auto ref_count = interop_sycl_queue.get_info<info::queue::reference_count>();
        EXPECT_EQ(ref_count, 2);
#endif

        auto sycl_queue = s.get_sycl_queue();
        EXPECT_EQ(sycl_queue, interop_sycl_queue);
    }

    // TODO: enable the following check when Intel(R) oneAPI DPC++ Compiler adds
    // support for it
#if 0
    auto ref_count = interop_sycl_queue.get_info<info::queue::reference_count>();
    EXPECT_EQ(ref_count, 1);
#endif
}

TEST_F(sycl_stream_test, InteropIncompatibleQueue) {
    SKIP_IF(!eng, "GPU device not found.");

    cl_device_id cpu_ocl_dev = find_ocl_device(CL_DEVICE_TYPE_CPU);
    SKIP_IF(!cpu_ocl_dev, "CPU device not found.");

    queue cpu_sycl_queue(cpu_selector {});
    SKIP_IF(cpu_sycl_queue.get_device().is_gpu(), "CPU-only device not found");

    catch_expected_failures([&] { stream s(*eng, cpu_sycl_queue); }, true,
            dnnl_invalid_arguments);
}

} // namespace dnnl
