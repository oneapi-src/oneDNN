/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

#include <CL/sycl.hpp>
#include <memory>

using namespace cl::sycl;

namespace mkldnn {
namespace {

enum class dev_kind { cpu, cpu_only, gpu, gpu_only, host_only };
enum class ctx_kind { cpu, cpu_only, gpu, gpu_only, host_only };

std::string to_string(dev_kind kind) {
    static const char *strs[]
            = { "CPU", "CPU-only", "GPU", "GPU-only", "Host-only" };
    return strs[static_cast<int>(kind)];
}

std::string to_string(ctx_kind kind) {
    static const char *strs[]
            = { "CPU", "CPU-only", "GPU", "GPU-only", "Host-only" };
    return strs[static_cast<int>(kind)];
}

struct sycl_engine_test_params {
    engine::kind eng_kind;
    dev_kind adev_kind;
    ctx_kind actx_kind;
    mkldnn_status_t expected_status;
};

} // namespace

class sycl_engine_test
    : public ::testing::TestWithParam<sycl_engine_test_params>
{
protected:
    virtual void SetUp() {
        for (auto &plat : platform::get_platforms()) {
            for (auto &dev : plat.get_devices()) {
                if (dev.is_gpu()) {
                    if (!gpu_dev) {
                        gpu_dev.reset(new device(dev));
                        gpu_ctx.reset(new context(*gpu_dev));
                    }
                    if (!gpu_only_dev && !dev.is_cpu() && !dev.is_host()) {
                        gpu_only_dev.reset(new device(dev));
                        gpu_only_ctx.reset(new context(*gpu_only_dev));
                    }
                } else if (dev.is_cpu()) {
                    if (!cpu_dev) {
                        cpu_dev.reset(new device(dev));
                        cpu_ctx.reset(new context(*cpu_dev));
                    }
                    if (!cpu_only_dev && !dev.is_gpu() && !dev.is_host()) {
                        cpu_only_dev.reset(new device(dev));
                        cpu_only_ctx.reset(new context(*cpu_only_dev));
                    }
                } else if (dev.is_host()) {
                    if (!host_only_dev && !dev.is_cpu() && !dev.is_gpu()) {
                        host_only_dev.reset(new device(dev));
                        host_only_ctx.reset(new context(*host_only_dev));
                    }
                }
            }
        }
    }

    virtual void TearDown() {}

    std::unique_ptr<device> cpu_dev, cpu_only_dev;
    std::unique_ptr<device> gpu_dev, gpu_only_dev;
    std::unique_ptr<device> host_only_dev;
    std::unique_ptr<context> cpu_ctx, cpu_only_ctx;
    std::unique_ptr<context> gpu_ctx, gpu_only_ctx;
    std::unique_ptr<context> host_only_ctx;
};

TEST_P(sycl_engine_test, BasicInterop) {
    auto param = GetParam();

    device *dev_ptr = nullptr;
    switch (param.adev_kind) {
    case dev_kind::cpu: dev_ptr = cpu_dev.get(); break;
    case dev_kind::cpu_only: dev_ptr = cpu_only_dev.get(); break;
    case dev_kind::gpu: dev_ptr = gpu_dev.get(); break;
    case dev_kind::gpu_only: dev_ptr = gpu_only_dev.get(); break;
    case dev_kind::host_only: dev_ptr = host_only_dev.get(); break;
    }
    context *ctx_ptr = nullptr;
    switch (param.actx_kind) {
    case ctx_kind::cpu: ctx_ptr = cpu_ctx.get(); break;
    case ctx_kind::cpu_only: ctx_ptr = cpu_only_ctx.get(); break;
    case ctx_kind::gpu: ctx_ptr = gpu_ctx.get(); break;
    case ctx_kind::gpu_only: ctx_ptr = gpu_only_ctx.get(); break;
    case ctx_kind::host_only: ctx_ptr = host_only_ctx.get(); break;
    }

    SKIP_IF(!dev_ptr, to_string(param.adev_kind) + " device not found");
    SKIP_IF(!ctx_ptr, to_string(param.actx_kind) + " context not found");

    auto &dev = *dev_ptr;
    auto &ctx = *ctx_ptr;

    catch_expected_failures(
            [&]() {
                engine eng(param.eng_kind, dev, ctx);
                if (param.expected_status != mkldnn_success) {
                    FAIL() << "Success not expected";
                }

                EXPECT_EQ(eng.get_sycl_device(), dev);
                EXPECT_EQ(eng.get_sycl_context(), ctx);
            },
            param.expected_status != mkldnn_success, param.expected_status);
}

INSTANTIATE_TEST_SUITE_P(Simple, sycl_engine_test,
        ::testing::Values(sycl_engine_test_params{ engine::kind::gpu,
                dev_kind::gpu, ctx_kind::gpu, mkldnn_success }));

INSTANTIATE_TEST_SUITE_P(InvalidArgs, sycl_engine_test,
        ::testing::Values(
                // SYCL CPU device is not yet supported
                sycl_engine_test_params{ engine::kind::cpu, dev_kind::cpu,
                        ctx_kind::cpu, mkldnn_invalid_arguments },
                // SYCL host device is not yet supported
                sycl_engine_test_params{ engine::kind::cpu, dev_kind::host_only,
                        ctx_kind::host_only, mkldnn_invalid_arguments },
                sycl_engine_test_params{ engine::kind::cpu, dev_kind::gpu,
                        ctx_kind::gpu, mkldnn_invalid_arguments },
                sycl_engine_test_params{ engine::kind::gpu, dev_kind::gpu_only,
                        ctx_kind::cpu_only, mkldnn_invalid_arguments },
                sycl_engine_test_params{ engine::kind::cpu, dev_kind::cpu_only,
                        ctx_kind::gpu_only, mkldnn_invalid_arguments }));

} // namespace mkldnn
