/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include <vector>
#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"

using namespace dnnl::graph;

struct api_stream_params {
    engine::kind eng_kind_;
};

class api_stream : public ::testing::TestWithParam<api_stream_params> {
public:
    void create_with_sycl() {
        auto param = ::testing::TestWithParam<api_stream_params>::GetParam();
        engine::kind kind = param.eng_kind_;

        std::unique_ptr<sycl::device> sycl_dev;
        std::unique_ptr<sycl::context> sycl_ctx;

        auto platform_list = sycl::platform::get_platforms();
        for (const auto &plt : platform_list) {
            auto device_list = plt.get_devices();
            for (const auto &dev : device_list) {
                if ((kind == engine::kind::gpu && dev.is_gpu())
                        || (kind == engine::kind::cpu && dev.is_cpu())) {
                    sycl_dev.reset(new sycl::device(dev));
                    sycl_ctx.reset(new sycl::context(*sycl_dev));
                    engine e = dnnl::graph::sycl_interop::make_engine(
                            *sycl_dev, *sycl_ctx);
                    // TODO(zixuan): Only added test for creation of stream.
                    // Further improvements will take place as stream
                    // functionality changes.
                    sycl::queue queue {*sycl_ctx, *sycl_dev};
                    stream s = dnnl::graph::sycl_interop::make_stream(e, queue);
                }
            }
        }
    }
};

TEST_P(api_stream, create_with_sycl) {
    create_with_sycl();
}

#ifdef DNNL_GRAPH_GPU_SYCL
INSTANTIATE_TEST_SUITE_P(api_stream_gpu, api_stream,
        ::testing::Values(api_stream_params {engine::kind::gpu}));
#endif

#ifdef DNNL_GRAPH_CPU_SYCL
INSTANTIATE_TEST_SUITE_P(api_stream_cpu, api_stream,
        ::testing::Values(api_stream_params {engine::kind::cpu}));
#endif
