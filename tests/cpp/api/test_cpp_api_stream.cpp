/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.hpp"

#include <vector>
#include <gtest/gtest.h>

#if DNNL_GRAPH_WITH_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#endif

TEST(api_stream, simple_create) {
    using namespace dnnl::graph;
    engine e {engine::kind::cpu, 0};
    stream s {e};
    ASSERT_EQ(e.get_device_id(), 0);
}

#if DNNL_GRAPH_WITH_SYCL
TEST(api_stream, create_with_sycl) {
    using namespace dnnl::graph;
    namespace sycl = cl::sycl;

    std::unique_ptr<sycl::device> sycl_dev;
    std::unique_ptr<sycl::context> sycl_ctx;

    auto platform_list = sycl::platform::get_platforms();
    for (const auto &plt : platform_list) {
        auto device_list = plt.get_devices();
        for (const auto &dev : device_list) {
            if (dev.is_gpu()) {
                sycl_dev.reset(new sycl::device(dev));
                sycl_ctx.reset(new sycl::context(*sycl_dev));
                engine e = dnnl::graph::sycl_interop::make_engine(
                        *sycl_dev, *sycl_ctx);
                // TODO(zixuan): Only added test for creation of stream. Further
                // improvements will take place as stream functionality changes.
                sycl::queue queue {*sycl_ctx, *sycl_dev};
                stream s = dnnl::graph::sycl_interop::make_stream(e, queue);
                ASSERT_EQ(e.get_device_id(), 0);
            }
        }
    }
}
#endif // DNNL_GRAPH_WITH_SYCL
