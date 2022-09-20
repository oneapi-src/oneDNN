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

#include <gtest/gtest.h>

#include <vector>

#include "api/test_api_common.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

using namespace dnnl::graph;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
TEST(tensor_test, gpu_engine) {
    SKIP_IF(api_test_engine_kind == dnnl_cpu, "skip sycl test for cpu engine.");
    sycl::device dev {dnnl::impl::sycl::compat::gpu_selector_v};
    sycl::context ctx {dev};

    dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);

    logical_tensor lt {0, logical_tensor::data_type::f32,
            logical_tensor::layout_type::any};
    tensor t {lt, eng, nullptr};

    ASSERT_EQ(t.get_engine().get_kind(), dnnl::engine::kind::gpu);
}
#endif
