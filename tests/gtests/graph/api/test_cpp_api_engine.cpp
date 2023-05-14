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

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "test_api_common.hpp"
#include "gtest/gtest.h"

TEST(APIEngine, DefaultEngineAllocator) {
    using namespace dnnl::graph;
    using dt = logical_tensor::data_type;
    using lt = logical_tensor::layout_type;

    SKIP_IF(DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Skip the case when CPU runtime is NONE or SYCL");

    graph g(engine::kind::cpu);

    const std::vector<int64_t> src_dims = {1, 10, 10, 3};
    const std::vector<int64_t> wei_dims = {3, 3, 3, 3};

    logical_tensor src = logical_tensor(0, dt::f32, src_dims, lt::strided);
    logical_tensor wei = logical_tensor(1, dt::f32, wei_dims, lt::strided);
    logical_tensor dst = logical_tensor(2, dt::f32, 4, lt::strided);

    op conv = op(3, op::kind::Convolution, "_conv");
    conv.set_attr<std::vector<int64_t>>(op::attr::strides, {1, 1});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv.set_attr<std::string>(op::attr::data_format, "NXC");
    conv.set_attr<std::string>(op::attr::weights_format, "OIX");
    conv.set_attr<int64_t>(op::attr::groups, 1);

    conv.add_inputs({src, wei});
    conv.add_outputs({dst});

    g.add_op(conv);
    g.finalize();

    auto parts = g.get_partitions();

    // use the native engine to compile and execute a graph partition. The
    // native engine should contain a default allocator.
    engine eng = engine(engine::kind::cpu, 0);
    auto cp = parts[0].compile({src, wei}, {dst}, eng);

    stream str = stream(eng);
    std::vector<float> src_data(product(src_dims));
    std::vector<float> wei_data(product(wei_dims));

    logical_tensor query_dst = cp.query_logical_tensor(2);
    size_t sz_dst = query_dst.get_mem_size();
    std::vector<float> dst_data(sz_dst / sizeof(float));

    tensor ts_src = tensor(src, eng, src_data.data());
    tensor ts_wei = tensor(wei, eng, wei_data.data());
    tensor ts_dst = tensor(query_dst, eng, dst_data.data());

    cp.execute(str, {ts_src, ts_wei}, {ts_dst});
    str.wait();
}
