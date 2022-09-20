/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include "api/test_api_common.hpp"
#include "test_allocator.hpp"

using namespace dnnl::graph;

#ifdef DNNL_WITH_SYCL
TEST(ApiExecute, ConvReLU) {
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;

    std::vector<int64_t> conv0_input_dims {1, 3, 227, 227};
    std::vector<int64_t> conv0_weight_dims {16, 3, 11, 11};

    dnnl::engine::kind ekind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(api_test_engine_kind == dnnl_cpu,
            "skip sycl api test for native cpu runtime.");
#endif
    graph g(ekind);
    logical_tensor conv0_src_desc {
            0, data_type::f32, conv0_input_dims, layout_type::strided};
    logical_tensor conv0_weight_desc {
            1, data_type::f32, conv0_weight_dims, layout_type::strided};
    logical_tensor conv0_dst_desc {2, data_type::f32, 4, layout_type::strided};
    op conv0(0, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
            {conv0_dst_desc}, "conv0");
    conv0.set_attr<std::vector<int64_t>>(op::attr::strides, {4, 4});
    conv0.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv0.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv0.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv0.set_attr<int64_t>(op::attr::groups, 1);
    conv0.set_attr<std::string>(op::attr::data_format, "NCX");
    conv0.set_attr<std::string>(op::attr::weights_format, "OIX");
    logical_tensor relu0_dst_desc {3, data_type::f32, 4, layout_type::strided};
    op relu0(1, op::kind::ReLU, {conv0_dst_desc}, {relu0_dst_desc}, "relu0");
    g.add_op(conv0);
    g.add_op(relu0);
    g.finalize();
    auto partition = g.get_partitions()[0];

    allocator alloc = sycl_interop::make_allocator(
            dnnl::graph::testing::sycl_malloc_wrapper,
            dnnl::graph::testing::sycl_free_wrapper);

    sycl::queue q = (ekind == dnnl::engine::kind::gpu)
            ? sycl::queue(dnnl::impl::sycl::compat::gpu_selector_v,
                    sycl::property::queue::in_order {})
            : sycl::queue(dnnl::impl::sycl::compat::cpu_selector_v,
                    sycl::property::queue::in_order {});

    dnnl::engine eng = sycl_interop::make_engine_with_allocator(
            q.get_device(), q.get_context(), alloc);

    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);

    std::vector<std::shared_ptr<void>> data_buffers;

    std::vector<logical_tensor> inputs = partition.get_input_ports();
    std::vector<logical_tensor> outputs = partition.get_output_ports();

    compiled_partition cp = partition.compile(inputs, outputs, eng);

    for (size_t idx = 0; idx < outputs.size(); ++idx) {
        size_t id = outputs[idx].get_id();
        outputs[idx] = cp.query_logical_tensor(id);
    }

    std::vector<tensor> inputs_ts, outputs_ts;
    inputs_ts.reserve(inputs.size());
    outputs_ts.reserve(outputs.size());
    for (const auto &in : inputs) {
        size_t mem_size = in.get_mem_size();
        data_buffers.push_back({});
        data_buffers.back().reset(::sycl::malloc_shared(mem_size,
                                          q.get_device(), q.get_context()),
                sycl_deletor {q.get_context()});
        inputs_ts.push_back(tensor {in, eng, data_buffers.back().get()});
    }

    for (const auto &out : outputs) {
        size_t mem_size = out.get_mem_size();
        data_buffers.push_back({});
        data_buffers.back().reset(::sycl::malloc_device(mem_size,
                                          q.get_device(), q.get_context()),
                sycl_deletor {q.get_context()});
        outputs_ts.push_back(tensor {out, eng, data_buffers.back().get()});
    }

    cp.execute(strm, inputs_ts, outputs_ts);
    strm.wait();
}

TEST(SyclApiExecute, ConvReLU) {
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;

    std::vector<int64_t> conv0_input_dims {1, 3, 227, 227};
    std::vector<int64_t> conv0_weight_dims {16, 3, 11, 11};

    dnnl::engine::kind ekind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(api_test_engine_kind == dnnl_cpu,
            "skip sycl api test for native cpu runtime.");
#endif
    graph g(ekind);
    logical_tensor conv0_src_desc {
            0, data_type::f32, conv0_input_dims, layout_type::strided};
    logical_tensor conv0_weight_desc {
            1, data_type::f32, conv0_weight_dims, layout_type::strided};
    logical_tensor conv0_dst_desc {2, data_type::f32, 4, layout_type::strided};
    op conv0(0, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
            {conv0_dst_desc}, "conv0");
    conv0.set_attr<std::vector<int64_t>>(op::attr::strides, {4, 4});
    conv0.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv0.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv0.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv0.set_attr<int64_t>(op::attr::groups, 1);
    conv0.set_attr<std::string>(op::attr::data_format, "NCX");
    conv0.set_attr<std::string>(op::attr::weights_format, "OIX");
    logical_tensor relu0_dst_desc {3, data_type::f32, 4, layout_type::strided};
    op relu0(1, op::kind::ReLU, {conv0_dst_desc}, {relu0_dst_desc}, "relu0");
    g.add_op(conv0);
    g.add_op(relu0);
    g.finalize();
    auto partition = g.get_partitions()[0];

    allocator alloc = sycl_interop::make_allocator(
            dnnl::graph::testing::sycl_malloc_wrapper,
            dnnl::graph::testing::sycl_free_wrapper);

    sycl::queue q = (ekind == dnnl::engine::kind::gpu)
            ? sycl::queue(dnnl::impl::sycl::compat::gpu_selector_v,
                    sycl::property::queue::in_order {})
            : sycl::queue(dnnl::impl::sycl::compat::cpu_selector_v,
                    sycl::property::queue::in_order {});

    dnnl::engine eng = sycl_interop::make_engine_with_allocator(
            q.get_device(), q.get_context(), alloc);

    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);

    std::vector<std::shared_ptr<void>> data_buffers;

    std::vector<logical_tensor> inputs = partition.get_input_ports();
    std::vector<logical_tensor> outputs = partition.get_output_ports();

    compiled_partition cp = partition.compile(inputs, outputs, eng);

    for (size_t idx = 0; idx < outputs.size(); ++idx) {
        size_t id = outputs[idx].get_id();
        outputs[idx] = cp.query_logical_tensor(id);
    }

    std::vector<tensor> inputs_ts, outputs_ts;
    inputs_ts.reserve(inputs.size());
    outputs_ts.reserve(outputs.size());
    for (const auto &in : inputs) {
        size_t mem_size = in.get_mem_size();
        data_buffers.push_back({});
        data_buffers.back().reset(::sycl::malloc_shared(mem_size,
                                          q.get_device(), q.get_context()),
                sycl_deletor {q.get_context()});
        inputs_ts.push_back(tensor {in, eng, data_buffers.back().get()});
    }

    for (const auto &out : outputs) {
        size_t mem_size = out.get_mem_size();
        data_buffers.push_back({});
        data_buffers.back().reset(::sycl::malloc_device(mem_size,
                                          q.get_device(), q.get_context()),
                sycl_deletor {q.get_context()});
        outputs_ts.push_back(tensor {out, eng, data_buffers.back().get()});
    }

    sycl_interop::execute(cp, strm, inputs_ts, outputs_ts);
    strm.wait();
}

#endif
