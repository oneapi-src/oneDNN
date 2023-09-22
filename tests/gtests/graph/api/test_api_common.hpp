/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_API_TEST_API_COMMON_HPP
#define GRAPH_API_TEST_API_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.h"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "tests/gtests/dnnl_test_macros.hpp"

#ifdef DNNL_WITH_SYCL
#include "sycl/sycl_compat.hpp"
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#endif

using dim_t = int64_t;
using dims_t = std::vector<dim_t>;

struct conv_attr_name_t {
    conv_attr_name_t()
        : strides("strides")
        , pads_begin("pads_begin")
        , pads_end("pads_end")
        , dilations("dilations")
        , groups("groups") {}
    conv_attr_name_t(std::string strides, std::string pads_begin,
            std::string pads_end, std::string dilations, std::string groups)
        : strides(std::move(strides))
        , pads_begin(std::move(pads_begin))
        , pads_end(std::move(pads_end))
        , dilations(std::move(dilations))
        , groups(std::move(groups)) {}
    std::string strides;
    std::string pads_begin;
    std::string pads_end;
    std::string dilations;
    std::string groups;
};

struct conv_attr_value_t {
    conv_attr_value_t(dims_t strides, dims_t pads_begin, dims_t pads_end,
            dims_t dilations, dims_t groups)
        : strides(std::move(strides))
        , pads_begin(std::move(pads_begin))
        , pads_end(std::move(pads_end))
        , dilations(std::move(dilations))
        , groups(std::move(groups)) {}
    dims_t strides;
    dims_t pads_begin;
    dims_t pads_end;
    dims_t dilations;
    dims_t groups;
};

struct conv_shapes_t {
    conv_shapes_t(dims_t input_dims, dims_t weight_dims, dims_t output_dims)
        : input_ndim(static_cast<dim_t>(input_dims.size()))
        , weight_ndim(static_cast<dim_t>(weight_dims.size()))
        , output_ndim(static_cast<dim_t>(output_dims.size()))
        , input_dims(std::move(input_dims))
        , weight_dims(std::move(weight_dims))
        , output_dims(std::move(output_dims)) {}
    dim_t input_ndim;
    dim_t weight_ndim;
    dim_t output_ndim;
    dims_t input_dims;
    dims_t weight_dims;
    dims_t output_dims;
};

struct conv_layout_t {
    dnnl_graph_layout_type_t input_layout;
    dnnl_graph_layout_type_t weight_layout;
    dnnl_graph_layout_type_t output_layout;
};

/*
    conv2d attribute:
    strides, pad_begin, pad_end, dilations, groups

    logic tensors:
    input, weight, output
*/
struct conv_params_t {
    dnnl_engine_kind_t engine;
    dnnl_graph_op_kind_t op_kind;
    dnnl_graph_partition_policy_t policy;
    dnnl_data_type_t data_type;
    conv_attr_name_t attr_name;
    conv_attr_value_t attr_value;
    conv_layout_t tensor_layout;
    conv_shapes_t tensor_dims;
};

extern dnnl_engine_kind_t api_test_engine_kind;

#ifdef DNNL_WITH_SYCL
struct allocator_handle_t {
    dnnl_graph_allocator_t allocator = nullptr;

    allocator_handle_t() = default;
    allocator_handle_t(const allocator_handle_t &other) = delete;
    allocator_handle_t &operator=(const allocator_handle_t &other) = delete;
    ~allocator_handle_t() { dnnl_graph_allocator_destroy(allocator); }
    explicit operator bool() const noexcept {
        return static_cast<bool>(allocator);
    }
};
static allocator_handle_t allocator_handle;

struct sycl_deletor {
    sycl_deletor() = delete;
    ::sycl::context ctx_;
    sycl_deletor(const ::sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) ::sycl::free(ptr, ctx_);
    }
};
#endif // DNNL_WITH_SYCL

struct engine_handle_t {
    dnnl_engine_t engine = nullptr;
    ~engine_handle_t() {};
    explicit operator bool() const noexcept {
        return static_cast<bool>(engine);
    }
};
static engine_handle_t engine_handle;

void api_test_dnnl_engine_create(
        dnnl_engine_t *engine, dnnl_engine_kind_t engine_kind);

void api_test_dnnl_graph_graph_create(
        dnnl_graph_graph_t *graph, dnnl_engine_kind_t engine_kind);

dnnl::engine &cpp_api_test_dnnl_engine_create(dnnl::engine::kind engine_kind);

inline dnnl_dim_t product(const std::vector<int64_t> &dims) {
    return dims.empty() ? 0
                        : std::accumulate(dims.begin(), dims.end(),
                                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
}

#endif
