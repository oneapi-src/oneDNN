/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#ifndef UTILS_HPP
#define UTILS_HPP

#include <functional>
#include <vector>

#include "interface/c_types_map.hpp"

namespace dnnl {
namespace graph {
namespace tests {
namespace unit {
namespace utils {

#define EXPECT_SUCCESS(expression) \
    EXPECT_EQ((expression), dnnl::graph::impl::status::success)

#define SKIP_IF(cond, msg) \
    do { \
        if (cond) { \
            std::cout << "[  SKIPPED ] " << (msg) << std::endl; \
            return; \
        } \
    } while (0)

static inline dnnl::graph::impl::logical_tensor_t logical_tensor_init(size_t id,
        dnnl::graph::impl::data_type_t dtype,
        dnnl::graph::impl::layout_type_t ltype
        = dnnl::graph::impl::layout_type::undef) {
    dnnl::graph::impl::logical_tensor_t val;
    val.id = id;
    val.data_type = dtype;
    val.layout_type = ltype;
    val.ndims = -1;

    return val;
}

static inline dnnl::graph::impl::logical_tensor_t logical_tensor_init(size_t id,
        std::vector<dnnl::graph::impl::dim_t> dims,
        dnnl::graph::impl::data_type_t dtype,
        dnnl::graph::impl::layout_type_t ltype
        = dnnl::graph::impl::layout_type::strided) {
    if (dims.size() == 0) { return logical_tensor_init(id, dtype); }
    dnnl::graph::impl::logical_tensor_t val;
    val.id = id;
    val.data_type = dtype;
    val.ndims = static_cast<int>(dims.size());

    // dims
    for (size_t d = 0; d < dims.size(); ++d) {
        val.dims[d] = dims[d];
    }

    // strides
    val.layout_type = ltype;
    if (ltype == dnnl::graph::impl::layout_type::strided) {
        val.layout.strides[val.ndims - 1] = 1;
        for (int s = val.ndims - 2; s >= 0; --s) {
            size_t si = static_cast<size_t>(s);
            val.layout.strides[si] = dims[si + 1] * val.layout.strides[si + 1];
        }
    }

    return val;
}

static inline dnnl::graph::impl::logical_tensor_t logical_tensor_init(size_t id,
        std::vector<dnnl::graph::impl::dim_t> dims,
        std::vector<dnnl::graph::impl::dim_t> strides,
        dnnl::graph::impl::data_type_t dtype) {
    dnnl::graph::impl::logical_tensor_t val;
    val.id = id;
    val.data_type = dtype;
    val.ndims = static_cast<int>(dims.size());

    // dims and strides
    for (size_t d = 0; d < dims.size(); ++d) {
        val.dims[d] = dims[d];
        val.layout.strides[d] = strides[d];
    }

    val.layout_type = dnnl::graph::impl::layout_type::strided;
    return val;
}

static inline std::vector<int64_t> compute_dense_strides(
        const std::vector<int64_t> &output_dims) {
    std::vector<int64_t> output_strides(output_dims.size());
    for (auto it = output_dims.begin(); it < output_dims.end(); ++it) {
        const auto val = std::accumulate(std::next(it), output_dims.end(), 1,
                std::multiplies<int64_t>());
        const auto dist = std::distance(output_dims.begin(), it);
        output_strides[static_cast<size_t>(dist)] = val;
    }
    return output_strides;
}

} // namespace utils
} // namespace unit
} // namespace tests
} // namespace graph
} // namespace dnnl

#endif
