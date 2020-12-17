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

#include "oneapi/dnnl/dnnl_graph.h"

#include "backend.hpp"
#include "c_types_map.hpp"
#include "logical_tensor.hpp"
#include "utils.hpp"

using namespace llga::impl;

size_t logical_tensor_wrapper::size() const {
    if (is_strided()) {
        size_t max_size = 0;
        for (int d = 0; d < ndims(); ++d) {
            dim_t strided_pdim = dims()[d];
            dim_t effective_stride = strided_pdim == 1 ? 1 : strides()[d];
            max_size = std::max<size_t>(max_size,
                    static_cast<size_t>(strided_pdim * effective_stride));
        }

        return max_size * data_type_size();
    } else if (is_opaque()) {
        return backend_manager::get_backend(
                static_cast<size_t>(lt->layout.layout_id))
                ->get_mem_size(*lt);
    } else {
        return (size_t)-1;
    }
}

bool logical_tensor_wrapper::is_identical(const logical_tensor_t &lhs,
        const logical_tensor_t &rhs, bool check_id) const {
    bool equal = check_id ? lhs.id == rhs.id : true;
    equal = equal && (lhs.ndims == rhs.ndims)
            && (lhs.data_type == rhs.data_type)
            && (lhs.layout_type == rhs.layout_type);

    if (!equal) return false;
    if (lhs.ndims == 0 || lhs.ndims == -1) return true;

    // check dims
    equal = std::equal(std::begin(lhs.dims), std::begin(lhs.dims) + lhs.ndims,
            std::begin(rhs.dims));
    if (!equal) return false;

    // check layout information
    if (lhs.layout_type == layout_type::strided) {
        return std::equal(std::begin(lhs.layout.strides),
                std::begin(lhs.layout.strides) + lhs.ndims,
                std::begin(rhs.layout.strides));
    } else if (lhs.layout_type == layout_type::opaque) {
        return lhs.layout.layout_id == rhs.layout.layout_id;
    } else {
        return true;
    }
}

bool logical_tensor_wrapper::is_equal(const logical_tensor_t &lhs,
        const logical_tensor_t &rhs, bool check_id) const {
    if (is_identical(lhs, rhs, check_id)) return true;

    // need to ask backend
    if (lhs.layout_type != rhs.layout_type) {
        // two logical_tensors' layout may be implicitly equal, only when
        // both of their layout types are opaque or strided
        bool layout_may_implicit_equal
                = (lhs.layout_type == layout_type::opaque
                          || lhs.layout_type == layout_type::strided)
                && (rhs.layout_type == layout_type::opaque
                        || rhs.layout_type == layout_type::strided);

        if (!layout_may_implicit_equal) return false;

        // call backend to check whether a opaque layout is actually
        // equal to a strided layout in backend's perspective
        size_t layout_id = lhs.layout_type == layout_type::opaque
                ? static_cast<size_t>(lhs.layout.layout_id)
                : static_cast<size_t>(rhs.layout.layout_id);
        auto backend = backend_manager::get_backend(layout_id);
        return backend->is_similar(lhs, rhs);
    }

    return false;
}

status_t DNNL_GRAPH_API dnnl_graph_logical_tensor_init(
        logical_tensor_t *logical_tensor, size_t tid, data_type_t dtype,
        int32_t ndims, layout_type_t ltype) {
    if (logical_tensor == nullptr || ndims > DNNL_GRAPH_MAX_NDIMS) {
        return status::invalid_argument;
    }

    auto val = logical_tensor_t();
    val.id = tid;
    val.ndims = ndims;
    val.data_type = dtype;
    val.layout_type = ltype;

    // dims and strides are undefined.
    *logical_tensor = val;

    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_logical_tensor_init_with_dims(
        logical_tensor_t *logical_tensor, size_t tid, data_type_t dtype,
        int32_t ndims, const dims_t dims, layout_type_t ltype) {
    if (!logical_tensor || ndims < 0) return status::invalid_argument;

    auto val = logical_tensor_t();
    val.id = tid;
    val.ndims = ndims;
    val.data_type = dtype;
    val.layout_type = ltype;

    if (ndims == 0) {
        val.dims[0] = 0;
        if (ltype == layout_type::strided) { val.layout.strides[0] = 0; }
    } else {
        if (!dims) return status::invalid_argument;

        std::copy(dims, dims + ndims, val.dims);
        // sanity check for dims
        bool sanity = true && std::all_of(dims, dims + ndims, [](int64_t v) {
            return v > 0;
        });
        if (sanity && ltype == layout_type::strided) {
            // initialize strides
            val.layout.strides[ndims - 1] = 1;
            for (int s = ndims - 2; s >= 0; --s) {
                val.layout.strides[s] = dims[s + 1] * val.layout.strides[s + 1];
            }
        } // else strides are undefined
    }
    *logical_tensor = val;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_logical_tensor_init_with_strides(
        logical_tensor_t *logical_tensor, size_t tid, data_type_t dtype,
        int32_t ndims, const dims_t dims, const dims_t strides) {
    if (!logical_tensor || ndims < 0) return status::invalid_argument;

    auto val = logical_tensor_t();
    val.id = tid;
    val.ndims = ndims;
    val.data_type = dtype;
    val.layout_type = layout_type::strided;

    if (ndims == 0) {
        val.dims[0] = 0;
        val.layout.strides[0] = 0;
    } else {
        if (utils::any_null(dims, strides)) return status::invalid_argument;

        std::copy(dims, dims + ndims, val.dims);
        std::copy(strides, strides + ndims, val.layout.strides);
    }

    *logical_tensor = val;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_logical_tensor_get_mem_size(
        const logical_tensor_t *logical_tensor, size_t *size) {
    if (utils::any_null(logical_tensor, size)) return status::invalid_argument;

    *size = logical_tensor_wrapper(logical_tensor).size();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_logical_tenosr_has_same_layout_and_dtype(
        const logical_tensor_t *lt1, const logical_tensor_t *lt2,
        uint8_t *is_same) {
    if (utils::any_null(lt1, lt2)) return status::invalid_argument;

    logical_tensor_wrapper ltw1 {lt1};
    logical_tensor_wrapper ltw2 {lt2};

    *is_same = 0;
    // Here, firstly we will check if these two logical tensors have the same
    // dtype and dims. Below are several cases will also be considered as same
    // layout:
    //  - when layout_type is the same:
    //      1. `strided` layout:  has the same `strides`
    //      2. `opaque` layout: has the same `layout_id`
    // - when layout_type is different:
    //      - only one case: one of them has `strided` layout while another has
    //          `opaque` layout but with default format
    if (ltw1 == ltw2) *is_same = 1;
    return status::success;
}

int DNNL_GRAPH_API dnnl_graph_logical_tensor_equal(
        const logical_tensor_t *lhs, const logical_tensor_t *rhs) {
    if (lhs == rhs) return 1;
    if (utils::any_null(lhs, rhs)) return 0;
    return logical_tensor_wrapper(lhs) == logical_tensor_wrapper(rhs);
}
