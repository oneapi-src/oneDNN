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

#include "oneapi/dnnl/dnnl_graph.h"

#include "graph/interface/backend.hpp"
#include "graph/interface/logical_tensor.hpp"
#include "graph/interface/partition_hashing.hpp"

using namespace dnnl::impl::graph;

size_t logical_tensor_wrapper_t::size() const {
    if (is_strided()) {
        // scalar (0-D tensor)
        if (ndims() == 0) { return data_type_size(); }

        // zero-volume tensor
        if (has_zero_dim()) { return 0U; }

        size_t max_size = 0;
        for (int d = 0; d < ndims(); ++d) {
            dim_t strided_pdim = dims()[d];
            dim_t effective_stride = strided_pdim == 1 ? 1 : strides()[d];
            max_size = std::max<size_t>(max_size,
                    static_cast<size_t>(strided_pdim * effective_stride));
        }

        return max_size * data_type_size();
    } else if (is_opaque()) {
        size_t layout_id = lt->layout.layout_id;
        auto backend
                = backend_registry_t::get_singleton().get_registered_backend(
                        layout_id);

        // Before pass a logical tensor to specific backend, we should remove
        // the encoded backend id from the layout id. Because each backend is
        // invisible about backend id for simplifying the backend integration
        logical_tensor_t new_lt = *lt;
        new_lt.layout.layout_id
                = backend_registry_t::extract_layout_id(layout_id);
        return backend->get_mem_size(new_lt);
    } else {
        return (size_t)-1;
    }
}

// Every bit should be same
bool logical_tensor_wrapper_t::is_identical(
        const logical_tensor_t &lhs, const logical_tensor_t &rhs) const {
    bool equal = (lhs.id == rhs.id) && (lhs.ndims == rhs.ndims)
            && (lhs.data_type == rhs.data_type)
            && (lhs.layout_type == rhs.layout_type)
            && (lhs.property == rhs.property);

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

// Check if underlying layouts of two logical tensors are the same. Need to
// involve backend. By default it will check id and data type since `check_id`
// and `check_dtype` are set to true.
bool logical_tensor_wrapper_t::is_similar(const logical_tensor_t &lhs,
        const logical_tensor_t &rhs, bool check_id, bool check_dtype) const {
    bool equal = check_id ? (lhs.id == rhs.id) : true;
    equal = equal && (check_dtype ? lhs.data_type == rhs.data_type : true);
    equal = equal && (lhs.ndims == rhs.ndims);
    equal = equal && (lhs.property == rhs.property);

    if (!equal) return equal;

    // check dims
    if (lhs.ndims > 0) {
        if (!std::equal(std::begin(lhs.dims), std::begin(lhs.dims) + lhs.ndims,
                    std::begin(rhs.dims)))
            return false;
    }

    if (lhs.layout_type == rhs.layout_type) {
        if (lhs.ndims <= 0) return true;
        // check layout information
        if (lhs.layout_type == layout_type::strided) {
            return std::equal(std::begin(lhs.layout.strides),
                    std::begin(lhs.layout.strides) + lhs.ndims,
                    std::begin(rhs.layout.strides));
        } else if (lhs.layout_type == layout_type::opaque) {
            return lhs.layout.layout_id == rhs.layout.layout_id;
        } else { // layout_type = undef or any.
            return true;
        }
    } else {
        // layout_type = undef or any will be treated as not equal.
        const bool layout_may_implicit_equal
                = (lhs.layout_type == layout_type::opaque
                          || lhs.layout_type == layout_type::strided)
                && (rhs.layout_type == layout_type::opaque
                        || rhs.layout_type == layout_type::strided);
        if (!layout_may_implicit_equal) return false;

        if (lhs.ndims <= 0) return true;

        // at least one layout is opaque, need to ask backend.
        // call backend to check whether a opaque layout is actually
        // equal to a strided layout in backend's perspective
        size_t layout_id = lhs.layout_type == layout_type::opaque
                ? lhs.layout.layout_id
                : rhs.layout.layout_id;
        auto backend
                = backend_registry_t::get_singleton().get_registered_backend(
                        layout_id);

        // Before pass a logical tensor to specific backend, we should remove
        // the encoded backend id from the layout id. Because each backend is
        // invisible about backend id for simplifying the backend integration.
        logical_tensor_t new_lt
                = lhs.layout_type == layout_type::opaque ? lhs : rhs;
        new_lt.layout.layout_id
                = backend_registry_t::extract_layout_id(layout_id);

        return lhs.layout_type == layout_type::opaque
                ? backend->compare_logical_tensor(new_lt, rhs)
                : backend->compare_logical_tensor(lhs, new_lt);
    }

    return false;
}

size_t logical_tensor_wrapper_t::hash() const noexcept {
    size_t seed = 0;
    seed = hash_combine(seed, this->id());
    const int32_t nd = this->ndims();
    seed = nd > 0 ? partition_hashing::get_array_hash(seed, this->dims(), nd)
                  : hash_combine(seed, nd);
    seed = hash_combine(seed, static_cast<size_t>(this->data_type()));
    seed = hash_combine(seed, static_cast<size_t>(this->layout_type()));
    // layout type
    switch (this->layout_type()) {
        case layout_type::undef:
        case layout_type::any: break;
        case layout_type::strided:
            if (nd > 0)
                seed = partition_hashing::get_array_hash(
                        seed, this->strides(), nd);
            break;
        case layout_type::opaque:
            seed = hash_combine(seed, this->layout_id());
            break;
        default: assertm(false, "unknown layout_type");
    }
    return seed;
}

status_t DNNL_API dnnl_graph_logical_tensor_init(
        logical_tensor_t *logical_tensor, size_t tid, data_type_t dtype,
        int32_t ndims, layout_type_t ltype, property_type_t ptype) {
    if (logical_tensor == nullptr || ndims > DNNL_MAX_NDIMS) {
        return status::invalid_arguments;
    }

    auto val = logical_tensor_t();
    val.id = tid;
    val.ndims = ndims;
    val.data_type = dtype;
    val.layout_type = ltype;
    val.property = ptype;

    // initialize the dims and strides
    std::fill(val.dims, val.dims + DNNL_MAX_NDIMS, DNNL_GRAPH_UNKNOWN_DIM);
    std::fill(val.layout.strides, val.layout.strides + DNNL_MAX_NDIMS,
            DNNL_GRAPH_UNKNOWN_DIM);

    *logical_tensor = val;

    return status::success;
}

status_t DNNL_API dnnl_graph_logical_tensor_init_with_dims(
        logical_tensor_t *logical_tensor, size_t tid, data_type_t dtype,
        int32_t ndims, const dims_t dims, layout_type_t ltype,
        property_type_t ptype) {
    if (!logical_tensor || ndims < 0) return status::invalid_arguments;

    auto val = logical_tensor_t();
    val.id = tid;
    val.ndims = ndims;
    val.data_type = dtype;
    val.layout_type = ltype;
    val.property = ptype;

    if (ndims == 0) {
        val.dims[0] = 0;
        if (ltype == layout_type::strided) { val.layout.strides[0] = 0; }
    } else {
        if (!dims) return status::invalid_arguments;

        std::copy(dims, dims + ndims, val.dims);
        // sanity check for dims
        bool sanity = true && std::all_of(dims, dims + ndims, [](int64_t v) {
            return v >= 0;
        });
        if (sanity && ltype == layout_type::strided) {
            // initialize strides
            val.layout.strides[ndims - 1] = 1;
            for (int s = ndims - 2; s >= 0; --s) {
                // replace 0 in shape to 1 when computing the strides
                val.layout.strides[s] = std::max<dim_t>(dims[s + 1], 1)
                        * val.layout.strides[s + 1];
            }
        } else {
            // initialize strides with -1
            std::fill(val.layout.strides, val.layout.strides + DNNL_MAX_NDIMS,
                    DNNL_GRAPH_UNKNOWN_DIM);
        }
    }
    *logical_tensor = val;
    return status::success;
}

status_t DNNL_API dnnl_graph_logical_tensor_init_with_strides(
        logical_tensor_t *logical_tensor, size_t tid, data_type_t dtype,
        int32_t ndims, const dims_t dims, const dims_t strides,
        property_type_t ptype) {
    if (!logical_tensor || ndims < 0) return status::invalid_arguments;

    auto val = logical_tensor_t();
    val.id = tid;
    val.ndims = ndims;
    val.data_type = dtype;
    val.layout_type = layout_type::strided;
    val.property = ptype;

    if (ndims == 0) {
        val.dims[0] = 0;
        val.layout.strides[0] = 0;
    } else {
        if (utils::any_null(dims, strides)) return status::invalid_arguments;

        std::copy(dims, dims + ndims, val.dims);
        std::copy(strides, strides + ndims, val.layout.strides);
    }

    *logical_tensor = val;
    return status::success;
}

status_t DNNL_API dnnl_graph_logical_tensor_get_mem_size(
        const logical_tensor_t *logical_tensor, size_t *size) {
    if (utils::any_null(logical_tensor, size)) return status::invalid_arguments;

    *size = logical_tensor_wrapper_t(logical_tensor).size();
    return status::success;
}

status_t DNNL_API dnnl_graph_logical_tensor_is_equal(
        const logical_tensor_t *lt1, const logical_tensor_t *lt2,
        uint8_t *is_equal) {
    if (utils::any_null(lt1, lt2, is_equal)) return status::invalid_arguments;

    logical_tensor_wrapper_t ltw1 {lt1};
    logical_tensor_wrapper_t ltw2 {lt2};

    // Here, firstly we will check if these two logical tensors have the same
    // dtype and dims. Below are several cases will also be considered as same
    // layout:
    //  - when layout_type is the same:
    //      1. `strided` layout:  has the same `strides`
    //      2. `opaque` layout: has the same `layout_id`
    // - when layout_type is different:
    //      - only one case: one of them has `strided` layout while another has
    //          `opaque` layout but with default format
    *is_equal = (ltw1 == ltw2) ? 1 : 0;
    return status::success;
}
