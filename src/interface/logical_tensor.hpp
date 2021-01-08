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

#ifndef INTERFACE_LOGICAL_TENSOR_HPP
#define INTERFACE_LOGICAL_TENSOR_HPP

#include <algorithm>
#include <assert.h>
#include <string>
#include <utility>
#include <vector>

#include "c_types_map.hpp"
#include "common.hpp"
#include "utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {

struct logical_tensor_wrapper {
    const logical_tensor_t *lt;

    // constructor
    logical_tensor_wrapper(const logical_tensor_t *other) : lt(other) {}
    logical_tensor_wrapper(const logical_tensor_t &other)
        : logical_tensor_wrapper(&other) {}

    // getter
    size_t id() const { return lt->id; }
    int32_t ndims() const { return lt->ndims; }
    data_type_t data_type() const { return lt->data_type; }
    layout_type_t layout_type() const { return lt->layout_type; }
    int64_t layout_id() const { return lt->layout.layout_id; }

    const dims_t &dims() const { return lt->dims; }
    const dims_t &strides() const { return lt->layout.strides; };

    // convenient method to return a std::vector
    std::vector<dim_t> vdims() const {
        return {lt->dims, lt->dims + lt->ndims};
    }

    // convenient method to return a std::vector
    std::vector<dim_t> vstrides() const {
        return {lt->layout.strides, lt->layout.strides + lt->ndims};
    }

    // checker
    bool is_any() const { return lt->layout_type == layout_type::any; }
    bool is_strided() const { return lt->layout_type == layout_type::strided; }
    bool is_opaque() const { return lt->layout_type == layout_type::opaque; }

    bool is_zero() const { return ndims() == 0; }

    bool has_zero_dim() const {
        for (int d = 0; d < ndims(); ++d) {
            if (dims()[d] == 0) return true;
        }

        return false;
    }

    bool is_shape_unknown() const {
        // TODO(lvtao): need to specify: DNNL_GRAPH_UNKNOWN_NDIMS?
        if (ndims() < 0) return true;

        for (int d = 0; d < ndims(); ++d) {
            if (dims()[d] < 0) return true;
        }

        return false;
    }

    // every bit should be same
    bool is_identical(const logical_tensor_wrapper &rhs) const {
        return is_identical(*(this->lt), *(rhs.lt), /* check_id */ true);
    }

    // layout info may implicit same in backend's perspective
    bool operator==(const logical_tensor_wrapper &rhs) const {
        return is_equal(*(this->lt), *(rhs.lt), /* check_id */ true);
    }

    bool operator!=(const logical_tensor_wrapper &rhs) const {
        return !operator==(rhs);
    }

    bool operator==(const logical_tensor_t &rhs) const {
        return operator==(logical_tensor_wrapper(rhs));
    }

    bool operator!=(const logical_tensor_t &rhs) const {
        return !operator==(rhs);
    }

    // equal, but have different id
    bool is_similar(const logical_tensor_wrapper &rhs) const {
        return is_equal(*(this->lt), *(rhs.lt), /* check_id */ false);
    }

    // return the size of data type
    size_t data_type_size() const { return size_of(data_type()); }

    // get memory size in byte
    size_t size() const;

    // get element number
    dim_t nelems() const {
        if (is_zero()) return 0;
        // TODO(lvtao): need to specify: DNNL_RUNTIME_DIM_VAL?
        if (is_shape_unknown()) return -1;
        return utils::array_product(dims(), static_cast<size_t>(ndims()));
    }

    std::vector<dim_t> get_weight_spatial_dims(const std::string &format) {
        std::vector<dim_t> spatial_dims = vdims();
        if (format == "OIX") {
            spatial_dims.erase(spatial_dims.begin(), spatial_dims.begin() + 2);
        } else if (format == "XIO") {
            spatial_dims.erase(spatial_dims.end() - 2, spatial_dims.end());
        } else {
            // For code completeness - return an empty vector in this case
            spatial_dims.clear();
        }

        return spatial_dims;
    }

    std::vector<dim_t> get_src_spatial_dims(const std::string &format) {
        std::vector<dim_t> spatial_dims = vdims();
        if (format == "NCX") {
            spatial_dims.erase(spatial_dims.begin(), spatial_dims.begin() + 2);
        } else if (format == "NXC") {
            spatial_dims.erase(spatial_dims.begin(), spatial_dims.begin() + 1);
            spatial_dims.erase(spatial_dims.end() - 1, spatial_dims.end());
        } else {
            spatial_dims.clear();
        }

        return spatial_dims;
    }

    dim_t get_weight_i(const std::string &format) {
        if (format == "OIX") {
            return dims()[1];
        } else if (format == "XIO") {
            return dims()[ndims() - 2];
        } else {
            // For code completeness
            return DNNL_GRAPH_UNKNOWN_DIM;
        }
    }

    dim_t get_weight_o(const std::string &format) {
        if (format == "OIX") {
            return dims()[0];
        } else if (format == "XIO") {
            return dims()[ndims() - 1];
        } else {
            // For code completeness
            return DNNL_GRAPH_UNKNOWN_DIM;
        }
    }

    dim_t get_src_n() {
        // `n` is always the first element for both `NCX` and `NXC`
        return dims()[0];
    }

    dim_t get_src_c(const std::string &format) {
        if (format == "NCX") {
            return dims()[1];
        } else if (format == "NXC") {
            return dims()[ndims() - 1];
        } else {
            // For code completeness
            return DNNL_GRAPH_UNKNOWN_DIM;
        }
    }

    logical_tensor_t reorder_data_dims_strides() {
        assert(lt->ndims != -1 && "data dims haven't be uninitialized.");
        // update input tensor's dims NXC
        // keep HW order
        logical_tensor_t cdata = *lt;
        int32_t i = 1, j = cdata.ndims - 1;
        while (i < j) {
            std::swap(cdata.dims[i], cdata.dims[j]);
            if (cdata.layout_type == layout_type::strided) {
                std::swap(cdata.layout.strides[i], cdata.layout.strides[j]);
            }
            ++i;
        }
        return cdata;
    }

    logical_tensor_t reorder_weight_dims_strides() { // XIO->OIX
        assert(lt->ndims != -1 && "data dims haven't be uninitialized.");
        logical_tensor_t cweight = *lt;
        int32_t i = 0, j = cweight.ndims - 1;
        while (i < j) {
            std::swap(cweight.dims[i], cweight.dims[j]);
            if (cweight.layout_type == layout_type::strided) {
                std::swap(cweight.layout.strides[i], cweight.layout.strides[j]);
            }
            ++i;
            --j;
        }
        // // keep HW order
        i = 2, j = cweight.ndims - 1;
        while (i < j) {
            std::swap(cweight.dims[i], cweight.dims[j]);
            if (cweight.layout_type == layout_type::strided) {
                std::swap(cweight.layout.strides[i], cweight.layout.strides[j]);
            }
            ++i;
            --j;
        }
        return cweight;
    }

private:
    bool is_identical(const logical_tensor_t &lhs, const logical_tensor_t &rhs,
            bool check_id = true) const;

    bool is_equal(const logical_tensor_t &lhs, const logical_tensor_t &rhs,
            bool check_id = true) const;
};

} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
