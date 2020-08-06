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

#ifndef LLGA_TENSOR_HPP
#define LLGA_TENSOR_HPP

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "llga_types.h"

#include "llga_api_detail.hpp"
#include "llga_base.hpp"

namespace llga {
namespace api {

class logical_tensor {
public:
    using dims_t = std::vector<llga_dim_t>;

    /// Constructs a logical tensor object
    ///
    /// @param in_type Data type
    /// @param layout_id Memory layout
    logical_tensor(llga_data_type_t in_type = llga_data_type_undef,
            llga_layout_id_t layout_id = llga_any);

    /// Constructs a logical tensor object
    ///
    /// @param ndims Number of tensor dimensions
    /// @param in_type Data type
    /// @param layout_id Memory layout
    logical_tensor(int ndims, llga_data_type_t in_type = llga_data_type_undef,
            llga_layout_id_t layout_id = llga_any);

    /// Constructs a logical tensor object
    ///
    /// @param adims Tensor dimensions
    /// @param in_type Data type
    /// @param layout_id Memory layout
    logical_tensor(const dims_t &adims,
            llga_data_type_t in_type = llga_data_type_undef,
            llga_layout_id_t layout_id = llga_any);

    /// Constructs a logical tensor object
    ///
    /// @param adims Tensor dimensions
    /// @param strides Tensor strides
    /// @param in_type Data type
    /// @param layout_id Memory layout
    logical_tensor(const dims_t &adims, const dims_t &strides,
            llga_data_type_t in_type = llga_data_type_undef,
            llga_layout_id_t layout_id = llga_any);

    /// Returns dimensions of the tensor
    ///
    /// @returns A copy of the dimensions vector
    dims_t get_dims() const;

    /// Set dimensions to this logical tensor
    ///
    /// @param dim New tensor dimensions
    void set_dims(dims_t dim);

    /// Returns unique id of the tensor
    ///
    /// @returns Id number
    uint64_t get_id() const;

    /// Returns data type of the tensor
    ///
    /// @returns The data type
    llga_data_type_t get_type() const;

    /// Returns strides of this logical tensor
    ///
    /// @returns A copy of strides vector
    dims_t get_strides() const;

    /// Set strides of this logical tensor
    ///
    /// @param strides New tensor strides
    void set_strides(dims_t strides) {
        check_succeed(llga_logical_tensor_set_strides(
                raw(), strides.size(), strides.data()));
    }

    /// Returns a flag indicates whether the tensor is using strided format
    ///
    /// @returns @c true if the strided format is used
    ///     @c false if opaque format is used
    bool is_valid_stride() const;

    /// Return a flag that indicates whether an logical tensor is presented
    ///
    /// @returns @c true if an optional logical tensor is presented
    bool is_avaliable() const;

    /// Returns a flag that indicates whether the logical tensor is constant for an OP
    ///
    /// @returns @c true if this logical tensor is constant for an OP
    bool is_constant() const;

    /// Returns size of tensor in bytes
    ///
    /// @returns The number of bytes required to allocate a tensor buffer
    ///     for the tensor object described by this logical tensor including
    ///     the padding size
    size_t get_size() const;

    /// Returns the layout of the tensor
    ///
    /// @returns Layout id
    llga_layout_id_t get_layout_id() const;

    /// Set the layout to the tensor desscribed by this logical tensor
    ///
    /// @param layout_id New layout id
    void set_layout_id(llga_layout_id_t layout_id);
};

class tensor {
public:
    using dims_t = std::vector<llga_dim_t>;

    /// Constructs a tensor object with data type
    ///
    /// @param data_type Data type
    tensor(llga_data_type_t data_type = llga_data_type_undef);

    /// Constructs a tensor object with data type, dims and data handle
    ///
    /// @param sizes Tensor dimensions
    /// @param handle Handle of memroy buffer to use as an underlying storage
    /// @param data_type Data type
    tensor(const dims_t &sizes, void *handle, llga_data_type_t data_type);

    /// Returns the underlying memory buffer with the specific type
    ///
    /// @tparam T Type of the request buffer
    /// @returns The underlying memory buffer
    template <typename T>
    std::add_pointer_t<T> get_data_handle() const;

    /// Sets the underlying memory buffer
    ///
    /// @param handle Data handle. For the CPU engine, the data handle
    ///     is a pointer to the actual data.
    void set_data_handle(void *handle);

    /// Returns the number of tensor's elements
    ///
    /// @returns Number of elements
    int64_t get_nelem() const;

    /// Returns the underlying logical tensor
    ///
    /// @returns A copy of logical tensor object
    logical_tensor get_logical_tensor();

    /// Returns unique id of this tensor
    ///
    /// @returns Unique id
    uint64_t get_id() const;
};

} // namespace api
} // namespace llga

#endif
