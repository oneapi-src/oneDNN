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
    llga_logical_tensor_t data;

public:
    using dims_t = std::vector<llga_dim_t>;

    /// Data Type
    enum class data_type {
        undef = llga_data_type_undef,
        /// 16-bit/half-precision floating point.
        f16 = llga_f16,
        /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
        bf16 = llga_bf16,
        /// 32-bit/single-precision floating point.
        f32 = llga_f32,
        /// 32-bit signed integer.
        s32 = llga_s32,
        /// 8-bit signed integer.
        s8 = llga_s8,
        /// 8-bit unsigned integer.
        u8 = llga_u8,
    };

    /// Layout type
    enum class layout_type {
        undef = llga_layout_type_undef,
        any = llga_layout_type_any,
        strided = llga_layout_type_strided,
        opaque = llga_layout_type_opaque,
    };

    /// Constructs a logical tensor object
    ///
    /// @param c_data A C API handle of logical tensor
    explicit logical_tensor(const llga_logical_tensor_t &c_data);

    /// Copy constructor
    logical_tensor(const logical_tensor &other) = default;

    /// Assignment operator
    logical_tensor &operator=(const logical_tensor &other) = default;

    /// Constructs a logical tensor object
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param ndims Number of dimension, -1 means it's unknown, 0 means scalar
    /// @param ltype Layout type
    logical_tensor(
            size_t tid, data_type dtype, int32_t ndims, layout_type ltype);

    /// Delegated construtor
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param ltype Layout type
    logical_tensor(
            size_t tid, data_type dtype, layout_type ltype = layout_type::undef);

    /// Constructs a logical tensor object
    ///
    /// @param tid Tensor id
    /// @param adims Tensor dimensions, -1 means a particular axis of dims is
    ///        unknown, or the axis can be deduced by its size and other axis.
    /// @param dtype Data type
    /// @param ltype Layout type
    logical_tensor(size_t tid, const dims_t &adims, data_type dtype,
            layout_type ltype);

    /// Constructs a logical tensor object
    ///
    /// @note The layout_type for this constructor will always be strided
    ///
    /// @param tid Tensor id
    /// @param adims Tensor dimensions, -1 means a particular axis of dims is
    /// @param strides Tensor strides
    /// @param dtype Data type
    logical_tensor(size_t tid, const dims_t &adims, const dims_t &strides,
            data_type dtype);

    /// Constructs an opaque logical tensor object which accepts layout id
    ///
    /// @param tid Tensor id
    /// @param adims Tensor dimensions, -1 means a particular axis of dims is
    ///        unknown, or the axis can be deduced by its size and other axis.
    /// @param lid Layout id
    /// @param dtype Data type
    logical_tensor(size_t tid, const dims_t &adims, size_t lid,
            data_type dtype);

    /// Returns dimensions of the logical tensor
    ///
    /// @returns A the dimensions vector
    dims_t get_dims() const;

    /// Returns unique id of the logical tensor
    ///
    /// @returns Id number
    size_t get_id() const;

    /// Returns data type of the logical tensor
    ///
    /// @returns The data type
    data_type get_data_type() const;

    /// Returns layout type of the logical tensor
    ///
    /// @returns The layout type
    layout_type get_layout_type() const;

    /// Returns the layout of the tensor
    ///
    /// @returns Layout id
    size_t get_layout_id() const;

    /// Returns strides of this logical tensor
    ///
    /// @returns A copy of strides vector
    dims_t get_strides() const;

    /// Get memory size required by this logical tensor
    ///
    /// @returns The memory size in bytes
    size_t get_mem_size() const;
};

class tensor {
    llga_logical_tensor_t data;
public:
    using dims_t = std::vector<llga_dim_t>;

    /// Default constructor. Constructs an empty object.
    tensor() = default;

    /// Constructs a tensor object according to the given logical tensor
    ///
    /// @param lt The given logical tensor
    /// @param handle Handle of memory buffer to use as an underlying storage,
    ///     if the ndims in the logical tensor is 0, data handle holds a scalar
    tensor(const logical_tensor &lt, void *handle);

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

    /// Returns the number of elements in the tensor
    ///
    /// @returns Number of element
    int64_t get_element_num() const;
};

} // namespace api
} // namespace llga

#endif
