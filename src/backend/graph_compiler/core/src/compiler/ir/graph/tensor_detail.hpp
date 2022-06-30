/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TENSOR_DETAIL_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TENSOR_DETAIL_HPP

#include <algorithm>
#include <functional>

#include <iosfwd>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/ir/sc_data_type.hpp>
#include <util/assert.hpp>
namespace sc {

namespace reflection {
template <typename T, typename Dummy>
struct type_registry;
}

struct SC_INTERNAL_API logical_tensor_t {
public:
    sc_data_type_t dtype_;

    logical_tensor_t() = default;

    bool operator==(const logical_tensor_t &other) const {
        return format_ == other.format_ && dtype_ == other.dtype_
                && plain_dims_ == other.plain_dims_
                && strides_ == other.strides_;
    }

    bool operator!=(const logical_tensor_t &other) const {
        return !(*this == other);
    }

    logical_tensor_t(const sc_data_format_t &format, const sc_dims &plain_dims,
            const sc_data_type_t &type, const sc_dims &strides = {})
        : dtype_(type)
        , format_(format)
        , plain_dims_(plain_dims)
        , strides_(strides) {
        internal_update();
    }

    // gets the dims, taking blocking into consideration, using cache
    const sc_dims &get_blocking_dims() const;
    // gets the logical dims in plain format
    const sc_dims &get_plain_dims() const { return plain_dims_; }
    // gets strides corresponding to each blocking dim
    const sc_dims &get_strides() const { return strides_; }
    // sets the logical dims in plain format
    void set_plain_dims(const sc_dims &plain_dims);
    // sets the logical dims in blocking format
    void set_blocking_dims(const sc_dims &blocking_dims);
    // gets the data format
    const sc_data_format_t &get_format() const { return format_; }
    // sets the data format and invalidate the cached blocking_dims
    void set_format(const sc_data_format_t &newv);
    // sets the strides
    void set_strides(const sc_dims &strides);
    // sets the data format and stride
    void set_format_and_stride(
            const sc_data_format_t &newv, const sc_dims &strides);
    // gets the size of the tensor in bytes
    size_t size() const;
    // judge whether the current logical tensor is dense
    bool is_dense();
    // print the tensor detail to string
    void to_string(std::ostream &os);
    // used to compute dense stride based on dims
    static sc_dims compute_dense_stride(const sc_dims &dims);

private:
    template <typename T, typename Dummy>
    friend struct reflection::type_registry;
    friend struct std::hash<sc::logical_tensor_t>;
    sc_data_format_t format_;
    // The real dims, which may be blocking.
    sc_dims dims_;
    // the logical dims in plain format
    sc_dims plain_dims_;
    // strides corresponding to each dim of `dims_`
    sc_dims strides_;
    // sync real dims based on plain dims and format
    void internal_update();
};
} // namespace sc

namespace std {
template <>
struct hash<sc::logical_tensor_t> {
    std::size_t operator()(const sc::logical_tensor_t &k) const;
};
} // namespace std
#endif
