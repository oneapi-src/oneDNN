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

#ifndef LLGA_OP_HPP
#define LLGA_OP_HPP

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

#include "interned_strings.hpp"
#include "llga_api_detail.hpp"
#include "tensor.hpp"

namespace llga {
namespace api {

class op {
public:
    /// Contructs an Op object based on input/output tensors and attributes
    ///
    /// @param kind Op Kind
    /// @param input Input logical tensor to be bound to this op
    /// @param output Output logical tensor to be bound to this op
    /// @param attr Attributes to be bound to this op
    /// @param debug_string
    op(llga_op_kind_t kind, std::vector<logical_tensor> &inputs,
            std::vector<logical_tensor> &outputs,
            std::map<std::string, llga_attribute_kind> &attr, const std::string &debug_string);

    /// Returns the number of input logical tensor in the Op
    ///
    /// @returns Number of inputs
    uint64_t get_num_inputs() const;

    /// Returns the number of output logical tensor in the Op
    ///
    /// @returns Number of outputs
    uint64_t get_num_outputs() const;

    /// Returns the specified input logical tensor
    ///
    /// @param index Index of the request logical tensor
    /// @returns A copy of logical tensor object
    logical_tensor get_input(uint64_t index);

    /// Returns the specified output logical tensor
    ///
    /// @param index Index of the request logical tensor
    /// @returns A copy of logical tensor object
    logical_tensor get_output(uint64_t index);

    /// Returns the kind of specified attribute in the Op
    ///
    /// @param name Name of the attribute
    /// @returns Attribute kind
    llga_attribute_kind get_attr_kind(const std::string &name) const;

    /// Returns the attribute's kind according to the specified Type (int32_t)
    ///
    /// @tparam Attr Attribute's type
    /// @returns The llga attribute kind
    template <typename Attr,
            requires<std::is_same<Attr, int32_t>::value> = true>
    constexpr static llga_attribute_kind attr_kind() noexcept;

    /// Returns the attribute's kind according to the specified Type (std::vector<int32_t>)
    ///
    /// @tparam Attr Attribute's type
    /// @returns The llga attribute kind
    template <typename Attr,
            requires<std::is_same<Attr, std::vector<int32_t>>::value> = true>
    constexpr static llga_attribute_kind attr_kind() noexcept;

    /// Returns the attribute's kind according to the specified Type (float)
    ///
    /// @tparam Attr Attribute's type
    /// @returns The llga attribute kind
    template <typename Attr, requires<std::is_same<Attr, float>::value> = true>
    constexpr static llga_attribute_kind attr_kind() noexcept;

    /// Returns the attribute's kind according to the specified Type (std::vector<float>)
    ///
    /// @tparam Attr Attribute's type
    /// @returns The llga attribute kind
    template <typename Attr,
            requires<std::is_same<Attr, std::vector<float>>::value> = true>
    constexpr static llga_attribute_kind attr_kind() noexcept;

    /// Returns the attribute's according to the name and type (int32_t)
    ///
    /// @tparam Attr Attribute's type
    /// @returns A copy of attribute object
    template <typename Attr,
            requires<std::is_same<Attr, int32_t>::value> = true>
    Attr get_attr(const std::string &name) const;

    /// Returns the attribute according to the name and type (std::vector<int32_t>)
    ///
    /// @tparam Attr Attribute's type
    /// @returns A copy of attribute object
    template <typename Attr,
            requires<std::is_same<Attr, std::vector<int32_t>>::value> = true>
    Attr get_attr(const std::string &name) const;

    /// Returns the attribute according to the name and type (float)
    ///
    /// @tparam Attr Attribute's type
    /// @returns A copy of attribute object
    template <typename Attr, requires<std::is_same<Attr, float>::value> = true>
    Attr get_attr(const std::string &name) const;

    /// Returns the attribute according to the name and type (std::vector<float>)
    ///
    /// @tparam Attr Attribute's type
    /// @returns A copy of attribute object
    template <typename Attr,
            requires<std::is_same<Attr, std::vector<float>>::value> = true>
    Attr get_attr(const std::string &name) const;

    /// Sets the attribute according to the name and type,
    ///     currently for int32_t and float type.
    ///
    /// @tparam Attr Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Attr, requires<std::is_fundamental<Attr>::value> = true>
    op &set_attr(const std::string &name, const Attr &a);

    /// Sets the attribute according to the name and type
    ///     currently for std::vector<int32_t> and std::vector<float> type.
    ///
    /// @tparam Attr Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Attr, requires<!std::is_fundamental<Attr>::value> = true>
    op &set_attr(const std::string &name, const Attr &a);

    /// Return the unique id of the Op
    ///
    /// @returns Unique id
    uint64_t get_id() const;

    /// Returns the Op kind
    ///
    /// @returns Op kind
    llga_op_kind_t get_kind() const;

    /// Returns the string format of the Op kind
    ///
    /// @returns Ok kind in string format
    std::string to_string() const;
};
} // namespace api
} // namespace llga

#endif
