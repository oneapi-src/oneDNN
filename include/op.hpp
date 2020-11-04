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
    /// enumeration for Op kind
    enum class kind {
        Abs = kAbs,
        Add = kAdd,
        AvgPool = kAvgPool,
        AvgPoolBackprop = kAvgPoolBackprop,
        BatchNormInference = kBatchNormInference,
        BatchNormForwardTraining = kBatchNormForwardTraining,
        BatchNormTrainingBackprop = kBatchNormTrainingBackprop,
        BiasAddBackprop = kBiasAddBackprop,
        Clamp = kClamp,
        ClampBackprop = kClampBackprop,
        Concat = kConcat,
        Convolution = kConvolution,
        ConvolutionBackpropData = kConvolutionBackpropData,
        ConvolutionBackpropFilters = kConvolutionBackpropFilters,
        Elu = kElu,
        EluBackprop = kEluBackprop,
        Exp = kExp,
        GELU = kGELU,
        GELUBackprop = kGELUBackprop,
        HardTanh = kHardTanh,
        LayerNorm = kLayerNorm,
        LayerNormBackprop = kLayerNormBackprop,
        Log = kLog,
        LogSoftmax = kLogSoftmax,
        LogSoftmaxBackprop = kLogSoftmaxBackprop,
        MatMul = kMatMul,
        Maximum = kMaximum,
        MaxPool = kMaxPool,
        MaxPoolBackprop = kMaxPoolBackprop,
        Minimum = kMinimum,
        Multiply = kMultiply,
        Pow = kPow,
        PowBackprop = kPowBackprop,
        ReLU = kReLU,
        ReLUBackprop = kReLUBackprop,
        Reshape = kReshape,
        Round = kRound,
        Sigmoid = kSigmoid,
        SigmoidBackprop = kSigmoidBackprop,
        SoftMax = kSoftMax,
        SoftMaxBackprop = kSoftMaxBackprop,
        SoftPlus = kSoftPlus,
        SoftPlusBackprop = kSoftPlusBackprop,
        Sqrt = kSqrt,
        SqrtBackprop = kSqrtBackprop,
        Square = kSquare,
        Tanh = kTanh,
        TanhBackprop = kTanhBackprop,
        Wildcard = kWildcard,
        BiasAdd = kBiasAdd,
        LastSymbol = kLastSymbol,
    };

    /// Constructs an OP object
    ///
    /// @param id The unique id of this op
    /// @param akind The op kind specifies which computation is represented by
    ///     the op, such as Convolution and ReLU.
    /// @param debug_string The string added for debug
    op(size_t id, kind akind, const std::string &debug_string);

    /// Contructs an Op object based on input/output tensors and attributes
    ///
    /// @param id The unique id of this op.
    /// @param akind The op kind specifies which computation is represented by
    ///     this op, such as Convolution and ReLU.
    /// @param inputs Input logical tensor to be bound to this op.
    /// @param outputs Output logical tensor to be bound to this op
    /// @param debug_string The string added for debug
    op(size_t id, kind akind, const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs,
            const std::string &debug_string);

    /// Adds input logical tensor to the op
    ///
    /// @param t Input logical tensor
    void add_input(const logical_tensor &t);

    /// Adds input logical tensors to the op
    ///
    /// @param ts The list of input logical tensors
    void add_inputs(const std::vector<logical_tensor> &ts);

    /// Adds output logical tensor to the op
    ///
    /// @param t Output logical tensor
    void add_output(const logical_tensor &t);

    /// Adds output logical tensors to the op
    ///
    /// @param ts The list of output logical tensors
    void add_outputs(const std::vector<logical_tensor> &ts);

    /// Returns the number of input logical tensor in the Op
    ///
    /// @returns Number of inputs
    uint64_t get_inputs_num() const;

    /// Returns the number of output logical tensor in the Op
    ///
    /// @returns Number of outputs
    uint64_t get_outputs_num() const;

    /// Returns the kind of specified attribute in the Op
    ///
    /// @param name Name of the attribute
    /// @returns Attribute kind
    llga_attribute_kind get_attr_kind(const std::string &name) const;

    /// Returns the attribute's according to the name and type (int64_t)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @returns A copy of attribute object
    template <typename Type,
            requires<std::is_same<Type, int64_t>::value> = true>
    Type get_attr(const std::string &name) const;

    /// Returns the attribute according to the name and type
    /// (std::vector<int64_t>)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @returns A copy of attribute object
    template <typename Type,
            requires<std::is_same<Type, std::vector<int64_t>>::value> = true>
    Type get_attr(const std::string &name) const;

    /// Returns the attribute according to the name and type (float)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @returns A copy of attribute object
    template <typename Type, requires<std::is_same<Type, float>::value> = true>
    Type get_attr(const std::string &name) const;

    /// Returns the attribute according to the name and type
    /// (std::vector<float>)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @returns A copy of attribute object
    template <typename Type,
            requires<std::is_same<Type, std::vector<float>>::value> = true>
    Type get_attr(const std::string &name) const;

    /// Returns the attribute according to the name and type (std::string)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @returns A copy of attribute object
    template <typename Type,
            requires<std::is_same<Type, std::string>::value> = true>
    Type get_attr(const std::string &name) const;

    /// Returns the attribute according to the name and type (bool)
    ///
    /// @tparam Attr Attribute's type
    /// @param name Attribute's name
    /// @returns A copy of attribute object
    template <typename Attr, requires<std::is_same<Attr, bool>::value> = true>
    Attr get_attr(const std::string &name) const;

    /// Sets the attribute according to the name and type (int64_t)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, int64_t>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Sets the attribute according to the name and type (float)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type, requires<std::is_same<Type, float>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Sets the attribute according to the name and type (bool)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type, requires<std::is_same<Type, bool>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Sets the attribute according to the name and type (string)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::string>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Sets the attribute according to the name and type
    /// (std::vector<int64_t>)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::vector<int64_t>>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Sets the attribute according to the name and type
    /// (std::vector<float>)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::vector<float>>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Returns the unique id of the Op
    ///
    /// @returns Unique id
    uint64_t get_id() const;

    /// Returns the concrete kind of this op
    ///
    /// @returns kind The op kind
    kind get_kind() const;

    /// Returns the string format of the Op id and kind
    ///
    /// @returns Op kind in string format
    std::string to_string() const;
};
} // namespace api
} // namespace llga

#endif
