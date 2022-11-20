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

#ifndef GRAPH_INTERFACE_OP_SCHEMA_HPP
#define GRAPH_INTERFACE_OP_SCHEMA_HPP

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "common/rw_mutex.hpp"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/op.hpp"

#include "graph/utils/attribute_value.hpp"

namespace dnnl {
namespace impl {
namespace graph {

using opset_version = size_t;
using shape_infer_fn = std::function<status_t(op_t *,
        std::vector<logical_tensor_t *> &, std::vector<logical_tensor_t *> &)>;
using type_constraint_fn = std::function<bool(const op_t *)>;

class op_schema_t {
public:
    op_schema_t();
    op_schema_t(op_kind_t op_name, opset_version version);

    /*! @brief op parameter representation, including input/output name,
     *  and description.
     */
    class op_parameter_t {
    public:
        op_parameter_t() = default;

        explicit op_parameter_t(std::string &&name, std::string &&description,
                std::string &&dtype_string)
            : name_(std::move(name))
            , description_(std::move(description))
            , dtype_string_(std::move(dtype_string))
            , is_initialized(true) {}

        // op parameter name.
        std::string name_;

        // op parameter description.
        std::string description_;

        // op parameter dtype string.
        std::string dtype_string_;

        // Flag marking whether this parameter has already been initialized
        bool is_initialized = false;
    };

    class attribute_t {
    public:
        attribute_t() = default;

        // constructor for optional attributes which need to have default value
        attribute_t(op_attr_t name, std::string &&description, bool required,
                attribute_kind_t attr_kind, utils::attribute_value_t value)
            : name_(name)
            , description_(std::move(description))
            , required_(false)
            , has_default_value_(true)
            , attr_kind_(attr_kind)
            , attr_(std::move(value)) {
            assertm(!required,
                    "this attribute should be an optional attribute "
                    "since default value is provided");
            UNUSED(required);
        }

        // constructor for required attributes or special optional attributes
        // that have no default value.
        attribute_t(op_attr_t name, std::string &&description, bool required,
                attribute_kind_t attr_kind)
            : name_(name)
            , description_(std::move(description))
            , required_(required)
            , has_default_value_(false)
            , attr_kind_(attr_kind) {}

        // op attribute name.
        op_attr_t name_;

        // op attribute description.
        std::string description_;

        // whether the attribute is required or not.
        bool required_;

        // some special optional attribute may not have default value.
        // i.e. momentum in BatchNormForwardTraining is an optional attribute,
        // but it should not have default value.
        bool has_default_value_;

        // attribute data type.
        attribute_kind_t attr_kind_;

        // default value for the attribute
        utils::attribute_value_t attr_;
    };

    enum class param_num_option { fixed, optional, variadic };
    /*! @brief Returns the op_kind of this op schema. */
    op_kind_t get_op_kind() const;

    /*! @brief Set the op_kind of this op schema. */
    op_schema_t &set_op_kind(op_kind_t kind);

    /*! @brief Returns the docstring of this op schema. */
    const std::string &get_doc() const;

    /*! @brief Set the docstring of this op schema. */
    op_schema_t &set_doc(std::string &&doc);

    /*! @brief Returns the since version of this op schema. */
    opset_version get_since_version() const;

    /*! \brief The earliest operator set version which this
     * operator was present in.
     */
    op_schema_t &since_version(opset_version n);

    /*! @brief Set num of inputs of the op schema. */
    op_schema_t &set_num_inputs(size_t input_num);

    /*! @brief Set num of inputs of the op schema for optional and variadic
     * inputs.
     */
    op_schema_t &set_num_inputs(std::set<size_t> &&input_num);

    /*! @brief Get num of inputs of the op schema. */
    std::set<size_t> get_num_inputs() const;

    /*! @brief Set num of outputs of the op schema. */
    op_schema_t &set_num_outputs(size_t output_num);

    /*! @brief Set num of outputs of the op schema for optional and variadic
     * outputs.
     */
    op_schema_t &set_num_outputs(std::set<size_t> &&output_num);

    /*! @brief Get num of outputs of the op schema.*/
    std::set<size_t> get_num_outputs() const;

    /*! @brief Set a particular input of the op schema. */
    op_schema_t &set_input(size_t in_offset, std::string &&in_name,
            std::string &&in_description, std::string &&dtype_string = "any");

    /*! @brief Set a particular output of the op schema. */
    op_schema_t &set_output(size_t out_offset, std::string &&out_name,
            std::string &&out_description, std::string &&dtype_string = "any");

    /*! @brief Enable commutative inputs */
    op_schema_t &set_commutative_inputs();

    /*! @brief Get whether the commutative inputs option is enabled or not */
    bool get_commutative_inputs() const;

    op_schema_t &set_type_constraints(
            std::string &&dtype_string, std::set<data_type_t> &&dtypes);

    /*! @brief Set a particular attribute of the op schema. */
    op_schema_t &set_attr(op_attr_t name, std::string &&description,
            bool required, attribute_kind_t attr_kind);

    /*! @brief Set a particular attribute of the op schema. */
    template <typename T>
    op_schema_t &set_attr(op_attr_t name, std::string &&description,
            bool required, attribute_kind_t attr_kind, T value) {
        assertm(attributes_.count(name) == 0,
                "provided attribute has already been set");
        attributes_[name] = attribute_t(
                name, std::move(description), required, attr_kind, {value});
        return *this;
    }

    /*! @brief Set a particular attribute of the op schema. */
    op_schema_t &set_attr(op_attr_t name, std::string &&description,
            bool required, attribute_kind_t attr_kind, const char *value);

    /*! @brief Set shape inference function of the op schema. */
    op_schema_t &set_shape_inference_function(shape_infer_fn fn);

    /*! @brief Get shape inference function of the op schema. */
    shape_infer_fn get_shape_inference_function() const;

    /*! @brief Set type constraint function of the op schema. */
    op_schema_t &set_type_constraint_function(type_constraint_fn fn);

    /*! @brief Get type constraint function of the op schema. */
    type_constraint_fn get_type_constraint_function() const;

    /*! @brief Get inputs of the op schema. */
    const std::vector<op_parameter_t> &get_inputs() const;

    /*! @brief Get outputs of the op schema. */
    const std::vector<op_parameter_t> &get_outputs() const;

    /*! @brief Get attributes of the op schema. */
    const std::unordered_map<op_attr_t, attribute_t> &get_attrs() const;

    /*! @brief Verify the op schema. */
    bool verify(const op_t *l_op, bool check_undefined_attrs = true) const;

    /*! @brief Infer shape with the op schema. */
    status_t shape_infer(op_t *n, std::vector<logical_tensor_t *> &inputs,
            std::vector<logical_tensor_t *> &outputs) const;

    /*! @brief Set inputs param option: fixed, optional and variadic. */
    op_schema_t &set_inputs_option(param_num_option option);

    /*! @brief Get inputs param option: fixed, optional and variadic. */
    param_num_option get_inputs_option() const;

    /*! @brief Set outputs param option: fixed, optional and variadic. */
    op_schema_t &set_outputs_option(param_num_option option);

    /*! @brief Get outputs param option: fixed, optional and variadic. */
    param_num_option get_outputs_option() const;
    void set_default_attribute(op_t *l_op) const;

    /*! @brief Add additional item. The item can be any type*/
    op_schema_t &set_additional_item(
            const std::string &key, const utils::any_t &value);

    /*! @brief Get additional item. The item can be any type*/
    const utils::any_t &get_additional_item(const std::string &key) const;

    /*! @brief Add additional item. The item can be specified by template*/
    template <typename T>
    op_schema_t &set_additional_item(const std::string &key, const T &value) {
        return set_additional_item(key, utils::any_t {value});
    }

    /*! @brief Get additional item. The item can be specified by template*/
    template <typename T>
    T get_additional_item(const std::string &key) const {
        return utils::any_cast<T>(get_additional_item(key));
    }

    bool has_additional_item(const std::string &key) const;

private:
    void verify_input_(size_t in_offset);
    void verify_output_(size_t out_offset);
    bool verify_param_num(size_t actual_num,
            const std::set<size_t> &expected_num,
            param_num_option option) const;
    bool verify_param_dtype(
            const std::vector<std::shared_ptr<value_t>> &actual_values,
            const std::vector<op_schema_t::op_parameter_t> &expected_params,
            param_num_option option,
            std::unordered_map<std::string, std::set<data_type_t>>
                    &dtype_constraints) const;
    bool verify_attributes(
            const std::unordered_map<op_attr_t, utils::attribute_value_t>
                    &actual_attrs,
            const std::unordered_map<op_attr_t, attribute_t> &expected_attrs,
            bool check_undefined_attrs) const;
    size_t get_max_valid_param_num(
            const std::set<size_t> &param_num, param_num_option option) const;

    std::string doc_ = "";
    op_kind_t op_kind_;
    opset_version version_;
    std::set<size_t> num_inputs_;
    std::set<size_t> num_outputs_;
    std::set<size_t> inputs_offset;
    std::set<size_t> outputs_offset;
    // allowed data types for each dtype string
    std::unordered_map<std::string, std::set<data_type_t>>
            op_parameter_dtype_map_;
    param_num_option inputs_option = param_num_option::fixed;
    param_num_option outputs_option = param_num_option::fixed;
    std::vector<op_parameter_t> inputs_;
    std::vector<op_parameter_t> outputs_;
    std::unordered_map<op_attr_t, attribute_t> attributes_;
    shape_infer_fn tensor_inference_function_ = nullptr;
    type_constraint_fn op_type_constraint_function_ = nullptr;
    bool commutative_inputs_enabled_ = false;
    // type erased key-value storage
    std::unordered_map<std::string, utils::any_t> additional_items_map_;
};

using op_kind_version_schema_map
        = std::unordered_map<op_kind_t, std::map<opset_version, op_schema_t>>;

class op_schema_registry_t {
public:
    class op_schema_registry_once_t {
    public:
        op_schema_registry_once_t(op_schema_t &&schema);
    };

    /*! @brief Get the latest schema for an op. */
    static const op_schema_t *get_op_schema(op_kind_t a_op_kind);

private:
    /* !@brief Returns the static op_kind_version_schema_map.*/
    static op_kind_version_schema_map &get_map_without_ensuring_registration();
    static op_kind_version_schema_map &get_map();
    static impl::utils::rw_mutex_t &get_rw_mutex();
};

#ifdef GNUC
#define DNNL_GRAPH_UNUSED __attribute__((__unused__))
#else
#define DNNL_GRAPH_UNUSED
#endif // GNUC

void register_schema(op_schema_t &&schema);

template <class T>
void register_opset_schema() {
    T::for_each_schema(register_schema);
}

template <typename T>
op_schema_t get_op_schema();

#define DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(opkind, version) \
    _dnnl_graph_op_schema_##opkind##_##version##_

#define DNNL_GRAPH_OP_SCHEMA(opkind, version, impl) \
    class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(opkind, version); \
    template <> \
    inline op_schema_t \
    get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(opkind, version)>() { \
        return (impl).set_op_kind(op_kind::opkind).since_version(version); \
    }

#define SET_MATMUL_COMMON_ATTRS \
    set_attr(op_attr::transpose_a, \
            "transposes dimensions ROW_INDEX_DIM and COL_INDEX_DIM " \
            "of the first input", \
            false, attribute_kind::b, false) \
            .set_attr(op_attr::transpose_b, \
                    "transposes dimensions ROW_INDEX_DIM and COL_INDEX_DIM " \
                    "of the second input", \
                    false, attribute_kind::b, false)

#define SET_CONV_COMMON_ATTRS \
    set_attr(op_attr::strides, "the distance to slide the filter", true, \
            attribute_kind::is) \
            .set_attr(op_attr::pads_begin, "top and left padding", true, \
                    attribute_kind::is) \
            .set_attr(op_attr::pads_end, "bottom and right padding", true, \
                    attribute_kind::is) \
            .set_attr(op_attr::dilations, \
                    "the distance in width and height between elements " \
                    "in the filter", \
                    true, attribute_kind::is) \
            .set_attr(op_attr::auto_pad, "how the padding is calculated", \
                    false, attribute_kind::s, "None") \
            .set_attr(op_attr::groups, \
                    "the number of groups input / output channels are " \
                    "divided into", \
                    false, attribute_kind::i, (int64_t)1) \
            .set_attr(op_attr::data_format, \
                    "the data format of input / output, the options are " \
                    "NCX and NXC", \
                    false, attribute_kind::s, "NXC") \
            .set_attr(op_attr::weights_format, \
                    "the format of weight, the options are OIX, XIO", false, \
                    attribute_kind::s, "XIO")

#define SET_CONVTRANSPOSE_COMMON_ATTRS \
    set_attr(op_attr::strides, "the distance to slide the filter", true, \
            attribute_kind::is) \
            .set_attr(op_attr::pads_begin, "top and left padding", true, \
                    attribute_kind::is) \
            .set_attr(op_attr::pads_end, "bottom and right padding", true, \
                    attribute_kind::is) \
            .set_attr(op_attr::dilations, \
                    "the distance in width and height between elements " \
                    "in the filter", \
                    true, attribute_kind::is) \
            .set_attr(op_attr::auto_pad, "how the padding is calculated", \
                    false, attribute_kind::s, "None") \
            .set_attr(op_attr::groups, \
                    "the number of groups input / output channels are " \
                    "divided into", \
                    false, attribute_kind::i, (int64_t)1) \
            .set_attr(op_attr::data_format, \
                    "the data format of input / output, the options are " \
                    "NCX and NXC", \
                    false, attribute_kind::s, "NXC") \
            .set_attr(op_attr::weights_format, \
                    "the format of weight, the options are IOX, XOI", false, \
                    attribute_kind::s, "XOI")

#define SET_REDUCE_COMMON_ATTRS \
    set_attr(op_attr::axes, \
            "specifies indices of input data, along which the reduction is " \
            "performed.", \
            false, attribute_kind::is, std::vector<int64_t>()) \
            .set_attr(op_attr::keep_dims, \
                    "if true, holds axes that are used for reduction.", false, \
                    attribute_kind::b, false)

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
