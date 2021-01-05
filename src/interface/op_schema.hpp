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

#ifndef LLGA_INTERFACE_OP_SCHEMA_HPP
#define LLGA_INTERFACE_OP_SCHEMA_HPP

#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "c_types_map.hpp"
#include "ir.hpp"
#include "op.hpp"

namespace dnnl {
namespace graph {
namespace impl {

using opset_version = size_t;
using shape_infer_fn = std::function<status_t(node_t *,
        std::vector<logical_tensor_t *> &, std::vector<logical_tensor_t *> &)>;

class op_schema {
public:
    op_schema();
    op_schema(std::string op_name, opset_version version);

    /*! @brief formal parameter representation, including input/output name,
     *  and description.
     */
    class formal_parameter {
    public:
        formal_parameter() = default;

        explicit formal_parameter(
                std::string name, const std::string &description)
            : name_(std::move(name))
            , description_(description)
            , is_initialized(true) {}

        // Formal parameter name.
        std::string name_;

        // Formal parameter description.
        std::string description_;

        // Flag marking whether this parameter has already been initialized
        bool is_initialized = false;
    };

    class attribute {
    public:
        attribute() = default;
        attribute(std::string name, std::string description, bool required);
        attribute(std::string name, std::string description, bool required,
                utils::any attr_value);

        std::string name_;
        std::string description_;
        utils::any attr_;
        bool required_;
    };

    enum class param_num_option { fixed, optional, variadic };
    /*! @brief Returns the name of this op schema. */
    const std::string &get_name() const;

    /*! @brief Set the name of this op schema. */
    op_schema &set_name(const std::string &name);

    /*! @brief Returns the docstring of this op schema. */
    const std::string &get_doc() const;

    /*! @brief Set the docstring of this op schema. */
    op_schema &set_doc(const std::string &doc);

    /*! @brief Returns the since version of this op schema. */
    opset_version get_since_version() const;

    /*! \brief The earliest operator set version which this
     * operator was present in.
     */
    op_schema &since_version(opset_version n);

    /*! @brief Set num of inputs of the op schema. */
    op_schema &set_num_inputs(size_t input_num);

    /*! @brief Set num of inputs of the op schema for optional and variadic
     * inputs.
     */
    op_schema &set_num_inputs(std::set<size_t> input_num);

    /*! @brief Get num of inputs of the op schema. */
    std::set<size_t> get_num_inputs() const;

    /*! @brief Set num of outputs of the op schema. */
    op_schema &set_num_outputs(size_t output_num);

    /*! @brief Set num of outputs of the op schema for optional and variadic
     * outputs.
     */
    op_schema &set_num_outputs(std::set<size_t> output_num);

    /*! @brief Get num of outputs of the op schema. */
    std::set<size_t> get_num_outputs() const;

    /*! @brief Set a particular input of the op schema. */
    op_schema &set_input(size_t in_offset, std::string in_name,
            const std::string &in_description);

    /*! @brief Set a particular output of the op schema. */
    op_schema &set_output(size_t out_offset, std::string out_name,
            const std::string &out_description);

    /*! @brief Set a particular attribute of the op schema. */
    op_schema &set_attr(std::string attr_name,
            const std::string &attr_description, bool required = true);

    /*! @brief Set a particular attribute of the op schema. */
    template <typename T>
    op_schema &set_attr(std::string attr_name,
            const std::string &attr_description, bool required, T attr_value) {
        assertm(attributes_.count(attr_name) == 0,
                "provided attribute has already been set");
        attributes_[attr_name]
                = attribute(attr_name, attr_description, required, attr_value);
        return *this;
    }

    /*! @brief Set a particular attribute of the op schema. */
    op_schema &set_attr(std::string attr_name,
            const std::string &attr_description, bool required,
            const char *attr_value);

    /*! @brief Set shape inference function of the op schema. */
    op_schema &set_shape_inference_function(shape_infer_fn fn);

    /*! @brief Get shape inference function of the op schema. */
    shape_infer_fn get_shape_inference_function() const;

    /*! @brief Get inputs of the op schema. */
    const std::vector<formal_parameter> &get_inputs() const;

    /*! @brief Get outputs of the op schema. */
    const std::vector<formal_parameter> &get_outputs() const;

    /*! @brief Get attributes of the op schema. */
    const std::unordered_map<std::string, attribute> &get_attrs() const;

    /*! @brief Verify the op schema. */
    bool verify(op_t *l_op) const;

    /*! @brief Infer shape with the op schema. */
    status_t shape_infer(node_t *n, std::vector<logical_tensor_t *> &inputs,
            std::vector<logical_tensor_t *> &outputs) const;

    /*! @brief Set inputs param option: fixed, optional and variadic. */
    op_schema &set_inputs_option(param_num_option option);

    /*! @brief Get inputs param option: fixed, optional and variadic. */
    param_num_option get_inputs_option() const;

    /*! @brief Set outputs param option: fixed, optional and variadic. */
    op_schema &set_outputs_option(param_num_option option);

    /*! @brief Get outputs param option: fixed, optional and variadic. */
    param_num_option get_outputs_option() const;
    void set_default_attribute(op_t *l_op) const;

private:
    void validate_input_(size_t in_offset);
    void validate_output_(size_t out_offset);
    bool verify_param_num(size_t actual_num, std::set<size_t> expected_num,
            param_num_option option) const;
    size_t get_max_valid_param_num(
            std::set<size_t> param_num, param_num_option option) const;

    std::string doc_;
    std::string name_;
    opset_version version_;
    std::set<size_t> num_inputs_;
    std::set<size_t> num_outputs_;
    std::set<size_t> inputs_offset;
    std::set<size_t> outputs_offset;
    param_num_option inputs_option = param_num_option::fixed;
    param_num_option outputs_option = param_num_option::fixed;
    std::vector<formal_parameter> inputs_;
    std::vector<formal_parameter> outputs_;
    std::unordered_map<std::string, attribute> attributes_;
    shape_infer_fn tensor_inference_function_;
};

using op_name_version_schema_map
        = std::unordered_map<std::string, std::map<opset_version, op_schema>>;

class op_schema_registry {
public:
    class op_schema_registry_once {
    public:
        op_schema_registry_once(op_schema &schema);
    };

    /*! @brief Get the latest schema for an op. */
    static const op_schema *get_op_schema(op_kind_t a_op_kind);

private:
    /* !@brief Returns the static op_name_version_schema_map.*/
    static op_name_version_schema_map &get_map_without_ensuring_registration();
    static op_name_version_schema_map &get_map();
};

#define DNNL_GRAPH_UNUSED __attribute__((__unused__))

void register_schema(op_schema &&schema);

template <class T>
void register_opset_schema() {
    T::for_each_schema(register_schema);
}

template <typename T>
op_schema get_op_schema();

#define DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(opname, version) \
    _dnnl_graph_op_schema_##opname##_##verion##_

#define DNNL_GRAPH_OP_SCHEMA(opname, version, impl) \
    class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(opname, version); \
    template <> \
    op_schema \
    get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(opname, version)>() { \
        return impl.set_name(#opname).since_version(version); \
    }

#define SET_MATMUL_COMMON_ATTRS \
    set_attr("transpose_a", \
            "transposes dimensions ROW_INDEX_DIM and COL_INDEX_DIM " \
            "of the first input", \
            false, false) \
            .set_attr("transpose_b", \
                    "transposes dimensions ROW_INDEX_DIM and COL_INDEX_DIM " \
                    "of the second input", \
                    false, false)

#define SET_CONV_COMMON_ATTRS \
    set_attr("strides", "the distance to slide the filter", true) \
            .set_attr("pads_begin", "top and left padding", true) \
            .set_attr("pads_end", "bottom and right padding", true) \
            .set_attr("dilations", \
                    "the distance in width and height between elements " \
                    "in the filter", \
                    true) \
            .set_attr("auto_pad", "how the padding is calculated", false, \
                    "None") \
            .set_attr("groups", \
                    "the number of groups input / output channels are " \
                    "divided into", \
                    false, (int64_t)1) \
            .set_attr("data_format", \
                    "the data format of input / output, the options are " \
                    "NCX and NXC", \
                    false, "NXC") \
            .set_attr("filter_format", \
                    "the format of weight, the options are OIX, XIO", false, \
                    "XIO")

} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
