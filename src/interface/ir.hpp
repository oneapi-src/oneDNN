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

#ifndef LLGA_INTERFACE_IR_HPP
#define LLGA_INTERFACE_IR_HPP

#include <algorithm>
#include <cassert>
#include <deque>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "c_types_map.hpp"
#include "id.hpp"
#include "op.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "utils/compatible.hpp"

namespace llga {
namespace impl {

class node_t;

template <bool B>
using requires = typename std::enable_if<B, bool>::type;

template <typename fvisit>
inline void dfs_visit(const std::vector<node_t *> &nodes, fvisit f_visit);

class value {
    using tensor = llga::impl::tensor;

private:
    /*! \brief The producer node of this value */
    node_t *producer_;
    /*! \brief The output offset of the producer node */
    size_t offset_;
    /*! \brief The tensor of this value */
    tensor *tensor_ = nullptr;

public:
    value(node_t *anode = nullptr,
            size_t offset = std::numeric_limits<size_t>::max())
        : producer_(anode), offset_(offset) {}
    /*! \brief Get the producer node */
    node_t *get_producer() const { return producer_; }
    /*! \brief Set the producer node */
    void set_producer(node_t *anode) { producer_ = anode; }
    /*! \brief Get the output offset of the producer node */
    size_t get_offset() const { return offset_; }
    /*! \brief Set the output offset of the producer node */
    void set_offset(size_t aoffset) { offset_ = aoffset; }
    /*! \brief Set this value
    * \param anode the producer node
    * \param aoffset the output offset of the producer node
    * \return void
    */
    void set_value(node_t *anode, size_t aoffset) {
        set_producer(anode);
        set_offset(aoffset);
    }
    /*! \brief Get the tensor of this value */
    tensor *get_tensor() const { return tensor_; }
    /*! \brief Set the tensor of this value */
    void set_tensor(tensor *t) { tensor_ = t; }
};

class attributes {
public:
    /*!
    * \brief Get a vector of all attributes names.
    */
    std::vector<std::string> attr_names() const {
        std::vector<std::string> names;
        for (auto &attr : attrs_) {
            names.push_back(attr.first);
        }
        return names;
    }

    /*!
    * \brief get the attributes number.
    */
    size_t num_attrs() const { return attrs_.size(); }

    /*!
    * \brief check the attribute type.
    */
    template <typename value_type>
    bool check_type(const utils::any &a) const {
        return a.type() == typeid(value_type);
    }

    /*!
    * \brief Register additional attributes.
    * \param attr_name The name of the attribute.
    * \param value The value to be set.
    * \tparam value_type The type of the value to be set.
    */
    template <typename value_type>
    status_t set_attr(const std::string &attr_name, // NOLINT(*)
            const value_type &value) {
        auto it = attrs_.find(attr_name);
        if (it != attrs_.end() && !check_type<value_type>(it->second)) {
            return status::invalid_type;
        }
        attrs_[attr_name] = utils::any(value);
        return status::success;
    }

    /*!
    * \brief Get additional registered attribute.
    * \param attr_name The name of the attribute.
    * \return An attribute of specified attr_name.
    * \tparam value_type The type of the attribute.
    */
    template <typename value_type>
    const value_type &get_attr(const std::string &attr_name) const {
        auto it = attrs_.find(attr_name);
        assertm(it != attrs_.end(), "don't have such attribute");
        assertm(check_type<value_type>(it->second) != false,
                "registered attribute has inconsistent types.");
        return utils::any_cast<const value_type &>(it->second);
    }

    /*!
    * \brief Check whether has this attribute.
    * \param attr_name The name of the attribute.
    * \return whether has this attribute.
    */
    bool has_attr(const std::string &attr_name) const {
        return attrs_.count(attr_name) != 0;
    }

    /*!
    * \brief Return attributes map for merging usage.
    * \return attributes map.
    */
    const std::unordered_map<std::string, utils::any> &get_attrs_map() const {
        return attrs_;
    }

    /*!
    * \brief Merge attributes from another node.
    * \param attrs_map The attributes map of another node.
    * \warning existing elements in original map will not be inserted.
    */
    void merge_attrs_map(
            const std::unordered_map<std::string, utils::any> &attrs_map) {
        attrs_.insert(attrs_map.begin(), attrs_map.end());
    }

    // /*!
    // * \brief compare attribute_a value equal attribute_b value .
    // *        value_type should be one of int64_t, float, string, bool,
    // *        vector<float>, vector<int64_t>. others may need overload "=="
    // * \param attr_a attribute_a value.
    // * \param attr_b attribute_b value.
    // */
    template <typename value_type>
    static bool compare_attr_value(
            utils::any const &attr_a, utils::any const &attr_b) {
        static_assert(std::is_same<float, value_type>::value
                        || std::is_same<int64_t, value_type>::value
                        || std::is_same<std::vector<float>, value_type>::value
                        || std::is_same<std::vector<int64_t>, value_type>::value
                        || std::is_same<std::string, value_type>::value
                        || std::is_same<bool, value_type>::value,
                "value_type should be one of int64_t, float, string, "
                "bool,vector<float>, vector<int64_t>");
        return utils::any_cast<const value_type &>(attr_a)
                == utils::any_cast<const value_type &>(attr_b);
    }

    /*!
    * \brief check if attribute_b has the same attrib value with same name of this node
    * \param attribute_b node to be compare
    * \param attr_name attribute name
    * \return bool: whether has the same value
    */
    bool is_same_attr_value(
            const attributes &attribute_b, const std::string &attr_name) const {
        const auto &atrr_a = get_attrs_map();
        const auto &atrr_b = attribute_b.get_attrs_map();
        auto it_a = atrr_a.find(attr_name);
        auto it_b = atrr_b.find(attr_name);
        if (it_a == atrr_a.end() || it_b == atrr_b.end()) {
            throw std::runtime_error("node a or b don't have such attribute - ["
                    + attr_name + "].\n");
        }

        if (std::type_index((it_a->second).type())
                != std::type_index((it_b->second).type())) {
            return false;
        } else {
            static const std::unordered_map<std::type_index,
                    std::function<bool(utils::any const &attr_a,
                            utils::any const &attr_b)>>
                    any_to_attr_cmp_fun_map {{std::type_index(typeid(float)),
                                                     compare_attr_value<float>},
                            {std::type_index(typeid(std::vector<float>)),
                                    compare_attr_value<std::vector<float>>},
                            {std::type_index(typeid(int64_t)),
                                    compare_attr_value<int64_t>},
                            {std::type_index(typeid(std::vector<int64_t>)),
                                    compare_attr_value<std::vector<int64_t>>},
                            {std::type_index(typeid(std::string)),
                                    compare_attr_value<std::string>},
                            {std::type_index(typeid(bool)),
                                    compare_attr_value<bool>}};
            auto it_type_fun = any_to_attr_cmp_fun_map.find(
                    std::type_index((it_a->second).type()));
            if (it_type_fun == any_to_attr_cmp_fun_map.end()) {
                std::string errors_str
                        = std::string("can't find pair <type, fun> for type")
                        + (it_a->second).type().name() + " , need add one ?\n";
                throw std::runtime_error(errors_str);
                return false;
            } else {
                // will return rslt of cmp any value by map std::functional
                return it_type_fun->second(it_a->second, it_b->second);
            }
        }
    }

    /*!
    * \brief check if nodeB has the same value for each attribute from this node
    * \param attribute_b attribute to be compare
    * \param expected a set of attribute which needn't to compare
    * \return bool: whether all attributes are equal
    */
    bool has_same_attr_values(const attributes &attribute_b,
            std::set<std::string> expected = {}) const {
        return std::all_of(attrs_.begin(), attrs_.end(),
                [&](const std::unordered_map<std::string,
                        utils::any>::value_type &attr) {
                    return expected.count(attr.first)
                            ? true
                            : is_same_attr_value(attribute_b, attr.first);
                });
    }

protected:
    ~attributes() = default;
    std::unordered_map<std::string, utils::any> attrs_;
};

class node_t : public id, public attributes {
    using pair_t = std::pair<size_t, size_t>;

private:
    /*! \brief The operator this node uses */
    op_t op_;
    /*! \brief The input values of this node */
    std::unordered_map<size_t, value> input_values_;
    /*! \brief The output nodes of this node */
    std::vector<node_t *> output_nodes_;
    /*! \brief name of this node */
    std::string name_;
    /*! \brief partition node ids in this node */
    std::vector<size_t> op_ids_;

    std::vector<logical_tensor_t> input_tensor_;
    std::vector<logical_tensor_t> output_tensor_;
    // Map from node input index -> (op id, op input offset)
    std::unordered_map<size_t, pair_t> input_tensor_map_;
    // Map from node output index -> (op id, op output offset)
    std::unordered_map<size_t, pair_t> output_tensor_map_;

public:
    node_t(op_kind_t kind) : op_(kind, std::string {}) {
        name_ = op_t::kind2str(kind) + "_" + std::to_string(id());
    }

    node_t(size_t op_id, op_kind_t kind) : op_(op_id, kind, std::string {}) {
        name_ = op_t::kind2str(kind) + "_" + std::to_string(id());
    }

    /*! \brief get the name of this node, with the form of {op_kind}_{id} */
    std::string get_name() const { return name_; }
    /*! \brief get op_kind of this node */
    op_kind_t get_op_kind() const { return op_.kind(); }
    /*! \brief get  op id of this node */
    size_t get_op_id() const { return op_.id(); }
    /*! \brief get num_inputs of this node */
    size_t num_inputs() const { return input_values_.size(); }
    /*! \brief get number of connected output_nodes of this node */
    size_t num_outputs() const { return output_nodes_.size(); }
    /*! \brief get number of input tensors of this node */
    size_t num_inputs_tensor() const { return input_tensor_.size(); }
    /*! \brief get number of output tensors of this node */
    size_t num_outputs_tensor() const { return output_tensor_.size(); }

    /*!
    * \brief Set input of this node.
    * \param offset The index of this node's inputs.
    * \param input_node The input node to this node.
    * \param input_offset The index of the input node's outputs.
    * \return
    */
    void set_input(size_t offset, node_t *input_node, size_t input_offset) {
        input_values_[offset] = value(input_node, input_offset);
        if (input_node != nullptr) input_node->add_output(this);
    }

    const std::unordered_map<size_t, pair_t> &get_input_tensor_map() const {
        return input_tensor_map_;
    }

    const std::unordered_map<size_t, pair_t> &get_output_tensor_map() const {
        return output_tensor_map_;
    }

    void add_input_tensors(const std::vector<logical_tensor_t> &input_tensor) {
        for (auto in_tensor : input_tensor) {
            input_tensor_map_[input_tensor_.size()]
                    = std::make_pair(op_.id(), input_tensor_.size());
            input_tensor_.push_back(in_tensor);
        }
    }
    /*!
     *\brief Add input tensor for fused node
     * \param in_tensor input tensor which will be added.
     * \param cur_node the current node
     * \param in_offset input tensor index
     * \return
     */
    void add_input_tensors(const logical_tensor_t &in_tensor, node_t *cur_node,
            size_t in_offset) {
        auto in_tensor_map = cur_node->get_input_tensor_map();
        assertm(in_tensor_map.find(in_offset) != in_tensor_map.end(),
                "find this key fail");
        input_tensor_map_[input_tensor_.size()] = in_tensor_map[in_offset];
        input_tensor_.push_back(in_tensor);
    }

    void add_output_tensors(
            const std::vector<logical_tensor_t> &output_tensor) {
        for (auto out_tensor : output_tensor) {
            output_tensor_map_[output_tensor_.size()]
                    = std::make_pair(op_.id(), output_tensor_.size());
            output_tensor_.push_back(out_tensor);
        }
    }
    /*!
     *\brief Add output tensor for fused node
     * \param output_tensor  ouput tensor list which will be added.
     * \param out_node out_tensor from out_node
     * \param out_offset out_tensor index
     * \return
     */
    void add_output_tensors(const logical_tensor_t &out_tensor,
            node_t *out_node, size_t out_offset) {
        auto out_tensor_map = out_node->get_output_tensor_map();
        assertm(out_tensor_map.find(out_offset) != out_tensor_map.end(),
                "find this key fail");
        output_tensor_map_[output_tensor_.size()] = out_tensor_map[out_offset];
        output_tensor_.push_back(out_tensor);
    }

    /*!
    * \brief Get the offset's input node of this node's.
    * \param offset The index of this node's inputs.
    * \return input node
    */
    node_t *get_input_node(size_t offset) {
        return input_values_[offset].get_producer();
    }

    /*!
    * \brief swap input value of this node's.
    * \param offset1 The first index of this node's input value.
    * \param offset2 The second index of this node's input value.
    */
    void swap_input_value(size_t offset1, size_t offset2) {
        std::swap(input_values_[offset1], input_values_[offset2]);
    }

    /*!
    * \brief The offset's input of this node is connected to the input_offset's output of input node.
    * \param offset The index of this node's inputs.
    * \return input_offset
    */
    size_t get_input_offset(size_t offset) {
        return input_values_[offset].get_offset();
    }

    /*!
    * \brief To find all input_value_offsets produced by a specific input node.
    * \param inode The specific input node.
    * \param offsets The indexes of this node's input values produced by inode.
    */
    void get_input_offsets(node_t *inode, std::vector<size_t> &offsets) {
        offsets.clear();
        for (auto &value : input_values_) {
            if (value.second.get_producer() == inode)
                offsets.push_back(value.second.get_offset());
        }
    }

    bool has_input_value(size_t idx) {
        auto iter = input_values_.find(idx);
        return iter != input_values_.end();
    }

    /*!
    * \brief Find the offset of an input node
    * \param node* the input node
    * \param input_offsets the found offsets of the input node
    * \return bool
    */
    bool find_input_nodes(node_t *in_node, std::vector<size_t> &input_offsets) {
        input_offsets.clear();
        for (auto &value : input_values_) {
            if (value.second.get_producer() == in_node)
                input_offsets.push_back(value.first);
        }

        if (input_offsets.size() > 0) {
            return true;
        } else {
            return false;
        }
    }

    /*!
    * \brief Add output node of this node.
    * \param output_node The output node to this node.
    * \return void
    */
    void add_output(node_t *output_node) {
        if (std::find(output_nodes_.begin(), output_nodes_.end(), output_node)
                == output_nodes_.end())
            output_nodes_.push_back(output_node);
    }

    /*!
    * \brief Get this node's output.
    * \param offset The index of this node's outputs.
    * \return output node
    */
    node_t *get_output_node(size_t offset) const {
        return output_nodes_[offset];
    }

    /*!
    * \brief Find output node with attr equals to pattern_node
    * \param node* The pointer of pattern_node
    * \param pattern_index the found index of the pattern_node
    * \param from_index* the search starts from from_index
    * \return bool
    */
    bool find_output_node(node_t *pattern_node, size_t *pattern_index,
            size_t from_index = 0) {
        if (from_index > num_outputs() - 1) { return false; }
        for (size_t i = from_index; i < num_outputs(); ++i) {
            if (output_nodes_[i]->get_op_kind()
                    == pattern_node->get_op_kind()) {
                *pattern_index = i;
                return true;
            }
        }
        return false;
    }

    /*!
    * \brief Swap two output nodes
    * \param offset1 The index of one output node.
    * \param offset2 The index of the other output node.
    * \return void
    */
    void swap_output_node(size_t offset1, size_t offset2) {
        if (offset1 > num_outputs() - 1 || offset2 > num_outputs() - 1) {
            return;
        }
        if (offset1 == offset2) return;
        std::swap(output_nodes_[offset1], output_nodes_[offset2]);
    }

    /*!
    * \brief delete the connection between an output node and this node.
    * \param output_node The output node to be deleted.
    * \return void
    */
    void remove_output(node_t *output_node) {
        for (auto it = output_nodes_.begin(); it != output_nodes_.end(); ++it) {
            if (*it == output_node) {
                output_nodes_.erase(it);
                break;
            }
        }
    }

    /*!
    * \brief Add llga op ids to this node.
    * \param id llga op id.
    * \return void
    */
    void add_op_ids(size_t id) {
        //TODO(llga): copy all the attrs from anode
        //so that anode can be deleted
        op_ids_.push_back(id);
    }

    /*!
    * \brief Add llga op ids to this node.
    * \param id_vec llga op id vector.
    * \return void
    */
    void add_op_ids(const std::vector<size_t> &id_vec) {
        for (auto id : id_vec)
            op_ids_.push_back(id);
    }

    /*!
    * \brief Get llga op ids in this node.
    * \return a vector of llga op ids
    */
    const std::vector<size_t> &get_op_ids() const { return op_ids_; }

    const logical_tensor_t &get_input_tensor(size_t offset) const {
        return input_tensor_[offset];
    }

    const logical_tensor_t &get_output_tensor(size_t offset) const {
        return output_tensor_[offset];
    }

    /*! \brief parse attributes from llga op.
    * \param l_op llga op.
    */
    void parse_op_attr(const op_t *l_op) {
        status_t state = status::success;
        for (const auto &a : l_op->attributes()) {
            attribute_kind_t kind;
            state = l_op->kind_of(a.first, kind);
            assertm(state == status::success,
                    "failed to query the kind of attribute according to the "
                    "name.");
            switch (kind) {
                case attribute_kind::i: {
                    const int64_t *ival {nullptr};
                    const auto **ret = &ival;
                    state = l_op->attr<int64_t>(a.first, ret);
                    assertm(state == status::success,
                            "failed to get attribute according to the name.");
                    state = this->set_attr<int64_t>(a.first, **ret);
                    assertm(state == status::success,
                            "failed to set attribute into this op.");
                } break;
                case attribute_kind::is: {
                    const std::vector<int64_t> *isval {nullptr};
                    const auto **ret = &isval;
                    state = l_op->attr<std::vector<int64_t>>(a.first, ret);
                    assertm(state == status::success,
                            "failed to get attribute according to the name.");
                    state = this->set_attr<std::vector<int64_t>>(
                            a.first, **ret);
                    assertm(state == status::success,
                            "failed to set attribute into this op.");
                } break;
                case attribute_kind::f: {
                    const float *fval {nullptr};
                    const auto **ret = &fval;
                    state = l_op->attr<float>(a.first, ret);
                    assertm(state == status::success,
                            "failed to get attribute according to the name.");
                    state = this->set_attr<float>(a.first, **ret);
                    assertm(state == status::success,
                            "failed to set attribute into this op.");
                } break;
                case attribute_kind::fs: {
                    const std::vector<float> *fsval {nullptr};
                    const auto **ret = &fsval;
                    state = l_op->attr<std::vector<float>>(a.first, ret);
                    assertm(state == status::success,
                            "failed to get attribute according to the name.");
                    state = this->set_attr<std::vector<float>>(a.first, **ret);
                    assertm(state == status::success,
                            "failed to set attribute into this op.");
                } break;
                case attribute_kind::s: {
                    const std::string *sval {nullptr};
                    const auto **ret = &sval;
                    state = l_op->attr<std::string>(a.first, ret);
                    assertm(state == status::success,
                            "failed to get attribute according to the name.");
                    state = this->set_attr<std::string>(a.first, **ret);
                    assertm(state == status::success,
                            "failed to set attribute into this op.");
                } break;
                case attribute_kind::b: {
                    const bool *bval {nullptr};
                    const auto **ret = &bval;
                    state = l_op->attr<bool>(a.first, ret);
                    assertm(state == status::success,
                            "failed to get attribute according to the name.");
                    state = this->set_attr<bool>(a.first, **ret);
                    assertm(state == status::success,
                            "failed to set attribute into this op.");
                } break;
                default: break;
            }
        }
        UNUSED(state);
    }
};

template <typename HashType, typename fvisit, typename hashfunc,
        typename indegree, typename getinput>
void post_order_dfs_visit(const std::vector<node_t *> &nodes, fvisit f_visit,
        hashfunc hash_func, indegree in_degree, getinput get_input) {
    std::deque<std::pair<node_t *, uint32_t>> queue;
    std::unordered_set<HashType> visited;
    for (auto n : nodes) {
        HashType head_hash = hash_func(n);
        if (visited.count(head_hash) == 0) {
            queue.push_front(std::make_pair(n, 0));
            visited.insert(head_hash);
        }
        while (!queue.empty()) {
            std::pair<node_t *, uint32_t> &front = queue.front();
            if (front.second == in_degree(front.first)) {
                f_visit(front.first);
                queue.pop_front();
            } else {
                node_t *input = get_input(front.first, front.second++);
                HashType input_hash = hash_func(input);
                if (visited.count(input_hash) == 0) {
                    queue.push_front(std::make_pair(input, 0));
                    visited.insert(input_hash);
                }
            }
        }
    }
}

template <typename fvisit>
inline void dfs_visit(const std::vector<node_t *> &nodes, fvisit f_visit) {
    post_order_dfs_visit<size_t>(
            nodes, [f_visit](node_t *n) { f_visit(n); }, // fvisit
            [](node_t *n) -> size_t { return n->id(); }, // hashfunc
            [](node_t *n) -> size_t { // indegree
                if (!n) return 0;
                return n->num_inputs();
            },
            [](node_t *n, size_t index) -> node_t * { // getinput
                return n->get_input_node(index);
            });
}

} // namespace impl
} // namespace llga

#endif
