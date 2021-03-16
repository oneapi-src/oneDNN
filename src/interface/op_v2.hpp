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

#ifndef INTERFACE_OP_V2_HPP
#define INTERFACE_OP_V2_HPP

#include <limits>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "attribute_value.hpp"
#include "c_types_map.hpp"
#include "internal_ops.hpp"
#include "logical_tensor.hpp"
#include "value.hpp"

#include "utils/compatible.hpp"

namespace dnnl {
namespace graph {
namespace impl {
/// forward declaration
class op_schema;
class partition_impl_t;
} // namespace impl
} // namespace graph
} // namespace dnnl

/******************************************************************************
 * op functionalities:
 *  1. support frontend API
 *      - create op with id, kind, name string.
 *      - get id, kind, etc.
 *      - add input logical tensors.
 *      - add output logical tensors.
 *      - set/get attributes of the op.
 *  2. as a node on the graph
 *      - input logical tensor -> value -> one producer.
 *      - output logical tensor -> value -> multiple consumers.
 *      - set/get producers and consumers.
 *      - verify the op is legitimate, with op schema.
 *  3. as an internal (fused) node on the graph
 *      - create with id (not provided by users, how to generate?), kind, name string.
 *      - merge attributes from the source ops.
 *      - contain the ids of source ops.
 *      - fused op/node -> partition.
 * 
 *****************************************************************************/

struct dnnl_graph_op : public std::enable_shared_from_this<dnnl_graph_op> {
public:
    using op_kind_t = dnnl::graph::impl::op_kind_t;
    using logical_tensor_t = dnnl::graph::impl::logical_tensor_t;
    using attribute_kind_t = dnnl::graph::impl::attribute_kind_t;
    using status_t = dnnl::graph::impl::status_t;
    using attribute_value_t = dnnl::graph::impl::attribute_value;
    using value_t = dnnl::graph::impl::value_t;
    using pair_t = std::pair<size_t, size_t>; // <op_id, input/output offset>

    const static size_t DEFAULT_ID = std::numeric_limits<size_t>::max();

    // create dnnl_graph_op with explicit id, kind, and string
    dnnl_graph_op(
            size_t id, op_kind_t kind, std::string name, bool internal = false);

    // create dnnl_graph_op with default id, only for internal use.
    dnnl_graph_op(op_kind_t kind, std::string name)
        : dnnl_graph_op(DEFAULT_ID, kind, std::move(name), true) {}

    // convenient function to make tests happy
    dnnl_graph_op(op_kind_t kind)
        : dnnl_graph_op(DEFAULT_ID, kind, kind2str(kind), true) {}

    // TODO(xxx): why? any problem with copy constructor?
    ~dnnl_graph_op() {
        for (size_t i = 0; i < inputs_.size(); ++i) {
            inputs_[i]->remove_consumer(*this, i);
        }

        for (auto &v : outputs_) {
            if (&v->get_producer() == this) { v->reset_producer(); }
        }
    }

    // which op produced this input?
    dnnl_graph_op *get_input_op(size_t index) {
        return &(inputs_[index]->get_producer());
    }

    // TODO(xxx): we need to improve by checking more information
    bool operator==(const dnnl_graph_op &other) const {
        return this->get_id() == other.get_id()
                && this->get_kind() == other.get_kind()
                && this->get_name() == other.get_name()
                && this->is_internal() == other.is_internal();
    }

    bool operator!=(const dnnl_graph_op &other) const {
        return !operator==(other);
    }

    // some getters
    op_kind_t get_kind() const { return kind_; }
    size_t get_id() const { return id_; }
    const std::string &get_name() const { return name_; }
    const dnnl::graph::impl::op_schema *get_schema() const { return schema_; }
    bool is_internal() const { return internal_; }

    // verify the op against the schema
    bool verify() const;

    ///////////////////////////////////////////////////////////////////////////
    // input values
    size_t num_inputs() const { return inputs_.size(); }

    // add an input value to the op
    void add_input(std::shared_ptr<value_t> value) {
        // setup the input_tensor_map_
        const size_t offset = inputs_.size();
        input_tensor_map_[offset] = std::make_pair(id_, offset);

        inputs_.push_back(value);
    }

    // frontend API, add an input logical tensor to the op
    void add_input(const logical_tensor_t &lt) {
        add_input(std::make_shared<value_t>(lt));
    }

    std::shared_ptr<value_t> get_input_value(size_t offset) const {
        return inputs_.at(offset);
    }

    const std::vector<std::shared_ptr<value_t>> &get_input_values() const {
        return inputs_;
    }

    void fill_and_connect_input(
            size_t index, dnnl_graph_op &op, size_t offset) {
        while (op.num_outputs() <= offset) {
            op.add_output(dnnl::graph::impl::zero_logical_tensor());
        }
        connect_input(index, op.get_output_value(offset));
    }

    void connect_input(size_t index, dnnl_graph_op &op, size_t offset) {
        connect_input(index, op.get_output_value(offset));
    }

    void connect_input(size_t index, std::shared_ptr<value_t> output) {
        output->add_consumer(*this, index);
        if (inputs_.size() <= index) { inputs_.resize(index + 1); }
        inputs_[index] = output;
    }

    void swap_input_values(size_t offset1, size_t offset2) {
        std::shared_ptr<value_t> input1 = inputs_[offset1];
        input1->remove_consumer(*this, offset1);
        std::shared_ptr<value_t> input2 = inputs_[offset2];
        input2->remove_consumer(*this, offset2);
        std::swap(inputs_[offset1], inputs_[offset2]);
        input1->add_consumer(*this, offset2);
        input2->add_consumer(*this, offset1);
    }

    ///////////////////////////////////////////////////////////////////////////
    // output values
    size_t num_outputs() const { return outputs_.size(); }

    void add_output(std::shared_ptr<value_t> value) {
        const size_t offset = outputs_.size();
        output_tensor_map_[offset] = std::make_pair(id_, offset);

        value->set_producer(*this);
        value->set_offset(offset);
        outputs_.push_back(value);
    }

    // frontend API, add an output logical tensor to the op
    void add_output(const logical_tensor_t &lt) {
        add_output(std::make_shared<value_t>(lt));
    }

    const std::vector<std::shared_ptr<value_t>> &get_output_values() const {
        return outputs_;
    }

    std::shared_ptr<value_t> get_output_value(size_t offset) const {
        return outputs_.at(offset);
    }

    size_t num_output_consumers(size_t offset) const {
        return get_output_value(offset)->get_consumers().size();
    }

    ///////////////////////////////////////////////////////////////////////////
    // attributes handling
    status_t kind_of(const std::string &name, attribute_kind_t &kind) const {
        const auto &found = attributes_.find(name);
        if (found == end(attributes_)) {
            return dnnl::graph::impl::status::invalid_argument;
        }

        kind = found->second.get_kind();
        return dnnl::graph::impl::status::success;
    }

    template <typename Attr>
    dnnl_graph_op &set_attr(const std::string &name, const Attr &a) {
        auto it = attributes_.find(name);
        if (it != end(attributes_)) {
            it->second = {a};
        } else {
            attributes_.insert({name, {a}});
        }
        return *this;
    }

    dnnl_graph_op &set_attr(
            const std::string &name, const attribute_value_t &a) {
        auto it = attributes_.find(name);
        if (it != end(attributes_)) {
            it->second = a;
        } else {
            attributes_.insert({name, a});
        }
        return *this;
    }

    template <typename value_type>
    value_type get_attr(const std::string &name) const {
        auto it = attributes_.find(name);
        assertm(it != attributes_.end(), "don't have such attribute");
        return it->second.get<value_type>();
    }

    template <typename Attr>
    status_t get_attr(const std::string &name, const Attr **attr) const {
        const auto &found = attributes_.find(name);
        if (found == end(attributes_)) {
            return dnnl::graph::impl::status::invalid_argument;
        }

        Attr &val = found->second.get<Attr>();
        *attr = &val;
        return dnnl::graph::impl::status::success;
    }

    bool has_attr(const std::string &attr_name) const {
        return attributes_.find(attr_name) != attributes_.end();
    }

    const std::unordered_map<std::string, attribute_value_t> &
    get_attributes() const {
        return attributes_;
    }

    size_t num_attributes() const { return attributes_.size(); }

    void merge_attributes(
            const std::unordered_map<std::string, attribute_value_t> &attrs) {
        attributes_.insert(attrs.begin(), attrs.end());
    }

    bool is_same_attr_value(
            const dnnl_graph_op &op_b, const std::string &attr_name) const {
        const auto &attr_a = get_attributes();
        const auto &attr_b = op_b.get_attributes();
        auto it_a = attr_a.find(attr_name);
        auto it_b = attr_b.find(attr_name);

        return it_b == attr_b.end() ? false : (it_a->second == it_b->second);
    }

    bool has_same_attr_values(const dnnl_graph_op &op_b,
            std::set<std::string> excepted = {}) const {
        return std::all_of(attributes_.begin(), attributes_.end(),
                [&](const std::pair<std::string, attribute_value_t> &attr) {
                    return excepted.count(attr.first)
                            ? true
                            : is_same_attr_value(op_b, attr.first);
                });
    }

    static const std::string &kind2str(op_kind_t kind) {
        // 0: Abs, ..., N: LastSymbol, 0x1234: any, ...
        const size_t k = static_cast<size_t>(kind);
        const size_t l
                = static_cast<size_t>(dnnl::graph::impl::op_kind::LastSymbol);
        const size_t a = static_cast<size_t>(dnnl::graph::impl::op_kind::any);
        if (k <= l) {
            return dnnl::graph::impl::op_kind::op_kind_strings.at(k);
        } else {
            return dnnl::graph::impl::op_kind::internal_op_strings.at(k - a);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // partition handling
    bool is_assigned_to_partition() const { return partition_ != nullptr; }

    void set_partition(dnnl::graph::impl::partition_impl_t *part) {
        partition_ = part;
    }

    dnnl::graph::impl::partition_impl_t *get_partition() const {
        return partition_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // As a fused op
    bool is_fused() const { return !op_ids_.empty(); }

    void add_op_ids(size_t id) { op_ids_.push_back(id); }

    void add_op_ids(const std::vector<size_t> &ids) {
        for (auto id : ids)
            op_ids_.push_back(id);
    }

    const std::vector<size_t> &get_op_ids() const { return op_ids_; }

    const std::unordered_map<size_t, pair_t> &get_input_tensor_map() const {
        return input_tensor_map_;
    }

    const std::unordered_map<size_t, pair_t> &get_output_tensor_map() const {
        return output_tensor_map_;
    }

    // Add an input logical tensor to the fused op. The logical tensor is from
    // one of original ops.
    void add_fused_input(dnnl_graph_op *op, size_t in_offset) {
        auto map = op->get_input_tensor_map();
        assertm(map.find(in_offset) != map.end(), "fail to find the key");
        input_tensor_map_[inputs_.size()] = map[in_offset];
        inputs_.push_back(std::make_shared<value_t>(
                op->get_input_value(in_offset)->get_logical_tensor()));
    }

    // Add an output logical tensor to the fused op. The logical tensor is from
    // one of original ops.
    void add_fused_output(dnnl_graph_op *op, size_t out_offset) {
        auto map = op->get_output_tensor_map();
        assertm(map.find(out_offset) != map.end(), "fail to find the key");
        output_tensor_map_[outputs_.size()] = map[out_offset];
        auto value = std::make_shared<value_t>(
                op->get_output_value(out_offset)->get_logical_tensor());
        value->set_producer(*this);
        value->set_offset(outputs_.size());
        outputs_.push_back(value);
    }

private:
    size_t id_ {};
    op_kind_t kind_ {};
    std::string name_ {};
    std::vector<std::shared_ptr<value_t>> inputs_ {};
    std::vector<std::shared_ptr<value_t>> outputs_ {};
    std::unordered_map<std::string, attribute_value_t> attributes_;

    const dnnl::graph::impl::op_schema *schema_;
    dnnl::graph::impl::partition_impl_t *partition_ {nullptr};
    bool internal_ {false};

    // fused op: we still need to represent a fused op
    // possibly we can remove these once the new backend API and new pattern
    // matcher is done.
    std::vector<size_t> op_ids_ {};
    // Map from the fused op input index -> (original op id, op input offset)
    std::unordered_map<size_t, pair_t> input_tensor_map_;
    // Map from the fused op output index -> (original op id, op output offset)
    std::unordered_map<size_t, pair_t> output_tensor_map_;
};

namespace dnnl {
namespace graph {
namespace impl {
template <typename OPS, typename FUN>
void topo_order_visit(const OPS &root_ops, const FUN &f) {
    std::stack<op_t *> todo;
    std::unordered_set<op_t *> visited;
    for (auto &op : root_ops) {
        todo.push(op);
    }

    while (!todo.empty()) {
        op_t *top = todo.top();
        if (visited.find(top) != visited.end()) {
            todo.pop();
            continue;
        }
        bool ready = true;
        auto &inputs = top->get_input_values();
        // Need to iterate backwards because some examples are broken and depend
        // on the partition order
        for (auto it = inputs.rbegin(); it != inputs.rend(); ++it) {
            if ((*it)->has_producer()) {
                op_t &producer = (*it)->get_producer();
                if (visited.find(&producer) == visited.end()) {
                    // Need to visit first
                    todo.push(&producer);
                    ready = false;
                }
            }
        }
        if (ready) {
            todo.pop();
            f(top);
            visited.insert(top);
        }
    }
}
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
