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

#ifndef INTERFACE_PARTITION_HPP
#define INTERFACE_PARTITION_HPP

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "c_types_map.hpp"
#include "engine.hpp"
#include "id.hpp"
#include "ir.hpp"
#include "logical_tensor.hpp"
#include "op.hpp"
#include "utils.hpp"

#include "interface/stream.hpp"
#include "utils/compatible.hpp"

namespace std {
template <>
struct hash<std::pair<size_t, size_t>> {
    size_t operator()(const std::pair<size_t, size_t> &v) const {
        size_t seed = 0;
        seed ^= std::hash<size_t> {}(v.first) + 0x9e3779b9 + (seed << 6)
                + (seed >> 2);
        seed ^= std::hash<size_t> {}(v.second) + 0x9e3779b9 + (seed << 6)
                + (seed >> 2);
        return seed;
    }
};
} // namespace std

namespace dnnl {
namespace graph {
namespace impl {
class executable;
} // namespace impl
} // namespace graph
} // namespace dnnl

namespace impl = dnnl::graph::impl;
using node_attrs = impl::attributes;

struct dnnl_graph_partition : public dnnl_graph_id {
public:
    friend struct dnnl_graph_compiled_partition;

    dnnl_graph_partition() = default;

    // enable copy
    dnnl_graph_partition(const dnnl_graph_partition &other)
        : dnnl_graph_id(other)
        , engine_kind_(other.engine_kind_)
        , ids_(other.ids_)
        , node_(impl::utils::make_unique<impl::node_t>(
                  other.node_->get_op_kind()))
        , inputs_(other.inputs_)
        , outputs_(other.outputs_)
        , inputs_map_(other.inputs_map_)
        , outputs_map_(other.outputs_map_) {
        node_->merge_attrs_map(other.node_->get_attrs_map());
    }

    // disable assign
    dnnl_graph_partition &operator=(const dnnl_graph_partition &other) = delete;

    ~dnnl_graph_partition() = default;

    bool is_initialized() { return node_ ? true : false; }

    void init(
            const impl::node_t *anode, const impl::engine_kind_t engine_kind) {
        engine_kind_ = engine_kind;
        node_ = impl::utils::make_unique<impl::node_t>(anode->get_op_kind());
        node_->merge_attrs_map(anode->get_attrs_map());
        add_op(anode->get_op_ids());
        add_tensors(anode);
        add_tensors_map(anode);
    }

    void init(impl::op_kind_t op_kind, const impl::engine_kind_t engine_kind,
            const impl::logical_tensor_t &input,
            const impl::logical_tensor_t &output);

    void add_tensors(const impl::node_t *anode) {
        for (size_t i = 0; i < anode->num_inputs_tensor(); ++i) {
            inputs_.push_back(anode->get_input_tensor(i));
        }
        for (size_t i = 0; i < anode->num_outputs_tensor(); ++i) {
            outputs_.push_back(anode->get_output_tensor(i));
        }
    }

    void add_tensors_map(const impl::node_t *anode) {
        for (auto kv : anode->get_input_tensor_map()) {
            inputs_map_[kv.second] = kv.first;
        }
        for (auto kv : anode->get_output_tensor_map()) {
            outputs_map_[kv.second] = kv.first;
        }
    }

    void add_op(size_t id) { ids_.insert(id); }

    void add_op(const std::vector<size_t> &ids) {
        ids_.insert(ids.begin(), ids.end());
    }

    size_t num_ops() const { return ids_.size(); }

    const std::unordered_set<size_t> &get_ops() const { return ids_; }

    const impl::node_t *node() const { return node_.get(); };

    size_t get_inputs_num() const { return inputs_.size(); }

    size_t get_outputs_num() const { return outputs_.size(); }

    const std::vector<impl::logical_tensor_t> &get_inputs() const {
        return inputs_;
    }

    const std::vector<impl::logical_tensor_t> &get_outputs() const {
        return outputs_;
    }

    impl::logical_tensor_t *find_input(size_t id, size_t offset) {
        auto p = std::make_pair(id, offset);

        auto v = inputs_map_.find(p);
        if (v != inputs_map_.end()) {
            return &(inputs_.at(v->second));
        } else {
            return nullptr;
        }
    }

    impl::logical_tensor_t *find_output(size_t id, size_t offset) {
        auto p = std::make_pair(id, offset);

        auto v = outputs_map_.find(p);
        if (v != outputs_map_.end()) {
            return &(outputs_.at(v->second));
        } else {
            return nullptr;
        }
    }

    impl::status_t compile(impl::compiled_partition_t *compiled_partition,
            std::vector<const impl::logical_tensor_t *> &inputs,
            std::vector<const impl::logical_tensor_t *> &outputs,
            const impl::engine_t *e = nullptr);

    impl::status_t infer_shape(
            std::vector<const impl::logical_tensor_t *> &inputs,
            std::vector<impl::logical_tensor_t *> &outputs);

    friend std::string to_string(const dnnl_graph_partition &p) {
        std::ostringstream os;

        const auto type_to_string = [](impl::data_type_t t) {
            switch (t) {
                case dnnl_graph_data_type_undef: return "undef";
                case dnnl_graph_f16: return "f16";
                case dnnl_graph_bf16: return "f16";
                case dnnl_graph_f32: return "f32";
                case dnnl_graph_s32: return "s32";
                case dnnl_graph_s8: return "s8";
                case dnnl_graph_u8: return "u8";
            }
        };

        const auto dims_to_string = [&](const std::vector<int64_t> &dims) {
            std::ostringstream oss;
            oss << "(";
            const char *delimer = "";
            for (const auto &d : dims) {
                oss << delimer << d;
                delimer = "x";
            }
            oss << ")";
            return oss.str();
        };

        os << "[ Partition ID: " << p.id() << '\n';
        os << " [ node: (";
        if (p.node_) {
            os << "ID: " << p.node_->id()
               << ", kind: " << impl::op_t::kind2str(p.node_->get_op_kind());
        }
        os << ") \n";

        os << "  [ inputs: ";
        const char *delimer = "";
        for (const auto &i : p.inputs_) {
            const impl::logical_tensor_wrapper v(i);
            os << delimer << "(ID: " << v.id() << "("
               << type_to_string(v.data_type()) << ":"
               << dims_to_string(v.vdims());
            delimer = ")), ";
        }
        os << " ]\n";

        os << "  [ outputs: ";
        delimer = "";
        for (const auto &o : p.outputs_) {
            const impl::logical_tensor_wrapper v(o);
            os << delimer << "(ID: " << v.id() << "("
               << type_to_string(v.data_type()) << ":"
               << dims_to_string(v.vdims());
            delimer = ")), ";
        }
        os << " ]\n";
        os << " ]\n";
        os << "]";

        return os.str();
    }

private:
    // Engine kind
    impl::engine_kind_t engine_kind_;

    // All the IDs of corresponding op_t objects
    std::unordered_set<size_t> ids_ {};

    // Fused node. Currently, only one node here
    std::unique_ptr<impl::node_t> node_ {nullptr};

    // All the input logical tensors of a partition
    std::vector<impl::logical_tensor_t> inputs_ {};

    // All the output logical tensors of a partition
    std::vector<impl::logical_tensor_t> outputs_ {};

    // Map from (op id, op input offset) -> partition input index
    std::unordered_map<std::pair<size_t, size_t>, size_t> inputs_map_ {};

    // Map from (op id, op output offset) -> partition output index
    std::unordered_map<std::pair<size_t, size_t>, size_t> outputs_map_ {};
};

///
/// \brief dnnl_graph_compiled_partition_t
///
struct dnnl_graph_compiled_partition : public dnnl_graph_id {
public:
    friend struct dnnl_graph_partition;

    using tensor_shape = std::vector<int64_t>;
    static constexpr tensor_shape::value_type unknown_shape {-1};

    dnnl_graph_compiled_partition(const impl::partition_t &src_partition)
        : src_partition_ {src_partition} {}

    ~dnnl_graph_compiled_partition() = default;

    const impl::partition_t &src_partition() { return src_partition_; }

    const std::vector<impl::inplace_pair_t> &get_inplace_pairs() const;

    impl::status_t execute(const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) const;

#if DNNL_GRAPH_WITH_SYCL
    impl::status_t execute_sycl(const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs,
            const cl::sycl::event *sycl_event) const;
#endif

    impl::status_t query_logical_tensor(
            size_t tid, impl::logical_tensor_t *lt) const;

private:
    const impl::partition_t src_partition_;

    // Executable pointer to run kernel
    std::shared_ptr<impl::executable> executable_;

    // All the input logical tensors of a partition.
    // compared with the inputs_ in partition, these
    // inputs may have richer info, such as shape
    // note: now we don't have to do this, but if we
    // store value instead of ptr (pinzhen's proposal),
    // we may need the above design
    std::vector<impl::logical_tensor_t> inputs_ {};

    // All the output logical tensors of a partition
    // ditto
    std::vector<impl::logical_tensor_t> outputs_ {};

    // this engine must be valid
    impl::engine_t engine_;
};

#endif
