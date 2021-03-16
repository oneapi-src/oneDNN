/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef BACKEND_DNNL_DNNL_PARTITION_IMPL_HPP
#define BACKEND_DNNL_DNNL_PARTITION_IMPL_HPP

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "interface/backend.hpp"
#include "interface/partition.hpp"

#include "dnnl_backend.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace {

impl::status_t get_ordered_inputs_outputs(const impl::op_t *work_op,
        const std::vector<impl::logical_tensor_t> &expected,
        const std::vector<impl::logical_tensor_t> &given,
        std::vector<impl::logical_tensor_t> &ordered,
        std::map<size_t, size_t> &permutation) {
    // FIXME(qun) Workaround: op in this list can have repeated inputs
    const std::set<impl::op_kind_t> s_whitelist {
            impl::op_kind::Multiply, impl::op_kind::Add};

    // to support arbitrary re-connection in FWK graph, we need to
    // find required and ordered input and output logical tensors from the
    // out-of-order inputs / outputs
    if (given.size() < expected.size()
            && s_whitelist.count(work_op->get_kind()) == 0) {
        return impl::status::miss_ins_outs;
    }

    ordered.reserve(expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        for (size_t j = 0; j < given.size(); j++) {
            if (expected[i].id == given[j].id) {
                ordered.emplace_back(given[j]);
                permutation.insert({i, j});
                break;
            }
        }
    }

    if (ordered.size() != expected.size()) return impl::status::miss_ins_outs;
    return impl::status::success;
}

//RCONT is for const or non-const logical_tensor_t
template <typename RCONT>
status_t get_ordered_inputs_outputs(const op_t *work_op,
        const std::vector<logical_tensor_t> &expected,
        const std::vector<RCONT *> &origin,
        std::vector<logical_tensor_t *> &ordered) {
    // FIXME(qun) Workaround: op in this list can have repeated inputs
    const std::set<impl::op_kind_t> s_whitelist {
            impl::op_kind::Multiply, impl::op_kind::Add};

    // to support arbitrary re-connection in FWK graph, we need to
    // find required and ordered input and output logical tensors from the
    // out-of-order inputs / outputs
    if (origin.size() < expected.size()
            && s_whitelist.count(work_op->get_kind()) == 0) {
        return status::miss_ins_outs;
    }

    ordered.reserve(expected.size());
    for (auto &&val : expected) {
        auto pos = std::find_if(origin.begin(), origin.end(),
                [&val](RCONT *in) -> bool { return in->id == val.id; });
        if (pos != origin.end()) {
            ordered.emplace_back(const_cast<logical_tensor_t *>(*pos));
        }
    }
    if (ordered.size() != expected.size()) return status::miss_ins_outs;
    return status::success;
}

} // namespace

class dnnl_compiled_partition_impl_t : public impl::compiled_partition_impl_t {
    friend class dnnl_backend;
    friend class dnnl_partition_impl_t;

public:
    dnnl_compiled_partition_impl_t(const impl::engine_t &engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs,
            const std::map<size_t, size_t> &perm_ins,
            const std::map<size_t, size_t> &perm_outs, kernel_ptr &kernel,
            const impl::op_t *node)
        : impl::compiled_partition_impl_t(
                engine, inputs, outputs, kernel->inplace_pairs_)
        , perm_ins_(perm_ins)
        , perm_outs_(perm_outs)
        , kernel_(kernel)
        , node_(node) {}

    virtual impl::status_t execute(const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        std::vector<impl::tensor_t> ordered_inputs, ordered_outputs;
        ordered_inputs.reserve(inputs_.size());
        ordered_outputs.reserve(outputs_.size());
        for (size_t i = 0; i < inputs_.size(); i++) {
            assertm(perm_ins_[i] < inputs.size()
                            && inputs[perm_ins_[i]].get_logical_tensor().id
                                    == inputs_[i].id,
                    "invalid inputs");
            ordered_inputs.emplace_back(inputs[perm_ins_[i]]);
        }
        for (size_t i = 0; i < outputs_.size(); i++) {
            assertm(perm_outs_[i] < outputs.size()
                            && outputs[perm_outs_[i]].get_logical_tensor().id
                                    == outputs_[i].id,
                    "invalid inputs");
            ordered_outputs.emplace_back(outputs[perm_outs_[i]]);
        }

        return kernel_->execute(
                node_, g_stream, ordered_inputs, ordered_outputs);
    }

#if DNNL_GRAPH_WITH_SYCL
    virtual impl::status_t execute_sycl(const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs,
            const cl::sycl::event *sycl_event) override {
        UNUSED(sycl_event);
        std::vector<impl::tensor_t> ordered_inputs, ordered_outputs;
        ordered_inputs.reserve(inputs_.size());
        ordered_outputs.reserve(outputs_.size());
        for (size_t i = 0; i < inputs_.size(); i++) {
            assertm(perm_ins_[i] < inputs.size()
                            && inputs[perm_ins_[i]].get_logical_tensor().id
                                    == inputs_[i].id,
                    "invalid inputs");
            ordered_inputs.emplace_back(inputs[perm_ins_[i]]);
        }
        for (size_t i = 0; i < outputs_.size(); i++) {
            assertm(perm_ins_[i] < inputs.size()
                            && outputs[perm_outs_[i]].get_logical_tensor().id
                                    == outputs_[i].id,
                    "invalid inputs");
            ordered_outputs.emplace_back(outputs[perm_outs_[i]]);
        }

        return kernel_->execute(
                node_, g_stream, ordered_inputs, ordered_outputs);
    }
#endif

private:
    std::map<size_t, size_t> perm_ins_;
    std::map<size_t, size_t> perm_outs_;
    kernel_ptr kernel_;
    const impl::op_t *node_;
};

class dnnl_partition_impl_t : public impl::partition_impl_t {
    friend class dnnl_backend;

public:
    dnnl_partition_impl_t(impl::engine_kind_t engine_kind)
        : impl::partition_impl_t(engine_kind) {};

    virtual ~dnnl_partition_impl_t() {};

    // deep copy
    dnnl_partition_impl_t(const dnnl_partition_impl_t &other)
        : impl::partition_impl_t(other)
        , fused_op_(impl::utils::make_unique<impl::op_t>(
                  other.fused_op_->get_kind()))
        , inputs_map_(other.inputs_map_)
        , outputs_map_(other.outputs_map_) {
        fused_op_->merge_attributes(other.fused_op_->get_attributes());
    }

    ///// The following are used only in backend for constructing object

    void init(const impl::op_t *anode) {
        fused_op_ = impl::utils::make_unique<impl::op_t>(anode->get_kind());
        fused_op_->merge_attributes(anode->get_attributes());
        add_tensors(anode);
        add_tensors_map(anode);
    }

    void add_op(const std::shared_ptr<op_t> &op) { ops_.emplace_back(op); }

    void add_op(const std::vector<std::shared_ptr<op_t>> &ops) {
        for (auto &op : ops) {
            add_op(op);
        }
    }

    void add_tensors(const impl::op_t *op) {
        for (size_t i = 0; i < op->num_inputs(); ++i) {
            inputs_.push_back(op->get_input_value(i)->get_logical_tensor());
        }
        for (size_t i = 0; i < op->num_outputs(); ++i) {
            outputs_.push_back(op->get_output_value(i)->get_logical_tensor());
        }
    }

    void add_tensors_map(const impl::op_t *anode) {
        for (auto kv : anode->get_input_tensor_map()) {
            inputs_map_[kv.second] = kv.first;
        }
        for (auto kv : anode->get_output_tensor_map()) {
            outputs_map_[kv.second] = kv.first;
        }
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

    /////////////// the followings are the implementation of interface

    virtual bool is_initialized() override { return fused_op_ != nullptr; }

    virtual std::shared_ptr<impl::partition_impl_t> clone() override {
        return std::make_shared<dnnl_partition_impl_t>(*this);
    }

    virtual const impl::backend *get_assigned_backend() const override {
        return &dnnl_backend::get_singleton();
    }

    virtual impl::status_t compile(
            impl::compiled_partition_t *compiled_partition,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs,
            const impl::engine_t *g_engine = nullptr) override {
        using ltw = impl::logical_tensor_wrapper;

        impl::op_t *fused_op = dynamic_cast<const dnnl_partition_impl_t *>(
                compiled_partition->src_partition().get_pimpl())
                                       ->get_fused_op();
        if (!fused_op) return status::compile_fail;

        // To support arbitrary re-connection in FWK graph, we need to
        // find required input and output logical tensors from the compile
        // function's parameters
        status_t ret;
        std::vector<impl::logical_tensor_t> ordered_inputs;
        std::map<size_t, size_t> perm_ins;
        ret = get_ordered_inputs_outputs(
                fused_op, inputs_, inputs, ordered_inputs, perm_ins);
        if (status::success != ret) return ret;

        std::vector<impl::logical_tensor_t> ordered_outputs;
        std::map<size_t, size_t> perm_outs;
        ret = get_ordered_inputs_outputs(
                fused_op, outputs_, outputs, ordered_outputs, perm_outs);
        if (status::success != ret) return ret;

        // Check if all the shapes are known
        // In the phase of compilation, all the output shape should be known.
        auto pos = std::find_if(ordered_outputs.begin(), ordered_outputs.end(),
                [&](const impl::logical_tensor_t &out) -> bool {
                    return ltw(out).is_shape_unknown();
                });
        if (pos != ordered_outputs.end()) { return status::invalid_argument; }

        // Infer attributes of the node, i.e.
        std::vector<impl::logical_tensor_t *> tmp_inputs, tmp_outputs;
        for (auto &in : ordered_inputs) {
            tmp_inputs.emplace_back(&in);
        }
        for (auto &out : ordered_outputs) {
            tmp_outputs.emplace_back(&out);
        }
        const op_schema *cur_op_schema
                = op_schema_registry::get_op_schema(fused_op->get_kind());
        if (cur_op_schema) {
            cur_op_schema->shape_infer(fused_op, tmp_inputs, tmp_outputs);
        }

        // create kernel
        auto kernel = dnnl_backend::get_singleton().create_kernel(*fused_op);
        if (!kernel) return status::compile_fail;

        // compile kernel
        ret = kernel->compile(
                fused_op, g_engine, ordered_inputs, ordered_outputs);
        if (ret != impl::status::success) return status::compile_fail;

        // wrapper kernel to dnnl_compiled_partition_impl_t
        auto pimpl = std::make_shared<dnnl_compiled_partition_impl_t>(*g_engine,
                ordered_inputs, ordered_outputs, perm_ins, perm_outs, kernel,
                fused_op);
        compiled_partition->init(pimpl);

        return status::success;
    }

    virtual impl::status_t infer_shape(
            std::vector<const impl::logical_tensor_t *> &inputs,
            std::vector<impl::logical_tensor_t *> &outputs) override {
        impl::status_t ret;

        std::vector<impl::logical_tensor_t *> ordered_inputs, ordered_outputs;
        ret = get_ordered_inputs_outputs(
                fused_op_.get(), inputs_, inputs, ordered_inputs);
        if (impl::status::success != ret) return ret;
        ret = get_ordered_inputs_outputs(
                fused_op_.get(), outputs_, outputs, ordered_outputs);
        if (impl::status::success != ret) return ret;

        const impl::op_schema *cur_op_schema
                = impl::op_schema_registry::get_op_schema(
                        fused_op_->get_kind());

        if (cur_op_schema) {
            // shape_infer will change node attrs, so in order to keep the node_
            // in partition unchanged, create a temp_node to hold these changes
            impl::op_t temp_node = impl::op_t(fused_op_->get_kind());
            temp_node.merge_attributes(fused_op_->get_attributes());
            return cur_op_schema->shape_infer(
                    &temp_node, ordered_inputs, ordered_outputs);
        } else {
            return impl::status::invalid_op;
        }
    }

    virtual std::string to_string() const override {
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
                default: return "undef";
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

        os << " [ node: (";
        if (fused_op_) {
            os << "ID: " << fused_op_->get_id()
               << ", kind: " << impl::op_t::kind2str(fused_op_->get_kind());
        }
        os << ") \n";

        os << "  [ inputs: ";
        const char *delimer = "";
        for (const auto &i : inputs_) {
            const impl::logical_tensor_wrapper v(i);
            os << delimer << "(ID: " << v.id() << "("
               << type_to_string(v.data_type()) << ":"
               << dims_to_string(v.vdims());
            delimer = ")), ";
        }
        os << " ]\n";

        os << "  [ outputs: ";
        delimer = "";
        for (const auto &o : outputs_) {
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

    impl::op_t *get_fused_op() const { return fused_op_.get(); };

private:
    // // Fused op. Currently, only one op here
    std::unique_ptr<impl::op_t> fused_op_ {nullptr};

    // Map from (op id, op input offset) -> partition input index
    std::unordered_map<std::pair<size_t, size_t>, size_t> inputs_map_;

    // Map from (op id, op output offset) -> partition output index
    std::unordered_map<std::pair<size_t, size_t>, size_t> outputs_map_;
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
