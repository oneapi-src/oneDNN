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

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/internal_ops.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace {

inline impl::status_t get_ordered_inputs_outputs(const impl::op_t *work_op,
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
            const std::shared_ptr<impl::op_t> &op)
        : impl::compiled_partition_impl_t(
                engine, inputs, outputs, kernel->inplace_pairs_)
        , perm_ins_(perm_ins)
        , perm_outs_(perm_outs)
        , kernel_(kernel)
        , op_(op)
        , use_subgraph_(false) {}

    // used in subgraph mode
    dnnl_compiled_partition_impl_t(const impl::engine_t &engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs,
            kernel_ptr &kernel)
        : impl::compiled_partition_impl_t(
                engine, inputs, outputs, kernel->inplace_pairs_)
        , kernel_(kernel)
        , use_subgraph_(true) {}

    impl::status_t execute(const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        if (use_subgraph_) {
            // In subgraph mode, we don't need to resort the inputs and outputs
            return kernel_->execute((const dnnl_partition_impl_t *)nullptr,
                    g_stream, inputs, outputs);
        } else {
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
                                && outputs[perm_outs_[i]]
                                                .get_logical_tensor()
                                                .id
                                        == outputs_[i].id,
                        "invalid inputs");
                ordered_outputs.emplace_back(outputs[perm_outs_[i]]);
            }

            return kernel_->execute(
                    op_.get(), g_stream, ordered_inputs, ordered_outputs);
        }
    }

#if DNNL_GRAPH_WITH_SYCL
    impl::status_t execute_sycl(const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs,
            const cl::sycl::event *sycl_event) override {
        UNUSED(sycl_event);
        if (use_subgraph_) {
            // In subgraph mode, we don't need to resort the inputs and outputs
            return kernel_->execute((const dnnl_partition_impl_t *)nullptr,
                    g_stream, inputs, outputs);
        } else {
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
                                && outputs[perm_outs_[i]]
                                                .get_logical_tensor()
                                                .id
                                        == outputs_[i].id,
                        "invalid inputs");
                ordered_outputs.emplace_back(outputs[perm_outs_[i]]);
            }

            return kernel_->execute(
                    op_.get(), g_stream, ordered_inputs, ordered_outputs);
        }
    }
#endif

private:
    std::map<size_t, size_t> perm_ins_;
    std::map<size_t, size_t> perm_outs_;
    kernel_ptr kernel_;
    const std::shared_ptr<impl::op_t> op_;
    bool use_subgraph_ {false};
};

class dnnl_partition_impl_t : public impl::partition_impl_t {
    friend class dnnl_backend;

public:
    dnnl_partition_impl_t(impl::engine_kind_t engine_kind)
        : impl::partition_impl_t(engine_kind) {};

    ~dnnl_partition_impl_t() override = default;

    // deep copy
    dnnl_partition_impl_t(const dnnl_partition_impl_t &other)
        : impl::partition_impl_t(other)
        , fused_op_(std::make_shared<impl::op_t>(other.fused_op_->get_kind()))
        , inputs_map_(other.inputs_map_)
        , outputs_map_(other.outputs_map_) {
        fused_op_->merge_attributes(other.fused_op_->get_attributes());
    }

    ///// The following are used only in backend for constructing object

    void init(const impl::op_t *aop) {
        fused_op_ = std::make_shared<impl::op_t>(aop->get_kind());
        fused_op_->merge_attributes(aop->get_attributes());
        add_tensors(aop);
        add_tensors_map(aop);
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

    void add_tensors_map(const impl::op_t *aop) {
        for (auto kv : aop->get_input_tensor_map()) {
            inputs_map_[kv.second] = kv.first;
        }
        for (auto kv : aop->get_output_tensor_map()) {
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

    bool is_initialized() const override { return fused_op_ != nullptr; }

    std::shared_ptr<impl::partition_impl_t> clone() override {
        return std::make_shared<dnnl_partition_impl_t>(*this);
    }

    const impl::backend *get_assigned_backend() const override {
        return &dnnl_backend::get_singleton();
    }

    impl::status_t compile(impl::compiled_partition_t *compiled_partition,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs,
            const impl::engine_t *g_engine) const override {
        using ltw = impl::logical_tensor_wrapper_t;

        static std::set<op_kind_t> subgraph_patterns {op_kind::int8_conv_relu,
                op_kind::int8_conv, op_kind::int8_conv_bias_relu,
                op_kind::int8_conv_bias, op_kind::int8_conv_bias_add,
                op_kind::int8_conv_bias_add_relu, op_kind::int8_conv_add_relu,
                op_kind::int8_convtranspose, op_kind::int8_convtranspose_bias,
                op_kind::int8_matmul, op_kind::int8_matmul_bias,
                op_kind::int8_matmul_relu, op_kind::int8_matmul_bias_relu,
                op_kind::int8_matmul_sigmoid, op_kind::int8_matmul_bias_sigmoid,
                op_kind::int8_matmul_gelu, op_kind::int8_matmul_bias_gelu,
                op_kind::int8_matmul_add, op_kind::int8_matmul_bias_add,
                op_kind::x8s8float_matmul_add,
                op_kind::x8s8float_matmul_bias_add, op_kind::x8x8float_matmul,
                op_kind::x8s8float_matmul_bias, op_kind::x8s8f32_matmul_relu,
                op_kind::x8s8f32_matmul_bias_relu,
                op_kind::x8s8f32_matmul_sigmoid,
                op_kind::x8s8f32_matmul_bias_sigmoid,
                op_kind::x8s8f32_matmul_gelu, op_kind::x8s8f32_matmul_bias_gelu,
                op_kind::int8_maxpool, op_kind::int8_avgpool,
                op_kind::x8s8f32_conv_relu, op_kind::x8s8f32_conv,
                op_kind::x8s8f32_conv_bias_relu, op_kind::x8s8f32_conv_bias,
                op_kind::x8s8f32_conv_bias_add_relu,
                op_kind::x8s8f32_conv_add_relu,
                op_kind::int8_quant_wei_conv_add_relu,
                op_kind::int8_quant_wei_conv_relu, op_kind::int8_quant_wei_conv,
                op_kind::int8_quant_wei_conv_bias_add_relu,
                op_kind::int8_quant_wei_conv_bias_relu,
                op_kind::int8_quant_wei_conv_bias,
                op_kind::int8_quant_wei_matmul,
                op_kind::int8_quant_wei_matmul_bias,
                op_kind::int8_quant_wei_matmul_add,
                op_kind::int8_quant_wei_matmul_bias_add,
                op_kind::int8_quant_wei_matmul_relu,
                op_kind::int8_quant_wei_matmul_bias_relu,
                op_kind::int8_quant_wei_matmul_sigmoid,
                op_kind::int8_quant_wei_matmul_bias_sigmoid,
                op_kind::int8_quant_wei_matmul_gelu,
                op_kind::int8_quant_wei_matmul_bias_gelu,
                op_kind::x8s8f32_quant_wei_conv,
                op_kind::x8s8f32_quant_wei_conv_relu,
                op_kind::x8s8f32_quant_wei_conv_bias,
                op_kind::x8s8f32_quant_wei_conv_bias_relu,
                op_kind::x8s8f32_quant_wei_conv_add_relu,
                op_kind::x8s8f32_quant_wei_conv_bias_add_relu,
                op_kind::x8s8f32_quant_wei_matmul_add,
                op_kind::x8s8f32_quant_wei_matmul_bias_add,
                op_kind::x8s8f32_quant_wei_matmul,
                op_kind::x8s8f32_quant_wei_matmul_bias,
                op_kind::x8s8f32_quant_wei_matmul_relu,
                op_kind::x8s8f32_quant_wei_matmul_bias_relu,
                op_kind::x8s8f32_quant_wei_matmul_sigmoid,
                op_kind::x8s8f32_quant_wei_matmul_bias_sigmoid,
                op_kind::x8s8f32_quant_wei_matmul_gelu,
                op_kind::x8s8f32_quant_wei_matmul_bias_gelu,
                op_kind::x8x8float_matmul_div,
                // f32 conv pattern
                impl::op_kind::Convolution, op_kind::conv_relu,
                op_kind::conv_add, op_kind::conv_add_relu,
                op_kind::conv_add_elu, op_kind::conv_add_relu6,
                op_kind::conv_bias, op_kind::conv_bias_elu,
                op_kind::conv_bias_relu, op_kind::conv_bias_sigmoid,
                op_kind::conv_bias_swish, op_kind::conv_bias_relu6,
                op_kind::conv_bias_hardtanh, op_kind::conv_bias_square,
                op_kind::conv_bias_tanh, op_kind::conv_bias_abs,
                op_kind::conv_bias_sqrt, op_kind::conv_bias_add,
                op_kind::conv_bias_add_elu, op_kind::conv_bias_add_relu,
                op_kind::conv_bias_add_relu6, op_kind::conv_bias_bn,
                op_kind::conv_bias_bn_add, op_kind::conv_bias_bn_add_relu,
                op_kind::conv_bias_bn_relu, op_kind::conv_bn,
                op_kind::conv_bn_add, op_kind::conv_bn_add_relu,
                op_kind::conv_bn_relu, impl::op_kind::ConvolutionBackpropData,
                // fp32 contranspose pattern
                impl::op_kind::ConvTranspose, op_kind::convtranspose_bias,
                // fp32 matmul pattern
                impl::op_kind::MatMul, op_kind::matmul_relu,
                op_kind::matmul_elu, op_kind::matmul_sigmoid,
                op_kind::matmul_hardtanh, op_kind::matmul_gelu,
                op_kind::matmul_bias, op_kind::matmul_bias_relu,
                op_kind::matmul_bias_gelu, op_kind::matmul_bias_relu6,
                op_kind::matmul_bias_elu, op_kind::matmul_bias_sigmoid,
                op_kind::matmul_bias_swish, op_kind::matmul_bias_hardtanh,
                op_kind::matmul_bias_add, op_kind::matmul_bias_add_relu,
                op_kind::matmul_bias_bn, op_kind::matmul_add,
                op_kind::matmul_add_gelu, op_kind::matmul_add_relu,
                op_kind::matmul_add_sigmoid,
                // f32 pooling pattern
                impl::op_kind::AvgPool, impl::op_kind::MaxPool,
                op_kind::avgpool_add, op_kind::maxpool_add,
                // fp32 eltwise pattern
                impl::op_kind::Abs, impl::op_kind::Elu, impl::op_kind::Exp,
                impl::op_kind::GELU, impl::op_kind::HardTanh,
                impl::op_kind::Log, impl::op_kind::Pow, impl::op_kind::ReLU,
                op_kind::relu_add, impl::op_kind::Round, impl::op_kind::Sqrt,
                impl::op_kind::Square, impl::op_kind::Tanh};

        // compile will transform the subgraph in partition, so we make
        // a copy
        auto part = std::make_shared<dnnl_partition_impl_t>(*this);

        std::shared_ptr<impl::op_t> fused_op = part->get_fused_op();
        if (!fused_op) return status::compile_fail;

        // create kernel
        auto kernel = dnnl_backend::get_singleton().create_kernel(*fused_op);
        if (!kernel) return status::compile_fail;

        status_t ret;

        // compile kernel. patterns in subgraph_patterns set will use subgraph
        // mode
        bool use_subgraph = subgraph_patterns.count(fused_op->get_kind());
        if (use_subgraph) {
            // In subgraph mode, we don't need to resort the inputs or outputs
            // FIXME(qun) will modify the outputs inside the compile, which
            // break the constant semantics
            ret = kernel->compile(part.get(), g_engine, inputs, outputs);
            if (ret != status::success) return ret;

            std::vector<impl::logical_tensor_t> ordered_inputs;
            std::vector<impl::logical_tensor_t> ordered_outputs;
            std::map<size_t, size_t> dummy;
            ret = get_ordered_inputs_outputs(
                    fused_op.get(), inputs_, inputs, ordered_inputs, dummy);
            if (status::success != ret) return ret;

            ret = get_ordered_inputs_outputs(
                    fused_op.get(), outputs_, outputs, ordered_outputs, dummy);
            if (status::success != ret) return ret;

            // wrapper kernel to dnnl_compiled_partition_impl_t
            auto pimpl = std::make_shared<dnnl_compiled_partition_impl_t>(
                    *g_engine, ordered_inputs, ordered_outputs, kernel);
            compiled_partition->init(pimpl);
        } else {
            // To support arbitrary re-connection in FWK graph, we need to
            // find required input and output logical tensors from the compile
            // function's parameters
            std::vector<impl::logical_tensor_t> ordered_inputs;
            std::map<size_t, size_t> perm_ins;
            ret = get_ordered_inputs_outputs(
                    fused_op.get(), inputs_, inputs, ordered_inputs, perm_ins);
            if (status::success != ret) return ret;

            std::vector<impl::logical_tensor_t> ordered_outputs;
            std::map<size_t, size_t> perm_outs;
            ret = get_ordered_inputs_outputs(fused_op.get(), outputs_, outputs,
                    ordered_outputs, perm_outs);
            if (status::success != ret) return ret;

            // Infer attributes of the node, i.e.
            std::vector<impl::logical_tensor_t *> tmp_inputs, tmp_outputs;
            for (auto &in : ordered_inputs) {
                tmp_inputs.emplace_back(&in);
            }
            for (auto &out : ordered_outputs) {
                tmp_outputs.emplace_back(&out);
            }
            const op_schema_t *cur_op_schema
                    = op_schema_registry_t::get_op_schema(fused_op->get_kind());
            if (cur_op_schema) {
                ret = cur_op_schema->shape_infer(
                        fused_op.get(), tmp_inputs, tmp_outputs);
                if (ret != status::success) return ret;
            }

            ret = kernel->compile(
                    fused_op.get(), g_engine, ordered_inputs, ordered_outputs);
            if (ret != status::success) return status::compile_fail;

            // wrapper kernel to dnnl_compiled_partition_impl_t
            auto pimpl = std::make_shared<dnnl_compiled_partition_impl_t>(
                    *g_engine, ordered_inputs, ordered_outputs, perm_ins,
                    perm_outs, kernel, fused_op);
            compiled_partition->init(pimpl);
        }

        return status::success;
    }

    impl::status_t infer_shape(
            std::vector<const impl::logical_tensor_t *> &inputs,
            std::vector<impl::logical_tensor_t *> &outputs) const override {
        impl::status_t ret;

        std::vector<impl::logical_tensor_t *> ordered_inputs, ordered_outputs;
        ret = get_ordered_inputs_outputs(
                fused_op_.get(), inputs_, inputs, ordered_inputs);
        if (impl::status::success != ret) return ret;
        ret = get_ordered_inputs_outputs(
                fused_op_.get(), outputs_, outputs, ordered_outputs);
        if (impl::status::success != ret) return ret;

        const impl::op_schema_t *cur_op_schema
                = impl::op_schema_registry_t::get_op_schema(
                        fused_op_->get_kind());

        if (cur_op_schema) {
            // shape_infer will change op attrs, so in order to keep the op_
            // in partition unchanged, create a temp_op to hold these changes
            impl::op_t temp_op = impl::op_t(fused_op_->get_kind());
            temp_op.merge_attributes(fused_op_->get_attributes());
            return cur_op_schema->shape_infer(
                    &temp_op, ordered_inputs, ordered_outputs);
        } else {
            return impl::status::invalid_op;
        }
    }

    std::string to_string() const override {
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

        os << " [ op: (";
        if (fused_op_) {
            os << "ID: " << fused_op_->get_id()
               << ", kind: " << impl::op_t::kind2str(fused_op_->get_kind());
        }
        os << ") \n";

        os << "  [ inputs: ";
        const char *delimer = "";
        for (const auto &i : inputs_) {
            const impl::logical_tensor_wrapper_t v(i);
            os << delimer << "(ID: " << v.id() << "("
               << type_to_string(v.data_type()) << ":"
               << dims_to_string(v.vdims());
            delimer = ")), ";
        }
        os << " ]\n";

        os << "  [ outputs: ";
        delimer = "";
        for (const auto &o : outputs_) {
            const impl::logical_tensor_wrapper_t v(o);
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

    const std::shared_ptr<impl::op_t> &get_fused_op() const {
        return fused_op_;
    };

private:
    // // Fused op. Currently, only one op here
    std::shared_ptr<impl::op_t> fused_op_ {nullptr};

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
