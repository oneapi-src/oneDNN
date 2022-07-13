/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "interface/backend.hpp"
#include "interface/partition.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/internal_ops.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace {

inline impl::status_t get_ordered_inputs_outputs(
        const std::vector<impl::logical_tensor_t> &expected,
        const std::vector<impl::logical_tensor_t> &given,
        std::vector<impl::logical_tensor_t> &ordered) {
    ordered.reserve(expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        for (size_t j = 0; j < given.size(); j++) {
            if (expected[i].id == given[j].id) {
                ordered.emplace_back(given[j]);
                break;
            }
        }
    }

    if (ordered.size() != expected.size())
        return impl::status::invalid_arguments;
    return impl::status::success;
}

} // namespace

class dnnl_compiled_partition_impl_t : public impl::compiled_partition_impl_t {
    friend class dnnl_backend;
    friend class dnnl_partition_impl_t;

public:
    dnnl_compiled_partition_impl_t(const impl::engine_t &engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs,
            kernel_ptr &kernel)
        : impl::compiled_partition_impl_t(
                engine, inputs, outputs, kernel->inplace_pairs_)
        , kernel_(kernel) {}

    impl::status_t execute(const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        // We don't need to resort the inputs and outputs
        return kernel_->execute((const dnnl_partition_impl_t *)nullptr,
                g_stream, inputs, outputs);
    }

#ifdef DNNL_GRAPH_WITH_SYCL
    impl::status_t execute_sycl(const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        // We don't need to resort the inputs and outputs
        return kernel_->execute_sycl((const dnnl_partition_impl_t *)nullptr,
                g_stream, inputs, outputs, sycl_deps, sycl_event);
    }
#endif

private:
    kernel_ptr kernel_;
};

class dnnl_partition_impl_t : public impl::partition_impl_t {
    friend class dnnl_backend;

public:
    dnnl_partition_impl_t(impl::engine_kind_t engine_kind,
            impl::fpmath_mode_t fpmath_mode, impl::partition_kind_t pkind)
        : impl::partition_impl_t(engine_kind, fpmath_mode, pkind) {}

    ~dnnl_partition_impl_t() override = default;

    ///// The following are used only in backend for constructing object

    void init(FCreateKernel kernel_creator) {
        init_inputs_outputs();

        // init kernel
        kernel_creator_ = std::move(kernel_creator);
    }

    void add_op(const std::shared_ptr<op_t> &op) { ops_.emplace_back(op); }

    // init backend partition's input/output logical tensors
    // based on ops in the partition
    void init_inputs_outputs() {
        inputs_.clear();
        outputs_.clear();
        std::unordered_set<impl::op_t *> visit;
        for (auto &cur_op : ops_) {
            visit.insert(cur_op.get());
        }

        for (auto &cur_op : ops_) {
            for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
                auto in_value = cur_op->get_input_value(j);
                if (!in_value->has_producer()
                        || !visit.count(&in_value->get_producer())) {
                    inputs_.push_back(in_value->get_logical_tensor());
                }
            }
            for (size_t j = 0; j < cur_op->num_outputs(); ++j) {
                auto out_value = cur_op->get_output_value(j);
                // if out_value has no consumer
                // OR any of its consumers are not inside the pattern
                // it is output tensor
                bool is_output = out_value->get_consumers().empty();
                for (auto &consumer : out_value->get_consumers()) {
                    if (!visit.count(&consumer.get_op())) {
                        is_output = true;
                        break;
                    }
                }
                if (is_output) {
                    outputs_.push_back(out_value->get_logical_tensor());
                }
            }
        }
    }

    FCreateKernel get_kernel_creator() const { return kernel_creator_; }

    /////////////// the followings are the implementation of interface

    bool is_initialized() const override { return kernel_creator_ != nullptr; }

    std::shared_ptr<impl::partition_impl_t> clone() const override {
        auto ret = std::make_shared<dnnl_partition_impl_t>(
                get_engine_kind(), get_fpmath_mode(), get_kind());
        ret->ops_ = impl::graph_t::deep_copy(ops_);
        ret->inputs_ = inputs_;
        ret->outputs_ = outputs_;
        ret->kernel_creator_ = kernel_creator_;
        ret->id_ = id_;
        return ret;
    }

    const impl::backend *get_assigned_backend() const override {
        return &dnnl_backend::get_singleton();
    }

    impl::status_t compile(impl::compiled_partition_t *compiled_partition,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs,
            const impl::engine_t *g_engine) const override {
        // compile will transform the subgraph in partition, so we make
        // a copy
        auto part = std::dynamic_pointer_cast<dnnl_partition_impl_t>(
                this->clone());

        // get kernel creator
        auto kernel_creator = part->get_kernel_creator();

        // This internal env var is used for test purpose. When setting
        // _DNNL_GRAPH_USE_LARGE_PARTITION_KERNEL to 1, all partitions will be
        // dispatched to the large partition kernel.
        if (impl::utils::getenv_int_internal("USE_LARGE_PARTITION_KERNEL", 0)) {
            kernel_creator = large_partition_kernel_creator;
        }

        kernel_ptr kernel = kernel_creator();
        if (!kernel) return status::unimplemented;

        status_t ret;

        // compile kernel.
        // FIXME(qun) will modify the outputs inside the compile, which
        // break the constant semantics
        ret = kernel->compile(part.get(), g_engine, inputs, outputs);
        if (ret != status::success) return ret;

        std::vector<impl::logical_tensor_t> ordered_inputs;
        std::vector<impl::logical_tensor_t> ordered_outputs;
        ret = get_ordered_inputs_outputs(inputs_, inputs, ordered_inputs);
        if (status::success != ret) return ret;

        ret = get_ordered_inputs_outputs(outputs_, outputs, ordered_outputs);
        if (status::success != ret) return ret;

        // wrapper kernel to dnnl_compiled_partition_impl_t
        auto pimpl = std::make_shared<dnnl_compiled_partition_impl_t>(
                *g_engine, ordered_inputs, ordered_outputs, kernel);
        compiled_partition->init(pimpl);

        return status::success;
    }

    impl::status_t infer_shape(
            std::vector<const impl::logical_tensor_t *> &inputs,
            std::vector<impl::logical_tensor_t *> &outputs) const override {
        UNUSED(inputs);
        UNUSED(outputs);
        return impl::status::success;
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

private:
    FCreateKernel kernel_creator_;
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
