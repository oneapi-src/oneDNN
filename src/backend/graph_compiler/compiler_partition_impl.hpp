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
#ifndef BACKEND_GRAPH_COMPILER_COMPILER_PARTITION_IMPL_HPP
#define BACKEND_GRAPH_COMPILER_COMPILER_PARTITION_IMPL_HPP

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "compiler/ir/graph/lowering.hpp"
#include "compiler/jit/cfake/cfake_jit.hpp"
#include "interface/backend.hpp"
#include "interface/partition.hpp"
#include "runtime/dynamic_dispatch/dynamic_tensor.hpp"
#include "runtime/memorypool.hpp"

#include "compiler_allocator.hpp"
#include "compiler_backend.hpp"
#include "compiler_graph.hpp"
#include "utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {

class compiler_partition_impl_t : public partition_impl_t {
    friend class compiler_backend_t;

public:
    compiler_partition_impl_t(impl::engine_kind_t engine_kind,
            impl::fpmath_mode_t fpmath_mode, impl::partition_kind_t pkind,
            std::string pname)
        : impl::partition_impl_t(engine_kind, fpmath_mode, pkind)
        , is_init_(true)
        , pname_(pname) {
        assertm(fpmath_mode == fpmath_mode::strict,
                "Compiler backend only allows fpmath mode: strict.");
    }

    virtual ~compiler_partition_impl_t() = default;

    bool is_initialized() const override;
    std::shared_ptr<impl::partition_impl_t> clone() const override;
    impl::status_t infer_shape(
            std::vector<const impl::logical_tensor_t *> &inputs,
            std::vector<impl::logical_tensor_t *> &outputs) const override;
    impl::status_t compile(impl::compiled_partition_t *compiled_partition,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs,
            const impl::engine_t *aengine,
            const impl::context_t *acontext) const override;

    const impl::backend *get_assigned_backend() const override {
        return &compiler_backend_t::get_singleton();
    }

    // add op to backend partition
    void add_op(const std::shared_ptr<op_t> &op) { ops_.emplace_back(op); }

    // add ops to backend partition
    void add_op(const std::vector<std::shared_ptr<op_t>> &ops) {
        for (auto &op : ops) {
            add_op(op);
        }
    }

    // add backend partition's input tensor
    void add_input_tensor(const std::shared_ptr<value_t> &v) {
        auto in_pos = std::find_if(inputs_.begin(), inputs_.end(),
                [&](const impl::logical_tensor_t &alt) -> bool {
                    return alt.id == v->get_logical_tensor().id;
                });
        if (in_pos == inputs_.end()) {
            inputs_.push_back(v->get_logical_tensor());
        }
    }

    // add backend partition's output tensor
    void add_output_tensor(const std::shared_ptr<value_t> &v) {
        auto out_pos = std::find_if(outputs_.begin(), outputs_.end(),
                [&](const impl::logical_tensor_t &alt) -> bool {
                    return alt.id == v->get_logical_tensor().id;
                });
        if (out_pos == outputs_.end()) {
            outputs_.push_back(v->get_logical_tensor());
        }
    }

    bool is_op_exist(const op_t *aop) {
        auto pos = std::find_if(ops_.begin(), ops_.end(),
                [&](const std::shared_ptr<impl::op_t> &cur) -> bool {
                    return cur->get_id() == aop->get_id();
                });
        return pos != ops_.end();
    }

    std::string get_name() const { return pname_; }

protected:
    bool is_init_ = false;
    mutable std::vector<std::shared_ptr<impl::op_t>> copied_ops_;
    mutable std::mutex mtx_;
    std::string pname_;
};
class compiler_compiled_partition_impl_t : public compiled_partition_impl_t {
public:
    compiler_compiled_partition_impl_t(const impl::engine_t &engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs,
            const std::shared_ptr<sc::jit_function_t> &jit_func,
            const std::shared_ptr<impl::compiler_impl::compiler_graph_engine_t>
                    &graph_engine,
            std::vector<sc::runtime::dynamic_tensor_t> &&dyn_inputs,
            std::vector<sc::runtime::dynamic_tensor_t> &&dyn_outputs,
            sc::sc_graph_t &&sc_graph);
    virtual ~compiler_compiled_partition_impl_t();
    impl::status_t execute(const impl::stream_t *astream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override;
    status_t query_dynamic_outputs(
            const std::vector<logical_tensor_t *> &out_lts,
            const std::vector<const logical_tensor_t *> &in_lts,
            const impl::context_t *acontext) const override;

private:
    std::shared_ptr<sc::jit_function_t> jit_func_;
    std::shared_ptr<impl::compiler_impl::compiler_graph_engine_t> graph_engine_;
    std::vector<sc::runtime::dynamic_tensor_t> dyn_inputs_, dyn_outputs_;
    mutable sc::sc_graph_t sc_graph_;
};

} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
#endif
