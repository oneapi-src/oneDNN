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
#include <memory>
#include <string>
#include <vector>

#include "compiler_partition_impl.hpp"
#include "utils/debug.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {

impl::status_t compiler_partition_impl_t::infer_shape(
        std::vector<const impl::logical_tensor_t *> &inputs,
        std::vector<impl::logical_tensor_t *> &outputs) const {
    UNUSED(inputs);
    UNUSED(outputs);
    return impl::status::not_ready; // will be ready in part3
}

impl::status_t compiler_partition_impl_t::compile(
        impl::compiled_partition_t *compiled_partition,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs,
        const impl::engine_t *aengine) const {
    UNUSED(compiled_partition);
    UNUSED(aengine);
    UNUSED(inputs);
    UNUSED(outputs);
    return impl::status::not_ready; // will be ready in part3
}

std::shared_ptr<impl::partition_impl_t> compiler_partition_impl_t::clone() {
    return std::make_shared<compiler_partition_impl_t>(*this);
}

compiler_partition_impl_t::compiler_partition_impl_t(
        const compiler_partition_impl_t &other)
    : impl::partition_impl_t(other) {
    is_init_ = other.is_init_;
}

bool compiler_partition_impl_t::is_initialized() const {
    return is_init_;
}

std::string compiler_partition_impl_t::to_string() const {
    std::ostringstream os;

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

    for (const auto &op : ops_) {
        os << " [ op: (";
        if (op) {
            os << "ID: " << op->get_id()
               << ", kind: " << impl::op_t::kind2str(op->get_kind()) << " ), ";
        }
    }
    os << " ] \n";

    os << "  [ inputs: ";
    const char *delimer = "";
    for (const auto &i : inputs_) {
        const impl::logical_tensor_wrapper_t v(i);
        os << delimer << "(ID: " << v.id() << "("
           << impl::utils::data_type2str(v.data_type()) << ":"
           << dims_to_string(v.vdims());
        delimer = ")), ";
    }
    os << " ]\n";

    os << "  [ outputs: ";
    delimer = "";
    for (const auto &o : outputs_) {
        const impl::logical_tensor_wrapper_t v(o);
        os << delimer << "(ID: " << v.id() << "("
           << impl::utils::data_type2str(v.data_type()) << ":"
           << dims_to_string(v.vdims());
        delimer = ")), ";
    }
    os << " ]\n";
    os << " ]\n";
    os << "]";

    return os.str();
}

} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
