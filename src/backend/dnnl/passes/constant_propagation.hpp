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
#ifndef BACKEND_DNNL_PASSES_CONSTANT_PROPAGATION_HPP
#define BACKEND_DNNL_PASSES_CONSTANT_PROPAGATION_HPP

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <interface/value.hpp>

#include "backend/dnnl/passes/utils.hpp"
#include "backend/dnnl/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace {
inline bool has_scratchpad(impl::op_kind_t kind) {
    const static std::set<impl::op_kind_t> ops {op_kind::dnnl_convolution,
            op_kind::dnnl_bn_folding, op_kind::dnnl_pool, op_kind::dnnl_matmul,
            op_kind::dnnl_convtranspose};
    return ops.count(kind) != 0;
}
}; // namespace

// Because we don't know which logical tensors (may be partition's ins/outs
// edges, or edges inside partition) will be set constant by FWK, so we have to
// do constant propagation bidirectionally
template <bool with_scratchpad = true>
impl::status_t constant_propagation(std::shared_ptr<subgraph_t> &sg) {
    using op_t = impl::op_t;
    using ltw = impl::logical_tensor_wrapper_t;

    bool changed;
    do {
        changed = false;
        impl::topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
            size_t scpad_num
                    = with_scratchpad && has_scratchpad(op->get_kind()) ? 1 : 0;

            bool all_inputs_are_constant = true;
            for (const auto &in : op->get_input_values()) {
                if (ltw(in->get_logical_tensor()).property_type()
                        != property_type::constant) {
                    all_inputs_are_constant = false;
                    break;
                }
            }

            bool all_outputs_are_constant = true;
            for (size_t i = 0; i < op->num_outputs() - scpad_num; i++) {
                auto out = op->get_output_value(i);
                if (ltw(out->get_logical_tensor()).property_type()
                        != property_type::constant) {
                    all_outputs_are_constant = false;
                    break;
                }
            }

            const bool is_constant
                    = all_inputs_are_constant || all_outputs_are_constant;
            op->set_attr<bool>("is_constant", is_constant);

            if (all_inputs_are_constant && !all_outputs_are_constant) {
                // propagate from in to out
                for (auto &out : op->get_output_values()) {
                    out->set_property(property_type::constant);
                }
                changed = changed || true;
            } else if (!all_inputs_are_constant && all_outputs_are_constant) {
                // propagate from out to in
                for (auto &in : op->get_input_values()) {
                    in->set_property(property_type::constant);
                }
                changed = changed || true;
            } else {
                changed = changed || false;
            }
            return status::success;
        });
    } while (changed);
    return impl::status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
