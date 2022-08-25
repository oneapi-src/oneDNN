/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "interface/value.hpp"

#include "backend/dnnl/internal_attrs.hpp"
#include "backend/dnnl/passes/utils.hpp"
#include "backend/dnnl/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace {
bool has_scratchpad(const impl::op_t *op) {
    // the following ops do not have scratchpad output by definition
    const static std::set<impl::op_kind_t> no_scratchpad_ops {
            op_kind::dnnl_constant_scales,
            op_kind::dnnl_constant_zps,
            op_kind::dnnl_add_zps,
            op_kind::dnnl_sub_zps,
            op_kind::dnnl_to_group,
            op_kind::dnnl_from_group,
            op_kind::dnnl_permute,
            op_kind::dnnl_squeeze,
            op_kind::dnnl_unsqueeze,
            op_kind::dnnl_transpose,
            op_kind::dnnl_reshape,
    };

    // the following ops may have scratchpad output if output size > 1
    const static std::set<impl::op_kind_t> may_have_scratchpad_ops {
            op_kind::dnnl_mul_scales,
            op_kind::dnnl_reorder,
    };

    const impl::op_kind_t kind = op->get_kind();
    const bool no_scratchpad = no_scratchpad_ops.count(kind)
            || (may_have_scratchpad_ops.count(kind) && op->num_outputs() == 1);

    return !no_scratchpad;
}
}; // namespace

impl::status_t constant_propagation(std::shared_ptr<subgraph_t> &sg) {
    using op_t = impl::op_t;
    using ltw = impl::logical_tensor_wrapper_t;

    // Because we don't know which logical tensors (may be partition's ins/outs
    // edges, or edges inside partition) will be set to constant by FWK, so we
    // have to do constant propagation bidirectionally
    bool changed;
    do {
        changed = false;
        auto ret = impl::topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
            // FIXME(xx): Because constant cache is only for inference
            // optimization, so we don't consider the workspace output
            // currently. But this may be a potential issue if we also use
            // constant cache in training scenarios
            size_t scpad_num = has_scratchpad(op) ? 1 : 0;

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
            op->set_attr<bool>(op_attr::is_constant, is_constant);

            // FIXME(xx): Currently, we consider that if the inputs of an op are
            // constant, then its outputs should also be constant, vice versa.
            // But this assumption may be broken if we have more ops in the
            // future. For example, if we have an Shape op, who returns the
            // shape of input tensor. In static shape scenarios, its output will
            // be constant, but its input can be variable. Such ops should have
            // special constant propagation rules
            if (all_inputs_are_constant && !all_outputs_are_constant) {
                // propagate from in to out
                for (size_t i = 0; i < op->num_outputs() - scpad_num; i++) {
                    auto out = op->get_output_value(i);
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

        if (ret != impl::status::success) return ret;
    } while (changed);
    return impl::status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
