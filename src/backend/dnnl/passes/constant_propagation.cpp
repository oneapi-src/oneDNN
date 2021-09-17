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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "dnnl.hpp"

#include <interface/value.hpp>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/passes/utils.hpp"
#include "backend/dnnl/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using ltw = impl::logical_tensor_wrapper;

namespace {
inline bool has_scratchpad(impl::op_kind_t kind) {
    const static std::set<impl::op_kind_t> ops {op_kind::dnnl_convolution,
            op_kind::dnnl_bn_folding, op_kind::dnnl_pool,
            impl::op_kind::MatMul};
    return ops.count(kind) != 0;
}
}; // namespace

// Because we don't know which logical tensors (may be partition's ins/outs
// edges, or edges inside partition) will be set constant by FWK, so we have to
// do constant propagation bidirectionally
void constant_propagation(std::vector<op_ptr> &subgraph, bool with_scratchpad) {
    impl::graph_t tmp_graph(subgraph);
    bool changed;
    do {
        changed = false;
        impl::topo_order_visit(tmp_graph.get_output_ops(), [&](op_t *op) {
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

            if (all_inputs_are_constant || all_outputs_are_constant) {
                op->set_attr<bool>("is_constant", true);
            } else {
                op->set_attr<bool>("is_constant", false);
            }

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
}

std::vector<value_t *> get_constant_block_output_values(
        const std::vector<op_ptr> &subgraph) {
    using ltw = impl::logical_tensor_wrapper;
    std::vector<value_t *> ret;
    for (auto &cur_op : subgraph) {
        auto out_vals = cur_op->get_output_values();
        for (auto &val : out_vals) {
            if (!ltw(val->get_logical_tensor()).is_constant()) continue;
            // if a constant value feed into a consumer whose output is not
            // constant, then the value is the final output of a constant block
            auto consumers = val->get_consumers();
            for (auto &csm : consumers) {
                // A consumer is not constant
                if (!csm.get_op().get_attr<bool>("is_constant")) {
                    ret.emplace_back(val.get());
                    break;
                }
            }
        }
    }
    return ret;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
