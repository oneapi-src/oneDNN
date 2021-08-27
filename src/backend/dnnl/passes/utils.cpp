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

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/passes/utils.hpp"
#include "backend/dnnl/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using value_ptr = std::shared_ptr<impl::value_t>;
using ltw = impl::logical_tensor_wrapper;

// this function fuse a op to its successor.
// you should guarantee that the op has only one successor
//
//   in_val
//     |
//   next_op         in_val
//     |      --->     |
//   base_op         base_op
//     |               |
//   out_val         out_val
void fuse_op_to_successor(op_t *op, std::vector<op_ptr> &subgraph) {
    assertm(op->num_inputs() == 1, "this op should have only one input value.");
    value_ptr in_val = op->get_input_value(0);
    in_val->remove_consumer(*op, 0);

    assertm(op->num_outputs() == 1,
            "this op should have only one output value.");
    value_ptr out_val = op->get_output_value(0);
    auto consumers = out_val->get_consumers();
    assertm(!consumers.empty() && consumers.size() == 1,
            "this op has zero consumer or more than one consumers.");

    op_t &successor = consumers[0].get_op();
    size_t offset = consumers[0].get_offset();
    in_val->add_consumer(successor, offset);
    successor.connect_input(offset, in_val);

    auto pos = std::find_if(subgraph.begin(), subgraph.end(),
            [op](const op_ptr &tmp) { return op == tmp.get(); });
    if (pos != subgraph.end()) subgraph.erase(pos);
}

//   in_val                  in_val     in_val2
//     |                         \       /
//   base_op  in_val2             base_op
//      \       /       --->         |
//       next_op                  out_val
//          |
//       out_val
void fuse_op_to_predecessor(
        op_t *op, std::vector<op_ptr> &subgraph, size_t in_offset) {
    value_ptr in_val = op->get_input_value(in_offset);
    assertm(op->num_outputs() == 1,
            "this op should have only one output value.");
    value_ptr out_val = op->get_output_value(0);

    op_t &predecessor = in_val->get_producer();
    size_t offset = in_val->get_offset();
    predecessor.connect_output(offset, out_val);

    for (size_t i = 0; i < op->num_inputs(); i++) {
        value_ptr tmp = op->get_input_value(i);
        if (tmp == in_val) { continue; }

        tmp->remove_consumer(*op, i);
        tmp->add_consumer(predecessor, predecessor.num_inputs());
        predecessor.add_input(tmp);
    }

    auto pos = std::find_if(subgraph.begin(), subgraph.end(),
            [op](const op_ptr &tmp) { return op == tmp.get(); });
    if (pos != subgraph.end()) subgraph.erase(pos);
}

//   in_val          in_val
//     |               |
//     |     ->    inserted_op
//     |               |
//     |             new_val
//     |               |
//  base_op         base_op
void insert_op_before(op_ptr &inserted_op, op_ptr &base_op, size_t offset) {
    value_ptr in_val = base_op->get_input_value(offset);
    in_val->remove_consumer(*base_op, offset);
    in_val->add_consumer(*inserted_op, inserted_op->num_inputs());
    inserted_op->add_input(in_val);

    impl::logical_tensor_t new_lt
            = impl::empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*inserted_op, 0, new_lt, true);
    inserted_op->add_output(new_val);

    new_val->add_consumer(*base_op, offset);
    base_op->connect_input(offset, new_val);
}

//   base_op         base_op
//     |               |
//     |             new_val
//     |               |
//     |     ->    inserted_op
//     |               |
//  out_val         out_value
void insert_op_after(op_ptr &inserted_op, op_ptr &base_op, size_t offset) {
    value_ptr out_val = base_op->get_output_value(offset);
    inserted_op->add_output(out_val);

    impl::logical_tensor_t new_lt
            = impl::empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*base_op, 0, new_lt, true);
    base_op->connect_output(offset, new_val);

    new_val->add_consumer(*inserted_op, inserted_op->num_inputs());
    inserted_op->add_input(new_val);
}

void set_given_inputs_outputs(std::vector<op_ptr> &subgraph,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs) {
    // set the inputs's layout to subgraph's inputs value
    auto graph_in_vals = impl::graph_t(subgraph).get_input_values();
    auto graph_out_vals = impl::graph_t(subgraph).get_output_values();

    for (auto &graph_in : graph_in_vals) {
        for (const auto &in : inputs) {
            if (graph_in->get_logical_tensor().id == in.id) {
                graph_in->set_logical_tensor(in);
            }
        }
    }

    for (auto &graph_out : graph_out_vals) {
        for (const auto &out : outputs) {
            if (graph_out->get_logical_tensor().id == out.id) {
                graph_out->set_logical_tensor(out);
            }
        }
    }
}

void set_all_layout_to_any(std::vector<op_ptr> &subgraph) {
    for (auto &cur_op : subgraph) {
        for (auto val : cur_op->get_input_values()) {
            val->set_layout_type(impl::layout_type::any);
        }

        for (auto val : cur_op->get_output_values()) {
            val->set_layout_type(impl::layout_type::any);
        }
    }
}

// Constant property should be set by users from API level, this function is
// just a workaround at this moment.
void set_weight_bias_constant(std::vector<op_ptr> &subgraph) {
    for (auto &op : subgraph) {
        if (!(op->get_kind() == impl::op_kind::MatMul
                    || op->get_kind() == impl::op_kind::Convolution
                    || op->get_kind() == op_kind::dnnl_convolution))
            continue;

        // set weight to be constant
        op->get_input_value(1)->set_property(property_type::constant);

        // set bias to be constant
        if (op->get_attr<bool>("with_bias")) {
            op->get_input_value(2)->set_property(property_type::constant);
        }
    }
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
