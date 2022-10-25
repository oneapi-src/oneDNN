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
#include "dynamic_utils.hpp"
#include <utility>
#include "dynamic_dispatch_key.hpp"
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <runtime/dynamic_dispatch/hash_dispatch_table.hpp>
#include <util/utils.hpp>
namespace sc {
void initialize_format_table_with_op(
        const sc_op_ptr &op, op_dispatch_tables_ptr &tb) {
    uint32_t inp_size = op->get_inputs().size();
    auto dispatch_keys = op->get_dispatch_key_set();
    if (op->isa<tunable_op_t>()) {
        auto set_format_by_keys
                = [&](const op_dispatch_key_base_t *dispatch_key) {
                      std::vector<runtime::dispatch_key> all_formats
                              = dispatch_key->convert_to_runtime_format_vec();
                      std::vector<runtime::dispatch_key> keys(
                              all_formats.begin(), all_formats.end() - 1);
                      runtime::dispatch_key value
                              = all_formats[all_formats.size() - 1];
                      tb->format_table_[keys] = {value};
                  };
        dispatch_keys->for_each_key_process(set_format_by_keys);
    } else {
        uint64_t unknown_fmt = 0;
        std::vector<runtime::dispatch_key> keys(inp_size, unknown_fmt);
        auto set_format_by_key
                = [&](const op_dispatch_key_base_t *dispatch_key) {
                      std::vector<runtime::dispatch_key> values
                              = dispatch_key->convert_to_runtime_format_vec();
                      // only one input format known.
                      for (uint32_t i = 0; i < inp_size; i++) {
                          keys.clear();
                          keys.resize(inp_size, unknown_fmt);
                          keys[i] = values[i];
                          tb->format_table_[keys] = values;
                      }
                      // all input format known
                      for (uint32_t i = 0; i < inp_size; i++) {
                          keys[i] = values[i];
                      }
                      tb->format_table_[keys] = values;
                  };
        dispatch_keys->for_each_key_process(set_format_by_key);
    }
}

void add_dispatch_symbol_to_kernel_table(op_dispatch_tables_ptr &tb,
        const op_dispatch_key_base_t *key, const std::string &value) {
    std::vector<runtime::dispatch_key> runtime_keys
            = key->convert_to_runtime_format_vec();
    tb->kernel_table_.insert(std::make_pair(runtime_keys, value));
}

bool can_op_be_dispatched(const sc_op_ptr &op) {
    return op->op_name_ != "input" && op->op_name_ != "output"
            && op->op_name_ != "constant"
            && op->get_dispatch_key_set()->size() > 1;
}

runtime::dynamic_tensor_t convert_graph_tensor_to_dynamic_tensor(
        const graph_tensor_ptr &in, void *data_ptr, sc_dim *shape_ptr) {
    runtime::dynamic_tensor_t ret;
    auto &plain_dims = in->details_.get_plain_dims();
    ret.data_ = data_ptr;
    ret.dims_ = shape_ptr;
    ret.ndims_ = static_cast<int>(plain_dims.size());
    ret.dtype_ = static_cast<uint32_t>(in->details_.dtype_.type_code_);
    ret.dyn_mask_ = 0;
    for (int i = 0; i < static_cast<int>(plain_dims.size()); i++) {
        if (is_dynamic_dim(plain_dims[i])) { ret.dyn_mask_ |= (1 << i); }
    }
    return ret;
}

expr divide_and_ceil(const expr &v, const expr &d) {
    return do_cast_and_fold((v + d - 1) / d);
}
} // namespace sc
