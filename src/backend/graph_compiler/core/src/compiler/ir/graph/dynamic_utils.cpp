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
#include <runtime/dynamic_dispatch/hash_dispatch_table.hpp>
#include <util/utils.hpp>
namespace sc {
std::vector<runtime::dispatch_key> convert_to_runtime_format_vec(
        const op_dispatch_key_t &key) {
    std::vector<runtime::dispatch_key> outs(key.in_out_formats_.size());
    bool var_block_empty = key.var_block_.empty();
    assert(var_block_empty
            || key.var_block_.size() == key.in_out_formats_.size());
    for (size_t i = 0; i < key.in_out_formats_.size(); i++) {
        sc_dim block0 = 0, block1 = 0;
        if (!var_block_empty) {
            block0 = key.var_block_[i][0];
            block1 = key.var_block_[i][1];
        } else {
            block0 = key.in_out_formats_[i].blocks_[0];
            block1 = key.in_out_formats_[i].blocks_[1];
        }
        outs[i] = runtime::dispatch_key(
                static_cast<uint64_t>(key.in_out_formats_[i].format_code_),
                block0, block1, key.impl_, key.in_out_formats_[i].is_plain());
    }
    return outs;
}

void initialize_format_table_with_op(
        const sc_op_ptr &op, op_dispatch_tables_ptr &tb) {
    uint32_t inp_size = op->get_inputs().size();
    auto dispatch_keys = op->get_dispatch_key_set();
    if (op->isa<tunable_op_t>()) {
        for (auto &dispatch_key : dispatch_keys->set_) {
            std::vector<runtime::dispatch_key> all_formats
                    = convert_to_runtime_format_vec(dispatch_key);
            std::vector<runtime::dispatch_key> keys(
                    all_formats.begin(), all_formats.end() - 1);
            runtime::dispatch_key value = all_formats[all_formats.size() - 1];
            tb->format_table_[keys] = {value};
        }
    } else {
        uint64_t unknown_fmt = 0;
        std::vector<runtime::dispatch_key> keys(inp_size, unknown_fmt);
        for (auto &dispatch_key : dispatch_keys->set_) {
            std::vector<runtime::dispatch_key> values
                    = convert_to_runtime_format_vec(dispatch_key);
            // only one input format known.
            for (uint32_t i = 0; i < inp_size; i++) {
                keys.resize(inp_size, unknown_fmt);
                keys[i] = values[i];
                tb->format_table_[keys] = values;
            }
            // all input format known
            for (uint32_t i = 0; i < inp_size; i++) {
                keys[i] = values[i];
            }
            tb->format_table_[keys] = values;
        }
    }
}

void add_dispatch_symbol_to_kernel_table(op_dispatch_tables_ptr &tb,
        const op_dispatch_key_t &key, const std::string &value) {
    std::vector<runtime::dispatch_key> runtime_keys
            = convert_to_runtime_format_vec(key);
    tb->kernel_table_.insert(std::make_pair(runtime_keys, value));
}

bool can_op_be_dispatched(const sc_op_ptr &op) {
    return op->op_name_ != "input" && op->op_name_ != "output"
            && op->op_name_ != "constant" && op->op_name_ != "tensor_view"
            && op->get_dispatch_key_set()->set_.size() > 1;
}

expr divide_and_ceil(const expr &v, const expr &d) {
    return do_cast_and_fold((v + d - 1) / d);
}
} // namespace sc
