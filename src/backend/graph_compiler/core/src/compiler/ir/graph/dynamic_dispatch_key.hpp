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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_DISPATCH_KEY_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_DISPATCH_KEY_HPP
#include <set>
#include <vector>
#include <compiler/ir/sc_data_format.hpp>
#include <runtime/dynamic_dispatch/ops/impl_type.hpp>
namespace sc {
// the dispatch key type for lowering. Will be used in a map in lowering. The
// key is this struct and the value is kernel.
struct op_dispatch_key_t {
    // Currently only need for tunable op. Size is same as in_out_formats, and
    // illustrate the config of input/outputs. E.g matmul_core config (M, N,
    // K)[32, 16, 64], we got {{32, 64}, {64, 16}, {32, 16}}.
    std::vector<std::vector<sc_dim>> var_block_;
    // a vector of input/output formats, order is input 0,1,..., output 0,1,...
    std::vector<sc_data_format_t> in_out_formats_;
    // the op can be dispatched as padding or not.
    int impl_ = impl_kind_t::normal;
    op_dispatch_key_t() = default;
    op_dispatch_key_t(const std::vector<sc_data_format_t> &formats,
            int impl = impl_kind_t::normal)
        : in_out_formats_(formats), impl_(impl) {}
    op_dispatch_key_t(const std::vector<std::vector<sc_dim>> &var_block,
            const std::vector<sc_data_format_t> &formats, bool impl = false)
        : var_block_(var_block), in_out_formats_(formats), impl_(impl) {}
    bool operator==(const op_dispatch_key_t &other) const {
        return var_block_ == other.var_block_
                && in_out_formats_ == other.in_out_formats_
                && impl_ == other.impl_;
    }
};

struct dispatch_key_cmper_t {
    bool operator()(
            const op_dispatch_key_t &key0, const op_dispatch_key_t &key1) const;
};

struct dispatch_key_set_t {
    using inner_set_t = std::set<op_dispatch_key_t, dispatch_key_cmper_t>;
    inner_set_t set_;
};

inline std::vector<int> get_default_impl_dispatch_candidates() {
    static std::vector<int> default_impl_candidates
            = {impl_kind_t::normal, impl_kind_t::no_padding};
    return default_impl_candidates;
}
} // namespace sc

#endif
