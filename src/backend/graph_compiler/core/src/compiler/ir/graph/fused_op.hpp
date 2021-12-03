/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSED_OP_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSED_OP_HPP

#include <memory>
#include <string>
#include <vector>
#include "graph_op.hpp"

namespace sc {
namespace horizontal_merge_type {
constexpr int no_merge = 0;
}
class fusion_manager;

// inputs: base op inputs, additional args inputs (should be in the same order
// of the input ops in fmgr)
// outputs: If need to keep base op output, base op output will be the first
// element in the outs. Then the output of fmgr
class fused_op_t : public graph_op_t {
public:
    std::shared_ptr<fusion_manager> mgr_;
    sc_graph_t main_op_;
    std::vector<bool> keep_outputs_ = {false};
    op_traits::post_fusion_acceptable_t *get_main_op() const;
    fused_op_t(const std::string &name, sc_graph_t &&main_op,
            std::shared_ptr<fusion_manager> fuse_mgr,
            const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    std::shared_ptr<sc_graph_t> get_graph() override;
    ir_module_ptr get_func(context_ptr ctx) override;
    bool is_valid(const context_ptr &) override;
    bool compare_contents(const sc_op *other) const override;
    size_t hash_contents() const override;
    ir_module_ptr try_get_func(const context_ptr &ctx, bool just_check,
            std::vector<sc_op_ptr> &out_failed);
};

class horizontal_fused_op_t : public graph_op_t {
public:
    horizontal_fused_op_t(const std::string &name,
            const std::vector<sc_op_ptr> &ops_to_merge,
            const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    std::vector<sc_op_ptr> ops_to_merge_;
    ir_module_ptr get_func(context_ptr ctx) override;
    std::shared_ptr<sc_graph_t> get_graph() override;
    void schedule_loops(const stmt &body);
};
} // namespace sc

#endif
