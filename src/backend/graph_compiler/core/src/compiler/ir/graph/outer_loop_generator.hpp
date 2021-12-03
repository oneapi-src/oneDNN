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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_OUTER_LOOP_GENERATOR_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_OUTER_LOOP_GENERATOR_HPP
#include <memory>
#include <vector>
#include "fusible_op.hpp"
#include <ops/body_generator.hpp>

namespace sc {

// generates the outer loops for an op. By default it will generates N-1 outer
// loops for the first input N-D tensor from higher dims to lower
class outer_loop_generator_t : public body_generator_base_t {
private:
    // decide which one is base input
    size_t base_inp_idx_;

public:
    outer_loop_generator_t(size_t base_inp_idx = 0);
    std::shared_ptr<void> get_default_config(context_ptr ctx) const override {
        return nullptr;
    }

    size_t get_base_inp_idx() const { return base_inp_idx_; }

    bool generate(context_ptr ctx, const void *config, fusion_manager *fusion,
            const std::vector<expr> &inputs, const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const override;

    void schedule_loops(context_ptr ctx, const void *config, stmt body,
            std::vector<for_loop> &fors) const override;
    float get_gflop() const override { return 0; }
};

class top_level_anchor_generator_t : public outer_loop_generator_t {
    bool generate(context_ptr ctx, const void *config, fusion_manager *fusion,
            const std::vector<expr> &inputs, const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const override;

    void schedule_loops(context_ptr ctx, const void *config, stmt body,
            std::vector<for_loop> &fors) const override;
};

// generates the fused function for the ops in fmgr. The order of function
// args will be out tensors followed by in tensors. In the arg list of
// in/out tensors, Ops with smaller op_id will be inserted before larger
// ones.
ir_module_ptr lower_fusion_manager(const context_ptr &ctx,
        outer_loop_generator_t *gen, sc_op *op, fusion_manager *fmgr,
        bool check_parallel);
ir_module_ptr try_lower_fusion_manager(const context_ptr &ctx,
        outer_loop_generator_t *gen, sc_op *op, fusion_manager *fmgr,
        bool check_parallel, bool just_check,
        std::vector<sc_op_ptr> &out_failed);
} // namespace sc

#endif
