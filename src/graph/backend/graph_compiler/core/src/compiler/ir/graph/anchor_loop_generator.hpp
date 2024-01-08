/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_ANCHOR_LOOP_GENERATOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_ANCHOR_LOOP_GENERATOR_HPP
#include <memory>
#include <vector>
#include "graph.hpp"
#include <ops/body_generator.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct fusion_anchor_t;

class anchor_loop_generator_t : public body_generator_base_t {
private:
    // decide which one is base graph tensor
    graph_tensor_ptr base_gt_;

public:
    anchor_loop_generator_t(const graph_tensor_ptr &base_gt);
    config_ptr get_default_config(context_ptr ctx) const override {
        return nullptr;
    }

    bool generate(context_ptr ctx, const void *config,
            fusion_anchor_mgr_t *fmgr, const std::vector<expr> &inputs,
            const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const override {
        return false;
    };

    bool create_outer_loop_anchor(
            fusion_anchor_mgr_t *fmgr, const context_ptr &ctx) const;

    bool create_inner_loop_anchor(fusion_anchor_mgr_t *fmgr,
            const std::shared_ptr<fusion_anchor_t> &parent_fanchor) const;

    void schedule_loops(context_ptr ctx, const void *config, stmt body,
            std::vector<for_loop> &fors) const override {};
    float get_gflop() const override { return 0; }
};

for_loop get_next_inner_loop(const for_loop &cur_loop);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
