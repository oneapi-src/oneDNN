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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TUNABLE_OP_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TUNABLE_OP_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <ops/body_generator.hpp>
#include <util/utils.hpp>

namespace sc {
class SC_INTERNAL_API tunable_op_t
    : public sc_op,
      public op_traits::copyable_t,
      public op_traits::may_quantize_t,
      public op_traits::post_fusion_acceptable_t {
public:
    tunable_op_t(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;

    bool is_valid(const context_ptr &) override;

    ir_module_ptr get_func(context_ptr ctx,
            const std::shared_ptr<fusion_manager> &fuse_mgr,
            const std::string &func_name) override {
        throw std::runtime_error("unimplemented");
    }
    ir_module_ptr get_func(context_ptr ctx) override;

    std::shared_ptr<void> get_config() { return config_data_; }

    void set_config(const std::shared_ptr<void> &config);
    void set_config_if_empty(context_ptr ctx, body_generator_base_t *p);

    virtual body_generator_ptr create_generator() = 0;

protected:
    std::shared_ptr<void> config_data_;
};

} // namespace sc

#endif
