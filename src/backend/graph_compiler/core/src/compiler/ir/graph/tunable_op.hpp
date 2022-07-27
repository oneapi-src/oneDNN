/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/trait/configurable.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <ops/body_generator.hpp>
#include <util/utils.hpp>

namespace sc {
class SC_INTERNAL_API tunable_op_t
    : public sc_op,
      public op_traits::copyable_t,
      public op_traits::may_quantize_t,
      public op_traits::post_fusion_acceptable_t,
      public op_traits::configurable_t,
      public op_traits::mixed_partition_acceptable {
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

    func_t get_func(mixed_parti_t *parti, const std::vector<expr> &ins,
            const std::vector<expr> &outs);

    config_ptr get_config() override { return config_data_; }

    void set_config(const config_ptr &config) override;
    void set_config_if_empty(context_ptr ctx, body_generator_base_t *p);

    config_ptr get_default_config(context_ptr ctx) override;

    void search_anchor(mixed_parti_t *parti) override;

    void commit_into_anchor(mixed_parti_t *parti) override;

    virtual body_generator_ptr create_generator() = 0;

    void create_mixed_partition(mixed_parti_t *parti) override;

    void append_mixed_partition(mixed_parti_t *parti) override;

    virtual void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map)
            = 0;

protected:
    config_ptr config_data_;
};

} // namespace sc

#endif
