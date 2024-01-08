/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TUNABLE_OP_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TUNABLE_OP_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <compiler/ir/graph/fusion_data.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/trait/configurable.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <ops/body_generator.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
struct op_dispatch_key_t;
struct impl_op_dispatch_key_t;

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
            const std::shared_ptr<fusion_anchor_mgr_t> &fuse_mgr,
            const std::string &func_name) override {
        throw std::runtime_error("unimplemented");
    }
    ir_module_ptr get_func(context_ptr ctx) override;

    func_t get_func(mixed_parti_t *parti, const std::vector<expr> &ins,
            const std::vector<expr> &outs);

    config_ptr get_config() override { return config_data_; }

    void set_config(const config_ptr &config) override;
    void set_config_if_empty(context_ptr ctx, body_generator_base_t *p);
    virtual void set_config_by_key(
            const op_dispatch_key_t &key, const context_ptr &ctx) {
        throw std::runtime_error("unimplemented");
    }
    virtual void set_internal_config_by_key(
            const impl_op_dispatch_key_t &key, const context_ptr &ctx) {
        throw std::runtime_error("unimplemented");
    }

    config_ptr get_default_config(context_ptr ctx) override;

    void search_anchor(mixed_parti_t *parti) override;

    void commit_into_anchor(fusion_anchor_t *committed_anchor) override;

    config_ptr_vec get_dynamic_config_candidates(
            const context_ptr &ctx) override;
    impl_kind_map convert_config_candidates_to_impl_map(
            const config_ptr_vec &configs) override;
    std::vector<int> get_impl_dispatch_candidates(
            const context_ptr &ctx) override;

    virtual body_generator_ptr create_generator() = 0;

    void create_mixed_partition(mixed_parti_t *parti) override;

    void append_mixed_partition(mixed_parti_t *parti) override;

    virtual infer_status_code infer_slice_ranges(
            const context_ptr &ctx, fslice_map &fsmap)
            = 0;

    void infer_binding_axis(binding_axis_map &bdax_map) override {}
    void pre_infer_binding_axis(binding_axis_map &bdax_map) override {}

protected:
    config_ptr config_data_;
    std::vector<config_ptr> dyn_config_candidates_;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
