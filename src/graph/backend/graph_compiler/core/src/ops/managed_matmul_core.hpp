/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_MANAGED_MATMUL_CORE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_MANAGED_MATMUL_CORE_HPP

#include <vector>
#include <compiler/ir/graph/trait/may_prefetch.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/graph/tunable_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

class SC_INTERNAL_API managed_matmul_core_op_t
    : public tunable_op_t,
      public op_traits::may_prefetch_t {
public:
    managed_matmul_core_op_t(const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs);
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    body_generator_ptr create_generator() override;
    float get_gflop() override;
    sc_dims get_batch_dims() const;
    sc_op_ptr do_compensations(
            sc_graph_t &mgr, const context_ptr &ctx) override;
    sc_op_ptr get_data_compensation(sc_graph_t &mgr);
    // reuse cast and reduce nodes to do s8s8 and weight compensations togethor
    std::vector<sc_op_ptr> get_s8s8_and_weight_compensation(
            sc_graph_t &mgr, bool s8s8_compensation);
    sc_op_ptr get_constant_compensation(sc_graph_t &mgr);
    shape_rl_vec get_dynamic_shape_relations() const override;
    bool need_dynamic_internal_query_impl() const override;
    ir_module_ptr get_internal_func(const context_ptr &ctx) override;

    std::vector<int> get_impl_dispatch_candidates(
            const context_ptr &ctx) override;
    dispatch_set_ptr get_internal_dispatch_key_set(
            const context_ptr &ctx) override;

    infer_status_code infer_slice_ranges(
            const context_ptr &ctx, fslice_map &fsmap) override {
        // TODO(XXX)
        return infer_status_code::FAIL;
    }

    std::vector<int> query_prefetch(const context_ptr &ctx, bool is_global,
            const std::vector<tensor_slice> &ins) override;

    void generate_prefetcher_body_for_tensor(const context_ptr &ctx,
            const std::vector<expr> &func_args, const std::vector<expr> &ins,
            const std::vector<int> &indices) override;
    void infer_binding_axis(binding_axis_map &bdax_map) override;
    void pre_infer_binding_axis(binding_axis_map &bdax_map) override;

    void set_config_by_key(
            const op_dispatch_key_t &key, const context_ptr &ctx) override;
    void set_internal_config_by_key(
            const impl_op_dispatch_key_t &key, const context_ptr &ctx) override;
    virtual sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins, // NOLINT
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;

private:
    int iim_block_ = -1;
    int iin_block_ = -1;
    int iik_block_ = -1;
};
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
