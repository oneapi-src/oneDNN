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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_MATMUL_CORE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_MATMUL_CORE_HPP

#include <vector>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/graph/tunable_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

struct blocking_axis_t;

class SC_INTERNAL_API matmul_core_op_t : public tunable_op_t {
public:
    matmul_core_op_t(const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs);
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    body_generator_ptr create_generator() override;
    float get_gflop() override;
    sc_dims get_batch_dims() const;
    static sc_dims get_batch_dims_impl(const sc_dims &, const sc_dims &);

    // Various broadcasts are supported:
    // (1, M, K) * (A, B, K, N) -> (A, B, M, K)
    // (A, B, M, K) * (A, 1, K, N) -> (A, B, M, K)
    // (A, B, M, K) * (1, B, K, N) -> (A, B, M, N)
    // (1, B, M, K) * (A, 1, K, N) -> (A, B, M, N)
    // (A, B, C, M, K) * (A, 1, C, K, N) -> (A, B, C, M, N).
    static sc_dims get_batch_dims_with_bc_impl(
            const sc_dims &, const sc_dims &);

    static sc_data_type_t infer_out_dtype(
            const std::vector<graph_tensor_ptr> &);
    sc_op_ptr do_compensations(
            sc_graph_t &mgr, const context_ptr &ctx) override;
    sc_op_ptr get_data_compensation(sc_graph_t &mgr);
    // reuse cast and reduce nodes to do s8s8 and weight compensations togethor
    std::vector<sc_op_ptr> get_s8s8_and_weight_compensation(
            sc_graph_t &mgr, bool s8s8_compensation);
    sc_op_ptr get_constant_compensation(sc_graph_t &mgr);

    infer_status_code infer_slice_ranges(
            const context_ptr &ctx, fslice_map &fsmap) override;

    void infer_binding_axis(binding_axis_map &bdax_map) override;
    void pre_infer_binding_axis(binding_axis_map &bdax_map) override;

    void set_config_by_key(
            const op_dispatch_key_t &key, const context_ptr &ctx) override;
    std::vector<int> get_impl_dispatch_candidates(
            const context_ptr &ctx) override;
    shape_rl_vec get_dynamic_shape_relations() const override;
    static shape_rl_vec get_shape_relations_impl(const sc_dims &data_plain_dims,
            const sc_dims &weight_plain_dims, const sc_dims &out_plain_dims);

    sc_dims batch_dims_;
};

blocking_axis_t get_mm_blocking_axis(const logical_tensor_t &inp,
        const logical_tensor_t &wei, const logical_tensor_t &out);

void infer_matmul_binding_axis(tunable_op_t *cur, binding_axis_map &bdax_map);
void pre_matmul_binding_axis(tunable_op_t *cur, binding_axis_map &bdax_map);

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
