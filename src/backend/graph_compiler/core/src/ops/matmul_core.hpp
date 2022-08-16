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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_MATMUL_CORE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_MATMUL_CORE_HPP

#include <vector>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/graph/tunable_op.hpp>

namespace sc {
namespace ops {

struct blocking_axes_t;

class SC_INTERNAL_API matmul_core_op_t
    : public tunable_op_t,
      public op_traits::batchwise_shrinkable_t {
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
    sc_op_ptr do_compensations(
            sc_graph_t &mgr, const context_ptr &ctx) override;
    sc_op_ptr get_data_compensation(sc_graph_t &mgr);
    // reuse cast and reduce nodes to do s8s8 and weight compensations togethor
    std::vector<sc_op_ptr> get_s8s8_and_weight_compensation(
            sc_graph_t &mgr, bool s8s8_compensation);
    sc_op_ptr get_constant_compensation(sc_graph_t &mgr);

    sc_dims get_bwise_fuse_shrink_dims() override;

    void collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map) override;

    void collect_shrinked_axes_map(
            int bw_size, gt2axes_map &bw_axes_map) override;

    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override;

    void set_config_by_key(const op_dispatch_key_t &key) override;
    std::vector<int> get_impl_dispatch_candidates() const override;
};

blocking_axes_t get_mm_blocking_axes(const logical_tensor_t &inp,
        const logical_tensor_t &wei, const logical_tensor_t &out);

} // namespace ops
} // namespace sc
#endif
