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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_BATCH_MATMUL_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_BATCH_MATMUL_HPP

#include <vector>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/graph/tunable_op.hpp>

namespace sc {
namespace ops {

namespace BMM_kind {
constexpr int Normal = 0;
constexpr int ATT_QK = 1;
constexpr int ATT_V = 2;
}; // namespace BMM_kind

class SC_INTERNAL_API batch_matmul_op_t : public tunable_op_t {
public:
    batch_matmul_op_t(const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs);
    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;
    body_generator_ptr create_generator() override;
    float get_gflop() override;
    sc_dims get_batch_dims();
    sc_op_ptr do_compensations(
            sc_graph_t &mgr, const context_ptr &ctx) override;
    sc_op_ptr get_s8s8_compensation(sc_graph_t &mgr);
    sc_op_ptr get_data_compensation(sc_graph_t &mgr);
    sc_op_ptr get_weight_compensation(sc_graph_t &mgr);
    sc_op_ptr get_constant_compensation(sc_graph_t &mgr);
};
} // namespace ops
} // namespace sc
#endif
