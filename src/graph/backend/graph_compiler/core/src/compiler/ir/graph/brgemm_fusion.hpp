/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_BRGEMM_FUSION_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_BRGEMM_FUSION_HPP
#include <vector>
#include "graph.hpp"
#include <compiler/ir/builtin.hpp>
#include <runtime/microkernel/cpu/brgemm_common.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
// When fusion manager executing `do_compute_blocks` pass, check if it can be
// registered to brgemm inside fusion, otherwise do trival computation.
struct brgemm_fusion_register {
    // register in brgemm fusion needs op infos, extra input tensor shape and
    // index, op output tensor index.
    bool register_op_infos(const sc_op_ptr &op, const expr &output,
            const expr &extra_in = get_ir_null(),
            const std::vector<expr> &extra_in_shape = std::vector<expr>());
    // only have one valid brgemm
    bool can_register_brgemm_fusion(const stmt &body);
    stmt remake_brgemm_intrinsic_by_fusion(
            stmt body, expr c_buf = get_ir_null()) const;
    void reset();
    bool can_register_next_ = true;
    expr last_out_;
    expr valid_brgemm_node_;
    sc_brgemm_postops_setting_t setting_;
    std::vector<expr> data_ = builtin::create_initialed_postops_data();
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
