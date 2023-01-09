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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_SHAPE_OF_TENSOR_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_SHAPE_OF_TENSOR_HPP

#include <functional>
#include <vector>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/runtime_op.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/ir_module.hpp>

namespace sc {
namespace ops {

// Get plain(may padded) shapes of a tensor
class shape_of_tensor_op_t : public runtime_op_t,
                             public op_traits::auto_copyable_t {
public:
    shape_of_tensor_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    ir_module_ptr get_func(context_ptr ctx) override;
    // the function is mainly for padded K in constant compensation calculation
    // for matmul
    void set_find_ltsr_func(
            const std::function<std::vector<graph_tensor_ptr>(const sc_op *)>
                    &func) {
        find_ltsr_func_ = func;
    }
    runtime_extra_info_t get_extra_lower_infos(
            sc_graph_t &graph, ir_module_ptr &m) override;

private:
    // indexes of real/padded_plain shapes
    std::vector<int> shape_idxs_;
    std::function<std::vector<graph_tensor_ptr>(const sc_op *)> find_ltsr_func_;
};
} // namespace ops
} // namespace sc

#endif
