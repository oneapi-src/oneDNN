/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
 ******************************************************************************/

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_MATMUL_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_MATMUL_HPP

#include <memory>
#include <vector>
#include "compiler/ir/graph/graph.hpp"
#include "compiler/ir/graph/graph_op.hpp"
#include "compiler/ir/graph/traits.hpp"

namespace sc {
namespace ops {

class matmul_op : public graph_op_t, public op_traits::auto_copyable_t {
public:
    matmul_op(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    std::shared_ptr<sc_graph_t> get_graph() override;
    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;
};

} // namespace ops
} // namespace sc

#endif
