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
 *******************************************************************************/

#include <memory>
#include <vector>
#include <unordered_map>

#include "interface/c_types_map.hpp"
#include "interface/value.hpp"

#include "backend/dnnl/legacy.hpp"
#include "backend/dnnl/subgraph/compile_ops.hpp"
#include "backend/dnnl/subgraph/op_executable.hpp"
#include "backend/dnnl/subgraph/passes.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_ptr = std::shared_ptr<impl::op_t>;

/// After the lower down, infer shape, infer type and layout propagation passes,
/// each op in the subgraph will has complete attributes and each edge will have
/// complete shape/dtype/layout information. We can create executable for these
/// ops.
impl::status_t compile_ops(std::vector<op_ptr> &subgraph,
        const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr,
        executable_mgr &exec_mgr) {
    for (auto &cur_op : subgraph) {
        if (cur_op->has_attr("executable_key")) continue; // already compiled

        int64_t key = exec_mgr.init_executable();
        cur_op->set_attr<int64_t>("executable_key", key);
        std::shared_ptr<op_executable> &prm = exec_mgr.get_executable(key);

        if (cur_op->get_kind() == op_kind::Convolution
                || cur_op->get_kind() == op_kind::dnnl_convolution) {
            prm = std::make_shared<conv_fwd_executable>(
                    cur_op, p_engine, prm_attr_mgr);
        } else if (cur_op->get_kind() == op_kind::MatMul) {
            prm = std::make_shared<matmul_executable>(
                    cur_op, p_engine, prm_attr_mgr);
        } else if (cur_op->get_kind() == op_kind::MaxPool) {
            prm = std::make_shared<pool_executable>(
                    cur_op, p_engine, prm_attr_mgr);
        } else if (cur_op->get_kind() == op_kind::mul_scales
                || cur_op->get_kind() == op_kind::Reorder) {
            prm = std::make_shared<reorder_executable>(
                    cur_op, p_engine, prm_attr_mgr);
        } else if (cur_op->get_kind() == op_kind::permute
                || cur_op->get_kind() == op_kind::to_group
                || cur_op->get_kind() == op_kind::expand) {
            // For preprocess ops. The memory_reparser will not do
            // computation, it only re-parses the existing buffer.
            prm = std::make_shared<memory_reparser>();
        } else {
            assertm(false, "unimplemented op, can't compile it");
            return impl::status::compile_fail;
        }
    }
    return impl::status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
