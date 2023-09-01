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

#include <utility>

#include "graph/utils/any.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/dnnl_backend.hpp"
#include "graph/backend/dnnl/dnnl_opset.hpp"
#include "graph/backend/dnnl/kernels/kernels.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

dnnl_backend::dnnl_backend(const std::string &name, float priority)
    : backend_t(name, priority) {
    register_op_schemas();
    register_passes();
}

bool dnnl_backend::register_op_schemas() {
    register_dnnl_opset_schema();
    return true;
}

bool dnnl_backend::register_passes() {
#define DNNL_BACKEND_REGISTER_PATTERN_CALL(pattern_class_, pattern_registry_) \
    pattern::register_##pattern_class_(pattern_registry_);

    DNNL_BACKEND_REGISTER_PATTERN_CALL(binary_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(bn_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(concat_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(conv_block_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(conv_post_ops, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(convtranspose_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(matmul_post_ops, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(sdp, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(single_op_pass, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(pool_post_ops, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(eltwise_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(quantize_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(interpolate_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(softmax_post_ops, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(layernorm_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(sum_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(reorder_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(shuffle_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(reduction_fusion, pass_registry_);
    pass_registry_.sort_passes();

#undef DNNL_BACKEND_REGISTER_PATTERN_CALL

    return true;
}

size_t dnnl_backend::get_mem_size(const logical_tensor_t &lt) const {
    auto md = make_dnnl_memory_desc(lt);
    return md.get_size();
}

bool dnnl_backend::compare_logical_tensor(
        const logical_tensor_t &lhs, const logical_tensor_t &rhs) const {
    auto md1 = make_dnnl_memory_desc(lhs);
    auto md2 = make_dnnl_memory_desc(rhs);
    return md1 == md2;
}

graph::utils::optional_t<size_t> dnnl_backend::set_mem_desc(
        const memory::desc &md) {
    return layout_id_manager_.set_mem_desc(md);
}

graph::utils::optional_t<memory::desc> dnnl_backend::get_mem_desc(
        const size_t &layout_id) const {
    return layout_id_manager_.get_mem_desc(layout_id);
}

kernel_ptr large_partition_kernel_creator() {
    return std::make_shared<larger_partition_kernel_t>();
}

kernel_ptr dummy_kernel_creator() {
    return std::make_shared<dummy_kernel_t>();
}
} // namespace dnnl_impl

// This function should be called by backend_registry_t
void register_dnnl_backend() {
    backend_registry_t::get_singleton().register_backend(
            &dnnl_impl::dnnl_backend::get_singleton());
}

} // namespace graph
} // namespace impl
} // namespace dnnl
