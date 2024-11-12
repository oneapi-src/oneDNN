/*******************************************************************************
 * Copyright 2020-2024 Intel Corporation
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
#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"
#include "graph/backend/dnnl/dnnl_opset.hpp"
#include "graph/backend/dnnl/kernels/kernels.hpp"
#include "graph/backend/dnnl/patterns/data_type_check_pass.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

dnnl_backend_t::dnnl_backend_t(const std::string &name, float priority)
    : backend_t(name, priority) {
    register_op_schemas();
}

bool dnnl_backend_t::register_op_schemas() {
    register_dnnl_opset_schema();
    return true;
}

pass::pass_registry_t dnnl_backend_t::register_passes() {
#define DNNL_BACKEND_REGISTER_PATTERN_CALL(pattern_class_, pattern_registry_) \
    pattern::register_##pattern_class_(pattern_registry_);

    pass::pass_registry_t pass_registry;
    DNNL_BACKEND_REGISTER_PATTERN_CALL(binary_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(bn_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(concat_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(conv_block_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(conv_post_ops, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(convtranspose_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(matmul_post_ops, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(sdp, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(single_op_pass, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(pool_post_ops, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(eltwise_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(quantize_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(interpolate_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(softmax_post_ops, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(layernorm_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(sum_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(reorder_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(shuffle_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(reduction_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(groupnorm_fusion, pass_registry);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(mlp, pass_registry);

    const std::vector<data_type_t> dtypes_to_check
            = {dnnl_bf16, dnnl_f16, dnnl_f8_e4m3, dnnl_f8_e5m2};
    auto check_pass_ptr = std::make_shared<pattern::dtype_check_pass_t>(
            "dnnl_backend", "dtype_check_pass", dtypes_to_check);
    pass_registry.register_pass(check_pass_ptr);

    pass_registry.sort_passes();

#undef DNNL_BACKEND_REGISTER_PATTERN_CALL

    return pass_registry;
}

pass::pass_registry_t dnnl_backend_t::pass_registry_
        = dnnl_backend_t::register_passes();

size_t dnnl_backend_t::get_mem_size(const logical_tensor_t &lt) const {
    auto md = make_dnnl_memory_desc(lt);
    return md.get_size();
}

bool dnnl_backend_t::compare_logical_tensor(
        const logical_tensor_t &lhs, const logical_tensor_t &rhs) const {
    auto md1 = make_dnnl_memory_desc(lhs);
    auto md2 = make_dnnl_memory_desc(rhs);
    return md1 == md2;
}

graph::utils::optional_t<size_t> dnnl_backend_t::set_mem_desc(
        const memory::desc &md) {
    return layout_id_manager_.set_mem_desc(md);
}

graph::utils::optional_t<memory::desc> dnnl_backend_t::get_mem_desc(
        const size_t &layout_id) const {
    return layout_id_manager_.get_mem_desc(layout_id);
}

} // namespace dnnl_impl

// This function should be called by backend_registry_t
status_t register_dnnl_backend() {
    const status_t ret = backend_registry_t::get_singleton().register_backend(
            &dnnl_impl::dnnl_backend_t::get_singleton());
    return ret;
}

} // namespace graph
} // namespace impl
} // namespace dnnl
