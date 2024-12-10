/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2024 Codeplay Software Limited
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

#include "gpu/generic/sycl/ref_inner_product.hpp"
#include "common/primitive_desc_iterator.hpp"

namespace dnnl::impl::gpu::generic::sycl {

status_t ref_inner_product_fwd_t::pd_t::init_matmul(impl::engine_t *engine) {
    matmul_desc_t matmul_desc;
    CHECK(matmul_desc_init(&matmul_desc, &src_md_reshaped, &weights_md_reshaped,
            &bias_md_reshaped, arg_md(DNNL_ARG_DST)));
    primitive_attr_t matmul_attr(*attr());

    primitive_desc_iterator_t it(engine,
            reinterpret_cast<op_desc_t *>(&matmul_desc), &matmul_attr, nullptr);
    if (!it.is_initialized()) return status::invalid_arguments;
    while (++it != it.end()) {
        matmul_pd = *it;
        if (matmul_pd) { break; }
    }
    if (!matmul_pd) { return status::invalid_arguments; }
    return status::success;
}

status_t ref_inner_product_fwd_t::init(impl::engine_t *engine) {
    std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
    CHECK(pd()->matmul_pd->create_primitive_nested(p, engine));
    matmul_primitive = p.first;
    return status::success;
}

status_t ref_inner_product_fwd_t::execute(const exec_ctx_t &ctx) const {
    nested_scratchpad_t nested_scratchpad(
            ctx, memory_tracking::names::key_nested, matmul_primitive);
    exec_ctx_t copied_ctx(ctx);
    copied_ctx.set_scratchpad_grantor(nested_scratchpad.grantor());
    return matmul_primitive->execute(copied_ctx);
}

} // namespace dnnl::impl::gpu::generic::sycl
