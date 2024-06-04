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

#include "gpu/sycl/ref_concat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;

status_t ref_concat_t::pd_t::init_conf() {
    conf_ = sycl_concat_conf_t();
    conf_.n = n_inputs();

    return status::success;
}

status_t ref_concat_t::init(engine_t *engine) {
    const size_t n = pd()->reorder_pds_.size();
    reorders_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        CHECK(pd()->reorder_pds_[i]->create_primitive(
                reorders_[i], engine, cache_blob()));
    }

    return status::success;
}

status_t ref_concat_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t r_args;
    std::unique_ptr<memory_t> p_reorder_mem_data;
    for (auto i = 0; i < pd()->conf_.n; ++i) {
        memory_arg_t dst_mem_arg = {ctx.args().at(DNNL_ARG_DST).mem, false};
        r_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_MULTIPLE_SRC + i);
        r_args[DNNL_ARG_DST] = dst_mem_arg;
        exec_ctx_t r_ctx(ctx, std::move(r_args));
        CHECK(reorders_[i]->execute(r_ctx));
    }
    return status::success;
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
