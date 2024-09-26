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

#include "gpu/generic/sycl/ref_sum_many_inputs.hpp"
#include "common/primitive_desc_iface.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_sum_many_inputs_t::pd_t::init_conf() {
    conf_ = sycl_sum_conf_t();
    conf_.n = n_inputs();

    return status::success;
}

status_t ref_sum_many_inputs_t::init(engine_t *engine) {
    const size_t n = pd()->base_pds_.size();
    base_prims_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        CHECK(pd()->base_pds_[i]->impl()->create_primitive(
                base_prims_[i], engine, cache_blob()));
    }

    return status::success;
}

status_t ref_sum_many_inputs_t::execute(const exec_ctx_t &ctx) const {
    memory_arg_t dst_mem_arg = {ctx.args().at(DNNL_ARG_DST).mem, false};
    memory_arg_t dst_read_mem_arg = {ctx.args().at(DNNL_ARG_DST).mem, true};

    int n_remaining = pd()->conf_.n;
    int in_arg_offset = 0;
    int i = 0;

    while (n_remaining > 0) {
        bool pass_in_dst = i > 0;
        int max_n_child_inputs = DNNL_REF_SUM_MAX_NUM_TENSORS - pass_in_dst;
        int args_handled = std::min(n_remaining, max_n_child_inputs);
        exec_args_t r_args;
        r_args[DNNL_ARG_DST] = dst_mem_arg;
        if (pass_in_dst) {
            r_args[DNNL_ARG_MULTIPLE_SRC + 0] = dst_read_mem_arg;
        }
        for (int j = 0; j < args_handled; j++) {
            r_args[DNNL_ARG_MULTIPLE_SRC + j + pass_in_dst]
                    = ctx.args().at(DNNL_ARG_MULTIPLE_SRC + j + in_arg_offset);
        }

        exec_ctx_t r_ctx(ctx, std::move(r_args));
        CHECK(base_prims_[i]->execute(r_ctx));

        n_remaining -= args_handled;
        in_arg_offset += args_handled;
        i++;
    }
    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
