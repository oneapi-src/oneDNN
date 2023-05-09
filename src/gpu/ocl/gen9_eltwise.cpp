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

#include <algorithm>

#include "gpu/ocl/gen9_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gen9_eltwise_jit_params_t::init(engine_t *engine,
        const memory_desc_wrapper data_d, alg_kind_t alg_kind_) {
    *this = {};
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    arch = compute_engine->device_info()->gpu_arch();
    data_type = data_d.data_type();
    alg_kind = alg_kind_;

    bool is_pre_xe_hp = arch < compute::gpu_arch_t::xe_hp;

    // Important hw features for code generation
    const int dt_size = (int)data_d.data_type_size();
    const int max_load_size = is_pre_xe_hp ? 128 : 256;

    // Heuristics chosen by experimentation
    // load_unroll hides computation overhead associated with kernel start
    // local_threads hides workgroup scheduling overhead
    const int load_unroll = is_pre_xe_hp ? 4 : 1;
    const int local_threads = is_pre_xe_hp ? 1 : 16;

    // Prefer loading multiple of max load size to reduce messages
    const int load_size = load_unroll * max_load_size;

    // Set simd size
    sub_group_size = compute_engine->device_info()->max_subgroup_size(
            data_d.data_type());

    // VECT_DATA_T only supports vector sizes up to 8
    vector_size = std::min(load_size / (dt_size * sub_group_size), 8);
    work_group_size = local_threads * sub_group_size;

    const int local_block_size = work_group_size * vector_size;
    with_overflow = (data_d.nelems(true) % local_block_size) != 0;

    return status::success;
}

compute::kernel_ctx_t gen9_eltwise_jit_params_t::get_kernel_ctx() const {
    compute::kernel_ctx_t kernel_ctx;

    kernel_ctx.set_data_type(data_type);
    def_eltwise_alg_kinds(kernel_ctx);

    kernel_ctx.define_int("WITH_ELTWISE", 1);
    kernel_ctx.define_int("ELTWISE_ALG", alg_kind);

    kernel_ctx.define_int("VECT_DT_N", vector_size);

    kernel_ctx.define_int("NELEMS_OVERFLOW", with_overflow);

    // attribute for wg-size and subgroup-size
    kernel_ctx.define_int("GWS_WITH_SG_DEFAULT", 1);
    // wg-size
    kernel_ctx.define_int("GWS_LWS0_DEFAULT", work_group_size);
    kernel_ctx.define_int("GWS_LWS1_DEFAULT", 1);
    kernel_ctx.define_int("GWS_LWS2_DEFAULT", 1);
    // subgroup-size
    kernel_ctx.define_int("GWS_SGS_DEFAULT", sub_group_size);

    return kernel_ctx;
}

status_t gen9_eltwise_fwd_t::pd_t::init_conf(engine_t *engine) {
    const memory_desc_wrapper data_d(use_dst() ? dst_md() : src_md());
    status_t status = conf.init(engine, data_d, this->desc()->alg_kind);
    return status;
}

status_t gen9_eltwise_fwd_t::execute_forward_dense(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;
    const memory_desc_wrapper data_d(
            pd()->use_dst() ? pd()->dst_md() : pd()->src_md());
    const dim_t nelems = data_d.nelems(true);
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.set(2, nelems);
    arg_list.set(3, alpha);
    arg_list.set(4, beta);

    dim_t lws = conf.work_group_size;
    dim_t total_wi = utils::div_up(nelems, conf.vector_size);
    compute::nd_range_t nd_range({utils::rnd_up(total_wi, lws)}, {lws});

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    if (!gpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                pd()->desc()->alg_kind, alpha, beta)) {
        CHECK(ctx.zero_pad_output(DNNL_ARG_DST));
    }

    return status;
}

status_t gen9_eltwise_bwd_t::pd_t::init_conf(engine_t *engine) {
    using namespace dnnl::impl::format_tag;

    const memory_desc_wrapper data_d(data_md());
    const memory_desc_wrapper diff_data_d(diff_src_md());

    // This kernel supports only matching data and diff formats
    if (data_d != diff_data_d) return status::unimplemented;

    return conf.init(engine, data_d, this->desc()->alg_kind);
}

status_t gen9_eltwise_bwd_t::execute_backward_dense(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;

    auto &src = pd()->use_dst() ? CTX_IN_STORAGE(DNNL_ARG_DST)
                                : CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    const auto &conf = pd()->conf;
    const memory_desc_wrapper data_d(pd()->data_md());
    const dim_t nelems = data_d.nelems(true);
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_src);
    arg_list.set(2, diff_dst);
    arg_list.set(3, nelems);
    arg_list.set(4, alpha);
    arg_list.set(5, beta);

    dim_t lws = conf.work_group_size;
    dim_t total_wi = utils::div_up(nelems, conf.vector_size);
    compute::nd_range_t nd_range({utils::rnd_up(total_wi, lws)}, {lws});

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    if (!gpu_eltwise_bwd_pd_t::eltwise_preserves_zero(
                pd()->desc()->alg_kind, alpha, beta)) {
        CHECK(ctx.zero_pad_output(DNNL_ARG_DIFF_SRC));
    }

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
