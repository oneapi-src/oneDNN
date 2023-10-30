/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/jit/pooling/gen_pooling.hpp"

#include <iostream>
#include <utility>

#include "common/c_types_map.hpp"
#include "common/impl_registration.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/tensor_config.hpp"
#include "gpu/jit/pooling/config.hpp"
#include "gpu/jit/pooling/pooling_kernel.hpp"
#include "gpu/jit/utils/utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t gen_pooling_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using namespace prop_kind;
    using namespace alg_kind;
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto arch = compute_engine->device_info()->gpu_arch();
    auto src_data_t = src_md()->data_type;
    auto dst_data_t = dst_md()->data_type;
    auto acc_data_t = desc()->accum_data_type;

    // TODO: add training(?), add bwd
    bool ok = set_default_params() == status::success
            && utils::one_of(
                    desc()->prop_kind, /*forward_training,*/ forward_inference)
            && utils::one_of(desc()->alg_kind, pooling_max,
                    pooling_avg_include_padding, pooling_avg_exclude_padding)
            && (utils::everyone_is(f32, src_data_t, dst_data_t, acc_data_t)
                    || utils::everyone_is(f16, src_data_t, dst_data_t)
                    || utils::everyone_is(bf16, src_data_t, dst_data_t)
                    || utils::everyone_is(u8, src_data_t, dst_data_t)
                    || utils::everyone_is(s8, src_data_t, dst_data_t))
            && IMPLICATION(utils::one_of(src_data_t, f16, s8, u8),
                    desc()->prop_kind == forward_inference)
            && attr_.set_default_formats(dst_md(0)) == status::success
            && !is_dilated() && !utils::one_of(f64, src_data_t, dst_data_t)
            && compute_engine->mayiuse(compute::device_ext_t::intel_subgroups)
            && IMPLICATION(src_data_t == f16,
                    compute_engine->mayiuse(compute::device_ext_t::khr_fp16)
                            && compute_engine->mayiuse(compute::device_ext_t::
                                            intel_subgroups_short))
            && IMPLICATION(src_data_t == bf16, // easier to refuse BF16 on XeHPG
                    arch != compute::gpu_arch_t::xe_hpg);
    if (!ok) return status::unimplemented;

    bool is_training = desc()->prop_kind == forward_training;
    if (desc()->alg_kind == pooling_max && is_training) init_default_ws(s32);

    const memory_desc_wrapper src_mdw(invariant_src_md());
    const memory_desc_wrapper dst_mdw(invariant_dst_md());
    if (src_mdw.ndims() != dst_mdw.ndims()) return status::unimplemented;
    if (src_mdw.has_runtime_dims_or_strides()) return status::unimplemented;

    pool_conf_t pool_conf;
    set_default_pool_conf(pool_conf, *desc(), *invariant_src_md(),
            *invariant_dst_md(), *attr());

    auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    bool large_grf_mode = gpu_attr && gpu_attr->threads_per_eu() == 4;
    exec_config_t exec_cfg(hw_config_t(engine, large_grf_mode));
    exec_cfg.set_regs(128);
    exec_cfg.set_simd(16);

    cfg = std::make_shared<pooling_config_t>(
            exec_cfg, pool_conf, src_mdw, dst_mdw);
    zero_points_config_t zp_cfg(this);
    cfg->set_zp_cfg(zp_cfg);

    if (!cfg->try_compute_grid()) return status::unimplemented;
    return init_kernel_info();
}

status_t gen_pooling_fwd_t::pd_t::init_kernel_info() {
    tensor_config_t tensor_cfg;
    tensor_cfg.add_tensor("src", DNNL_ARG_SRC, true, false,
            cfg->src_layout().user(), cfg->src_layout().user());
    tensor_cfg.add_tensor("dst", DNNL_ARG_DST, true, true,
            cfg->dst_layout().user(), cfg->dst_layout().user());

    init_extra_tensors(
            cfg->zp_cfg(), *attr(), *dst_md(), /*ic=*/1, /*oc=*/1, tensor_cfg);

    kernel_info = std::make_shared<kernel_info_t>();
    kernel_info->set_nd_range(pooling_kernel_t<>::nd_range(*cfg));

    // Initialize kernel arguments.
    for (auto &t : tensor_cfg.tensors()) {
        ir_assert(!t.needs_reorder);
        ir_assert(!t.needs_zero_out);

        int user_arg_key = t.arg_key;
        auto user_buf = make_buffer(t.name);

        if (user_arg_key == -1) {
            ir_assert(!t.needs_reorder);
            ir_assert(!t.needs_zero_out);
            ir_error_not_expected();
            continue;
        }

        kernel_info->register_user_arg(user_buf, user_arg_key,
                /*is_input=*/t.is_input && !t.is_output);
    }
    return status::success;
}

status_t gen_pooling_fwd_t::init(engine_t *engine) {
    kernel_ = make_kernel<pooling_kernel_t>(this, engine, *pd()->cfg,
            "gen_pooling_fwd", *pd()->kernel_info, grf_mode_t::any, *pd());
    return (kernel_) ? status::success : status::runtime_error;
}

status_t gen_pooling_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto &info = *pd()->kernel_info;

    std::vector<memory_storage_wrapper_t> storage_list;
    info.init_memory_storage_list(storage_list, ctx, this);

    compute::kernel_arg_list_t arg_list;
    info.set_args(arg_list, storage_list);

    return parallel_for(ctx, info.nd_range(), kernel_, arg_list);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
