/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "gpu/intel/jit/pooling/gen_pooling.hpp"

#include <iostream>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/post_ops.hpp"
#include "gpu/intel/jit/ir/tensor_config.hpp"
#include "gpu/intel/jit/pooling/pooling_kernel.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "ngen_register_allocator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

status_t gen_pooling_fwd_t::pd_t::init(impl::engine_t *engine) {
    using namespace data_type;
    using namespace prop_kind;
    using namespace alg_kind;
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto arch = compute_engine->device_info()->gpu_arch();
    auto src_data_t = src_md()->data_type;
    auto dst_data_t = dst_md()->data_type;
    auto acc_data_t = desc()->accum_data_type;

    // TODO: add training(?), add bwd
    VDISPATCH_POOLING_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_POOLING(utils::one_of(desc()->prop_kind,
                              /*forward_training,*/ forward_inference),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_POOLING(
            utils::one_of(desc()->alg_kind, pooling_max,
                    pooling_avg_include_padding, pooling_avg_exclude_padding),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_POOLING(
            (utils::everyone_is(f32, src_data_t, dst_data_t, acc_data_t)
                    || utils::everyone_is(f16, src_data_t, dst_data_t)
                    || utils::everyone_is(bf16, src_data_t, dst_data_t)
                    || utils::everyone_is(u8, src_data_t, dst_data_t)
                    || utils::everyone_is(s8, src_data_t, dst_data_t)),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_POOLING(IMPLICATION(utils::one_of(src_data_t, f16, s8, u8),
                              desc()->prop_kind == forward_inference),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_POOLING_SC(
            attr_.set_default_formats(dst_md(0)), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
            "does not support dilations");
    VDISPATCH_POOLING(!utils::one_of(f64, src_data_t, dst_data_t),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_POOLING(
            compute_engine->mayiuse(compute::device_ext_t::intel_subgroups),
            VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");
    VDISPATCH_POOLING(
            IMPLICATION(src_data_t == f16,
                    compute_engine->mayiuse(compute::device_ext_t::khr_fp16)
                            && compute_engine->mayiuse(compute::device_ext_t::
                                            intel_subgroups_short)),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_POOLING(IMPLICATION(src_data_t == bf16,
                              arch >= compute::gpu_arch_t::xe_hpc),
            VERBOSE_UNSUPPORTED_DT_CFG);

    src = std::make_shared<layout_t>(invariant_src_md());
    dst = std::make_shared<layout_t>(invariant_dst_md());
    VDISPATCH_POOLING(src->ndims() == dst->ndims(), VERBOSE_INCONSISTENT_NDIMS,
            "src->ndims()", "dst_ndims()");

    pool_conf = std::make_shared<pool_conf_t>();
    set_default_pool_conf(*pool_conf, *desc(), *invariant_src_md(),
            *invariant_dst_md(), *attr());

    auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    hw_t hw(engine);
    exec_cfg = std::make_shared<exec_config_t>(hw);
    exec_cfg->set_regs(hw.prefer_large_grf(gpu_attr) ? 256 : 128);
    exec_cfg->set_simd(16);

    VDISPATCH_POOLING(pooling_config_t::check_compatibility(*pool_conf,
                              *exec_cfg, *src, attr()->post_ops_, dst->type()),
            "incompatible pooling configuration");
    return status::success;
}

status_t gen_pooling_fwd_t::init(impl::engine_t *engine) {
    cfg_ = pooling_config_t(
            *pd()->exec_cfg, *pd()->pool_conf, *pd()->src, *pd()->dst);
    zero_points_config_t zp_cfg(pd());
    cfg_.set_zp_cfg(zp_cfg);
    cfg_.compute_grid();

    if (auto blob = cache_blob()) {
        int32_t version;
        CHECK(blob.get_value((uint8_t *)&version, sizeof(version)));
        while (version--)
            cfg_.cut();
    }

    tensor_config_t tensor_cfg;
    tensor_cfg.add_tensor("src", DNNL_ARG_SRC, true, false,
            cfg_.src_layout().user(), cfg_.src_layout().user());
    tensor_cfg.add_tensor("dst", DNNL_ARG_DST, true, true,
            cfg_.dst_layout().user(), cfg_.dst_layout().user());

    init_extra_tensors(cfg_.zp_cfg(), *pd()->attr(), nullptr, *pd()->dst_md(),
            /* ic = */ 1, /* oc = */ 1, tensor_cfg);

    kernel_info_ = kernel_info_t();
    kernel_info_.set_nd_range(cfg_.nd_range());

    // Initialize kernel arguments.
    for (auto &t : tensor_cfg.tensors()) {
        gpu_assert(!t.needs_reorder);
        gpu_assert(!t.needs_zero_out);

        if (t.arg_key == DNNL_ARG_UNDEF) {
            gpu_error_not_expected();
            continue;
        }
        kernel_info_.register_user_arg(make_buffer(t.name), t.arg_key,
                /*is_input=*/t.is_input && !t.is_output);
    }

    while (!kernel_) {
        try {
            kernel_ = make_kernel<pooling_kernel_t>(
                    this, engine, cfg_, "gen_pooling_fwd", kernel_info_, *pd());
            break;
        } catch (const ngen::out_of_registers_exception &exc) {
            UNUSED(exc);
            gpu_warning() << "loop too large: cut and retry!";
            kernel_ = {};
            if (!cfg_.cut()) {
                gpu_error_not_expected() << "minimal loop too large!";
                break;
            }
        } catch (const std::exception &exc) {
            gpu_error_not_expected() << exc.what();
            kernel_ = {};
            break;
        }
    }
    set_version(cfg_.n_cuts());
    return (kernel_) ? status::success : status::runtime_error;
}

status_t gen_pooling_fwd_t::execute(const exec_ctx_t &ctx) const {
    std::vector<memory_storage_wrapper_t> storage_list;
    kernel_info_.init_memory_storage_list(storage_list, ctx, this);

    compute::kernel_arg_list_t arg_list;
    kernel_info_.set_args(arg_list, storage_list);

    return parallel_for(ctx, cfg_.nd_range(), kernel_, arg_list);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
