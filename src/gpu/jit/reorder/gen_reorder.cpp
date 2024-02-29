/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#include "gpu/jit/reorder/gen_reorder.hpp"

#include <iostream>
#include <utility>

#include "common/c_types_map.hpp"
#include "common/impl_registration.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/tensor_config.hpp"
#include "gpu/jit/reorder/config.hpp"
#include "gpu/jit/reorder/reorder_kernel.hpp"
#include "gpu/jit/utils/utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t gen_reorder_t::pd_t::init(
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
    const auto src_dt = src_md()->data_type;
    const auto dst_dt = dst_md()->data_type;
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto *device_info = compute_engine->device_info();
    zero_points_config_t zp_cfg(this);

    auto post_ops_ok = [&]() {
        const auto &po = attr()->post_ops_;
        memory_desc_t dst_md_type(*dst_md());
        for (int i = 0; i < po.len(); i++)
            if (po.entry_[i].is_binary()) {
                dst_md_type.data_type = po.entry_[i].binary.src1_desc.data_type;
                if (!dnnl_memory_desc_equal(
                            &dst_md_type, &po.entry_[i].binary.src1_desc))
                    return false;
            }
        return true;
    };
    auto scales_ok = [&]() {
        return (attr()->scales_.get(DNNL_ARG_SRC).mask_ == 0)
                && (attr()->scales_.get(DNNL_ARG_DST).mask_ == 0);
    };
    auto zps_ok = [&]() {
        return (!zp_cfg.do_src_compensation || zp_cfg.is_common_src_zero_point)
                && (!zp_cfg.do_dst_compensation
                        || zp_cfg.is_common_dst_zero_point);
    };
    auto is_bf16_or_f32_or_bf8 = [](data_type_t dt) {
        return utils::one_of(
                dt, data_type::bf16, data_type::f32, data_type::f8_e5m2);
    };
    bool any_hf8 = utils::one_of(data_type::f8_e4m3, dst_dt, src_dt);
    auto skip_mask = dnnl_primitive_attr::skip_mask_t::post_ops
            | dnnl_primitive_attr::skip_mask_t::zero_points_runtime
            | dnnl_primitive_attr::skip_mask_t::scales_runtime;
    using namespace data_type;
    VDISPATCH_REORDER(
            src_engine == dst_engine && src_engine->kind() == engine_kind::gpu,
            VERBOSE_BAD_ENGINE_KIND);
    VDISPATCH_REORDER(
            utils::one_of(src_dt, f32, f16, bf16, f8_e5m2, s32, s8, u8, f64),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(
            utils::one_of(dst_dt, f32, f16, bf16, f8_e5m2, s32, s8, u8, f64),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(
            IMPLICATION(src_dt == data_type::f16 || dst_dt == data_type::f16,
                    device_info->has_native(data_type::f16)),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_REORDER(IMPLICATION(src_dt == data_type::bf16,
                              is_bf16_or_f32_or_bf8(dst_dt)),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_REORDER(IMPLICATION(dst_dt == data_type::bf16,
                              is_bf16_or_f32_or_bf8(src_dt)),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_REORDER(
            IMPLICATION(utils::one_of(data_type::f8_e5m2, src_dt, dst_dt),
                    device_info->has_native(data_type::f8_e5m2)),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_REORDER(
            IMPLICATION(src_dt == data_type::f64 || dst_dt == data_type::f64,
                    device_info->has_native(data_type::f64)),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_REORDER(
            attr()->has_default_values(skip_mask), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_REORDER(extra_ok(), VERBOSE_UNSUPPORTED_MD_FLAG, "extra_ok");
    VDISPATCH_REORDER(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_REORDER(scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_REORDER(zps_ok(), VERBOSE_UNSUPPORTED_ZP_CFG);
    VDISPATCH_REORDER(!any_hf8, VERBOSE_UNSUPPORTED_DT);

    memory_desc_wrapper src_mdw {src_md()};
    memory_desc_wrapper dst_mdw {dst_md()};
    VDISPATCH_REORDER(!src_mdw.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_REORDER(src_mdw.ndims() == dst_mdw.ndims(),
            VERBOSE_INCONSISTENT_MDS, "src", "dst");
    int ndims = src_mdw.ndims();

    layout_t src_layout {src_mdw, /*do_normalize=*/false};
    layout_t dst_layout {dst_mdw, /*do_normalize=*/false};

    VDISPATCH_REORDER(!(src_layout.elems() == 0 || dst_layout.elems() == 0),
            VERBOSE_EMPTY_TENSOR, "src_layout || dst_layout");

    std::vector<dim_t> dims(ndims);
    for (int i = 0; i < ndims; ++i)
        dims[i] = std::max(src_layout.dim(i), dst_layout.dim(i));

    auto check_layout = [&](const layout_t &l) {
        for (auto &eb : l.enumerated_blocks()) {
            auto &b = eb.second;
            if (l.is_outermost(eb)) {
                dim_t inner = l.dim(b.dim_idx) / b.block;
                if (dims[b.dim_idx] % inner) return false;
            }
        }
        return true;
    };

    VDISPATCH_REORDER(
            check_layout(src_layout), "unsupported src tensor layout");
    VDISPATCH_REORDER(
            check_layout(dst_layout), "unsupported dst tensor layout");
    VDISPATCH_REORDER(compute_engine->mayiuse_ngen_kernels(),
            VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "ngen_kernels");
    auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    hw_t hw(engine);
    exec_config_t exec_cfg(hw);
    exec_cfg.set_regs(hw.prefer_large_grf(gpu_attr) ? 256 : 128);
    exec_cfg.set_simd(16);
    cfg = std::make_shared<reorder_config_t>(exec_cfg, src_layout, dst_layout);
    cfg->set_zp_cfg(zp_cfg);
    VDISPATCH_REORDER_SC(
            init_kernel_info(), "kernel initialization unsuccessful");

    return status::success;
}

status_t gen_reorder_t::pd_t::init_kernel_info() {
    tensor_config_t tensor_cfg;
    tensor_cfg.add_tensor("src", DNNL_ARG_SRC, true, false,
            cfg->src_layout().user(), cfg->src_layout().user());
    tensor_cfg.add_tensor("dst", DNNL_ARG_DST, true, true,
            cfg->dst_layout().user(), cfg->dst_layout().user());

    // TODO: set input/output channels here to enable per-channel ZPs/scales
    init_extra_tensors(
            cfg->zp_cfg(), *attr(), *dst_md(), /*ic=*/1, /*oc=*/1, tensor_cfg);

    kernel_info = std::make_shared<kernel_info_t>();
    auto nd_range = reorder_kernel_t<>::nd_range(cfg->exec_cfg(),
            cfg->src_layout().user(), cfg->dst_layout().user());
    auto global_range = nd_range.global_range();
    constexpr int max = std::numeric_limits<int>::max();
    // This case *probably* overflowed in int32 precision.
    // Skip until we have a proper fix.
    if (global_range[0] > max || global_range[1] > max || global_range[2] > max)
        return status::unimplemented;
    kernel_info->set_nd_range(nd_range);

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

status_t gen_reorder_t::init(engine_t *engine) {
    auto &cfg = *pd()->cfg;
    auto &info = *pd()->kernel_info;

    kernel_ = make_kernel<reorder_kernel_t>(this, engine, cfg, "gen_reorder",
            info, /*require_dpas=*/false, grf_mode_t::any, pd());
    if (!kernel_) return status::runtime_error;
    return status::success;
}

status_t gen_reorder_t::execute(const exec_ctx_t &ctx) const {
    auto &info = *pd()->kernel_info;

    std::vector<memory_storage_wrapper_t> storage_list;
    info.init_memory_storage_list(storage_list, ctx, this);

    compute::kernel_arg_list_t arg_list;
    info.set_args(arg_list, storage_list);

    CHECK(parallel_for(ctx, info.nd_range(), kernel_, arg_list));
    return status::success;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
