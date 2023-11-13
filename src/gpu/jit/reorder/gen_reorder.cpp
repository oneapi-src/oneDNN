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
    auto is_bf16_or_f32 = [](data_type_t dt) {
        return utils::one_of(dt, data_type::bf16, data_type::f32);
    };
    bool has_native_bf16 = device_info->has_native(data_type::bf16);
    auto skip_mask = dnnl_primitive_attr::skip_mask_t::post_ops
            | dnnl_primitive_attr::skip_mask_t::zero_points_runtime
            | dnnl_primitive_attr::skip_mask_t::scales_runtime;
    bool ok = src_engine == dst_engine && src_engine->kind() == engine_kind::gpu
            && IMPLICATION(src_dt == data_type::f16 || dst_dt == data_type::f16,
                    device_info->has_native(data_type::f16))
            && IMPLICATION(src_dt == data_type::bf16,
                    has_native_bf16 && is_bf16_or_f32(dst_dt))
            && IMPLICATION(dst_dt == data_type::bf16,
                    has_native_bf16 && is_bf16_or_f32(src_dt))
            && IMPLICATION(src_dt == data_type::f64 || dst_dt == data_type::f64,
                    device_info->has_native(data_type::f64))
            && attr()->has_default_values(skip_mask) && extra_ok()
            && post_ops_ok() && scales_ok() && zps_ok();
    if (!ok) return status::unimplemented;

    memory_desc_wrapper src_mdw {src_md()};
    memory_desc_wrapper dst_mdw {dst_md()};
    if (src_mdw.has_runtime_dims_or_strides()) return status::unimplemented;
    if (src_mdw.ndims() != dst_mdw.ndims()) return status::unimplemented;
    int ndims = src_mdw.ndims();

    layout_t src_layout {src_mdw, /*do_normalize=*/false};
    layout_t dst_layout {dst_mdw, /*do_normalize=*/false};

    if (src_layout.elems() == 0 || dst_layout.elems() == 0)
        return status::unimplemented;

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

    if (!check_layout(src_layout)) return status::unimplemented;
    if (!check_layout(dst_layout)) return status::unimplemented;
    if (!compute_engine->mayiuse_ngen_kernels()) return status::unimplemented;
    auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    bool large_grf_mode = gpu_attr && gpu_attr->threads_per_eu() == 4;
    exec_config_t exec_cfg(hw_config_t(engine, large_grf_mode));
    exec_cfg.set_regs(128);
    exec_cfg.set_simd(16);
    cfg = std::make_shared<reorder_config_t>(exec_cfg, src_layout, dst_layout);
    cfg->set_zp_cfg(zp_cfg);
    CHECK(init_kernel_info());

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
    kernel_info->set_nd_range(reorder_kernel_t<>::nd_range(cfg->exec_cfg(),
            cfg->src_layout().user(), cfg->dst_layout().user()));

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
