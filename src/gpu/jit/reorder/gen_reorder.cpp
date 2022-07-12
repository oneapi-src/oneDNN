/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "common/impl_registration.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
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
    bool ok = src_engine == dst_engine && src_engine->kind() == engine_kind::gpu
            && IMPLICATION(src_dt == data_type::f16 || dst_dt == data_type::f16,
                    compute_engine->mayiuse(compute::device_ext_t::khr_fp16))
            && IMPLICATION(src_dt == data_type::bf16,
                    utils::one_of(dst_dt, data_type::bf16, data_type::f32))
            && IMPLICATION(dst_dt == data_type::bf16,
                    utils::one_of(src_dt, data_type::bf16, data_type::f32))
            && IMPLICATION(src_dt == data_type::f64 || dst_dt == data_type::f64,
                    compute_engine->mayiuse(compute::device_ext_t::khr_fp64))
            && attr()->has_default_values() && extra_ok();
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
    cfg = std::make_shared<reorder_config_t>(engine, src_md(), dst_md());
    cfg->exec_cfg.set_regs(128);
    cfg->exec_cfg.set_simd(16);
    CHECK(init_kernel_info());

    return status::success;
}

status_t gen_reorder_t::pd_t::init_kernel_info() {
    auto &info = kernel_info;
    auto elems = cfg->dst_layout.elems();

    info = std::make_shared<kernel_info_t>();
    auto src_buf = make_buffer("src_user");
    auto dst_buf = make_buffer("dst_user");
    info->register_user_arg(src_buf, DNNL_ARG_SRC, /*is_input=*/true);
    info->register_user_arg(dst_buf, DNNL_ARG_DST, /*is_input=*/false);
    auto elems_var = var_t::make(type_t::u32(), "elems");
    info->register_internal_arg(elems_var, uint32_t(elems));
    info->set_nd_range(reorder_kernel_t<>::nd_range(
            cfg->exec_cfg, cfg->src_layout, cfg->dst_layout));

    return status::success;
}

status_t gen_reorder_t::init(engine_t *engine) {
    auto &cfg = *pd()->cfg;
    auto &info = *pd()->kernel_info;

    kernel_ = make_kernel<reorder_kernel_t>(this, engine, cfg.exec_cfg,
            "gen_reorder", info, cfg.src_layout, cfg.dst_layout, false,
            grf_mode_t::any);
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
