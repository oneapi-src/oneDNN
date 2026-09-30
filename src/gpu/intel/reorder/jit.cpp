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

#include "gpu/intel/reorder/jit.hpp"

#include <utility>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/post_ops.hpp"
#include "gpu/intel/jit/ir/tensor_config.hpp"
#include "gpu/intel/reorder/jit/config.hpp"
#include "gpu/intel/reorder/jit/kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {

using namespace jit;

status_t gen_t::pd_t::init(const impl::engine_t *engine,
        const impl::engine_t *src_engine, const impl::engine_t *dst_engine) {
    const auto src_dt = src_md()->data_type;
    const auto dst_dt = dst_md()->data_type;
    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);
    auto *device_info = intel_engine->device_info();
    using namespace data_type;

    auto post_ops_ok = [&]() {
        const auto &po = attr()->post_ops_;
        memory_desc_t dst_md_type(*dst_md());
        for (int i = 0; i < po.len(); i++)
            if (po.entry_[i].is_binary()) {
                dst_md_type.data_type = po.entry_[i].binary.src1_desc.data_type;
                if (!dnnl_memory_desc_equal(
                            &dst_md_type, &po.entry_[i].binary.src1_desc))
                    return false;
                if (po.entry_[i].is_binary_with_ternary_op()) return false;
            }
        return true;
    };
    auto scales_ok = [&]() {
        const bool src_scale_ok
                = attr()->scales_.has_default_values(DNNL_ARG_SRC)
                || attr()->scales_.get_mask(DNNL_ARG_SRC) == 0;
        const bool dst_scale_ok
                = attr()->scales_.has_default_values(DNNL_ARG_DST)
                || attr()->scales_.get_mask(DNNL_ARG_DST) == 0;
        return src_scale_ok && dst_scale_ok;
    };
    VDISPATCH_REORDER(
            src_engine == dst_engine && src_engine->kind() == engine_kind::gpu,
            VERBOSE_BAD_ENGINE_KIND);
    VDISPATCH_REORDER(utils::one_of(src_dt, f32, f16, bf16, f8_e5m2, f8_e4m3,
                              f4_e2m1, s32, s8, u8, s4, u4, u2, f64),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(utils::one_of(dst_dt, f32, f16, bf16, f8_e5m2, f8_e4m3,
                              f4_e2m1, s32, s8, u8, s4, u4, u2, f64),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(IMPLICATION(src_dt == f64 || dst_dt == f64,
                              device_info->has_native(f64)),
            VERBOSE_UNSUPPORTED_DT_CFG);

    using sm = dnnl_primitive_attr::skip_mask_t;
    auto skip_mask = sm::post_ops | sm::zero_points | sm::scales
            | sm::rounding_mode | sm::gpu_attr;
    VDISPATCH_REORDER(
            attr()->has_default_values(skip_mask), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_REORDER(extra_ok(true), VERBOSE_UNSUPPORTED_MD_FLAG, "extra_ok");
    VDISPATCH_REORDER(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_REORDER(scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);

    // Create `zp_cfg` object after all relevant zp checks passing to avoid
    // potential issues with the generator for unsupported features.
    zero_points_config_t zp_cfg(this);
    auto zps_ok = [&]() {
        return (!zp_cfg.do_src_compensation || zp_cfg.is_common_src_zero_point)
                && (!zp_cfg.do_dst_compensation
                        || zp_cfg.is_common_dst_zero_point);
    };
    VDISPATCH_REORDER(zps_ok(), VERBOSE_UNSUPPORTED_ZP_CFG);

    memory_desc_wrapper src_mdw {src_md()};
    memory_desc_wrapper dst_mdw {dst_md()};
    VDISPATCH_REORDER(!src_mdw.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_REORDER(src_mdw.ndims() == dst_mdw.ndims(),
            VERBOSE_INCONSISTENT_MDS, "src", "dst");
    int ndims = src_mdw.ndims();

    layout_t src_layout = make_layout(*src_md());
    layout_t dst_layout = make_layout(*dst_md());

    VDISPATCH_REORDER(!(src_layout.elems() == 0 || dst_layout.elems() == 0),
            VERBOSE_EMPTY_TENSOR, "src_layout || dst_layout");

    std::vector<dim_t> dims(ndims);
    for (int i = 0; i < ndims; ++i)
        dims[i] = std::max(src_layout.elems(i), dst_layout.elems(i));

    auto check_layout = [&](const layout_t &l) {
        for (auto &b : l.blocks()) {
            if (l.is_outermost(b)) {
                dim_t inner = l.elems(b.idx) / b.size;
                if (dims[b.idx] % inner) return false;
            }
        }
        return true;
    };

    VDISPATCH_REORDER(
            check_layout(src_layout), VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    VDISPATCH_REORDER(
            check_layout(dst_layout), VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");
    VDISPATCH_REORDER(intel_engine->mayiuse_ngen_kernels(),
            VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "ngen_kernels");

    auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    dsl::hw_t hw(make_ir_hw(engine));
    dsl::kernel::options_t options(hw);
    options.set_regs(prefer_large_grf(hw, gpu_attr) ? 256 : 128);
    options.set_require_dpas(prefer_dpas(hw, gpu_attr));
    options.set_simd(16);
    cfg = std::make_shared<config_t>(options, src_layout, dst_layout);
    cfg->set_zp_cfg(zp_cfg);

    auto byte_aligned = [&](const layout_t &layout) {
        auto dims = cfg->tiles().front().values();
        dim_t contiguous_inner_elems = 1;
        dim_t innermost_elems = 1;
        const auto packing = layout.type().packing();
        if (packing == 1) return true;
        for (auto &b : layout.blocks()) {
            if (b.size == 1) continue;
            if ((dim_t)b.stride != contiguous_inner_elems) break;
            if (innermost_elems == 1) innermost_elems = b.size;
            if (b.size > dims[b.idx]) {
                if (b.size % dims[b.idx] == 0)
                    contiguous_inner_elems *= dims[b.idx];
                break;
            }
            contiguous_inner_elems *= b.size;
            dims[b.idx] /= b.size;
        }
        return innermost_elems % packing == 0
                || contiguous_inner_elems % (4 * packing) == 0;
    };

    VDISPATCH_REORDER(byte_aligned(cfg->src_layout().user()),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    VDISPATCH_REORDER(byte_aligned(cfg->dst_layout().user()),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");
    VDISPATCH_REORDER_SC(
            init_kernel_info(), "kernel initialization unsuccessful");
    VDISPATCH_REORDER_SC(maybe_create_zp_precompute_conv_pd(dst_engine),
            "failed to create nested zp precompute convolution");
    return status::success;
}

status_t gen_t::pd_t::init_kernel_info() {
    tensor_config_t tensor_cfg;
    tensor_cfg.add_tensor("src", DNNL_ARG_SRC, true, false,
            cfg->src_layout().user(), cfg->src_layout().user());
    tensor_cfg.add_tensor("dst", DNNL_ARG_DST, true, true,
            cfg->dst_layout().user(), cfg->dst_layout().user());

    // TODO: set input/output channels here to enable per-channel ZPs/scales
    init_extra_tensors(cfg->zp_cfg(), *attr(), nullptr, *dst_md(), /*ic=*/1,
            /*oc=*/1, tensor_cfg);

    kernel_info = std::make_shared<kernel_info_t>();
    auto nd_range = cfg->nd_range();
    auto global_range = nd_range.global_range();
    constexpr int max = std::numeric_limits<int>::max();
    // This case *probably* overflowed in int32 precision.
    // Skip until we have a proper fix.
    if (global_range[0] > max || global_range[1] > max || global_range[2] > max)
        return status::unimplemented;
    kernel_info->set_nd_range(nd_range);

    // Initialize kernel arguments.
    for (auto &t : tensor_cfg.tensors()) {
        gpu_assert(!t.needs_reorder);
        gpu_assert(!t.needs_zero_out);

        if (t.arg_key == DNNL_ARG_UNDEF) {
            gpu_error_not_expected();
            continue;
        }
        kernel_info->register_user_arg(make_buffer(t.name), t.arg_key,
                /*is_input=*/t.is_input && !t.is_output);
    }
    return status::success;
}

status_t gen_t::init(impl::engine_t *engine) {
    CHECK(pd()->maybe_create_zp_precompute_conv(
            zp_precomp_conv_, engine, this));

    auto &cfg = *pd()->cfg;
    auto &info = *pd()->kernel_info;

    kernel_ = make_kernel<kernel_t>(
            this, engine, cfg, "gen_reorder", info, pd());
    if (!kernel_) return status::runtime_error;
    return status::success;
}

status_t gen_t::execute(const exec_ctx_t &ctx) const {
    auto &info = *pd()->kernel_info;

    std::vector<memory_storage_wrapper_t> storage_list;
    info.init_memory_storage_list(storage_list, ctx, this);

    compute::kernel_arg_list_t arg_list;
    info.set_args(arg_list, storage_list);

    CHECK(parallel_for(ctx, info.nd_range(), kernel_, arg_list));
    CHECK(pd()->maybe_exec_zp_precompute_conv(ctx, zp_precomp_conv_));
    return status::success;
}

} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
