/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include "gpu/intel/ocl/gemm/gemm_with_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t gemm_with_post_ops_t::pd_t::init(impl::engine_t *engine) {
    using namespace data_type;

    const auto &d = desc();
    const auto attr_skip_mask = primitive_attr_t::skip_mask_t::scales_runtime
            | primitive_attr_t::skip_mask_t::scales_runtime_data_type
            | primitive_attr_t::skip_mask_t::post_ops
            | primitive_attr_t::skip_mask_t::fpmath_mode
            | primitive_attr_t::skip_mask_t::zero_points_runtime
            | primitive_attr_t::skip_mask_t::zero_points_runtime_data_type;

    bool wei_decomp = (utils::one_of(d->c_type(), f32, f16, bf16)
                              && utils::one_of(d->a_type(), u8, s8, u4, s4)
                              && utils::one_of(d->b_type(), f16, f32, bf16))
            && attr()->mayiconvert(d->a_type(), f32);
    VDISPATCH_GEMM(
            d->c_desc.ndims <= 4, VERBOSE_UNSUPPORTED_MD_FLAG, "c_desc.ndims");
    VDISPATCH_GEMM(!utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m(), d->n(), d->k()),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_GEMM(attr()->has_default_values(attr_skip_mask),
            VERBOSE_UNSUPPORTED_ATTR);

    const primitive_attr_t *attributes_with_po = attr();
    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
        if (attr()->scales_.has_default_values(arg)) continue;

        const auto &mask = attr()->scales_.get_mask(arg);
        if (arg == DNNL_ARG_WEIGHTS && !wei_decomp)
            VDISPATCH_GEMM((mask == 0 || mask == (1 << (dst_md()->ndims - 1))),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        else
            VDISPATCH_GEMM((mask == 0), VERBOSE_UNSUPPORTED_SCALES_CFG);
    }
    attr_info_ = attr_info_t::create(attributes_with_po);

    VDISPATCH_GEMM(d->sum_ab == sum_ab::sum_none, VERBOSE_UNSUPPORTED_FEATURE,
            "bias reduction");

    const auto impl_list = engine->get_implementation_list(op_desc());
    int current_impl_idx
            = impl_list_item_t::find<ocl::gemm_with_post_ops_t::pd_t>(
                    impl_list);

    primitive_desc_iterator_t it_gemm_with_po(engine, op_desc(),
            attributes_with_po, nullptr,
            current_impl_idx /* skip implementation */);
    if (!it_gemm_with_po.is_initialized()) return status::invalid_arguments;
    gemm_pd_ = *(++it_gemm_with_po);
    // exit if gemm kernel support post ops
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto arch = compute_engine->device_info()->gpu_arch();
    bool is_xe_hp = arch >= compute::gpu_arch_t::xe_hp;
    auto skip_impl = is_xe_hp ? "ocl" : "ref";
    VDISPATCH_GEMM(
            !(gemm_pd_ && strstr(gemm_pd_->name(), skip_impl) == nullptr),
            VERBOSE_PRIMITIVE_CREATION_FAIL, gemm_pd_->name());
    auto gemm_desc = *desc();
    auto dst_type = gemm_desc.c_desc.data_type;
    gemm_desc.c_desc.data_type = engine->mayiuse_f16_accumulator_with_f16()
                    && utils::one_of(data_type::f16, gemm_desc.a_desc.data_type,
                            gemm_desc.b_desc.data_type)
            ? data_type::f32
            : gemm_desc.acc_type;
    use_reorder = dst_md(0)->data_type != gemm_desc.c_desc.data_type;
    gemm_desc.bias_desc = glob_zero_md;
    // Setup empty attributes but keep zero points for gemm.
    primitive_attr_t attributes_without_po = *attr();
    attributes_without_po.set_post_ops(post_ops_t());
    attributes_without_po.scales_ = scales_t();
    attributes_without_po.zero_points_ = zero_points_t();
    const auto &zp = attributes_with_po->zero_points_;
    int src_mask = zp.get_mask(DNNL_ARG_SRC);
    int wei_mask = zp.get_mask(DNNL_ARG_WEIGHTS);
    if (!zp.has_default_values(DNNL_ARG_SRC)) {
        attributes_without_po.zero_points_.set(DNNL_ARG_SRC, src_mask);
    }
    if (!zp.has_default_values(DNNL_ARG_WEIGHTS)) {
        const auto dt = attr()->zero_points_.get_data_type(DNNL_ARG_WEIGHTS);
        attributes_without_po.zero_points_.set(
                DNNL_ARG_WEIGHTS, wei_mask, dt, 0, {});
    }

    primitive_desc_iterator_t it_gemm_without_po(engine,
            reinterpret_cast<const op_desc_t *>(&gemm_desc),
            &attributes_without_po, nullptr,
            current_impl_idx /* skip implementation */);
    if (!it_gemm_without_po.is_initialized()) return status::invalid_arguments;
    gemm_pd_ = *(++it_gemm_without_po);
    VDISPATCH_GEMM(
            !(!gemm_pd_ || strstr(gemm_pd_->name(), skip_impl) != nullptr),
            VERBOSE_PRIMITIVE_CREATION_FAIL, gemm_pd_ ? gemm_pd_->name() : "");

    //set tags for end user
    desc_.a_desc = *gemm_pd_->arg_md(DNNL_ARG_SRC_0);
    desc_.b_desc = *gemm_pd_->arg_md(DNNL_ARG_SRC_1);
    desc_.c_desc = *gemm_pd_->arg_md(DNNL_ARG_DST);
    desc_.c_desc.data_type = dst_type;
    desc_.acc_type = gemm_desc.c_desc.data_type;
    CHECK(attr_.set_default_formats(dst_md(0)));
    VDISPATCH_GEMM(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    compute::kernel_ctx_t kernel_ctx;
    use_scratchpad_with_post_op_worker = use_reorder
            || attributes_with_po->post_ops_.find(primitive_kind_t::dnnl_sum)
                    != -1;
    auto ndims = gemm_pd_->dst_md()->ndims;
    dispatch_ = compute_engine->create_dispatch(gemm_pd_->dst_md());
    dispatch_.define_dim("D0", 0, gemm_pd_->dst_md()->padded_dims[0]);
    dispatch_.define_dim("D1", 1, gemm_pd_->dst_md()->padded_dims[1]);
    dispatch_.define_dim("D3", ndims > 3 ? 3 : 0,
            ndims > 3 ? gemm_pd_->dst_md()->padded_dims[3] : 1);
    dispatch_.define_dim("D2", ndims > 2 ? 2 : 0,
            ndims > 2 ? gemm_pd_->dst_md()->padded_dims[2] : 1);
    dispatch_.generate();

    init_scratchpad();

    return status::success;
}
status_t gemm_with_post_ops_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    auto c_type = dst_md(0)->data_type;
    const auto src_info = memory_desc_info_t::create(gemm_pd_->dst_md(0));
    const auto bias_info = [&]() {
        // If no bias, just default to same layout as dst - any valid layout will work, it's just a dummy
        auto info = memory_desc_info_t::create(
                with_bias() ? src_md(2) : dst_md(0));
        if (info.data_type == data_type::undef) info.data_type = data_type::f32;
        return info;
    }();

    def_memory_desc_info(kernel_ctx, src_info, "SRC", false);
    def_memory_desc_info(kernel_ctx, bias_info, "BIAS", false);
    def_memory_desc_info(
            kernel_ctx, memory_desc_info_t::create(dst_md(0)), "DST", false);

    int ndims = src_info.ndims;
    kernel_ctx.set_data_type(c_type);

    const auto &attr_scales = attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const bool with_dst_scales = !attr_scales.has_default_values(DNNL_ARG_DST);
    auto is_int_type = [](data_type_t t) {
        return utils::one_of(t, data_type::s8, data_type::u8, data_type::s32);
    };
    data_type_t acc_type = desc_.acc_type;
    if (desc_.acc_type == data_type::s32) {
        if (with_src_scales || with_wei_scales
                || !is_int_type(bias_info.data_type)
                || !is_int_type(dst_md(0)->data_type)) {
            acc_type = data_type::f32;
        }
    }
    def_data_type(kernel_ctx, acc_type, "ACC");

    kernel_ctx.define_int("NDIMS", ndims);
    CHECK(def_attr_info(kernel_ctx, attr_info_, attr()->post_ops_,
            *gemm_pd_->dst_md(), false));
    kernel_ctx.define_int("A_SCALES", with_src_scales);
    kernel_ctx.define_int("B_SCALES", with_wei_scales);
    kernel_ctx.define_int("C_SCALES", with_dst_scales);
    kernel_ctx.define_int("DST_ZERO_POINT",
            !attr()->zero_points_.has_default_values(DNNL_ARG_DST));
    def_dispatch(kernel_ctx, dispatch_);
    return status::success;
}

void gemm_with_post_ops_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();
    if (use_scratchpad_with_post_op_worker) {
        memory_desc_wrapper dst_mdw(dst_md());
        scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer,
                dst_mdw.size(), types::data_type_size(desc_.acc_type));
    }
    scratchpad.book(memory_tracking::names::key_nested_multiple,
            gemm_pd_->scratchpad_registry());
}

status_t gemm_with_post_ops_t::execute(const gemm_exec_ctx_t &ctx) const {
    std::unique_ptr<memory_t, memory_deleter_t> c_mem_before_po_worker;
    status_t exec_status;
    gemm_exec_args_t g_args(ctx.args());

    if (pd()->use_scratchpad()) {
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_tmp_buffer);
        auto tmp_md = *(pd()->dst_md(0));
        tmp_md.data_type = pd()->desc()->acc_type;
        CHECK(safe_ptr_assign(c_mem_before_po_worker,
                new memory_t(ctx.stream()->engine(), &tmp_md,
                        std::move(scratchpad))));

        g_args.c = c_mem_before_po_worker->memory_storage();
    }
    exec_ctx_t tmp_exec_ctx(ctx.stream());
    tmp_exec_ctx.set_resource_mapper(ctx.get_resource_mapper());
    tmp_exec_ctx.set_scratchpad_grantor(&ctx.get_scratchpad_grantor());
    nested_scratchpad_t g_ns(tmp_exec_ctx,
            memory_tracking::names::key_nested_multiple, gemm_prim_);

    gemm_exec_ctx_t gemm_ex_ctx
            = gemm_exec_ctx_t(ctx.stream(), g_args, ctx.desc());
    gemm_ex_ctx.set_resource_mapper(ctx.get_resource_mapper());
    gemm_ex_ctx.set_scratchpad_grantor(g_ns.grantor());

    exec_status = gpu_gemm(gemm_prim_)->execute(gemm_ex_ctx);
    CHECK(exec_status);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(
            ctx.stream()->engine());
    auto arch = compute_engine->device_info()->gpu_arch();
    // Workaround correctness issue on Gen9
    if (arch == compute::gpu_arch_t::gen9) ctx.stream()->wait();
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0,
            pd()->use_scratchpad() ? *c_mem_before_po_worker->memory_storage()
                                   : GEMM_CTX_ARG_STORAGE(c));
    arg_list.set(1, GEMM_CTX_ARG_STORAGE(bias));
    arg_list.set(2, GEMM_CTX_ARG_STORAGE(c));
    const auto &args = ctx.args();
    int idx = append_post_ops_to_arg_list_gemm(
            args.exec_args, arg_list, 3, pd()->attr()->post_ops_);
    //a/b tensors are swapped for gemm
    arg_list.set(idx++, GEMM_CTX_ARG_STORAGE(b_scales));
    arg_list.set(idx++, GEMM_CTX_ARG_STORAGE(a_scales));
    arg_list.set(idx++, GEMM_CTX_ARG_STORAGE(c_scales));
    arg_list.set(idx++,
            pd()->attr()->scales_.get_mask(DNNL_ARG_WEIGHTS) > 0 ? 1 : 0);
    arg_list.set(idx, GEMM_CTX_ARG_STORAGE(c_zero_point));
    auto nd_range = pd()->dispatch_.nd_range();
    exec_status = parallel_for(ctx, nd_range, post_process_kernel_, arg_list);
    return exec_status;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
