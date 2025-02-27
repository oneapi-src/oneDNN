/*******************************************************************************
* Copyright 2025 Intel Corporation
* Copyright 2025 Codeplay Software Limited
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

#include "gpu/generic/sycl/ref_group_normalization.hpp"

namespace dnnl::impl::gpu::generic::sycl {
status_t ref_group_normalization_fwd_t::pd_t::init(impl::engine_t *engine) {
    using namespace data_type;
    VDISPATCH_GNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_GNORM(is_fwd(), VERBOSE_BAD_PROPKIND);

    auto src_mdw = memory_desc_wrapper(arg_md(DNNL_ARG_SRC));
    auto dst_mdw = memory_desc_wrapper(arg_md(DNNL_ARG_DST));
    auto src_dt = src_mdw.data_type();
    auto dst_dt = dst_mdw.data_type();

    VDISPATCH_GNORM(utils::one_of(src_dt, f32, bf16, f16, s8, u8),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_GNORM(utils::one_of(dst_dt, f32, bf16, f16, s8, u8),
            VERBOSE_UNSUPPORTED_DT);

    VDISPATCH_GNORM(attr_.set_default_formats(dst_md()) == status::success,
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_GNORM(sycl_post_ops_t::post_ops_ok(attr()),
            "sycl post op initialization returns false");

    const primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::scales_runtime
            | primitive_attr_t::skip_mask_t::post_ops;
    VDISPATCH_GNORM(
            attr()->has_default_values(attr_mask), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_GNORM(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_GNORM(check_scale_shift_data_type(),
            "scale / shift data type must be data_type::f32");
    VDISPATCH_GNORM(
            sycl_post_ops_t::post_ops_ok(attr()), VERBOSE_UNSUPPORTED_POSTOP);

    const auto &dims = src_mdw.dims();
    const auto num_groups = desc()->groups;
    VDISPATCH_GNORM(dims[1] % num_groups == 0,
            "number of groups must divide the channels evenly");

    auto batch_size = static_cast<std::size_t>(dims[0]);
    auto group_range = static_cast<std::size_t>(desc()->groups);

    auto device = utils::downcast<const impl::xpu::sycl::engine_impl_t *>(
            engine->impl())
                          ->device();

    // To avoid using excess registers error
    auto local_range = std::max(std::size_t(64),
            device.get_info<::sycl::info::device::max_work_group_size>() / 8);
    launch_range = ::sycl::nd_range<2>(
            {batch_size * local_range, group_range}, {local_range, 1});

    conf_ = sycl_group_norm_conf_t();
    conf_.src_desc = xpu::sycl::md_t(arg_md(DNNL_ARG_SRC));
    conf_.dst_desc = xpu::sycl::md_t(arg_md(DNNL_ARG_DST));
    conf_.use_global_stats = stats_is_src();
    conf_.num_groups = static_cast<int32_t>(group_range);
    conf_.num_channels_per_group = static_cast<int32_t>(dims[1] / group_range);
    conf_.use_scale = use_scale();
    conf_.use_shift = use_shift();
    conf_.src_scaling = !attr()->scales_.has_default_values(DNNL_ARG_SRC);
    conf_.dst_scaling = !attr()->scales_.has_default_values(DNNL_ARG_DST);
    conf_.eta = desc()->group_norm_epsilon;
    conf_.post_ops = {attr(), dst_mdw};
    return status::success;
}

status_t ref_group_normalization_fwd_t::init(impl::engine_t *engine) {
    auto kid = ::sycl::get_kernel_id<group_norm_fwd_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_group_normalization_fwd_t::execute(const exec_ctx_t &ctx) const {
    exec_args_t cloned_args(ctx.args());
    // Circumventing the const pointers when mean and value
    // use_global_stats is set, to avoid creating 2 kernels just
    // for this
    cloned_args[DNNL_ARG_MEAN]
            = memory_arg_t {cloned_args[DNNL_ARG_MEAN].mem, false};
    cloned_args[DNNL_ARG_VARIANCE]
            = memory_arg_t {cloned_args[DNNL_ARG_VARIANCE].mem, false};

    exec_ctx_t exec_ctx(ctx.stream(), std::move(cloned_args));
    auto &conf_ = pd()->conf_;
    auto launch_range = pd()->launch_range;

    parallel_for(exec_ctx, kernel_, [&](::sycl::handler &cgh) {
        ::sycl::local_accessor<float, 1> local_memory(
                launch_range.get_local_range()[0] + 2, cgh);
        cgh.parallel_for(launch_range,
                group_norm_fwd_t(conf_, local_memory, cgh, exec_ctx));
    });
    return status::success;
}

} // namespace dnnl::impl::gpu::generic::sycl
