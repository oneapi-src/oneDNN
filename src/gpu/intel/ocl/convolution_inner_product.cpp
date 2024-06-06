/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include "gpu/intel/ocl/convolution_inner_product.hpp"
#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/reorder.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"

using namespace dnnl::impl::memory_tracking;

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

static int adjust_dims(dims_t &dims, const memory_desc_t *dst, int ndims) {
    utils::array_copy(&dims[0], &dst->dims[0], dst->ndims);
    int max_dims = nstl::max(3, nstl::max(ndims, dst->ndims));
    utils::array_set(&dims[dst->ndims], 1, max_dims - dst->ndims);
    return max_dims;
}

status_t convolution_inner_product_fwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    const inner_product_desc_t &ipd = *desc();

    const auto *src_md = invariant_src_md();
    const auto *wei_md = invariant_wei_md();
    const auto *dst_md = invariant_dst_md();

    convolution_desc_t cd;
    memory_desc_t conv_src_md, conv_wei_md, conv_dst_md, ip_dst_md;

    conf.ndims = src_md->ndims;
    conf.attr_info = attr_info_t::create(attr());

    dims_t dims;

    int max_dims = adjust_dims(dims, dst_md, conf.ndims);

    CHECK(memory_desc_init_by_tag(
            conv_dst_md, max_dims, dims, dst_md->data_type, format_tag::any));

    auto init_md = [&](memory_desc_t &out_md, const memory_desc_t *in_md) {
        max_dims = adjust_dims(dims, in_md, conf.ndims);
        if (in_md->format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(
                    out_md, max_dims, dims, in_md->data_type, format_tag::any));
        } else {
            out_md = *in_md;
            out_md.ndims = max_dims;

            utils::array_copy(&out_md.dims[0], &dims[0], max_dims);
        }
        return status::success;
    };

    init_md(conv_src_md, src_md);
    init_md(conv_wei_md, wei_md);

    dim_t strides[] = {1, 1, 1};
    dim_t padding[] = {0, 0, 0};
    dim_t padding_r[] = {0, 0, 0};

    alg_kind_t alg = alg_kind::convolution_direct;
    CHECK(conv_desc_init(&cd, ipd.prop_kind, alg, &conv_src_md, &conv_wei_md,
            invariant_bia_md(), &conv_dst_md, &strides[0], nullptr, &padding[0],
            &padding_r[0]));

    primitive_attr_t conv_attr(*attr());
    if (!conv_attr.is_initialized()) return status::out_of_memory;

    primitive_desc_iterator_t it(engine, (op_desc_t *)&cd, &conv_attr, nullptr);
    if (!it.is_initialized()) return status::out_of_memory;
    cpd_ = *(++it);
    if (!cpd_) return status::unimplemented;
    std::string impl_name(cpd_->name());
    if (impl_name.find("ref") != std::string::npos)
        return status::unimplemented;

    auto src_conv = *cpd_->src_md();
    auto wei_conv = *cpd_->weights_md();
    auto dst_conv = *cpd_->dst_md();

    CHECK(memory_desc_init_by_tag(ip_dst_md, conv_dst_md.ndims,
            conv_dst_md.dims, dst_md->data_type,
            utils::pick(conv_dst_md.ndims - 2, format_tag::nc, format_tag::ncw,
                    format_tag::nchw, format_tag::ncdhw)));

    if (dst_conv != ip_dst_md
            && dst_conv.format_desc.blocking.inner_nblks > 0) {
        conf.reorder_dst = true;
        primitive_attr_t r_attr(default_attr());
        if (!r_attr.is_initialized()) return status::out_of_memory;
        CHECK(reorder_primitive_desc_create(
                rpd_dst_, engine, &dst_conv, &ip_dst_md, &r_attr));

        if (conf.attr_info.with_sum) {
            primitive_attr_t r_attr(default_attr());
            if (!r_attr.is_initialized()) return status::out_of_memory;
            CHECK(reorder_primitive_desc_create(
                    rpd_postop_, engine, &ip_dst_md, &dst_conv, &r_attr));
        }
    }

    if (src_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_blocking_desc(
                src_md_, src_conv.format_desc.blocking));
    }
    if (weights_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_blocking_desc(
                weights_md_, wei_conv.format_desc.blocking));
    }

    memory_desc_wrapper src_d(src_md_);
    memory_desc_wrapper dst_d(dst_md_);
    if (conv_src_md.format_desc.blocking.inner_nblks < 2
            && conv_wei_md.format_desc.blocking.inner_nblks < 2
            && src_d.size() + dst_d.size() >= 20000000)
        return status::unimplemented;

    return status::success;
}

status_t convolution_inner_product_fwd_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();
    if (conf.reorder_dst) {
        memory_desc_wrapper md_d(*cpd_->dst_md());
        scratchpad.book(memory_tracking::names::key_iprod_dst_reorder,
                md_d.size(), 1, OCL_BUFFER_ALIGNMENT);
        scratchpad.book(memory_tracking::names::key_nested_multiple + 1,
                rpd_dst_->scratchpad_registry());
        if (conf.attr_info.with_sum)
            scratchpad.book(memory_tracking::names::key_nested_multiple + 2,
                    rpd_postop_->scratchpad_registry());
    }
    scratchpad.book(memory_tracking::names::key_nested_multiple,
            cpd_->scratchpad_registry());
    return status::success;
}

status_t convolution_inner_product_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    const auto &conf = pd()->conf;

    auto src = ctx.input(DNNL_ARG_SRC);
    auto wei = ctx.input(DNNL_ARG_WEIGHTS);
    auto bia = ctx.input(DNNL_ARG_BIAS);
    auto dst = ctx.output(DNNL_ARG_DST);

    std::unique_ptr<memory_t> wspace_dst;
    auto exec_reorder = [&](memory_t *in, memory_t *out,
                                const std::shared_ptr<impl::primitive_t> &prim,
                                int r_num) -> status_t {
        exec_args_t r_args;
        r_args[DNNL_ARG_FROM] = memory_arg_t {in, true};
        r_args[DNNL_ARG_TO] = memory_arg_t {out, false};
        exec_ctx_t r_ctx(ctx, std::move(r_args));
        nested_scratchpad_t ns(
                ctx, memory_tracking::names::key_nested_multiple + r_num, prim);
        r_ctx.set_scratchpad_grantor(ns.grantor());
        return prim->execute(r_ctx);
    };
    if (conf.reorder_dst) {
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_iprod_dst_reorder);
        CHECK(safe_ptr_assign(wspace_dst,
                new memory_t(ctx.stream()->engine(), pd()->cpd_->dst_md(),
                        std::move(scratchpad))));
    }

    if (pd()->conf.attr_info.with_sum && conf.reorder_dst) {
        CHECK(exec_reorder(dst, wspace_dst.get(), postop_reorder_, 2));
    }

    exec_args_t c_args;
    c_args[DNNL_ARG_SRC] = memory_arg_t {src, true};
    c_args[DNNL_ARG_WEIGHTS] = memory_arg_t {wei, true};
    c_args[DNNL_ARG_BIAS] = memory_arg_t {bia, true};
    c_args[DNNL_ARG_DST]
            = memory_arg_t {conf.reorder_dst ? wspace_dst.get() : dst, false};

    const auto &args = ctx.args();
    for (int idx = 0; idx < pd()->attr()->post_ops_.len(); ++idx) {
        if (pd()->attr()->post_ops_.entry_[idx].is_binary()) {
            c_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1]
                    = args.at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                            | DNNL_ARG_SRC_1);
        }
    }

    exec_ctx_t c_ctx(ctx, std::move(c_args));
    nested_scratchpad_t ns(
            ctx, memory_tracking::names::key_nested_multiple, conv_);
    c_ctx.set_scratchpad_grantor(ns.grantor());
    CHECK(conv_->execute(c_ctx));

    if (conf.reorder_dst) {
        CHECK(exec_reorder(wspace_dst.get(), dst, dst_reorder_, 1));
    }

    return status::success;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
