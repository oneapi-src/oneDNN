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

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_ncsp_conv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {
format_tag_t get_ncsp_tag(int ndims) {
    using namespace format_tag;
    switch (ndims) {
        case 3: return ncw;
        case 4: return nchw;
        case 5: return ncdhw;
        default: assert("invalid ndims"); return undef;
    }
}
format_tag_t get_nspc_tag(int ndims) {
    using namespace format_tag;
    switch (ndims) {
        case 3: return nwc;
        case 4: return nhwc;
        case 5: return ndhwc;
        default: assert("invalid ndims"); return undef;
    }
}
} // namespace

status_t ncsp_convolution_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using namespace utils;

    // TODO: enable attributes (could be tricky for binary-like postops)
    const bool ok = is_fwd()
            && set_default_alg_kind(alg_kind::convolution_direct)
            && attr()->has_default_values() && !has_zero_dim_memory()
            && memory_desc_matches_tag(*src_md(), get_ncsp_tag(ndims()))
            && memory_desc_matches_tag(*dst_md(), get_ncsp_tag(ndims()));
    if (!ok) return status::unimplemented;

    // create a convolution descriptor with activations in nspc format
    convolution_desc_t nspc_conv_d = convolution_desc_t();
    format_tag_t nspc_tag = get_nspc_tag(ndims());
    nspc_src_md_ = *src_md();
    nspc_dst_md_ = *dst_md();
    CHECK(memory_desc_init_by_tag(nspc_src_md_, nspc_tag));
    CHECK(memory_desc_init_by_tag(nspc_dst_md_, nspc_tag));

    const convolution_desc_t *ncsp_conv_d = desc();
    CHECK(conv_desc_init(&nspc_conv_d, ncsp_conv_d->prop_kind,
            ncsp_conv_d->alg_kind, &nspc_src_md_, &ncsp_conv_d->weights_desc,
            &ncsp_conv_d->bias_desc, &nspc_dst_md_, ncsp_conv_d->strides,
            ncsp_conv_d->dilates, ncsp_conv_d->padding[0],
            ncsp_conv_d->padding[1]));

    primitive_desc_iterator_t it(engine,
            reinterpret_cast<const op_desc_t *>(&nspc_conv_d), attr(), nullptr);
    if (!it.is_initialized()) return status::out_of_memory;

    if (++it == it.end()) return status::unimplemented;
    nspc_conv_pd_ = *it;

    if (weights_md_.format_kind == format_kind::any)
        weights_md_ = *nspc_conv_pd_->weights_md(0);
    if (bias_md_.format_kind == format_kind::any)
        bias_md_ = *nspc_conv_pd_->weights_md(1);

    CHECK(reorder_primitive_desc_create(
            src_reorder_pd_, engine, src_md(), &nspc_src_md_));
    if (with_sum_)
        CHECK(reorder_primitive_desc_create(
                dst_pre_reorder_pd_, engine, dst_md(), &nspc_dst_md_));
    CHECK(reorder_primitive_desc_create(
            dst_post_reorder_pd_, engine, &nspc_dst_md_, dst_md()));

    init_name();
    init_scratchpad();
    return status::success;
}

void ncsp_convolution_fwd_t::pd_t::init_scratchpad() {
    using namespace memory_tracking::names;
    auto scratchpad = scratchpad_registry().registrar();
    const memory_desc_wrapper dst_mdw(dst_md());
    const memory_desc_wrapper src_mdw(src_md());
    scratchpad.book(
            key_conv_ncsp_dst, dst_mdw.nelems(), sizeof(dst_mdw.data_type()));
    scratchpad.book(
            key_conv_ncsp_src, src_mdw.nelems(), sizeof(src_mdw.data_type()));
    scratchpad.book(key_nested, nspc_conv_pd_->scratchpad_registry());
    scratchpad.book(key_nested, src_reorder_pd_->scratchpad_registry());
    if (with_sum_)
        scratchpad.book(key_nested, dst_pre_reorder_pd_->scratchpad_registry());
    scratchpad.book(key_nested, dst_post_reorder_pd_->scratchpad_registry());
}

status_t ncsp_convolution_fwd_t::init(engine_t *engine) {
    if (pd()->nspc_conv_pd_)
        CHECK(pd()->nspc_conv_pd_->create_primitive(nspc_conv_p_, engine));
    if (pd()->src_reorder_pd_)
        CHECK(pd()->src_reorder_pd_->create_primitive(src_reorder_p_, engine));
    if (pd()->dst_pre_reorder_pd_)
        CHECK(pd()->dst_pre_reorder_pd_->create_primitive(
                dst_pre_reorder_p_, engine));
    if (pd()->dst_post_reorder_pd_)
        CHECK(pd()->dst_post_reorder_pd_->create_primitive(
                dst_post_reorder_p_, engine));
    return status::success;
}

status_t ncsp_convolution_fwd_t::reorder_activations(const exec_ctx_t &ctx,
        const std::shared_ptr<primitive_t> prim, engine_t *engine,
        const memory_arg_t &in, const memory_arg_t &out) const {
    using namespace memory_tracking::names;
    exec_args_t r_args;
    r_args[DNNL_ARG_SRC] = in;
    r_args[DNNL_ARG_DST] = out;
    exec_ctx_t r_ctx(ctx, std::move(r_args));

    nested_scratchpad_t ns(ctx, key_nested, prim);
    r_ctx.set_scratchpad_grantor(ns.grantor());
    CHECK(prim->execute(r_ctx));

    return status::success;
}

status_t ncsp_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {

    using namespace memory_tracking::names;
    engine_t *engine = ctx.stream()->engine();
    auto scratchpad = ctx.get_scratchpad_grantor();

    // initialize nspc src memory
    auto nspc_src_mem = scratchpad.get_memory_storage(key_conv_ncsp_src);
    memory_t nspc_src(engine, &(pd()->nspc_src_md_), std::move(nspc_src_mem));

    // initialize nspc dst memory
    auto nspc_dst_mem = scratchpad.get_memory_storage(key_conv_ncsp_dst);
    memory_t nspc_dst(engine, &(pd()->nspc_dst_md_), std::move(nspc_dst_mem));

    // reorder src from ncsp to nspc
    CHECK(reorder_activations(ctx, src_reorder_p_, engine,
            ctx.args().at(DNNL_ARG_SRC), {&nspc_src, false}));

    // maybe reorder dst from ncsp to nspc
    if (pd()->dst_pre_reorder_pd_)
        CHECK(reorder_activations(ctx, dst_pre_reorder_p_, engine,
                ctx.args().at(DNNL_ARG_DST), {&nspc_dst, false}));

    // execute nspc convolution
    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_DST] = {&nspc_dst, false};
    conv_args[DNNL_ARG_SRC] = {&nspc_src, true};
    conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
    if (pd()->with_bias()) conv_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);

    exec_ctx_t nspc_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(
            ctx, memory_tracking::names::key_nested, nspc_conv_p_);
    nspc_ctx.set_scratchpad_grantor(ns.grantor());
    CHECK(nspc_conv_p_->execute(nspc_ctx));

    // reorder dst from nspc to ncsp
    CHECK(reorder_activations(ctx, dst_post_reorder_p_, engine,
            {&nspc_dst, false}, ctx.args().at(DNNL_ARG_DST)));

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
