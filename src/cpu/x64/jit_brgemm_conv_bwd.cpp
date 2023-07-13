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
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_brgemm_conv_bwd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {
status_t weights_axes_permutation(
        memory_desc_t *o_md, const memory_desc_t *i_md, bool with_groups) {
    int perm[DNNL_MAX_NDIMS] {}; // bwd conv to fwd conv weight permutation
    for (int d = 0; d < DNNL_MAX_NDIMS; ++d)
        perm[d] = d;
    nstl::swap(perm[0 + with_groups], perm[1 + with_groups]);

    return memory_desc_permute_axes(*o_md, *i_md, perm);
}

status_t fwd_conv_desc_create(
        convolution_desc_t *fwd_conv_d, const convolution_desc_t *bwd_conv_d) {
    // create a new weights descriptor with OC and IC transposed;
    // spatial inversion is handled by inverting indices on-the-fly
    memory_desc_t fwd_weights_md;
    const memory_desc_t &bwd_weights_md = bwd_conv_d->weights_desc;
    const bool with_groups
            = bwd_weights_md.ndims == bwd_conv_d->diff_src_desc.ndims + 1;
    CHECK(weights_axes_permutation(
            &fwd_weights_md, &bwd_weights_md, with_groups));

    // create a fwd convolution descriptor with padding adjusted
    // to the perspective of backward propagation, namely:
    // - left padding replaced by left overflow
    // - right padding replaced by right overflow
    const int ndims_spatial = bwd_conv_d->diff_src_desc.ndims - 2;
    dims_t overflow_l;
    dims_t overflow_r;
    dim_t ks = 1;
    for (int i = 0; i < ndims_spatial; i++) {
        // only unit strides are allowed for bwd-to-fwd conversion
        if (bwd_conv_d->strides[i] != 1) return status::unimplemented;
        const dim_t K
                = bwd_weights_md.dims[bwd_weights_md.ndims - ndims_spatial + i];
        ks *= K;
        const dim_t D = bwd_conv_d->dilates[i];
        const dim_t PL = bwd_conv_d->padding[0][i]; // left padding
        const dim_t PR = bwd_conv_d->padding[1][i]; // right padding
        constexpr dim_t S = 1;
        // the following relations hold for unit stride only
        overflow_l[i] = ((K - 1) * (D + 1) - PL) / S;
        overflow_r[i] = ((K - 1) * (D + 1) - PR) / S;
    }

    CHECK(conv_desc_init(fwd_conv_d, prop_kind::forward_training,
            alg_kind::convolution_direct, &bwd_conv_d->diff_dst_desc,
            &fwd_weights_md, &bwd_conv_d->bias_desc, &bwd_conv_d->diff_src_desc,
            bwd_conv_d->strides, bwd_conv_d->dilates, overflow_l, overflow_r));

    // HACK: Set diff_src_desc and diff_dst_desc as a signal to the primitive
    //       descriptor cache that we are using the bwd-via-fwd version of
    //       fwd conv and thus need a separate cache entry. Only needed for
    //       non-1x1 convs due to spatial inversion of weights. This assumes
    //       that external users only use the API to create conv descs, and
    //       relies on common/convolution.cpp only setting the expected mem descs.
    // TODO: Pass this information via attributes or integrate the bwd-via-fwd
    //       method directly into fwd conv implementations.
    const bool with_spatial_inversion = ks > 1;
    if (with_spatial_inversion) {
        fwd_conv_d->diff_src_desc = fwd_conv_d->src_desc;
        fwd_conv_d->diff_dst_desc = fwd_conv_d->dst_desc;
    }
    return status::success;
}
} // namespace

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_t<isa>::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using namespace utils;

    const bool ok = is_bwd_d()
            && set_default_alg_kind(alg_kind::convolution_direct)
            && attr()->has_default_values() && !has_zero_dim_memory();
    if (!ok) return status::unimplemented;

    convolution_desc_t fwd_conv_d = convolution_desc_t();
    CHECK(fwd_conv_desc_create(&fwd_conv_d, desc()));

    primitive_desc_iterator_t it(engine,
            reinterpret_cast<const op_desc_t *>(&fwd_conv_d), attr(), nullptr);
    if (!it.is_initialized()) return status::out_of_memory;

    while (++it != it.end()) {
        fwd_pd_ = *it;
        using fwd_1x1_conv_pd_t =
                typename brgemm_1x1_convolution_fwd_t<isa>::pd_t;
        const auto pd_1x1 = dynamic_cast<fwd_1x1_conv_pd_t *>((*it).get());
        if (pd_1x1 != nullptr) break; // 1x1 implementation found

        constexpr bool use_inversion = true; // invert weights' spatial indices
        using fwd_conv_pd_t =
                typename brgemm_convolution_fwd_t<isa, use_inversion>::pd_t;
        const auto pd = dynamic_cast<fwd_conv_pd_t *>((*it).get());
        if (pd != nullptr) break; // non-1x1 implementation found
    }
    if (it == it.end()) return status::unimplemented;

    if (weights_md_.format_kind == format_kind::any)
        CHECK(weights_axes_permutation(
                &weights_md_, fwd_pd_->weights_md(), with_groups()));
    if (diff_src_md_.format_kind == format_kind::any)
        diff_src_md_ = *fwd_pd_->dst_md();
    if (diff_dst_md_.format_kind == format_kind::any)
        diff_dst_md_ = *fwd_pd_->src_md();
    if (bias_md_.format_kind == format_kind::any)
        bias_md_ = *fwd_pd_->weights_md(1);

    init_name();
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(
            memory_tracking::names::key_nested, fwd_pd_->scratchpad_registry());

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_t<isa>::init(engine_t *engine) {
    return pd()->fwd_pd_->create_primitive(fwd_p_, engine);
}

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_DST] = args.at(DNNL_ARG_DIFF_SRC);
    conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
    conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
    if (pd()->with_bias()) conv_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);

    exec_ctx_t fwd_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(ctx, memory_tracking::names::key_nested, fwd_p_);
    fwd_ctx.set_scratchpad_grantor(ns.grantor());
    return fwd_p_->execute(fwd_ctx);
}

template struct brgemm_convolution_bwd_t<avx2>;
template struct brgemm_convolution_bwd_t<avx2_vnni_2>;
template struct brgemm_convolution_bwd_t<avx512_core>;
template struct brgemm_convolution_bwd_t<avx512_core_bf16>;
template struct brgemm_convolution_bwd_t<avx512_core_fp16>;
template struct brgemm_convolution_bwd_t<avx512_core_amx>;
template struct brgemm_convolution_bwd_t<avx512_core_amx_fp16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
