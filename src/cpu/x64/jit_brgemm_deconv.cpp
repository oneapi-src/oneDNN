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

#include "cpu/x64/jit_brgemm_deconv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {
status_t weights_axes_permutation(
        memory_desc_t *o_md, const memory_desc_t *i_md, bool with_groups) {
    int perm[DNNL_MAX_NDIMS] {}; // deconv to conv weight permutation
    for (int d = 0; d < DNNL_MAX_NDIMS; ++d)
        perm[d] = d;
    nstl::swap(perm[0 + with_groups], perm[1 + with_groups]);

    return memory_desc_permute_axes(*o_md, *i_md, perm);
}

status_t fwd_conv_desc_create(const deconvolution_desc_t *fwd_deconv_d,
        convolution_desc_t *fwd_conv_d) {
    const memory_desc_t &fwd_weights_md = fwd_deconv_d->weights_desc;
    // create a fwd convolution descriptor with padding adjusted
    // to the perspective of backward propagation, namely:
    // - left padding replaced by left overflow
    // - right padding replaced by right overflow
    const int ndims_spatial = fwd_deconv_d->dst_desc.ndims - 2;
    dims_t overflow_l;
    dims_t overflow_r;
    dim_t ks = 1;
    for (int i = 0; i < ndims_spatial; i++) {
        // only unit strides are allowed for bwd-to-fwd conversion
        if (fwd_deconv_d->strides[i] != 1) return status::unimplemented;
        const dim_t K
                = fwd_weights_md.dims[fwd_weights_md.ndims - ndims_spatial + i];
        ks *= K;
        const dim_t D = fwd_deconv_d->dilates[i];
        const dim_t PL = fwd_deconv_d->padding[0][i]; // left padding
        const dim_t PR = fwd_deconv_d->padding[1][i]; // right padding
        constexpr dim_t S = 1;
        // the following relations hold for unit stride only
        overflow_l[i] = ((K - 1) * (D + 1) - PL) / S;
        overflow_r[i] = ((K - 1) * (D + 1) - PR) / S;
    }

    CHECK(conv_desc_init(fwd_conv_d, prop_kind::forward_training,
            alg_kind::convolution_direct, &fwd_deconv_d->src_desc,
            &fwd_weights_md, &fwd_deconv_d->bias_desc, &fwd_deconv_d->dst_desc,
            fwd_deconv_d->strides, fwd_deconv_d->dilates, overflow_l,
            overflow_r));

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

status_t bwd_conv_desc_create(const deconvolution_desc_t *fwd_deconv_d,
        convolution_desc_t *bwd_conv_d) {
    const memory_desc_t *src_md, *dst_md, *deconv_weights_d;
    memory_desc_t src_md_patched;
    const auto src_dt = fwd_deconv_d->dst_desc.data_type;

    CHECK(memory_desc_init_by_md_and_dt(
            src_md_patched, fwd_deconv_d->dst_desc, src_dt));
    src_md = &src_md_patched;
    dst_md = &fwd_deconv_d->src_desc;
    deconv_weights_d = &fwd_deconv_d->weights_desc;

    /* create weights desc for convolution */
    memory_desc_t conv_weights_d;
    const bool with_groups = deconv_weights_d->ndims == src_md->ndims + 1;
    CHECK(weights_axes_permutation(
            &conv_weights_d, deconv_weights_d, with_groups));

    CHECK(conv_desc_init(bwd_conv_d, prop_kind::backward_data,
            alg_kind::convolution_direct, src_md, &conv_weights_d,
            &fwd_deconv_d->bias_desc, dst_md, fwd_deconv_d->strides,
            fwd_deconv_d->dilates, fwd_deconv_d->padding[0],
            fwd_deconv_d->padding[1]));

    // HACK: Set src_desc and dst_desc as a signal to the primitive
    //       descriptor cache that we are using the deconv version of bwd conv
    //       and thus need a separate cache entry (this will also disallow calling
    //       bwd_d conv with postops). This assumes that external users only use
    //       the API to create conv descs, and relies on common/convolution.cpp
    //       only setting the expected mem descs.
    // TODO: Pass this information via attributes or integrate this method
    //       directly into bwd conv implementations.
    bwd_conv_d->src_desc = bwd_conv_d->diff_src_desc;
    bwd_conv_d->dst_desc = bwd_conv_d->diff_dst_desc;
    return status::success;
}
} // namespace

template <typename implementation_pd>
status_t check_embedded_impl_init(primitive_desc_iterator_t &it) {
    const auto pd = dynamic_cast<implementation_pd *>((*it).get());
    if (pd != nullptr) return status::success; // implementation found
    return status::unimplemented;
}

template <cpu_isa_t isa>
status_t brgemm_deconvolution_fwd_t<isa>::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using namespace utils;
    using namespace format_tag;
    using smask_t = primitive_attr_t::skip_mask_t;
    const deconvolution_desc_t *fwd_deconv_d = desc();
    const auto src_type = fwd_deconv_d->src_desc.data_type;
    const auto dst_type = fwd_deconv_d->dst_desc.data_type;
    const bool is_int8 = utils::one_of(src_type, s8, u8);

    auto skip_mask = smask_t::post_ops | smask_t::sum_dt;
    if (is_int8)
        skip_mask |= smask_t::scales_runtime | smask_t::zero_points_runtime;

    const bool ok = is_fwd()
            && (desc()->alg_kind & alg_kind::deconvolution_direct)
            && attr()->has_default_values(skip_mask, dst_type)
            && attr()->post_ops_.check_sum_consistency(dst_type, is_int8)
            && attr_scales_ok() && post_ops_ok() && zero_points_ok()
            && !has_zero_dim_memory();
    if (!ok) return status::unimplemented;

    convolution_desc_t conv_d = convolution_desc_t();

    assert(src_type != data_type::undef);

    const int ndims_spatial = fwd_deconv_d->dst_desc.ndims - 2;
    for (int i = 0; i < ndims_spatial; i++) {
        if (fwd_deconv_d->strides[i] != 1) {
            has_strides_ = true;
            break;
        }
    }

    //TODO: Enable zero points support for bwd w/ stride on AMX
    const bool has_strides_with_zero_point = has_strides_
            && !everyone_is(brgemm_broadcast_t::none, get_zp_type(DNNL_ARG_SRC),
                    get_zp_type(DNNL_ARG_DST));
    if (is_superset(isa, avx512_core_amx) && has_strides_with_zero_point)
        return status::unimplemented;

    if (has_strides_) {
        CHECK(bwd_conv_desc_create(fwd_deconv_d, &conv_d));
        primitive_desc_iterator_t it(engine,
                reinterpret_cast<const op_desc_t *>(&conv_d), attr(), nullptr);
        if (!it.is_initialized()) return status::out_of_memory;

        while (++it != it.end()) {
            conv_pd_ = *it;
            // flag used to enable post-ops and properly disable zero-points
            constexpr bool is_deconv = true;
            if (check_embedded_impl_init<
                        typename brgemm_convolution_bwd_strided_t<isa,
                                is_deconv>::pd_t>(it)
                    == status::success)
                break;
        }
        if (it == it.end()) return status::unimplemented;
    } else {
        CHECK(fwd_conv_desc_create(fwd_deconv_d, &conv_d));

        primitive_desc_iterator_t it(engine,
                reinterpret_cast<const op_desc_t *>(&conv_d), attr(), nullptr);
        if (!it.is_initialized()) return status::out_of_memory;

        while (++it != it.end()) {
            conv_pd_ = *it;
            // try 1x1 fwd convolution
            if (check_embedded_impl_init<
                        typename brgemm_1x1_convolution_fwd_t<isa>::pd_t>(it)
                    == status::success)
                break;
            // try non-1x1 fwd convolution with invert weights' spatial indices
            constexpr bool use_inversion = true;
            if (check_embedded_impl_init<typename brgemm_convolution_fwd_t<isa,
                            use_inversion>::pd_t>(it)
                    == status::success)
                break;
        }
        if (it == it.end()) return status::unimplemented;
    }

    if (weights_md_.format_kind == format_kind::any) {
        if (has_strides_) {
            CHECK(weights_axes_permutation(
                    &weights_md_, conv_pd_->weights_md(), with_groups()));
            const bool is_signed_input = src_type == s8;
            const bool scale_adjust_required = is_signed_input
                    && !isa_has_s8s8(isa) && !isa_has_int8_vnni(isa);
            // Set flags after weights_axes_permutation call,
            // because this function expects flags to be zero
            if (scale_adjust_required)
                weights_md_.extra.flags = 0 | memory_extra_flags::scale_adjust;
        } else
            weights_md_ = *conv_pd_->weights_md();
    }
    if (src_md_.format_kind == format_kind::any) {
        if (has_strides_)
            src_md_ = *conv_pd_->diff_dst_md();
        else
            src_md_ = *conv_pd_->src_md();
    }
    if (dst_md_.format_kind == format_kind::any) {
        if (has_strides_)
            dst_md_ = *conv_pd_->diff_src_md();
        else
            dst_md_ = *conv_pd_->dst_md();
    }
    attr_.set_default_formats(&dst_md_);
    if (bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md_, x));

    init_name();
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_nested,
            conv_pd_->scratchpad_registry());

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_deconvolution_fwd_t<isa>::init(engine_t *engine) {
    return pd()->conv_pd_->create_primitive(conv_p_, engine);
}

template <cpu_isa_t isa>
status_t brgemm_deconvolution_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto &args = ctx.args();
    exec_args_t conv_args(args);
    if (pd()->has_strides_) {
        conv_args[DNNL_ARG_DIFF_SRC] = args.at(DNNL_ARG_DST);
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args.erase(DNNL_ARG_DST);
        conv_args.erase(DNNL_ARG_SRC);
    }

    exec_ctx_t conv_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(ctx, memory_tracking::names::key_nested, conv_p_);
    conv_ctx.set_scratchpad_grantor(ns.grantor());
    return conv_p_->execute(conv_ctx);
}

template struct brgemm_deconvolution_fwd_t<avx2>;
template struct brgemm_deconvolution_fwd_t<avx2_vnni>;
template struct brgemm_deconvolution_fwd_t<avx2_vnni_2>;
template struct brgemm_deconvolution_fwd_t<avx512_core>;
template struct brgemm_deconvolution_fwd_t<avx512_core_vnni>;
template struct brgemm_deconvolution_fwd_t<avx512_core_bf16>;
template struct brgemm_deconvolution_fwd_t<avx512_core_fp16>;
template struct brgemm_deconvolution_fwd_t<avx512_core_amx>;
template struct brgemm_deconvolution_fwd_t<avx512_core_amx_fp16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
