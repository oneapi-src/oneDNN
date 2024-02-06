/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

#include <cmath>
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/lrn/jit_uni_lrn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

static constexpr int MAX_LOCAL_SIZE = 32u;

static dnnl_dim_t compute_n_summands(
        dnnl_dim_t size, int ndims, const dnnl_alg_kind_t &alg_kind) {
    return alg_kind == alg_kind::lrn_across_channels
            ? size
            : std::pow(size, ndims - 2);
};

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_t<isa, d_type>::jit_uni_lrn_fwd_t(const pd_t *apd)
    : primitive_t(apd)
    , ker_(nullptr)
    , ker_first_(nullptr)
    , ker_last_(nullptr) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_t<isa, d_type>::~jit_uni_lrn_fwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_lrn_fwd_t<isa, d_type>::init(engine_t *engine) {
    using namespace alg_kind;

    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int ndims = memory_desc_wrapper(pd()->src_md()).ndims();
    const int ls = pd()->desc()->local_size;
    const float K = pd()->desc()->lrn_k;
    const auto pk = pd()->desc()->prop_kind;
    const auto ak = pd()->desc()->alg_kind;
    const auto dat_tag = pd()->dat_tag_;
    const float A = pd()->desc()->lrn_alpha / compute_n_summands(ls, ndims, ak);

    if (dat_tag == nChw8c && ls == 5 && ak == lrn_across_channels) {
        ker_ = utils::make_unique<jit_uni_lrn_fwd_kernel_t<isa, d_type>>(
                nchw8c_across_t(H, W, 0), A, K, pk);
        ker_first_ = utils::make_unique<jit_uni_lrn_fwd_kernel_t<isa, d_type>>(
                nchw8c_across_t(H, W, -1), A, K, pk);
        ker_last_ = utils::make_unique<jit_uni_lrn_fwd_kernel_t<isa, d_type>>(
                nchw8c_across_t(H, W, +1), A, K, pk);
    } else if (one_of(dat_tag, nhwc, nChw8c, nChw16c)
            && ak == lrn_within_channel) {

        ker_ = utils::make_unique<jit_uni_lrn_fwd_kernel_t<isa, d_type>>(
                within_config_t(H, W, C, ls, dat_tag), A, K, pk);
    } else if (dat_tag == nchw && ls == 5 && ak == lrn_across_channels) {
        ker_ = utils::make_unique<jit_uni_lrn_fwd_kernel_t<isa, d_type>>(
                nchw_across_t(C, H * W, 0), A, K, pk);
        const int remind = (H * W) % VECTOR_LENGTH;
        if (remind != 0) {
            ker_last_
                    = utils::make_unique<jit_uni_lrn_fwd_kernel_t<isa, d_type>>(
                            nchw_across_t(C, H * W, remind), A, K, pk);
        }
    } else {
        ker_ = utils::make_unique<jit_uni_lrn_fwd_kernel_t<isa, d_type>>(
                nhwc_across_t(C), A, K, pk);
    }
    CHECK(ker_->create_kernel());
    if (ker_first_) CHECK(ker_first_->create_kernel());
    if (ker_last_) CHECK(ker_last_->create_kernel());
    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_lrn_fwd_t<isa, d_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace alg_kind;

    status_t status = status::success;

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
    CHECK(status);
    auto ws = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_WORKSPACE, status);
    CHECK(status);

    const int N = pd()->MB();
    const int C = pd()->C();
    const int HW = pd()->H() * pd()->W();
    const int ls = pd()->desc()->local_size;

    const auto ak = pd()->desc()->alg_kind;
    const auto dat_tag = pd()->dat_tag_;
    const auto ker_first = ker_first_.get();
    const auto ker = ker_.get();
    const auto ker_last = ker_last_.get();

    if (dat_tag == nChw8c && ls == 5 && ak == lrn_across_channels) {
        parallel_nd(N, C / VECTOR_LENGTH, [&](dim_t n, dim_t c8) {
            const auto offset = n * HW * C + c8 * HW * VECTOR_LENGTH;
            auto ws_ptr = ws ? &ws[offset] : nullptr;
            jit_args_fwd_t args {&src[offset], &dst[offset], ws_ptr, nullptr};
            if (c8 == 0)
                (*ker_first)(&args);
            else if (c8 == C / VECTOR_LENGTH - 1)
                (*ker_last)(&args);
            else
                (*ker)(&args);
        });
    } else if (one_of(dat_tag, nhwc, nChw8c, nChw16c)
            && ak == lrn_within_channel) {
        parallel_nd(N, C / VECTOR_LENGTH, [&](dim_t n, dim_t c) {
            const std::size_t offset = dat_tag == nhwc
                    ? n * HW * C + c * VECTOR_LENGTH
                    : n * HW * C + c * HW * VECTOR_LENGTH;
            auto ws0_ptr = ws ? &ws[offset] : nullptr;
            auto ws1_ptr = ws ? &ws[offset + N * C * HW] : nullptr;
            jit_args_fwd_t args {&src[offset], &dst[offset], ws0_ptr, ws1_ptr};
            (*ker)(&args);
        });
    } else if (dat_tag == nchw && ls == 5 && ak == lrn_across_channels) {
        parallel_nd(N, (HW + VECTOR_LENGTH - 1) / VECTOR_LENGTH,
                [&](dim_t n, dim_t hw8) {
                    const auto offset = n * HW * C + hw8 * VECTOR_LENGTH;
                    auto ws0_ptr = ws ? &ws[offset] : nullptr;
                    jit_args_fwd_t args {
                            &src[offset], &dst[offset], ws0_ptr, nullptr};

                    if ((hw8 + 1) * VECTOR_LENGTH > HW)
                        (*ker_last)(&args);
                    else
                        (*ker)(&args);
                });
    } else { // nhwc
        parallel_nd(N, HW, [&](dim_t n, dim_t hw) {
            const auto offset = n * HW * C + hw * C;
            auto ws_ptr = ws ? &ws[offset] : nullptr;
            jit_args_fwd_t args {&src[offset], &dst[offset], ws_ptr, nullptr};
            (*ker)(&args);
        });
    }
    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_lrn_fwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace prop_kind;
    using namespace alg_kind;

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper dst_d(dst_md());

    VDISPATCH_LRN(is_fwd(), VERBOSE_BAD_PROPKIND);

    // disabling verbose dispatch checks for unsupported isa for better readability
    if (!mayiuse(isa)) return status::unimplemented;

    VDISPATCH_LRN(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_LRN(everyone_is(d_type, src_d.data_type(), dst_d.data_type()),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LRN(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_LRN(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_LRN(src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
    VDISPATCH_LRN(src_d.ndims() == 4, VERBOSE_BAD_NDIMS, "src", src_d.ndims());
    VDISPATCH_LRN(src_d.dims()[1] % VECTOR_LENGTH == 0
                    && src_d.dims()[1] >= 2 * VECTOR_LENGTH,
            "src has inconsistent dimensions with vector length");
    VDISPATCH_LRN(desc()->lrn_beta == 0.75, VERBOSE_BAD_PARAM, "lrn_beta");

    dat_tag_ = memory_desc_matches_one_of_tag(
            *src_md(), nChw16c, nChw8c, nchw, nhwc);

    const int HW = src_d.dims()[2] * src_d.dims()[3];

    const bool args_ok_across = true && desc()->alg_kind == lrn_across_channels
            && desc()->local_size == 5 && one_of(dat_tag_, nChw8c, nchw, nhwc)
            && everyone_is(data_type::f32, src_d.data_type())
            /* SSE41: prevent loads smaller than the size of xmm registers,
         * otherwise it will result in an illegal memory read (seg-fault)
         * due to protected memory. */
            && IMPLICATION(isa == sse41 && dat_tag_ == nchw, HW >= 4)
            && !is_superset(isa, avx512_core);

    const int jit_max_local_size = 5; // bigger size triggers too big code size
    const bool args_ok_within = true && desc()->alg_kind == lrn_within_channel
            && desc()->local_size <= (jit_max_local_size <= MAX_LOCAL_SIZE
                               ? jit_max_local_size
                               : MAX_LOCAL_SIZE)
            && src_d.dims()[2] >= desc()->local_size
            && src_d.dims()[3] >= desc()->local_size
            && IMPLICATION(d_type == data_type::bf16,
                    mayiuse(avx512_core) || mayiuse(avx2_vnni_2))
            && IMPLICATION(d_type == data_type::f16,
                    mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2))
            && (is_superset(isa, avx512_core) ? one_of(dat_tag_, nhwc, nChw16c)
                                              : one_of(dat_tag_, nhwc, nChw8c));

    const auto status
            = args_ok_across || args_ok_within ? success : unimplemented;

    if (desc()->prop_kind == forward_training && status == success) {
        dims_t ws_dims = {MB(), C(), H(), 2 * W()};
        memory_desc_init_by_tag(ws_md_, 4, ws_dims, d_type, dat_tag_);
    }

    return status;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_bwd_t<isa, d_type>::jit_uni_lrn_bwd_t(const pd_t *apd)
    : primitive_t(apd)
    , ker_(nullptr)
    , ker_first_(nullptr)
    , ker_last_(nullptr) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_bwd_t<isa, d_type>::~jit_uni_lrn_bwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_lrn_bwd_t<isa, d_type>::init(engine_t *engine) {
    using namespace alg_kind;
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int &ls = pd()->desc()->local_size;
    const auto &ak = pd()->desc()->alg_kind;
    const int ndims = memory_desc_wrapper(pd()->src_md()).ndims();
    const float A = pd()->desc()->lrn_alpha / compute_n_summands(ls, ndims, ak);
    const float &B = pd()->desc()->lrn_beta;
    const auto &dat_tag = pd()->dat_tag_;

    if (one_of(dat_tag, nhwc, nChw8c, nChw16c) && ak == lrn_within_channel) {
        ker_ = utils::make_unique<jit_uni_lrn_bwd_kernel_t<isa, d_type>>(
                within_config_t(H, W, C, ls, dat_tag), A, B);
    } else {
        int use_h_parallelism = 0; // XXX
        if (C / VECTOR_LENGTH == 1) {
            ker_ = utils::make_unique<jit_uni_lrn_bwd_kernel_t<isa, d_type>>(
                    nchw8c_across_t(H, W, 3), A, B, use_h_parallelism);
        } else {
            ker_ = utils::make_unique<jit_uni_lrn_bwd_kernel_t<isa, d_type>>(
                    nchw8c_across_t(H, W, 0), A, B, use_h_parallelism);
            ker_first_
                    = utils::make_unique<jit_uni_lrn_bwd_kernel_t<isa, d_type>>(
                            nchw8c_across_t(H, W, -1), A, B, use_h_parallelism);
            ker_last_
                    = utils::make_unique<jit_uni_lrn_bwd_kernel_t<isa, d_type>>(
                            nchw8c_across_t(H, W, +1), A, B, use_h_parallelism);
        }
    }
    CHECK(ker_->create_kernel());
    if (ker_first_) CHECK(ker_first_->create_kernel());
    if (ker_last_) CHECK(ker_last_->create_kernel());
    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_lrn_bwd_t<isa, d_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const data_t *, DNNL_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const int N = pd()->MB();
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const auto ak = pd()->desc()->alg_kind;
    const auto &dat_tag = pd()->dat_tag_;

    static constexpr bool use_h_parallelism = false; // XXX

    const auto ker = ker_.get();
    const auto ker_first = ker_first_.get();
    const auto ker_last = ker_last_.get();
    const auto tensor_size = N * C * H * W;

    if (one_of(dat_tag, nhwc, nChw8c, nChw16c)
            && ak == alg_kind::lrn_within_channel) {
        parallel_nd(N, C / VECTOR_LENGTH, [&](dim_t n, dim_t c) {
            const std::size_t offset = dat_tag == nhwc
                    ? n * H * W * C + c * VECTOR_LENGTH
                    : n * H * W * C + c * H * W * VECTOR_LENGTH;
            jit_args_bwd_t args {&src[offset], &diff_dst[offset], &ws[offset],
                    &ws[offset + tensor_size], &diff_src[offset]};
            (*ker)(&args);
        });
    } else if (use_h_parallelism) {
        parallel_nd(N, C / VECTOR_LENGTH, H, [&](dim_t n, dim_t c8, dim_t h) {
            const std::size_t offset = n * C * H * W
                    + c8 * H * W * VECTOR_LENGTH + h * W * VECTOR_LENGTH;
            jit_args_bwd_t args {&src[offset], &diff_dst[offset], &ws[offset],
                    nullptr, &diff_src[offset]};
            if (C / VECTOR_LENGTH == 1)
                (*ker)(&args);
            else if (c8 == 0)
                (*ker_first)(&args);
            else if (c8 == C / VECTOR_LENGTH - 1)
                (*ker_last)(&args);
            else
                (*ker)(&args);
        });
    } else {
        parallel_nd(N, C / VECTOR_LENGTH, [&](dim_t n, dim_t c8) {
            const std::size_t offset
                    = n * C * H * W + c8 * H * W * VECTOR_LENGTH;
            jit_args_bwd_t args {&src[offset], &diff_dst[offset], &ws[offset],
                    nullptr, &diff_src[offset]};
            if (C / VECTOR_LENGTH == 1)
                (*ker)(&args);
            else if (c8 == 0)
                (*ker_first)(&args);
            else if (c8 == C / VECTOR_LENGTH - 1)
                (*ker_last)(&args);
            else
                (*ker)(&args);
        });
    }
    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_lrn_bwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace prop_kind;
    using namespace alg_kind;

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper diff_src_d(diff_src_md());
    const memory_desc_wrapper diff_dst_d(diff_dst_md());

    VDISPATCH_LRN(!is_fwd(), VERBOSE_BAD_PROPKIND);

    // disabling verbose dispatch checks for unsupported isa for better readability
    if (!mayiuse(avx512_core)) return status::unimplemented;

    VDISPATCH_LRN(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_LRN(utils::everyone_is(d_type, src_d.data_type(),
                          diff_src_d.data_type(), diff_dst_d.data_type()),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LRN(src_d.ndims() == 4, VERBOSE_BAD_NDIMS, "src", src_d.ndims());
    VDISPATCH_LRN(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_LRN(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_LRN(src_d == diff_dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
    VDISPATCH_LRN(diff_dst_d == diff_src_d, VERBOSE_INCONSISTENT_MDS,
            "diff_src", "diff_dst");
    VDISPATCH_LRN((src_d.dims()[1] % VECTOR_LENGTH == 0
                          && src_d.dims()[1] >= 2 * VECTOR_LENGTH),
            "src has inconsistent dimensions with vector length");
    VDISPATCH_LRN(desc()->lrn_beta == 0.75, VERBOSE_BAD_PARAM, "lrn_beta");

    dat_tag_ = memory_desc_matches_one_of_tag(
            *src_md(), nChw16c, nChw8c, nchw, nhwc);

    const dims_t ws_dims = {MB(), C(), H(), 2 * W()};
    memory_desc_init_by_tag(ws_md_, 4, ws_dims, d_type, dat_tag_);

    VDISPATCH_LRN(compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);

    const bool args_ok_across = true && desc()->alg_kind == lrn_across_channels
            && desc()->local_size == 5 && utils::one_of(dat_tag_, nChw8c)
            && everyone_is(data_type::f32, src_d.data_type())
            && !is_superset(isa, avx512_core);

    const int jit_max_local_size = 5; // bigger size triggers too big code size
    const bool args_ok_within = true && desc()->alg_kind == lrn_within_channel
            && desc()->local_size <= (jit_max_local_size <= MAX_LOCAL_SIZE
                               ? jit_max_local_size
                               : MAX_LOCAL_SIZE)
            && src_d.dims()[2] >= desc()->local_size
            && src_d.dims()[3] >= desc()->local_size
            && IMPLICATION(d_type == data_type::bf16, mayiuse(avx512_core))
            && IMPLICATION(d_type == data_type::f16, mayiuse(avx512_core_fp16))
            && (isa == avx512_core ? one_of(dat_tag_, nhwc, nChw16c)
                                   : one_of(dat_tag_, nhwc, nChw8c));

    return args_ok_across || args_ok_within ? success : unimplemented;
}

template struct jit_uni_lrn_fwd_t<avx512_core, data_type::f32>;
template struct jit_uni_lrn_fwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_lrn_fwd_t<avx512_core_fp16, data_type::f16>;
template struct jit_uni_lrn_fwd_t<avx2_vnni_2, data_type::bf16>;
template struct jit_uni_lrn_fwd_t<avx2_vnni_2, data_type::f16>;
template struct jit_uni_lrn_fwd_t<avx2, data_type::f32>;
template struct jit_uni_lrn_fwd_t<sse41, data_type::f32>;
template struct jit_uni_lrn_bwd_t<avx512_core, data_type::f32>;
template struct jit_uni_lrn_bwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_lrn_bwd_t<avx512_core_fp16, data_type::f16>;
template struct jit_uni_lrn_bwd_t<avx2, data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
