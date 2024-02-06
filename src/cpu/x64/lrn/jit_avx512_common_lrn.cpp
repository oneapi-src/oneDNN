/*******************************************************************************
* Copyright 2017-2024 Intel Corporation
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
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/lrn/jit_avx512_common_lrn.hpp"
#include "cpu/x64/lrn/lrn_executor_factory.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static constexpr int vsize = 16;

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;

template <data_type_t d_type>
status_t jit_avx512_common_lrn_fwd_t<d_type>::pd_t::init(engine_t *engine) {
    using namespace prop_kind;
    using namespace alg_kind;

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper dst_d(dst_md());

    VDISPATCH_LRN(is_fwd(), VERBOSE_BAD_PROPKIND);

    // disabling verbose dispatch checks for unsupported isa for better readability
    if (!mayiuse(avx512_core)) return status::unimplemented;

    VDISPATCH_LRN(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_LRN(everyone_is(d_type, src_d.data_type(), dst_d.data_type()),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LRN(IMPLICATION(d_type == f16, mayiuse(avx512_core_fp16)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_LRN(src_d.ndims() == 4, VERBOSE_BAD_NDIMS, "src", src_d.ndims());
    VDISPATCH_LRN(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_LRN(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_LRN(src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");

    const auto fmt_tag
            = src_d.matches_one_of_tag(format_tag::nhwc, format_tag::nChw16c);

    VDISPATCH_LRN(
            desc()->alg_kind == lrn_across_channels, VERBOSE_BAD_ALGORITHM);
    VDISPATCH_LRN(desc()->local_size >= 1 && desc()->local_size <= 16,
            VERBOSE_BAD_PARAM, "local_size");
    VDISPATCH_LRN((desc()->lrn_beta == 0.75 || desc()->lrn_beta == 1.0),
            VERBOSE_BAD_PARAM, "lrn_beta");
    VDISPATCH_LRN(src_d.matches_tag(fmt_tag), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_LRN(
            IMPLICATION(fmt_tag == format_tag::nChw16c,
                    src_d.dims()[1] % vsize == 0 && desc()->local_size == 5),
            "unsupported format tag, dimension and local_size combination");

    if (desc()->prop_kind == forward_training) {
        dims_t ws_dims = {MB(), C(), H(), 2 * W()};
        memory_desc_init_by_tag(ws_md_, 4, ws_dims, d_type, fmt_tag);
    }

    return success;
}

template <data_type_t d_type>
jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_fwd_t(
        const pd_t *apd)
    : primitive_t(apd)
    , lrn_executor_(lrn::lrn_executor_factory_t::create_executor<d_type,
              typename jit_avx512_common_lrn_fwd_t<d_type>::pd_t>(
              pd(), lrn::direction::forward)) {}

template <data_type_t d_type>
jit_avx512_common_lrn_fwd_t<d_type>::~jit_avx512_common_lrn_fwd_t() = default;

template struct jit_avx512_common_lrn_fwd_t<f32>;
template struct jit_avx512_common_lrn_fwd_t<bf16>;
template struct jit_avx512_common_lrn_fwd_t<f16>;

template <data_type_t d_type>
status_t jit_avx512_common_lrn_bwd_t<d_type>::pd_t::init(engine_t *engine) {
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
    VDISPATCH_LRN(IMPLICATION(d_type == f16, mayiuse(avx512_core_fp16)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_LRN(src_d.ndims() == 4, VERBOSE_BAD_NDIMS, "src", src_d.ndims());
    VDISPATCH_LRN(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_LRN(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_LRN(
            src_d == diff_dst_d, VERBOSE_INCONSISTENT_MDS, "src", "diff_dst");
    VDISPATCH_LRN(diff_dst_d == diff_src_d, VERBOSE_INCONSISTENT_MDS,
            "diff_src", "diff_dst");

    const dims_t ws_dims = {MB(), C(), H(), 2 * W()};
    const auto fmt_tag
            = src_d.matches_one_of_tag(format_tag::nhwc, format_tag::nChw16c);
    memory_desc_init_by_tag(ws_md_, 4, ws_dims, d_type, fmt_tag);
    VDISPATCH_LRN(compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);

    VDISPATCH_LRN(
            desc()->alg_kind == lrn_across_channels, VERBOSE_BAD_ALGORITHM);
    VDISPATCH_LRN(desc()->local_size >= 1 && desc()->local_size <= 16,
            VERBOSE_BAD_PARAM, "local_size");
    VDISPATCH_LRN((desc()->lrn_beta == 0.75 || desc()->lrn_beta == 1.0),
            VERBOSE_BAD_PARAM, "lrn_beta");
    VDISPATCH_LRN(src_d.matches_tag(fmt_tag), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_LRN(
            IMPLICATION(fmt_tag == format_tag::nChw16c,
                    src_d.dims()[1] % vsize == 0 && desc()->local_size == 5),
            "unsupported format tag, dimension and local_size combination");
    return success;
}

template <data_type_t d_type>
jit_avx512_common_lrn_bwd_t<d_type>::jit_avx512_common_lrn_bwd_t(
        const pd_t *apd)
    : primitive_t(apd)
    , lrn_executor_(lrn::lrn_executor_factory_t::create_executor<d_type,
              typename jit_avx512_common_lrn_bwd_t<d_type>::pd_t>(
              pd(), lrn::direction::backward)) {}

template <data_type_t d_type>
jit_avx512_common_lrn_bwd_t<d_type>::~jit_avx512_common_lrn_bwd_t() = default;

template struct jit_avx512_common_lrn_bwd_t<f32>;
template struct jit_avx512_common_lrn_bwd_t<bf16>;
template struct jit_avx512_common_lrn_bwd_t<f16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
