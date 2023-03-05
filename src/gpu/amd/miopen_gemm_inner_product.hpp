/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_AMD_MIOPEN_GEMM_INNER_PRODUCT_HPP
#define GPU_AMD_MIOPEN_GEMM_INNER_PRODUCT_HPP

#include <miopen/miopen.h>

#include "common/c_types_map.hpp"
#include "common/inner_product_pd.hpp"
#include "common/primitive.hpp"
#include "gpu/amd/miopen_gemm_inner_product_impl.hpp"
#include "gpu/amd/miopen_inner_product.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {
namespace {

inline bool gemm_consitency_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    using namespace utils;

    auto strides_compatible = [&]() {
        bool ok = true;
        auto w_str = wei_d.blocking_desc().strides;
        auto d_str = src_d.blocking_desc().strides;
        for (int i = 1; i < src_d.ndims() - 1; i++) {
            ok = ok && w_str[i] / d_str[i] == w_str[i + 1] / d_str[i + 1];
        }
        return ok && one_of(w_str[1] / d_str[1], 1, wei_d.padded_dims()[0]);
    };

    auto inner_blk_compatible = [&]() {
        auto d_inner_blks = src_d.blocking_desc().inner_blks;
        auto w_inner_blks = wei_d.blocking_desc().inner_blks;
        auto d_inner_idxs = src_d.blocking_desc().inner_idxs;
        auto w_inner_idxs = wei_d.blocking_desc().inner_idxs;

        int d_inner_nblks = src_d.blocking_desc().inner_nblks;
        int w_inner_nblks = wei_d.blocking_desc().inner_nblks;

        bool ok = true;

        if ((wei_d.blocking_desc().strides[0] == 1) && (w_inner_nblks > 0)) {
            ok = ok && wei_d.dims()[0] / w_inner_blks[w_inner_nblks - 1] == 1
                    && w_inner_idxs[w_inner_nblks - 1] == 0;
            w_inner_nblks--;
        }

        ok = ok && d_inner_nblks == w_inner_nblks;
        bool supported_block_size = (d_inner_nblks == 0
                || (d_inner_nblks == 1 && d_inner_idxs[0] == w_inner_idxs[0]
                        && w_inner_idxs[0] == 1
                        && d_inner_blks[0] == w_inner_blks[0]
                        && d_inner_blks[0] == 4
                        && src_d.data_type() == data_type::s8));
        ok = ok && supported_block_size;
        for (int d = 1; d < w_inner_nblks; d++)
            ok = ok && (d_inner_blks[d] == w_inner_blks[d] == 0)
                    && (d_inner_idxs[d] == w_inner_idxs[d] == 0);
        return ok;
    };

    return true && src_d.is_blocking_desc() && wei_d.is_blocking_desc()
            && src_d.ndims() == wei_d.ndims() && inner_blk_compatible()
            && strides_compatible() && dst_d.matches_tag(format_tag::nc)
            && src_d.only_padded_dim(1) && wei_d.only_padded_dim(1)
            && src_d.padded_dims()[1] == wei_d.padded_dims()[1];
}

inline bool reorder_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    using namespace format_tag;
    using namespace utils;

    return true
            && ((src_d.matches_tag(nwc)
                        && (wei_d.matches_one_of_tag(oiw, iwo) != undef))
                    || (src_d.matches_tag(ncw)
                            && (wei_d.matches_one_of_tag(wio, owi) != undef))
                    || (src_d.matches_tag(nhwc),
                            (wei_d.matches_one_of_tag(oihw, ihwo) != undef))
                    || (src_d.matches_tag(nchw)
                            && (wei_d.matches_one_of_tag(ohwi, hwio) != undef))
                    || (src_d.matches_tag(ndhwc)
                            && (wei_d.matches_one_of_tag(oidhw, idhwo)
                                    != undef))
                    || (src_d.matches_tag(ncdhw)
                            && (wei_d.matches_one_of_tag(odhwi, dhwio)
                                    != undef)))
            && dst_d.matches_tag(nc);
}

inline bool dense_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    return true && src_d.is_dense(true) && dst_d.is_dense()
            && wei_d.is_dense(true);
}

status_t template_set_default_params(memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t *bias_md, int ndims) {
    using namespace format_tag;

    auto init_md = [&](memory_desc_t &out_md, const memory_desc_t &in_md) {
        format_tag_t md_tag;
        if (memory_desc_matches_one_of_tag(in_md, ab, abc, abcd, abcde))
            md_tag = utils::pick(ndims - 2, ab, abc, abcd, abcde);
        else if (memory_desc_matches_one_of_tag(in_md, acb, acdb, acdeb))
            md_tag = utils::pick(ndims - 3, cba, cdba, cdeba);
        else if (memory_desc_matches_one_of_tag(in_md, ba, cba, cdba, cdeba))
            md_tag = utils::pick(ndims - 2, ab, acb, acdb, acdeb);
        else {
            memory_desc_wrapper md_desc_wrapper(in_md);
            return memory_desc_init_by_blocking_desc(
                    out_md, md_desc_wrapper.blocking_desc());
        }
        return memory_desc_init_by_tag(out_md, md_tag);
    };
    if (src_md.format_kind == format_kind::any
            && weights_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(
                src_md, utils::pick(ndims - 2, nc, ncw, nchw, ncdhw)));
        CHECK(memory_desc_init_by_tag(
                weights_md, utils::pick(ndims - 2, oi, oiw, oihw, oidhw)));
    } else if (src_md.format_kind == format_kind::any) {
        CHECK(init_md(src_md, weights_md));
    } else if (weights_md.format_kind == format_kind::any) {
        CHECK(init_md(weights_md, src_md));
    }
    if (dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, nc));
    }
    if (bias_md->format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(*bias_md, x));
    }
    return status::success;
}

} // namespace

struct miopen_gemm_inner_product_fwd_t : public miopen_inner_product_fwd_t {
    using miopen_inner_product_fwd_t::miopen_inner_product_fwd_t;
    using parrent_pd_t = miopen_inner_product_fwd_t::pd_t;

    struct pd_t : public parrent_pd_t {
        using parrent_pd_t::parrent_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:gemm", miopen_gemm_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace data_type;

            bool ok = is_fwd() && (set_default_params() == status::success);
            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) return status::success;
            bool gemm_compatible
                    = gemm_consitency_check(src_md(), weights_md(), dst_md());
            bool need_reorder = gemm_compatible
                    ? false
                    : reorder_check(src_md(), weights_md(), dst_md());

            using sm_t = primitive_attr_t::skip_mask_t;
            const auto attr_skip_mask = sm_t::oscale_runtime | sm_t::post_ops;

            bool with_eltwise
                    = attr()->post_ops_.find(primitive_kind::eltwise) != -1;
            bool with_sum = attr()->post_ops_.find(primitive_kind::sum) != -1;

            data_type_t src_dt = src_md()->data_type;
            data_type_t dst_dt = dst_md()->data_type;
            data_type_t wei_dt = weights_md(0)->data_type;

            data_type_t bia_dt = data_type::undef;
            if (with_bias()) bia_dt = weights_md(1)->data_type;

            bool f32_case = utils::everyone_is(f32, src_dt, wei_dt, dst_dt);
            bool f16_case = utils::everyone_is(f16, src_dt, wei_dt, dst_dt);
            bool s8_case
                    = utils::everyone_is(s8, src_dt, wei_dt) && (dst_dt == s32);
            bool bf16_case = utils::everyone_is(bf16, src_dt, wei_dt)
                    && (utils::one_of(dst_dt, bf16, f32));

            ok = (f32_case || f16_case || bf16_case || s8_case)
                    && IMPLICATION(with_bias(),
                            (IMPLICATION(f32_case, (bia_dt == f32))
                                    && IMPLICATION(f16_case, (bia_dt == f16))
                                    && IMPLICATION(s8_case, (bia_dt == s32))
                                    && IMPLICATION(
                                            bf16_case, (bia_dt == f32))));
            ok = ok && memory_format_ok(src_md())
                    && memory_format_ok(weights_md(0))
                    && memory_format_ok(dst_md())
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            utils::one_of(src_md_.data_type, s8)
                                    && attr()->output_scales_.mask_ == 0)
                    && attr()->has_default_values(attr_skip_mask)
                    && attr_post_ops_ok(attr(), s8_case)
                    && dense_check(src_md(), weights_md(), dst_md())
                    && (gemm_compatible || need_reorder);
            if (!ok) return status::unimplemented;

            inner_product_impl_.reset(
                    new miopen_gemm_inner_product_fwd_impl_t());

            return inner_product_impl_->init(engine, this, with_eltwise,
                    with_eltwise, with_sum, need_reorder);
        }

        status_t set_default_params() {
            return template_set_default_params(
                    src_md_, weights_md_, dst_md_, &bias_md_, ndims());
        }
    };

    const pd_t *pd() const override {
        return (const pd_t *)primitive_t::pd().get();
    }
};

struct miopen_gemm_inner_product_bwd_data_t
    : public miopen_inner_product_bwd_data_t {
    using miopen_inner_product_bwd_data_t::miopen_inner_product_bwd_data_t;
    using parent_pd_t = miopen_inner_product_bwd_data_t::pd_t;

    struct pd_t : public parent_pd_t {
        using parent_pd_t::parent_pd_t;

        DECLARE_COMMON_PD_T(
                "hip:miopen:gemm", miopen_gemm_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            bool ok = true && this->desc()->prop_kind == backward_data
                    && set_default_params() == status::success;
            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) return status::success;
            bool gemm_compatible = gemm_consitency_check(
                    diff_src_md(), weights_md(), diff_dst_md());
            bool need_reorder = gemm_compatible
                    ? false
                    : reorder_check(diff_src_md(), weights_md(), diff_dst_md());

            data_type_t diff_src_dt = diff_src_md()->data_type;
            data_type_t diff_dst_dt = diff_dst_md()->data_type;
            data_type_t wei_dt = weights_md(0)->data_type;

            bool f32_case
                    = utils::everyone_is(f32, diff_src_dt, wei_dt, diff_dst_dt);
            bool bf16_case = utils::everyone_is(bf16, diff_dst_dt, wei_dt)
                    && (utils::one_of(diff_src_dt, bf16, f32));

            ok = ok && (f32_case || bf16_case) && attr()->has_default_values()
                    && dense_check(diff_src_md(), weights_md(), diff_dst_md())
                    && (gemm_compatible || need_reorder);
            if (!ok) return status::unimplemented;

            inner_product_impl_.reset(
                    new miopen_gemm_inner_product_bwd_data_impl_t());

            return inner_product_impl_->init(
                    engine, this, false, false, false, need_reorder);
        }

        status_t set_default_params() {
            return template_set_default_params(diff_src_md_, weights_md_,
                    diff_dst_md_, &glob_zero_md, ndims());
        }
    };

    const pd_t *pd() const override {
        return (const pd_t *)primitive_t::pd().get();
    }
};

struct miopen_gemm_inner_product_bwd_weights_t
    : public miopen_inner_product_bwd_weights_t {
    using miopen_inner_product_bwd_weights_t::
            miopen_inner_product_bwd_weights_t;
    using parent_pd_t = miopen_inner_product_bwd_weights_t::pd_t;

    struct pd_t : public parent_pd_t {
        using parent_pd_t::parent_pd_t;

        DECLARE_COMMON_PD_T(
                "hip:miopen:gemm", miopen_gemm_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            bool ok = true && this->desc()->prop_kind == backward_weights
                    && set_default_params() == status::success;
            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) return status::success;
            bool gemm_compatible = gemm_consitency_check(
                    src_md(), diff_weights_md(), diff_dst_md());
            bool need_reorder = gemm_compatible
                    ? false
                    : reorder_check(src_md(), diff_weights_md(), diff_dst_md());

            data_type_t src_dt = src_md()->data_type;
            data_type_t diff_dst_dt = diff_dst_md()->data_type;
            data_type_t diff_wei_dt = diff_weights_md(0)->data_type;

            bool f32_case
                    = utils::everyone_is(f32, src_dt, diff_dst_dt, diff_wei_dt);
            bool bf16_case = utils::everyone_is(bf16, src_dt, diff_dst_dt)
                    && (utils::one_of(diff_wei_dt, bf16, f32));

            ok = ok && (f32_case || bf16_case) && attr()->has_default_values()
                    && dense_check(src_md(), diff_weights_md(), diff_dst_md())
                    && (gemm_compatible || need_reorder);

            if (bf16_case) ok = ok && (!with_bias());

            if (!ok) return status::unimplemented;
            inner_product_impl_.reset(
                    new miopen_gemm_inner_product_bwd_weights_impl_t());
            return inner_product_impl_->init(
                    engine, this, false, false, false, need_reorder);
        }

        status_t set_default_params() {
            return template_set_default_params(src_md_, diff_weights_md_,
                    diff_dst_md_, &diff_bias_md_, ndims());
        }
    };

    const pd_t *pd() const override {
        return (const pd_t *)primitive_t::pd().get();
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif