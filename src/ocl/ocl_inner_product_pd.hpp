/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef OCL_INNER_PRODUCT_FWD_PD_HPP
#define OCL_INNER_PRODUCT_FWD_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/inner_product_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "ocl/ocl_engine.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

namespace {
inline bool dense_consitency_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    using namespace format_tag;
    using namespace utils;
    return true
            && IMPLICATION(src_d.matches_tag(ncw), wei_d.matches_tag(oiw))
            && IMPLICATION(src_d.matches_tag(nchw), wei_d.matches_tag(oihw))
            && IMPLICATION(src_d.matches_tag(ncdhw), wei_d.matches_tag(oidhw))
            && IMPLICATION(
                       src_d.matches_tag(nc), wei_d.matches_one_of_tag(oi, io))
            && dst_d.matches_tag(nc) && src_d.is_dense(true) && dst_d.is_dense()
            && wei_d.is_dense(true);
}

inline bool dense_gemm_consitency_check(const memory_desc_wrapper &src_d,
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
    return true && src_d.is_blocking_desc() && wei_d.is_blocking_desc()
            && src_d.ndims() == wei_d.ndims()
            && src_d.blocking_desc().inner_nblks
            == wei_d.blocking_desc().inner_nblks
            && utils::one_of(src_d.blocking_desc().inner_nblks, 0, 1)
            && array_cmp(src_d.blocking_desc().inner_blks,
                       wei_d.blocking_desc().inner_blks,
                       wei_d.blocking_desc().inner_nblks)
            && array_cmp(src_d.blocking_desc().inner_idxs,
                       wei_d.blocking_desc().inner_idxs,
                       wei_d.blocking_desc().inner_nblks)
            && strides_compatible()
            && dst_d.matches_tag(format_tag::nc)
            && src_d.only_padded_dim(1)
            && wei_d.only_padded_dim(1)
            && src_d.padded_dims()[1] == wei_d.padded_dims()[1]
            && src_d.is_dense(true)
            && dst_d.is_dense()
            && wei_d.is_dense(true);
}

status_t template_set_default_params(memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t *bias_md, int ndims) {
    using namespace format_tag;

    auto matching_tag = [&](const memory_desc_t &md) {
        if (memory_desc_matches_one_of_tag(md, ba, cba, cdba, cdeba))
            return utils::pick(ndims - 2, ab, acb, acdb, acdeb);
        if (memory_desc_matches_one_of_tag(md, acb, acdb, acdeb))
            return utils::pick(ndims - 3, cba, cdba, cdeba);
        auto src_tag = memory_desc_matches_one_of_tag(md, ab, abc, abcd,
                abcde, aBcd16b, aBcde16b, aBcd8b, aBcde8b, aBcd4b, aBcde4b);
        return src_tag;
    };
    if (src_md.format_kind == format_kind::any
            && weights_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(
                src_md, utils::pick(ndims - 2, nc, ncw, nchw, ncdhw)));
        CHECK(memory_desc_init_by_tag(weights_md,
                utils::pick(ndims - 2, oi, oiw, oihw, oidhw)));
    } else if (src_md.format_kind == format_kind::any)
         CHECK(memory_desc_init_by_tag(src_md, matching_tag(weights_md)));
    else if (weights_md.format_kind == format_kind::any)
         CHECK(memory_desc_init_by_tag(weights_md, matching_tag(src_md)));
    if (dst_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md, nc));
    if (bias_md->format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(*bias_md, x));
    return status::success;
}

} // namespace

struct ocl_inner_product_fwd_pd_t : public inner_product_fwd_pd_t {
    using inner_product_fwd_pd_t::inner_product_fwd_pd_t;
protected:
    status_t set_default_params() {
        return template_set_default_params(src_md_, weights_md_, dst_md_,
                &bias_md_, ndims());
    }

};

struct ocl_inner_product_bwd_data_pd_t : public inner_product_bwd_data_pd_t {
    using inner_product_bwd_data_pd_t::inner_product_bwd_data_pd_t;
protected:
    status_t set_default_params() {
        return template_set_default_params(diff_src_md_, weights_md_,
                diff_dst_md_, &glob_zero_md, ndims());
    }
};

struct ocl_inner_product_bwd_weights_pd_t
    : public inner_product_bwd_weights_pd_t {
    using inner_product_bwd_weights_pd_t::inner_product_bwd_weights_pd_t;
protected:
    status_t set_default_params() {
        return template_set_default_params(src_md_, diff_weights_md_,
                diff_dst_md_, &diff_bias_md_, ndims());
    }
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
