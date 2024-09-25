/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef CPU_MATMUL_REF_MATMUL_INT8_HPP
#define CPU_MATMUL_REF_MATMUL_INT8_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

struct ref_matmul_int8_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ref_int8:any", ref_matmul_int8_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto bia_type = weights_md(1)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            VDISPATCH_MATMUL(
                    is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(
                    utils::one_of(src_type, s8, u8), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(utils::one_of(wei_type, s8, u8, s4, u4),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(IMPLICATION(with_bias(),
                                     utils::one_of(bia_type, f32, bf16, f16,
                                             s32, s8, u8)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(
                    utils::one_of(dst_type, f32, bf16, f16, s32, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(
                    attr()->has_default_values(smask_t::scales_runtime_data_type
                                    | smask_t::scales_runtime_groups
                                    | smask_t::zero_points_runtime_data_type
                                    | smask_t::zero_points_runtime_groups
                                    | smask_t::post_ops | smask_t::sum_dt,
                            dst_type),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(attr_.post_ops_.check_sum_consistency(dst_type,
                                     /* is_int8 */ true),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_MATMUL(
                    ref_post_ops_t::primitive_kind_ok(attr()->post_ops_),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_MATMUL(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_MATMUL(attr_zero_points_ok(), VERBOSE_UNSUPPORTED_ZP_CFG);
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            return status::success;
        }

    private:
        bool attr_zero_points_ok() const {
            int mask_src = 0, mask_wei = 0, mask_dst = 0;
            CHECK_BOOL(attr()->zero_points_.get(DNNL_ARG_SRC, &mask_src));
            CHECK_BOOL(attr()->zero_points_.get(DNNL_ARG_WEIGHTS, &mask_wei));
            CHECK_BOOL(attr()->zero_points_.get(DNNL_ARG_DST, &mask_dst));

            const auto src_group_ndims
                    = attr()->zero_points_.get_groups_ndims(DNNL_ARG_SRC);
            const auto src_group_dims
                    = attr()->zero_points_.get_groups(DNNL_ARG_SRC);
            const bool src_m_group_ok
                    = IMPLICATION(src_group_ndims == 2, src_group_dims[0] == 1);
            const bool src_k_group_ok
                    = IMPLICATION(src_group_ndims == 2 && src_group_dims[1] > 1,
                            K() % src_group_dims[1] == 0);

            const auto wei_group_ndims
                    = attr()->zero_points_.get_groups_ndims(DNNL_ARG_WEIGHTS);
            const auto wei_group_dims
                    = attr()->zero_points_.get_groups(DNNL_ARG_WEIGHTS);
            const bool wei_k_group_ok
                    = IMPLICATION(wei_group_ndims == 2 && wei_group_dims[0] > 1,
                            K() % wei_group_dims[0] == 0);
            const bool wei_n_group_ok
                    = IMPLICATION(wei_group_ndims == 2 && wei_group_dims[1] > 1,
                            N() % wei_group_dims[1] == 0);

            bool mask_src_ok = utils::one_of(
                    mask_src, 0, src_qmask_K(), src_qmask_M() + src_qmask_K());
            bool mask_dst_ok = utils::one_of(mask_dst, 0, wei_qmask_N());

            return mask_src_ok && mask_dst_ok
                    && utils::one_of(wei_group_ndims, 0, 2)
                    && IMPLICATION(wei_group_ndims == 2,
                            utils::one_of(
                                    1, wei_group_dims[0], wei_group_dims[1])
                                    && wei_k_group_ok && wei_n_group_ok)
                    && IMPLICATION(src_group_ndims == 2,
                            utils::one_of(
                                    1, src_group_dims[0], src_group_dims[1])
                                    && src_m_group_ok && src_k_group_ok);
        }
    };

    ref_matmul_int8_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
