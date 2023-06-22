/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_SYCL_REF_ELTWISE_HPP
#define GPU_SYCL_REF_ELTWISE_HPP

#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/sycl/sycl_gpu_primitive.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct ref_sycl_eltwise_fwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_eltwise_fwd_pd_t {
        using gpu_eltwise_fwd_pd_t::gpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_sycl_eltwise_fwd_t);

        status_t init(engine_t *engine) {
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            const bool ok = is_fwd()
                    && check_data_types(
                            src_md()->data_type, dst_md()->data_type)
                    && (src_md()->format_desc.blocking.inner_nblks == 0)
                    && attr()->has_default_values(sm::post_ops)
                    && set_default_formats_common() && src_d == dst_d
                    && attr_.set_default_formats(dst_md(0)) == status::success
                    && post_ops_ok();

            if (!ok) return status::unimplemented;
            return init_conf();
        }

        sycl_eltwise_conf_t conf_;
        dim_t wg_thr;

    private:
        status_t init_conf();

        static bool check_data_types(
                const data_type_t &src_dt, const data_type_t &dst_dt) {
            using namespace data_type;

            for (auto t : {src_dt, dst_dt}) {
                if (!utils::one_of(t, f32, bf16, f16, s32, s8, u8))
                    return false;
            }

            return true;
        }

        bool post_ops_ok() const {
            for (int i = 0; i < attr()->post_ops_.len(); i++) {
                const auto &e = attr()->post_ops_.entry_[i];
                if (!IMPLICATION(e.is_binary(),
                            utils::one_of(e.binary.alg, alg_kind::binary_add,
                                    alg_kind::binary_div, alg_kind::binary_mul,
                                    alg_kind::binary_sub, alg_kind::binary_max,
                                    alg_kind::binary_min, alg_kind::binary_ge,
                                    alg_kind::binary_gt, alg_kind::binary_le,
                                    alg_kind::binary_lt, alg_kind::binary_eq,
                                    alg_kind::binary_ne))) {

                    return false;
                }
            }
            return attr()->post_ops_.len() <= sycl_post_ops_t::max_post_ops
                    && attr()->post_ops_.has_default_values(
                            {primitive_kind::eltwise, primitive_kind::binary,
                                    primitive_kind::sum});
        }
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward(const exec_ctx_t &ctx) const;
    compute::kernel_t kernel_;
};

struct ref_sycl_eltwise_bwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_eltwise_bwd_pd_t {
        using gpu_eltwise_bwd_pd_t::gpu_eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_sycl_eltwise_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());

            const bool ok = !is_fwd()
                    && check_data_types(data_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type)
                    && (data_md()->format_desc.blocking.inner_nblks == 0)
                    && attr()->has_default_values()
                    && set_default_formats_common() && diff_dst_d == diff_src_d;

            if (!ok) return status::unimplemented;
            return init_conf();
        }

        sycl_eltwise_conf_t conf_;
        dim_t wg_thr;

    private:
        status_t init_conf();

        static bool check_data_types(const data_type_t &data_dt,
                const data_type_t &diff_src_dt,
                const data_type_t &diff_dst_dt) {
            using namespace data_type;

            for (auto t : {data_dt, diff_src_dt, diff_dst_dt}) {
                if (!utils::one_of(t, f32, bf16, f16)) return false;
            }

            return true;
        }
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
