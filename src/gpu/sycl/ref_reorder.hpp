/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_SYCL_REF_REORDER_HPP
#define GPU_SYCL_REF_REORDER_HPP

#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/sycl/sycl_gpu_primitive.hpp"
#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct ref_reorder_t : public gpu::sycl::primitive_t {
    using gpu::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_reorder_pd_t {
        using gpu_reorder_pd_t::gpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_reorder_t);

        status_t init(impl::engine_t *engine, impl::engine_t *src_engine,
                impl::engine_t *dst_engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            const bool ok = check_data_types(src_d, dst_d)
                    && check_formats(src_d, dst_d)
                    && attr()->has_default_values(
                            sm::scales_runtime | sm::post_ops)
                    && post_ops_ok();
            if (!ok) return status::unimplemented;

            for (int i = 0; i < dst_d.ndims(); i++) {
                if (dst_d.dims()[i] > INT_MAX) { return status::unimplemented; }
            }

            return init_conf();
        }

        sycl_reorder_conf_t conf_;

    private:
        DECLARE_GPU_REORDER_CREATE();

        status_t init_conf();

        bool post_ops_ok() const {
            for (int i = 0; i < attr()->post_ops_.len(); i++) {
                if (!attr()->post_ops_.entry_[i].is_sum()) { return false; }
            }
            return attr()->post_ops_.len() <= sycl_post_ops_t::max_post_ops
                    && attr()->post_ops_.has_default_values(
                            {primitive_kind::sum});
        }

        static bool check_data_types(const memory_desc_wrapper &src,
                const memory_desc_wrapper &dst) {
            using namespace data_type;

            const auto src_dt = src.data_type();
            const auto dst_dt = dst.data_type();

            for (auto t : {src_dt, dst_dt}) {
                if (!utils::one_of(t, f32, bf16, f16, s8, u8)) return false;
            }

            return true;
        }

        static bool check_formats(const memory_desc_wrapper &src,
                const memory_desc_wrapper &dst) {
            using namespace format_tag;

            for (const auto &mdw : {src, dst}) {
                if (!mdw.is_plain()) { return false; }
            }
            return true;
        }
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
