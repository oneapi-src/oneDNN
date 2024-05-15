/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_SYCL_REF_BINARY_HPP
#define GPU_SYCL_REF_BINARY_HPP

#include "gpu/gpu_binary_pd.hpp"
#include "gpu/sycl/sycl_gpu_primitive.hpp"
#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "sycl/sycl_stream.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct ref_binary_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_binary_pd_t {
        using gpu_binary_pd_t::gpu_binary_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_binary_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper src0_d(src_md(0));
            const memory_desc_wrapper src1_d(src_md(1));
            const memory_desc_wrapper dst_d(dst_md());

            const bool ok = set_default_params() == status::success
                    && attr_.set_default_formats(dst_md()) == status::success
                    && check_data_types(src0_d, src1_d, dst_d)
                    && check_formats(src0_d, src1_d, dst_d)
                    && attr()->has_default_values(
                            sm::scales_runtime | sm::post_ops)
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask())
                    && post_ops_ok();
            if (!ok) return status::unimplemented;

            return init_conf();
        }

        sycl_binary_conf_t conf_;

    private:
        status_t init_conf();

        bool check_scales_mask() const {
            const std::vector<int> supported_args
                    = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1};
            return attr_scales_ok(supported_args);
        }

        bool post_ops_ok() const {
            // Dw conv post-ops are not supported.
            return attr()->post_ops_.len() <= sycl_post_ops_t::max_post_ops
                    && attr()->post_ops_.has_default_values(
                            {primitive_kind::eltwise, primitive_kind::binary,
                                    primitive_kind::prelu,
                                    primitive_kind::sum});
        }

        static bool check_data_types(const memory_desc_wrapper &src0,
                const memory_desc_wrapper &src1,
                const memory_desc_wrapper &dst) {
            using namespace data_type;

            const auto src0_dt = src0.data_type();
            const auto src1_dt = src1.data_type();
            const auto dst_dt = dst.data_type();

            for (auto t : {src0_dt, src1_dt, dst_dt}) {
                if (!utils::one_of(t, f32, bf16, f16, s8, u8)) return false;
            }

            return IMPLICATION(utils::one_of(bf16, src0_dt, src1_dt, dst_dt),
                    src0_dt == dst_dt && src1_dt == dst_dt);
        }

        static bool check_formats(const memory_desc_wrapper &src0,
                const memory_desc_wrapper &src1,
                const memory_desc_wrapper &dst) {
            using namespace format_tag;

            for (const auto &mdw : {src0, src1, dst}) {
                if (mdw.matches_one_of_tag(a, ab, abc, abcd, abcde) == undef) {
                    return false;
                }
            }
            return true;
        }
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    intel::compute::kernel_t kernel_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
