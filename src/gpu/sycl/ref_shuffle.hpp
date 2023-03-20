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

#ifndef GPU_SYCL_REF_SHUFFLE_HPP
#define GPU_SYCL_REF_SHUFFLE_HPP

#include "gpu/gpu_shuffle_pd.hpp"
#include "gpu/sycl/sycl_gpu_primitive.hpp"
#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "gpu/sycl/sycl_types.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct ref_shuffle_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_shuffle_pd_t {
        using gpu_shuffle_pd_t::gpu_shuffle_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_shuffle_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;
            const memory_desc_wrapper data_d(src_md(0));
            const memory_desc_wrapper dst_d(dst_md(0));

            const bool ok = src_md()->data_type == dst_md()->data_type
                    && (src_md()->data_type == bf16 || src_md()->data_type == s8
                            || src_md()->data_type == f16
                            || src_md()->data_type == f32)
                    && (src_md(0)->format_desc.blocking.inner_nblks == 0)
                    && attr()->has_default_values()
                    && set_default_formats_common()

                    && IMPLICATION(!is_fwd(), set_default_formats_common());
            if (!ok) return status::unimplemented;
            return init_conf();
        }

        status_t init_conf();
        sycl_shuffle_conf_t conf_;
    };

    status_t init(engine_t *engine) override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute(const exec_ctx_t &ctx) const override;
    compute::kernel_t kernel_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
