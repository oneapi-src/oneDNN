/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_REF_SHUFFLE_HPP
#define GPU_GENERIC_SYCL_REF_SHUFFLE_HPP

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_shuffle_pd.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_shuffle_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_shuffle_pd_t {
        using gpu_shuffle_pd_t::gpu_shuffle_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_shuffle_t);

        status_t init(impl::engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;
            auto src_data_md = invariant_src_md();
            auto dst_data_md = invariant_dst_md();

            VDISPATCH_SHUFFLE(src_data_md->data_type == dst_data_md->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_SHUFFLE(
                    (utils::one_of(src_data_md->data_type, bf16, f16, f32)
                            || (is_fwd()
                                    && utils::one_of(src_data_md->data_type,
                                            s32, s8, u8))),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SHUFFLE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SHUFFLE(
                    IMPLICATION(is_fwd(),
                            src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_SHUFFLE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SHUFFLE(memory_desc_wrapper(src_data_md)
                            == memory_desc_wrapper(dst_data_md),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_SHUFFLE(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");
            return init_conf();
        }

        status_t init_conf();
        sycl_shuffle_conf_t conf_;
    };

    status_t init(impl::engine_t *engine) override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute(const exec_ctx_t &ctx) const override;
    kernel_t kernel_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
