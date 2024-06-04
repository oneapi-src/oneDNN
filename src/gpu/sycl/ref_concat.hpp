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

#ifndef GPU_SYCL_REF_CONCAT_HPP
#define GPU_SYCL_REF_CONCAT_HPP

#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/reorder_pd.hpp"
#include "gpu/gpu_concat_pd.hpp"
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

struct ref_concat_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_concat_pd_t {
        using gpu_concat_pd_t::gpu_concat_pd_t;

        DECLARE_CONCAT_PD_t("dpcpp:ref:any", ref_concat_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper dst_d(dst_md());

            const bool ok = set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            const int n = n_inputs();
            reorder_pds_.resize(n);
            reorder_mds_.resize(n);
            dims_t offset = {0, 0, 0, 0, 0, 0};
            for (auto i = 0; i < n; ++i) {
                const memory_desc_wrapper src_d_n(src_md(i));

                CHECK(memory_desc_init_submemory(
                        reorder_mds_[i], *dst_md(), src_d_n.dims(), offset));
                primitive_attr_t r_attr;
                CHECK(reorder_primitive_desc_create(reorder_pds_[i], engine,
                        src_md(i), &reorder_mds_[i], &r_attr));
                offset[concat_dim()] += src_d_n.dims()[concat_dim()];
            }

            return init_conf();
        }

        sycl_concat_conf_t conf_;
        std::vector<memory_desc_t> reorder_mds_;
        std::vector<std::shared_ptr<primitive_desc_t>> reorder_pds_;

    private:
        status_t init_conf();
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::shared_ptr<primitive_t>> reorders_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
