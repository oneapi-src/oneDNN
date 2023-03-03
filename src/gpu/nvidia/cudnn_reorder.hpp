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

#ifndef GPU_NVIDIA_CUDNN_REORDER_HPP
#define GPU_NVIDIA_CUDNN_REORDER_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/nvidia/cudnn_reorder_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_reorder_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public gpu_reorder_pd_t {
        using gpu_reorder_pd_t::gpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_reorder_t);

        // Function to verify data and memory format
        bool valid_data_n_mem_format(engine_t *engine) const {
            auto sycl_dev
                    = utils::downcast<impl::sycl::sycl_engine_base_t *>(engine)
                              ->device();

            bool ok = utils::one_of(src_md()->data_type, data_type::s8,
                              data_type::bf16, data_type::f16, data_type::f32)
                    && utils::one_of(dst_md()->data_type, data_type::s8,
                            data_type::bf16, data_type::f16, data_type::f32)
                    // f16<->bf16 cases are not supported.
                    && IMPLICATION(src_md()->data_type == data_type::bf16,
                            dst_md()->data_type != data_type::f16)
                    && IMPLICATION(src_md()->data_type == data_type::f16,
                            dst_md()->data_type != data_type::bf16)
                    && IMPLICATION(
                            utils::one_of(data_type::bf16, src_md()->data_type,
                                    dst_md()->data_type),
                            has_bf16_support(sycl_dev));

            if (!ok) return false;

            // Nvidia only supports blocking for Int8
            if (!utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks > 0)
                return false;
            if (!utils::one_of(dst_md()->data_type, data_type::s8)
                    && dst_md()->format_desc.blocking.inner_nblks > 0)
                return false;

            // Nvidia supports blocking only on channel dimension C
            if (dst_md()->format_desc.blocking.inner_nblks > 1
                    || src_md()->format_desc.blocking.inner_nblks > 1)
                return false;
            if (utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks == 1) {
                ok = ok && memory_desc_matches_nchw_vect_c(src_md());
            }
            int blks = dst_md()->format_desc.blocking.inner_nblks;
            if (utils::one_of(dst_md()->data_type, data_type::s8)
                    && blks == 1) {
                ok = ok && memory_desc_matches_nchw_vect_c(dst_md());
            }
            return ok;
        }

        bool scales_ok() const {
            const auto &scales = attr()->scales_;
            const auto &supported_args = {DNNL_ARG_FROM, DNNL_ARG_TO};
            if (!scales.has_default_values(supported_args)) return false;
            // cuDNN does not support scaling per dimension.
            for (auto arg : supported_args)
                if (scales.get(arg).mask_ != 0) return false;
            return true;
        }

        bool post_ops_ok() const {
            // only sum post-op is supported
            const auto &p = attr()->post_ops_;
            return p.len() == 0 || (p.len() == 1 && p.entry_[0].is_sum(false));
        }

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
            const auto attr_skip_mask
                    = primitive_attr_t::skip_mask_t::scales_runtime
                    | primitive_attr_t::skip_mask_t::post_ops;
            bool ok = engine == dst_engine
                    && src_engine->kind() == engine_kind::gpu
                    && valid_data_n_mem_format(engine)
                    && attr()->has_default_values(attr_skip_mask) && scales_ok()
                    && post_ops_ok();
            if (!ok) return status::unimplemented;

            if (has_different_block_size(src_md(), dst_md())) {
                reorder_.reset(new cudnn_reorder_ex_t());
            } else {
                reorder_.reset(new cudnn_reorder_stride_t());
            }

            return reorder_->init(this);
        }
        std::shared_ptr<cudnn_reorder_generic_t> reorder_;

    private:
        DECLARE_GPU_REORDER_CREATE();
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
