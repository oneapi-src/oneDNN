/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2020-2022 Codeplay Software Limited
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

#ifndef GPU_AMD_MIOPEN_REORDER_HPP
#define GPU_AMD_MIOPEN_REORDER_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/reorder_pd.hpp"
#include "gpu/amd/miopen_reorder_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_reorder_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public reorder_pd_t {
        using reorder_pd_t::reorder_pd_t;
        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_reorder_t);

        // Function to verify data and memory format.
        bool valid_data_n_mem_format() const {
            bool ok = src_md()->ndims == dst_md()->ndims
                    // MIOpen doesn't support conversion between data types.
                    && src_md()->data_type == dst_md()->data_type
                    && utils::one_of(src_md()->data_type, data_type::f32);

            // MIOpen only supports blocking for Int8
            if (!utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks > 0)
                return false;

            if (!utils::one_of(dst_md()->data_type, data_type::s8)
                    && dst_md()->format_desc.blocking.inner_nblks > 0)
                return false;

            // MIOpen supports blocking only on channel dimension C
            if (dst_md()->format_desc.blocking.inner_nblks > 1
                    || src_md()->format_desc.blocking.inner_nblks > 1)
                return false;

            if (utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks == 1) {
                ok = ok && memory_desc_matches_nchw_vect_c(src_md());
            }

            if (utils::one_of(dst_md()->data_type, data_type::s8)
                    && dst_md()->format_desc.blocking.inner_nblks == 1) {
                ok = ok && memory_desc_matches_nchw_vect_c(dst_md());
            }

            return ok;
        }

        bool scales_ok() const {
            const auto &scales = attr()->scales_;
            const auto &supported_args = {DNNL_ARG_FROM, DNNL_ARG_TO};
            if (!scales.has_default_values(supported_args)) return false;
            // MIOpen does not support scaling per dimension.
            for (auto arg : supported_args)
                if (scales.get(arg).mask_ != 0) return false;
            return true;
        }

        bool post_ops_ok() const {
            // only sum post-op is supported
            const auto &p = attr()->post_ops_;
            return p.len() == 0 || (p.len() == 1 && p.entry_[0].is_sum(false));
        }

        bool check_for_zero_dims() const {
            return has_zero_dims(src_md(0)->dims, src_md(0)->ndims)
                    || has_zero_dims(dst_md()->dims, dst_md()->ndims);
        }

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
            const auto attr_skip_mask
                    = primitive_attr_t::skip_mask_t::scales_runtime
                    | primitive_attr_t::skip_mask_t::post_ops;
            const bool ok = true && (engine == dst_engine)
                    && src_engine->kind() == engine_kind::gpu
                    && valid_data_n_mem_format()
                    && attr()->has_default_values(attr_skip_mask) && scales_ok()
                    && post_ops_ok();

            if (!ok) return status::unimplemented;

            reorder_.reset(new miopen_reorder_stride_t());
            return reorder_->init(this);
        }

        std::shared_ptr<miopen_reorder_generic_t> reorder_;

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            auto _pd = make_unique_pd<pd_t>(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            CHECK(_pd->init(engine, src_engine, dst_engine));
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd.release());
        }
        friend dnnl::impl::impl_list_item_t;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
