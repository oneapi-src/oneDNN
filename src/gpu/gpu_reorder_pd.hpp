/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_GPU_REORDER_PD_HPP
#define GPU_GPU_REORDER_PD_HPP

#include "common/reorder_pd.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_reorder_pd_t : public reorder_pd_t {
    using reorder_pd_t::reorder_pd_t;

protected:
    bool attr_ok() const {
        using sm = dnnl_primitive_attr::skip_mask_t;
        return attr()->has_default_values(sm::zero_points_runtime
                       | sm::scales_runtime | sm::post_ops)
                && post_ops_ok() && zero_points_ok();
    }

    bool zero_points_ok() const {
        const auto &zp = attr()->zero_points_;

        using namespace data_type;
        bool ok = IMPLICATION(!utils::one_of(src_md()->data_type, s8, u8),
                zp.has_default_values(DNNL_ARG_SRC));
        if (!ok) return false;
        ok = IMPLICATION(!utils::one_of(dst_md()->data_type, s8, u8),
                zp.has_default_values(DNNL_ARG_DST));
        if (!ok) return false;

        if (!zp.has_default_values(DNNL_ARG_SRC)) {
            int mask_src = zp.get_mask(DNNL_ARG_SRC);
            ok = mask_src == 0;
            if (!ok) return false;
        }
        if (!zp.has_default_values(DNNL_ARG_DST)) {
            int mask_dst = zp.get_mask(DNNL_ARG_DST);
            ok = mask_dst == 0;
            if (!ok) return false;
        }

        return true;
    }

    bool post_ops_ok() const {
        const auto &post_ops = attr()->post_ops_;
        return post_ops.len() == 0
                || (post_ops.len() == 1
                        && post_ops.entry_[0].kind == primitive_kind::sum);
    }

    bool extra_ok(bool accept_conv_asymm = false) const {
        if (!accept_conv_asymm)
            return (src_md()->extra.flags == memory_extra_flags::none)
                    && (dst_md()->extra.flags == memory_extra_flags::none);
        return check_md_extra_flags_compensation_gpu(src_md()->extra.flags)
                && check_md_extra_flags_compensation_gpu(dst_md()->extra.flags);
    }

    status_t maybe_create_zp_precompute_conv_pd(impl::engine_t *dst_engine);

public:
    status_t maybe_create_zp_precompute_conv(
            std::shared_ptr<impl::primitive_t> &zp_precomp_conv,
            impl::engine_t *engine, gpu::primitive_t *primitive) const;

    status_t maybe_exec_zp_precompute_conv(const exec_ctx_t &ctx,
            const std::shared_ptr<impl::primitive_t> &zp_precomp_conv) const;

private:
    bool do_zp_precomp_conv_ = false;
    std::shared_ptr<primitive_desc_t> zp_precomp_conv_pd_;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#define DECLARE_GPU_REORDER_CREATE() \
    static status_t create(reorder_pd_t **reorder_pd, \
            dnnl::impl::engine_t *engine, const primitive_attr_t *attr, \
            dnnl::impl::engine_t *src_engine, const memory_desc_t *src_md, \
            dnnl::impl::engine_t *dst_engine, const memory_desc_t *dst_md) { \
        auto _pd = make_unique_pd<pd_t>( \
                attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md); \
        if (_pd == nullptr) return status::out_of_memory; \
        CHECK(_pd->init(engine, src_engine, dst_engine)); \
        CHECK(_pd->init_scratchpad_md()); \
        return safe_ptr_assign(*reorder_pd, _pd.release()); \
    } \
    friend dnnl::impl::impl_list_item_t;

#endif
