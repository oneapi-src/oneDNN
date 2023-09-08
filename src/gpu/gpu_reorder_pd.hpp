/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_reorder_pd_t : public reorder_pd_t {
    using reorder_pd_t::reorder_pd_t;

protected:
    bool attr_ok() const {
        return attr()->has_default_values(
                       dnnl_primitive_attr::skip_mask_t::zero_points_runtime
                       | dnnl_primitive_attr::skip_mask_t::scales_runtime
                       | dnnl_primitive_attr::skip_mask_t::post_ops)
                && post_ops_ok();
    }

    bool post_ops_ok() const {
        const auto &post_ops = attr()->post_ops_;
        return post_ops.len() == 0
                || (post_ops.len() == 1
                        && post_ops.entry_[0].kind == primitive_kind::sum);
    }

    bool extra_ok() const {
        return src_md()->extra.flags == 0 && dst_md()->extra.flags == 0;
    }
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#define DECLARE_GPU_REORDER_CREATE() \
    static status_t create(reorder_pd_t **reorder_pd, engine_t *engine, \
            const primitive_attr_t *attr, engine_t *src_engine, \
            const memory_desc_t *src_md, engine_t *dst_engine, \
            const memory_desc_t *dst_md) { \
        auto _pd = make_unique_pd<pd_t>( \
                attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md); \
        if (_pd == nullptr) return status::out_of_memory; \
        CHECK(_pd->init(engine, src_engine, dst_engine)); \
        CHECK(_pd->init_scratchpad_md()); \
        return safe_ptr_assign(*reorder_pd, _pd.release()); \
    } \
    friend dnnl::impl::impl_list_item_t;

#endif
