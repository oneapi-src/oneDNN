/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef GPU_AMD_MIOPEN_POOLING_HPP
#define GPU_AMD_MIOPEN_POOLING_HPP

#include "common/c_types_map.hpp"
#include "common/pooling_pd.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/amd/miopen_pooling_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_pooling_common_t {

    template <typename pd_t>
    void init_ws(const pd_t *pd, memory_desc_t &ws_md, size_t ws_size_miopen) {
        bool is_fwd = pd->is_fwd();
        memory_desc_wrapper src_wrap(is_fwd ? pd->src_md() : pd->diff_src_md());
        memory_desc_wrapper dst_wrap(is_fwd ? pd->dst_md() : pd->diff_dst_md());

        const auto src_size = src_wrap.size();
        const auto dst_size = dst_wrap.size();

        const size_t ws_size = src_size + dst_size + ws_size_miopen;
        dims_t dims = {(dim_t)ws_size};

        memory_desc_init_by_tag(
                ws_md, ws_size ? 1 : 0, dims, data_type::u8, format_tag::a);
    }
};

struct miopen_pooling_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public pooling_fwd_pd_t, public miopen_pooling_common_t {
        using pooling_fwd_pd_t::pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_pooling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace format_tag;
            assert(engine->kind() == engine_kind::gpu);
            auto src_dt = src_md()->data_type;

            bool ok = true && is_fwd()
                    && utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference)
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && utils::one_of(src_dt, f16, f32)
                    && src_dt == dst_md()->data_type
                    && IMPLICATION(utils::one_of(src_dt, f16),
                            desc()->prop_kind == forward_inference)
                    && !is_dilated() && attr()->has_default_values()
                    && set_default_params() == status::success
                    && check_format();

            if (!ok) return status::unimplemented;

            if (has_zero_dim_memory()) return status::success;

            pooling_impl_.reset(new miopen_pooling_fwd_impl_t());
            CHECK(pooling_impl_->init(this));

            if (is_training())
                init_ws(this, ws_md_, pooling_impl_->get_ws_size_miopen());

            return status::success;
        }

        bool check_format() const {
            // Only abx formats are supported
            return (memory_desc_wrapper(src_md()).matches_one_of_tag(
                            format_tag::abc, format_tag::abcd,
                            format_tag::abcde)
                    && memory_desc_wrapper(dst_md()).matches_one_of_tag(
                            format_tag::abc, format_tag::abcd,
                            format_tag::abcde));
        }

        bool is_training() const {
            return desc_.prop_kind == prop_kind::forward_training;
        }

        std::shared_ptr<miopen_pooling_impl_base_t> pooling_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct miopen_pooling_bwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public pooling_bwd_pd_t, public miopen_pooling_common_t {
        using pooling_bwd_pd_t::pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_pooling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace format_tag;
            assert(engine->kind() == engine_kind::gpu);
            bool ok = true && !is_fwd()
                    && set_default_params() == status::success
                    && desc()->prop_kind == backward_data
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && (utils::everyone_is(data_type::f32,
                                diff_dst_md()->data_type,
                                diff_src_md()->data_type)
                            || utils::everyone_is(data_type::f16,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type))
                    && !is_dilated() && attr()->has_default_values()
                    && check_format();

            if (!ok) return status::unimplemented;

            if (has_zero_dim_memory()) { return status::success; };
            pooling_impl_.reset(new miopen_pooling_bwd_impl_t());
            CHECK(pooling_impl_->init(this));

            init_ws(this, ws_md_, pooling_impl_->get_ws_size_miopen());
            if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            return status::success;
        }

        bool check_format() const {
            // Only abx formats are supported
            return (memory_desc_wrapper(diff_src_md())
                            .matches_one_of_tag(format_tag::abc,
                                    format_tag::abcd, format_tag::abcde)
                    && memory_desc_wrapper(diff_dst_md())
                               .matches_one_of_tag(format_tag::abc,
                                       format_tag::abcd, format_tag::abcde));
        }

        std::shared_ptr<miopen_pooling_impl_base_t> pooling_impl_;
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
