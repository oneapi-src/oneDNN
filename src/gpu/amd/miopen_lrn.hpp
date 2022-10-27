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

#ifndef GPU_AMD_MIOPEN_LRN_HPP
#define GPU_AMD_MIOPEN_LRN_HPP

#include <miopen/miopen.h>

#include "common/c_types_map.hpp"
#include "common/lrn_pd.hpp"
#include "common/primitive.hpp"
#include "gpu/amd/miopen_lrn_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_lrn_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public lrn_fwd_pd_t {
        using lrn_fwd_pd_t::lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_lrn_fwd_t);

        status_t init(engine_t *) {
            using namespace data_type;
            bool ok = is_fwd()
                    // MIOpen LRN implementation within channel supports only 2D spatial.
                    && IMPLICATION(
                            desc()->alg_kind == alg_kind::lrn_within_channel,
                            ndims() == 4)
                    && utils::one_of(src_md()->data_type, f32, f16)
                    && src_md()->data_type == dst_md()->data_type
                    && attr()->has_default_values() && desc_.local_size % 2
                    && set_default_formats_common() && check_format()
                    && memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md());
            if (!ok) return status::unimplemented;

            if (has_zero_dim_memory()) return status::success;

            lrn_impl_.reset(new miopen_lrn_fwd_impl_t());
            CHECK(lrn_impl_->init(this));
            return init_ws();
        }

        status_t init_ws() {
            if (!is_training()) return status::success;
            const size_t miopen_ws_size = lrn_impl_->get_workspace_size();
            const size_t dst_size = memory_desc_wrapper(dst_md()).size();
            const size_t ws_size = miopen_ws_size + dst_size;
            dims_t dims = {(dim_t)ws_size};
            return memory_desc_init_by_tag(ws_md_, ws_size ? 1 : 0, dims,
                    data_type::u8, format_tag::a);
        }

        bool check_format() const {
            // Only abx formats are supported
            return (memory_desc_wrapper(src_md()).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,
                            format_tag::abcd)
                    && memory_desc_wrapper(dst_md()).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,
                            format_tag::abcd));
        }

        bool is_training() const {
            return desc_.prop_kind == prop_kind::forward_training;
        }

        std::shared_ptr<miopen_lrn_impl_base_t> lrn_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct miopen_lrn_bwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public lrn_bwd_pd_t {
        using lrn_bwd_pd_t::lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_lrn_bwd_t);

        status_t init(engine_t *) {
            using namespace data_type;
            bool ok = !is_fwd()
                    // MIOpen LRN implementation within channel supports only 2D spatial.
                    && IMPLICATION(
                            desc()->alg_kind == alg_kind::lrn_within_channel,
                            ndims() == 4)
                    && utils::one_of(src_md()->data_type, f32, f16)
                    && utils::everyone_is(src_md()->data_type,
                            diff_dst_md()->data_type, diff_src_md()->data_type)
                    && set_default_formats_common()
                    && attr()->has_default_values() && desc_.local_size % 2
                    && check_format()
                    && memory_desc_wrapper(diff_src_md())
                            == memory_desc_wrapper(diff_dst_md());
            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) { return status::success; };

            lrn_impl_.reset(new miopen_lrn_bwd_impl_t());
            CHECK(lrn_impl_->init(this));
            CHECK(init_ws());
            if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            return status::success;
        }

        status_t init_ws() {
            const size_t miopen_ws_size = lrn_impl_->get_workspace_size();
            // XXX: assumption is that src_md() and dst_md() are always the same.
            const size_t dst_size = memory_desc_wrapper(src_md()).size();
            const size_t ws_size = miopen_ws_size + dst_size;

            dims_t dims = {(dim_t)ws_size};
            return memory_desc_init_by_tag(ws_md_, ws_size ? 1 : 0, dims,
                    data_type::u8, format_tag::a);
        }

        bool check_format() const {
            // Only abx formats are supported
            return (memory_desc_wrapper(src_md()).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,
                            format_tag::abcd)
                    && memory_desc_wrapper(diff_src_md())
                               .matches_one_of_tag(format_tag::a,
                                       format_tag::ab, format_tag::abc,
                                       format_tag::abcd)
                    && memory_desc_wrapper(diff_dst_md())
                               .matches_one_of_tag(format_tag::a,
                                       format_tag::ab, format_tag::abc,
                                       format_tag::abcd));
        }

        std::shared_ptr<miopen_lrn_impl_base_t> lrn_impl_;
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
