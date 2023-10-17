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

#ifndef GPU_NVIDIA_CUDNN_POOLING_HPP
#define GPU_NVIDIA_CUDNN_POOLING_HPP

#include "common/c_types_map.hpp"
#include "common/pooling_pd.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/nvidia/cudnn_pooling_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_pooling_common_t {
    template <typename pd_t>
    void init_ws(const pd_t *pd, memory_desc_t &ws_md) {
        memory_desc_wrapper src_wrap(pd->invariant_src_md());
        memory_desc_wrapper dst_wrap(pd->invariant_dst_md());

        const auto src_size = src_wrap.size();
        const auto dst_size = dst_wrap.size();
        const dims_t ws_size = {(dim_t)(src_size + dst_size)};

        memory_desc_init_by_tag(
                ws_md, 1, ws_size, data_type::u8, format_tag::x);
    }
};

struct cudnn_pooling_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public pooling_fwd_pd_t, public cudnn_pooling_common_t {
        using pooling_fwd_pd_t::pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_pooling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace format_tag;

            assert(engine->kind() == engine_kind::gpu);
            auto src_dt = src_md()->data_type;
            auto *sycl_engine
                    = utils::downcast<impl::sycl::sycl_engine_base_t *>(engine);

            bool ok = true && is_fwd()
                    && utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference)
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && utils::one_of(src_dt, s8, f16, f32, bf16)
                    && src_dt == dst_md()->data_type
                    && IMPLICATION(utils::one_of(src_dt, f16),
                            desc()->prop_kind == forward_inference)
                    && IMPLICATION(src_dt == s8, desc()->accum_data_type == s32)
                    && !is_dilated() && attr()->has_default_values()
                    && set_default_params() == status::success && blocking_ok()
                    && IMPLICATION(
                            utils::one_of(data_type::bf16, src_md()->data_type,
                                    dst_md()->data_type),
                            has_bf16_support(sycl_engine->device()));
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (is_training) init_ws(this, ws_md_);

            if (has_zero_dim_memory()) return status::success;

            pooling_impl_.reset(new cudnn_pooling_fwd_impl_t());
            return pooling_impl_->init(this);
        }

        bool blocking_ok() const {
            if (!utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks > 0)
                return false;

            if (src_md()->format_desc.blocking.inner_nblks > 1) return false;

            if (utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks == 1) {
                return memory_desc_matches_nchw_vect_c(src_md())
                        && memory_desc_matches_nchw_vect_c(dst_md());
            }

            return true;
        }

        std::shared_ptr<cudnn_pooling_impl_base_t> pooling_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cudnn_pooling_bwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public pooling_bwd_pd_t, public cudnn_pooling_common_t {
        using pooling_bwd_pd_t::pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_pooling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace format_tag;
            assert(engine->kind() == engine_kind::gpu);
            auto *sycl_engine
                    = utils::downcast<impl::sycl::sycl_engine_base_t *>(engine);

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
                                    diff_src_md()->data_type)
                            || utils::everyone_is(data_type::bf16,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type))
                    && !is_dilated() && attr()->has_default_values()
                    && no_blocking()
                    && IMPLICATION(utils::one_of(data_type::bf16,
                                           diff_dst_md()->data_type,
                                           diff_src_md()->data_type),
                            has_bf16_support(sycl_engine->device()));
            if (!ok) return status::unimplemented;

            init_ws(this, ws_md_);
            if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;

            if (has_zero_dim_memory()) { return status::success; };

            pooling_impl_.reset(new cudnn_pooling_bwd_impl_t());
            return pooling_impl_->init(this);
        }

        bool no_blocking() const {
            return diff_src_md()->format_desc.blocking.inner_nblks
                    + diff_dst_md()->format_desc.blocking.inner_nblks
                    == 0;
        }

        std::shared_ptr<cudnn_pooling_impl_base_t> pooling_impl_;
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
