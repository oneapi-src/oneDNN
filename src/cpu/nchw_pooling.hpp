/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#ifndef CPU_NCHW_POOLING_HPP
#define CPU_NCHW_POOLING_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_isa_traits.hpp"
#include "cpu_pooling_pd.hpp"

#include "bfloat16.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t d_type>
struct nchw_pooling_fwd_t : public primitive_impl_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nchw:any", nchw_pooling_fwd_t);

        status_t init() {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::ncw, format_tag::nchw, format_tag::ncdhw);

            bool ok = true
                    && IMPLICATION(
                            d_type == data_type::bf16, mayiuse(avx512_core))
                    && set_default_params() == status::success && is_fwd()
                    && utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                            alg_kind::pooling_avg_include_padding,
                            alg_kind::pooling_avg_exclude_padding)
                    && !has_zero_dim_memory()
                    && utils::everyone_is(
                            d_type, src_md()->data_type, dst_md()->data_type)
                    && attr()->has_default_values()
                    && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
                    && memory_desc_matches_tag(*dst_md(), desired_fmt_tag);
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == prop_kind::forward_training;
            if (desc()->alg_kind == alg_kind::pooling_max && is_training)
                init_default_ws();

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (src_md()->data_type == data_type::bf16) {
                size_t src_sz_ = ID() * IH() * IW() * C() * MB();
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(key_pool_src_bf16cvt, sizeof(float) * src_sz_);
            }
        }
    };

    nchw_pooling_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    typedef typename prec_traits<d_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

template <data_type_t d_type>
struct nchw_pooling_bwd_t : public primitive_impl_t {
    struct pd_t : public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nchw:any", nchw_pooling_bwd_t);

        status_t init() {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::ncw, format_tag::nchw, format_tag::ncdhw);

            using namespace prop_kind;
            using namespace alg_kind;
            bool ok = true
                    && IMPLICATION(
                            d_type == data_type::bf16, mayiuse(avx512_core))
                    && set_default_params() == status::success && !is_fwd()
                    && utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                            alg_kind::pooling_avg_include_padding,
                            alg_kind::pooling_avg_exclude_padding)
                    && !has_zero_dim_memory()
                    && utils::everyone_is(d_type, diff_dst_md()->data_type,
                            diff_src_md()->data_type)
                    && attr()->has_default_values()
                    && memory_desc_matches_tag(*diff_dst_md(), desired_fmt_tag)
                    && memory_desc_matches_tag(*diff_src_md(), desired_fmt_tag);
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == pooling_max) {
                bool ws_ok
                        = true && hint_fwd_pd_ && hint_fwd_pd_->workspace_md();
                if (!ws_ok) return status::unimplemented;

                const auto &ws_blk
                        = hint_fwd_pd_->workspace_md()->format_desc.blocking;
                ws_ok = ws_ok && ws_blk.inner_nblks <= 1
                        && IMPLICATION(ws_blk.inner_nblks == 1,
                                ws_blk.inner_idxs[0] == 1);
                if (!ws_ok) return status::unimplemented;

                ws_md_ = *hint_fwd_pd_->workspace_md();
            }

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (diff_dst_md()->data_type == data_type::bf16) {
                size_t dst_sz_ = OD() * OH() * OW();
                size_t src_sz_ = ID() * IH() * IW();
                size_t nthrs = dnnl_get_max_threads();
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(
                        key_pool_src_bf16cvt, sizeof(float) * src_sz_ * nthrs);
                scratchpad.book(
                        key_pool_dst_bf16cvt, sizeof(float) * dst_sz_ * nthrs);
            }
        }
    };

    nchw_pooling_bwd_t(const pd_t *apd) : primitive_impl_t(apd) {}
    typedef typename prec_traits<d_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward(ctx);
        return status::success;
    }

private:
    void execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
