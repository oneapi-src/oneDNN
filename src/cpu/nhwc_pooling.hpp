/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CPU_NHWC_POOLING_HPP
#define CPU_NHWC_POOLING_HPP

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

namespace nhwc_pooling {
size_t strided_offset(const int _n, const size_t _sn, const int _d,
        const size_t _sd, const int _h, const size_t _sh, const int _w,
        const size_t _sw);
}

template <data_type_t d_type>
struct nhwc_pooling_fwd_t : public primitive_impl_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nhwc:any", nhwc_pooling_fwd_t);

        status_t init() {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

            using namespace prop_kind;
            using namespace alg_kind;
            bool ok = true
                    && IMPLICATION(
                            d_type == data_type::bf16, mayiuse(avx512_core))
                    && set_default_params() == status::success && is_fwd()
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && utils::everyone_is(
                            d_type, src_md()->data_type, dst_md()->data_type)
                    && attr()->has_default_values()
                    && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
                    && memory_desc_matches_tag(*dst_md(), desired_fmt_tag);
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training) {
                init_default_ws();
            }

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (src_md()->data_type == data_type::bf16) {
                size_t bf16cvt_sz_ = C() * dnnl_get_max_threads();
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(
                        key_pool_src_bf16cvt, sizeof(float) * bf16cvt_sz_);
                scratchpad.book(
                        key_pool_dst_bf16cvt, sizeof(float) * bf16cvt_sz_);
            }
        }
    };

    nhwc_pooling_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    typedef typename prec_traits<d_type>::type data_t;
    typedef typename prec_traits<data_type::f32>::type ker_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void array_div_by_const(const int n, const ker_data_t *src,
            const size_t num, ker_data_t *dst) const;
    void array_add(const int n, const ker_data_t *src, ker_data_t *dst) const;

    template <bool use_workspace>
    void array_nhwc_max(const int n, ker_data_t *dst, const ker_data_t *src,
            unsigned char *ws, const size_t ws_offset, const data_type_t ws_dt,
            const int index) const {
        assert(!((use_workspace == false) ^ (!ws))); // ensure ws pointer exists
        PRAGMA_OMP_SIMD()
        for (int oc = 0; oc < n; ++oc) {
            auto s = src[oc];
            ker_data_t mv = dst[oc];

            // update index of maximum
#if defined __INTEL_COMPILER
            if ((use_workspace) && (s > mv)) {
                // if (ws && (s > mv)) {
                assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
                if (ws_dt == data_type::u8) {
                    assert(0 <= index && index <= 255);
                    ws[ws_offset + oc] = index;
                } else
                    reinterpret_cast<int *>(ws)[ws_offset + oc] = index;
            }
#else
            // Need to add explicit predicates for GCC to vectorize this.
            // And although the resulting code is ugly, it is still 4 times
            // faster than scalar
            if (use_workspace) {
                // if (ws) {
                assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);

                if (ws_dt == data_type::u8) {
                    assert(0 <= index && index <= 255);
                    unsigned char predicate = (s > mv) ? 0xff : 0;
                    unsigned char current_value = ws[ws_offset + oc];
                    current_value = (predicate & (unsigned char)index)
                            | ((~predicate) & current_value);
                    ws[ws_offset + oc] = current_value;
                } else {
                    auto wint = reinterpret_cast<int *>(ws);
                    unsigned int predicate = (s > mv) ? 0xffffffff : 0;
                    unsigned int current_value = wint[ws_offset + oc];
                    current_value = (predicate & (unsigned int)index)
                            | ((~predicate) & current_value);
                    wint[ws_offset + oc] = current_value;
                }
            }
#endif
            // update maximum
            dst[oc] = nstl::max(s, mv);
        }
    }

    template <bool use_workspace>
    void array_nhwc_initialize(const int n, ker_data_t *dst, unsigned char *ws,
            const size_t ws_offset, const data_type_t ws_dt) const {
        assert(!((use_workspace == false) ^ (!ws))); // ensure ws pointer exists
        for (int oc = 0; oc < n; ++oc) {
            if (use_workspace) {
                assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
                if (ws_dt == data_type::u8) {
                    ws[ws_offset + oc] = 0;
                } else
                    reinterpret_cast<int *>(ws)[ws_offset + oc] = 0;
            }
            dst[oc] = nstl::numeric_limits<data_t>::lowest();
        }
    }

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

template <impl::data_type_t d_type>
struct nhwc_pooling_bwd_t : public primitive_impl_t {
    struct pd_t : public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nhwc:any", nhwc_pooling_bwd_t);

        status_t init() {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

            using namespace prop_kind;
            using namespace alg_kind;
            bool ok = true
                    && IMPLICATION(
                            d_type == data_type::bf16, mayiuse(avx512_core))
                    && set_default_params() == status::success && !is_fwd()
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && utils::everyone_is(d_type, diff_dst_md()->data_type,
                            diff_src_md()->data_type)
                    && attr()->has_default_values()
                    && memory_desc_matches_tag(*diff_dst_md(), desired_fmt_tag)
                    && memory_desc_matches_tag(*diff_src_md(), desired_fmt_tag);
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == pooling_max) {
                init_default_ws();
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (diff_src_md()->data_type == data_type::bf16) {
                size_t bf16cvt_sz_ = C() * dnnl_get_max_threads();
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(
                        key_pool_src_bf16cvt, sizeof(float) * bf16cvt_sz_);
                scratchpad.book(
                        key_pool_dst_bf16cvt, sizeof(float) * bf16cvt_sz_);
            }
        }
    };

    nhwc_pooling_bwd_t(const pd_t *apd) : primitive_impl_t(apd) {}
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
