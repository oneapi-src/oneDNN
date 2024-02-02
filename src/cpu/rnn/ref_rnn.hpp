/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

#ifndef CPU_RNN_REF_RNN_HPP
#define CPU_RNN_REF_RNN_HPP

#include <assert.h>
#include <tuple>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm/os_blas.hpp"

#include "cpu/rnn/cpu_rnn_pd.hpp"
#include "cpu/rnn/postgemm_dispatcher.hpp"
#if DNNL_X64
#include "cpu/x64/rnn/rnn_brgemm_utils.hpp"
#endif
#include "cpu/rnn/rnn_utils.hpp"
namespace dnnl {
namespace impl {
namespace cpu {

namespace {
template <typename gates_t, typename acc_t>
// The loop body needs to be put in a function as some versions of icc have
// an issue with lambdas & macros inside omp simd loops
inline void body_loop(int i, int k, const gates_t *ws_gates, acc_t *diff_bias,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position) {
    if (rnn.diff_weights_overwrite && (cell_position & rnn_utils::last_iter))
        diff_bias[i * rnn.dhc + k] = 0.0f;
    for (int j = 0; j < rnn.mb; j++)
        diff_bias[i * rnn.dhc + k]
                += ws_gates[j * rnn.scratch_gates_ld + i * rnn.dhc + k];
}
} // namespace

template <typename gates_t, typename acc_t>
void gates_reduction(const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const gates_t *ws_gates_,
        acc_t *diff_bias_) {

    // @todo block k on simd-width to enable vectorization in
    // parallel_nd path
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP && _OPENMP >= 201307 \
        && (!defined(__INTEL_COMPILER) || __INTEL_COMPILER < 1910)
#pragma omp parallel for simd collapse(2)
    for (int i = 0; i < rnn.n_gates; i++)
        for (int k = 0; k < rnn.dhc; k++)
            body_loop(i, k, ws_gates_, diff_bias_, rnn, cell_position);
#else
    parallel_nd(rnn.n_gates, rnn.dhc, [&](dim_t i, dim_t k) {
        body_loop(i, k, ws_gates_, diff_bias_, rnn, cell_position);
    });
#endif
}

template <impl::data_type_t src_type, impl::data_type_t weights_type,
        impl::data_type_t acc_type>
struct _ref_rnn_fwd_t;

template <impl::data_type_t src_type, impl::data_type_t weights_type,
        impl::data_type_t acc_type>
struct _ref_rnn_bwd_t;

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
struct _ref_rnn_common_t : public primitive_t {
    static constexpr impl::data_type_t scratch_type
            = aprop == prop_kind::forward ? acc_type : src_type;

    using fwd_t = _ref_rnn_fwd_t<src_type, weights_type, acc_type>;
    using bwd_t = _ref_rnn_bwd_t<src_type, weights_type, acc_type>;
    using impl_t = typename utils::conditional<aprop == prop_kind::forward,
            fwd_t, bwd_t>::type;
    using postgemm_t = typename utils::conditional<aprop == prop_kind::forward,
            rnn_postgemm_fwd_t<src_type, scratch_type, acc_type>,
            rnn_postgemm_bwd_t<src_type, scratch_type, acc_type>>::type;

    /* These types are defined for each element in the cell execution */
    typedef typename prec_traits<src_type>::type src_layer_t;
    typedef typename prec_traits<src_type>::type src_iter_t;
    typedef typename prec_traits<src_type>::type dst_layer_t;
    typedef typename prec_traits<src_type>::type dst_iter_t;
    typedef typename prec_traits<weights_type>::type weights_t;
    typedef typename prec_traits<src_type>::type gemm_data_t;
    typedef typename prec_traits<acc_type>::type gemm_acc_t;
    typedef typename prec_traits<scratch_type>::type scratch_t;
    typedef typename prec_traits<src_type>::type ht_t;
    typedef typename prec_traits<src_type>::type gates_t;

    using class_name
            = _ref_rnn_common_t<aprop, src_type, weights_type, acc_type>;
#if DNNL_X64
    using ref_rnn_brgemm_t = x64::rnn_brgemm_utils::rnn_brgemm_t<aprop>;
#endif

    typedef rnn_cell_execution_sig((class_name::*cell_execution_f));
    typedef rnn_grid_execution_sig((class_name::*grid_execution_f));
    typedef rnn_merged_layer_execution_sig(
            (class_name::*merged_layer_execution_f));

    typedef rnn_gemm_sig((class_name::*gemm_t));
    typedef rnn_bias_prepare_sig((class_name::*bias_prepare_t));
    typedef rnn_bias_finalize_sig((class_name::*bias_finalize_t));
    typedef rnn_weights_assign_sig((class_name::*weights_assign_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    cpu_rnn_fwd_pd_t, cpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {
        using base_pd_t::base_pd_t;

        const char *impl_name() const {
#if DNNL_X64
            using namespace dnnl::impl::cpu::x64;
            return rnn_.is_brgemm
                    ? JIT_IMPL_NAME_HELPER("brgemm:", rnn_.brgemm_isa, "")
                    : "ref";
#else
            return "ref";
#endif
        }

        DECLARE_COMMON_PD_T(impl_name(), impl_t, USE_GLOBAL_SCRATCHPAD);

        status_t init_ref(engine_t *engine) {
            using namespace prop_kind;
            using namespace utils;
            using namespace format_tag;
            using namespace rnn_utils;
            const alg_kind_t cell_kind = this->desc()->cell_kind;

            const data_type_t src_layer_dt
                    = this->desc()->src_layer_desc.data_type;
            const data_type_t weights_iter_dt
                    = this->desc()->weights_iter_desc.data_type;
            const data_type_t weights_layer_dt
                    = this->desc()->weights_layer_desc.data_type;

            bool ok = true
                    && one_of(cell_kind, alg_kind::vanilla_rnn,
                            alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                            alg_kind::lbr_gru, alg_kind::vanilla_augru,
                            alg_kind::lbr_augru)
                    && IMPLICATION(aprop == prop_kind::forward,
                            one_of(this->desc()->prop_kind, forward_training,
                                    forward_inference))
                    && IMPLICATION(aprop == backward,
                            one_of(this->desc()->prop_kind, backward))
                    && src_layer_dt == src_type
                    && everyone_is(
                            weights_type, weights_iter_dt, weights_layer_dt)
                    && this->set_default_params() == status::success
                    && this->with_bias();

            if (!ok) return status::unimplemented;

            rnn_ = zero<decltype(rnn_)>();
            rnn_.is_brgemm = false;
            ok = init_conf<class_name>(rnn_, *this->desc(), *this->attr(),
                    this->src_md(0), this->src_md(1), this->src_md(2),
                    this->weights_md(0), this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION), this->dst_md(0),
                    this->dst_md(1), this->dst_md(2),
                    this->arg_md(DNNL_ARG_BIAS));
            if (!ok) return status::unimplemented;

            if (rnn_.is_bf16_conf()) {
                if (!utils::one_of(
                            rnn_.bias_dt, data_type::bf16, data_type::f32)
                        || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt
                        || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                                data_type::bf16, data_type::f32))
                    return status::unimplemented;
            } else if (rnn_.bias_dt != data_type::f32
                    || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                            data_type::f32)
                    || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt)
                return status::unimplemented;

            /* check that no data shift have been passed to s8s8 lstm */
            if (!IMPLICATION(rnn_.is_signed_int8_conf(),
                        this->attr()->rnn_data_qparams_.shift_ == 0.f))
                return status::unimplemented;

            /* INT8 cases with non-trivial strides are not supported */
            if (rnn_.is_int8_conf()
                    && !(rnn_.src_layer_is_trivial_stride
                            && rnn_.dst_layer_is_trivial_stride))
                return status::unimplemented;

            /* check that only supported attr have been passed */
            primitive_attr_t::skip_mask_t attr_mask
                    = primitive_attr_t::skip_mask_t::rnn_tparams;
            if (weights_layer_dt == data_type::s8)
                attr_mask = attr_mask
                        | primitive_attr_t::skip_mask_t::rnn_data_qparams
                        | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                        | primitive_attr_t::skip_mask_t::
                                rnn_weights_projection_qparams;
            ok = ok && this->attr()->has_default_values(attr_mask);
            if (!ok) return status::unimplemented;

            // Set weights descriptors to desired format
            memory_desc_t new_weights_layer_md = *this->weights_md(0);
            CHECK(set_expected_desc(rnn_, new_weights_layer_md,
                    rnn_utils::weights_type_t::layer));
            if (this->weights_layer_md_.format_kind == format_kind::any) {
                this->weights_layer_md_ = new_weights_layer_md;
            } else if (this->weights_layer_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_layer_md_ != new_weights_layer_md)
                    return status::unimplemented;
            }

            memory_desc_t new_weights_iter_md = *this->weights_md(1);
            CHECK(set_expected_desc(rnn_, new_weights_iter_md,
                    rnn_utils::weights_type_t::iter));
            if (this->weights_iter_md_.format_kind == format_kind::any) {
                this->weights_iter_md_ = new_weights_iter_md;
            } else if (this->weights_iter_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_iter_md_ != new_weights_iter_md)
                    return status::unimplemented;
            }

            if (rnn_.is_lstm_projection) {
                memory_desc_t new_weights_projection_md
                        = *this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION);
                CHECK(set_expected_desc(rnn_, new_weights_projection_md,
                        rnn_utils::weights_type_t::projection));
                if (this->weights_projection_md_.format_kind
                        == format_kind::any) {
                    this->weights_projection_md_ = new_weights_projection_md;
                } else if (this->weights_projection_md_.format_kind
                        == format_kind::rnn_packed) {
                    if (this->weights_projection_md_
                            != new_weights_projection_md)
                        return status::unimplemented;
                }
            }

            CHECK(this->check_layout_consistency(false /*is_brgemm*/));

            set_conf<class_name>(rnn_, *this->desc(), this->weights_md(0),
                    this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                    this->diff_weights_md(0), this->diff_weights_md(1),
                    this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));
            set_workspace_sizes<class_name>(rnn_, *this->desc());
            return status::success;
        }

        status_t init_brgemm(engine_t *engine) {
            using namespace prop_kind;
            using namespace utils;
            using namespace format_tag;
            using namespace rnn_utils;
#if DNNL_X64
            using namespace x64;
            const alg_kind_t cell_kind = this->desc()->cell_kind;

            const data_type_t src_layer_dt
                    = this->desc()->src_layer_desc.data_type;
            const data_type_t weights_iter_dt
                    = this->desc()->weights_iter_desc.data_type;
            const data_type_t weights_layer_dt
                    = this->desc()->weights_layer_desc.data_type;

            bool is_f32 = everyone_is(data_type::f32, src_layer_dt,
                    weights_iter_dt, weights_layer_dt);
            bool is_impl_bf16
                    = everyone_is(data_type::bf16, src_type, weights_type);
            bool is_fpmath_bf16 = one_of(this->attr()->fpmath_.mode_,
                    fpmath_mode::bf16, fpmath_mode::any);
            bool allow_down_conversion_to_bf16
                    = is_f32 && is_fpmath_bf16 && is_impl_bf16;

            bool ok = one_of(cell_kind, alg_kind::vanilla_rnn,
                              alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                              alg_kind::lbr_gru, alg_kind::vanilla_augru,
                              alg_kind::lbr_augru)
                    && IMPLICATION(aprop == prop_kind::forward,
                            one_of(this->desc()->prop_kind, forward_training,
                                    forward_inference))
                    // LBR is not supported for training in brgemm
                    && IMPLICATION(one_of(cell_kind, alg_kind::lbr_gru,
                                           alg_kind::lbr_augru),
                            this->desc()->prop_kind == forward_inference)
                    && IMPLICATION(aprop == backward,
                            one_of(this->desc()->prop_kind, backward))
                    // TODO: Enable diff_weights_overwrite support
                    && IMPLICATION(aprop == backward,
                            this->diff_weights_overwrite() == false)
                    // cell_type (or src_type) and primitive data type should
                    // match, except for the bf32 case.
                    && IMPLICATION(!allow_down_conversion_to_bf16,
                            src_layer_dt == src_type
                                    && everyone_is(weights_type,
                                            weights_iter_dt, weights_layer_dt))
                    && this->set_default_params() == status::success
                    && this->with_bias();

            if (!ok) return status::unimplemented;

            rnn_ = zero<decltype(rnn_)>();
            rnn_.is_brgemm = true;
            ok = init_conf<class_name>(rnn_, *this->desc(), *this->attr(),
                    this->src_md(0), this->src_md(1), this->src_md(2),
                    this->weights_md(0), this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION), this->dst_md(0),
                    this->dst_md(1), this->dst_md(2),
                    this->arg_md(DNNL_ARG_BIAS));

            ok = ok
                    && IMPLICATION(one_of(this->desc()->prop_kind,
                                           forward_training, backward),
                            (rnn_.is_xf16_conf() || rnn_.is_f32_conf()));

            if (!ok) return status::unimplemented;

            // Support for GRU / AUGRU cell in BRGEMM-based implementation is
            // limited by forward_inference pass for now, all_f32 is disabled
            // due to performance degradation.
            // TODO: Improve GRU / AUGRU coverage in BRGEMM-based implementation
            ok = IMPLICATION(rnn_.is_orig_gru,
                    this->desc()->prop_kind == forward_inference
                            && !rnn_.is_cell_dt_f32());
            if (!ok) return status::unimplemented;

            if (rnn_.is_cell_dt_f32()
                    && utils::one_of(this->desc()->prop_kind, backward,
                            forward_training))
                return status::unimplemented;

            if (!(IMPLICATION((cell_kind == alg_kind::vanilla_lstm
                                      && rnn_.is_lstm_projection),
                        this->desc()->prop_kind == forward_inference)))
                return status::unimplemented;

            if (rnn_.is_bf16_conf()) {
                if (!mayiuse(avx512_core_bf16)
                        || !utils::one_of(
                                rnn_.bias_dt, data_type::bf16, data_type::f32)
                        || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt
                        || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                                data_type::bf16, data_type::f32))
                    return status::unimplemented;
            } else if (rnn_.is_f16_conf()) {
                if (!mayiuse(avx512_core_amx_fp16)
                        || !utils::one_of(
                                rnn_.bias_dt, data_type::f16, data_type::f32)
                        || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt
                        || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                                data_type::f16, data_type::f32))
                    return status::unimplemented;
            } else if (rnn_.bias_dt != data_type::f32
                    || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                            data_type::f32)
                    || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt)
                return status::unimplemented;
            const auto isa = get_max_cpu_isa();
            if (rnn_.is_signed_int8_conf()
                    && !is_superset(isa, avx512_core_amx))
                return status::unimplemented;
            if (rnn_.is_int8_conf() && !is_superset(isa, avx512_core_vnni))
                return status::unimplemented;
            if (rnn_.is_f32_conf() && !is_superset(isa, avx2))
                return status::unimplemented;

            /* check that no shift have been passed to s8s8 amx lstm */
            if (!IMPLICATION(rnn_.is_signed_int8_conf(),
                        this->attr()->rnn_data_qparams_.shift_ == 0))
                return status::unimplemented;

            /* INT8 cases with non-trivial strides are not supported */
            if (rnn_.is_int8_conf()
                    && !(rnn_.src_layer_is_trivial_stride
                            && rnn_.dst_layer_is_trivial_stride))
                return status::unimplemented;

            /* check that only supported attr have been passed */
            primitive_attr_t::skip_mask_t attr_mask
                    = primitive_attr_t::skip_mask_t::rnn_tparams;
            if (weights_layer_dt == data_type::s8)
                attr_mask = attr_mask
                        | primitive_attr_t::skip_mask_t::rnn_data_qparams
                        | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                        | primitive_attr_t::skip_mask_t::
                                rnn_weights_projection_qparams
                        | primitive_attr_t::skip_mask_t::fpmath_mode;
            ok = ok && this->attr()->has_default_values(attr_mask);
            if (!ok) return status::unimplemented;

            set_conf<class_name>(rnn_, *this->desc(), this->weights_md(0),
                    this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                    this->diff_weights_md(0), this->diff_weights_md(1),
                    this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));

            CHECK(ref_rnn_brgemm_t::configure_brgemm(rnn_,
                    this->desc()->cell_kind, sizeof(src_layer_t),
                    sizeof(scratch_t)));

            // must be called after configure_brgemm()
            set_workspace_sizes<class_name>(rnn_, *this->desc());

            // Only AMX LSTM supports s8s8 now
            if (rnn_.is_signed_int8_conf() && !rnn_.is_cell_int8_amx())
                return status::unimplemented;

            // Set weights descriptors to desired format
            memory_desc_t new_weights_layer_md = *this->weights_md(0);
            CHECK(set_expected_desc(rnn_, new_weights_layer_md,
                    rnn_utils::weights_type_t::layer));
            if (this->weights_layer_md_.format_kind == format_kind::any) {
                this->weights_layer_md_ = new_weights_layer_md;
            } else if (this->weights_layer_md_ != new_weights_layer_md) {
                return status::unimplemented;
            }

            memory_desc_t new_weights_iter_md = *this->weights_md(1);
            CHECK(set_expected_desc(rnn_, new_weights_iter_md,
                    rnn_utils::weights_type_t::iter));
            if (this->weights_iter_md_.format_kind == format_kind::any) {
                this->weights_iter_md_ = new_weights_iter_md;
            } else if (this->weights_iter_md_ != new_weights_iter_md) {
                return status::unimplemented;
            }
            if (rnn_.is_lstm_projection) {
                memory_desc_t new_weights_projection_md
                        = *this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION);
                CHECK(set_expected_desc(rnn_, new_weights_projection_md,
                        rnn_utils::weights_type_t::projection));
                if (this->weights_projection_md_.format_kind
                        == format_kind::any) {
                    this->weights_projection_md_ = new_weights_projection_md;
                } else if (this->weights_projection_md_
                        != new_weights_projection_md) {
                    return status::unimplemented;
                }
            }
            if (rnn_.is_unsigned_int8_conf()) {
                const memory_desc_wrapper &weights_layer_d(
                        this->weights_layer_md_);
                const memory_desc_wrapper &weights_iter_d(
                        this->weights_iter_md_);
                const auto &pdims_l = weights_layer_d.padded_dims();
                const auto &pdims_i = weights_iter_d.padded_dims();
                rnn_.weights_layer_comp_offset = rnn_.n_layer * rnn_.n_dir
                        * rnn_.n_gates * pdims_l[2] * pdims_l[4];
                rnn_.weights_iter_comp_offset = rnn_.n_layer * rnn_.n_dir
                        * rnn_.n_gates * pdims_i[2] * pdims_i[4];
                if (rnn_.is_lstm_projection) {
                    const memory_desc_wrapper &weights_proj_d(
                            this->weights_projection_md_);
                    const auto &pdims_p = weights_proj_d.padded_dims();
                    rnn_.weights_projection_comp_offset = rnn_.n_layer
                            * rnn_.n_dir * pdims_p[2] * pdims_p[3];
                } else {
                    rnn_.weights_projection_comp_offset = 0;
                }
            }
            CHECK(this->check_layout_consistency(true /*is_brgemm*/));

            if (rnn_.is_bf32()) {
                const memory_desc_wrapper weights_layer_d(
                        this->weights_layer_md_);
                memory_desc_t weights_layer_md;
                const memory_desc_wrapper weights_iter_d(
                        this->weights_iter_md_);
                memory_desc_t weights_iter_md;

                const auto bf16_tag = rnn_.n_block == 64
                        ? format_tag::ldgOI64o2i
                        : format_tag::ldgOI32o2i;
                CHECK(memory_desc_init_by_tag(weights_layer_md,
                        weights_layer_d.ndims(), weights_layer_d.dims(),
                        data_type::bf16, bf16_tag));
                CHECK(reorder_primitive_desc_create(bf32_wei_layer_reorder_pd_,
                        engine, weights_layer_d.md_, &weights_layer_md,
                        nullptr));

                CHECK(memory_desc_init_by_tag(weights_iter_md,
                        weights_iter_d.ndims(), weights_iter_d.dims(),
                        data_type::bf16, bf16_tag));
                CHECK(reorder_primitive_desc_create(bf32_wei_iter_reorder_pd_,
                        engine, weights_iter_d.md_, &weights_iter_md, nullptr));
            }

            return status::success;
#else
            return status::unimplemented;
#endif
        }

        status_t init(engine_t *engine) {
            status_t st = init_brgemm(engine);
            if (st != status::success) {
                rnn_.is_brgemm = false;
                st = init_ref(engine);
            }
            if (st == status::success) {
                size_t scratchpad_sz {0}, ws_sz {0};
                get_scratchpad_and_workspace_sizes(rnn_, scratchpad_sz, ws_sz);

                init_scratchpad(scratchpad_sz);
                // initialize the workspace if needed
                if (rnn_.is_training) {
                    dims_t ws_dims = {(dim_t)ws_sz};
                    CHECK(memory_desc_init_by_tag(this->ws_md_, 1, ws_dims,
                            data_type::u8, format_tag::x));
                }
            }
            return st;
        }

        rnn_utils::rnn_conf_t rnn_;
#if DNNL_X64
        std::shared_ptr<primitive_desc_t> bf32_wei_layer_reorder_pd_;
        std::shared_ptr<primitive_desc_t> bf32_wei_iter_reorder_pd_;
#endif
    protected:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();

            {
                static constexpr size_t data_size
                        = 1; // "true" data size already incorporated
                static constexpr size_t data_align
                        = alignof(float); // "worst" case scenario
                static constexpr size_t perf_align = 4096;
                scratchpad.book(key_rnn_space, scratchpad_sz, data_size,
                        data_align, perf_align);
            }

            const int max_nparts
                    = utils::one_of(this->cell_kind(), alg_kind::vanilla_gru,
                              alg_kind::vanilla_augru)
                    ? 2
                    : 1;
            const int ptr_wei_sz = rnn_.n_layer * rnn_.n_dir * max_nparts;
            scratchpad.template book<float *>(
                    key_rnn_ptrs_wei_layer, ptr_wei_sz);
            scratchpad.template book<float *>(
                    key_rnn_ptrs_wei_iter, ptr_wei_sz);
            scratchpad.template book<float *>(
                    key_rnn_ptrs_wei_projection, ptr_wei_sz);

            const auto bias_dt_size = types::data_type_size(
                    this->arg_md(DNNL_ARG_BIAS)->data_type);
            scratchpad.template book<void *>(
                    key_rnn_ptrs_bia, ptr_wei_sz * bias_dt_size);

            scratchpad.template book<scratch_t>(
                    key_rnn_gates, rnn_.scratch_gates_size);
            scratchpad.template book<ht_t>(key_rnn_ht, rnn_.scratch_ht_size);
            scratchpad.template book<gemm_acc_t>(
                    key_rnn_diff_ht, rnn_.scratch_diff_ht_size);
            scratchpad.template book<scratch_t>(
                    key_rnn_cell, rnn_.scratch_cell_size);

#if DNNL_X64
            if (rnn_.is_brgemm) {
                ref_rnn_brgemm_t::init_scratchpad(rnn_, scratchpad,
                        sizeof(gemm_acc_t), alignof(gemm_acc_t));
                if (rnn_.is_bf32()) {
                    scratchpad.book(key_nested_multiple + 0,
                            bf32_wei_layer_reorder_pd_->scratchpad_registry());
                    scratchpad.book(key_nested_multiple + 1,
                            bf32_wei_iter_reorder_pd_->scratchpad_registry());
                }
            }
#endif
        }
    };

    _ref_rnn_common_t(const pd_t *apd)
        : primitive_t(apd), rnn_postgemm_(nullptr) {}

    status_t init(engine_t *engine) override {
        /// @todo set max_feature_size assuming that we limit the number of
        /// iterations and layer to one if slc != dhc and sic != dhc
        /// respectively

        bias_preparation_func = &class_name::bias_prepare;
        bias_finalization_func = &class_name::bias_finalize;

        const auto set_gemm_funcs
                = [](bool packed_gemm, gemm_t &g, weights_assign_t &a,
                          bool is_brgemm) {
                      if (packed_gemm) {
                          g = &class_name::packed_gemm;
                          a = &class_name::assign_packed_weights;
                      } else {
                          g = (!is_brgemm) ? &class_name::gemm : nullptr;
                          a = &class_name::assign_weights;
                      }
                  };
        set_gemm_funcs(pd()->rnn_.use_iter_packed_gemm, gemm_iter_func,
                weights_iter_assign_func, pd()->rnn_.is_brgemm);

        set_gemm_funcs(pd()->rnn_.use_layer_packed_gemm, gemm_layer_func,
                weights_layer_assign_func, pd()->rnn_.is_brgemm);

        if (pd()->rnn_.is_lstm_projection) {
            set_gemm_funcs(pd()->rnn_.use_projection_packed_gemm,
                    gemm_projection_func, weights_projection_assign_func,
                    pd()->rnn_.is_brgemm);
        }

        rnn_postgemm_ = new postgemm_t(pd()->rnn_, pd());
        assert(rnn_postgemm_ != nullptr);
        CHECK(rnn_postgemm_->init(pd()->rnn_));
        if (pd()->rnn_.is_brgemm)
            cell_func = &class_name::cell_execution_brgemm;
        else {
            switch (pd()->cell_kind()) {
                case alg_kind::vanilla_rnn:
                case alg_kind::vanilla_lstm:
                    cell_func = &class_name::cell_execution_ref;
                    break;
                case alg_kind::vanilla_gru:
                case alg_kind::vanilla_augru:
                    cell_func = &class_name::cell_execution_gru;
                    break;
                case alg_kind::lbr_augru:
                case alg_kind::lbr_gru:
                    cell_func = &class_name::cell_execution_gru_lbr;
                    break;
                default: break;
            }
        }

        merged_layer_func = pd()->rnn_.is_brgemm && pd()->rnn_.merge_gemm_layer
                        && aprop == prop_kind::forward
                ? &class_name::merged_layer_brgemm
                : &class_name::merged_layer_execution_ref;
        grid_computation = &class_name::linear_execution;

        size_t scratchpad_size, workspace_size;
        rnn_utils::set_offsets(pd()->rnn_, ws_gates_offset_, ws_ht_offset_,
                ws_states_layer_offset_, ws_states_iter_offset_,
                ws_states_iter_c_offset_, ws_diff_states_layer_offset_,
                ws_diff_states_iter_offset_, ws_diff_states_iter_c_offset_,
                ws_grid_comp_offset_, ws_bias_offset_, scratch_gates_offset_,
                scratch_ht_offset_, scratch_diff_ht_offset_,
                scratch_cell_offset_, scratchpad_size, workspace_size);
#if DNNL_X64
        const auto rnn = pd()->rnn_;
        if (rnn.is_brgemm) {
            if (rnn.is_bf32()) {

                CHECK(pd()->bf32_wei_layer_reorder_pd_->create_primitive(
                        bf32_wei_layer_reorder_, engine));

                CHECK(pd()->bf32_wei_iter_reorder_pd_->create_primitive(
                        bf32_wei_iter_reorder_, engine));
            }
            return rnn_brgemm_.init_kernels(rnn, src_type, weights_type);
        }
#endif
        return status::success;
    }

    virtual ~_ref_rnn_common_t() { delete rnn_postgemm_; }

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
#if DNNL_X64
    ref_rnn_brgemm_t rnn_brgemm_;
    std::shared_ptr<primitive_t> bf32_wei_layer_reorder_;
    std::shared_ptr<primitive_t> bf32_wei_iter_reorder_;
#endif

    template <typename input_t>
    void copy_init_layer(const rnn_utils::rnn_conf_t &rnn,
            src_layer_t *ws_states_layer_, gemm_acc_t *ws_diff_states_layer_,
            const input_t *xt_, const gemm_acc_t *diff_dst_layer) const;

    template <typename input_t>
    void copy_init_iter(const rnn_utils::rnn_conf_t &rnn,
            src_iter_t *ws_states_iter_, void *ws_states_iter_c_,
            gemm_acc_t *ws_diff_states_iter_,
            gemm_acc_t *ws_diff_states_iter_c_, const input_t *src_iter_,
            const void *src_iter_c_, const gemm_acc_t *diff_dst_iter_,
            const float *diff_dst_iter_c_) const;

    template <typename dst_layer_dt, typename dst_iter_dt>
    void copy_res_layer(const rnn_utils::rnn_conf_t &rnn,
            dst_layer_dt *dst_layer_, gemm_acc_t *diff_src_layer_,
            const dst_iter_dt *dst_iter_, const src_layer_t *ws_states_layer_,
            const gemm_acc_t *ws_diff_states_layer_) const;

    template <typename prim_dst_iter_t, typename prim_dst_layer_t>
    void copy_res_iter(const rnn_utils::rnn_conf_t &rnn,
            prim_dst_iter_t *dst_iter_, void *dst_iter_c_,
            gemm_acc_t *diff_src_iter_, float *diff_src_iter_c_,
            const prim_dst_layer_t *dst_layer_,
            const src_iter_t *ws_states_iter_, const void *ws_states_iter_c,
            const gemm_acc_t *ws_diff_states_iter_,
            const gemm_acc_t *ws_diff_states_iter_c_) const;

    rnn_grid_execution_sig(linear_execution);
    virtual rnn_cell_execution_sig(cell_execution_ref) = 0;
    virtual rnn_merged_layer_execution_sig(merged_layer_execution_ref) = 0;
    virtual rnn_cell_execution_sig(cell_execution_brgemm) = 0;
    virtual rnn_merged_layer_execution_sig(merged_layer_brgemm) = 0;
    virtual rnn_cell_execution_sig(cell_execution_gru) = 0;
    virtual rnn_cell_execution_sig(cell_execution_gru_lbr) = 0;
    virtual rnn_gemm_sig(gemm) = 0;
    virtual rnn_gemm_sig(packed_gemm) = 0;
    rnn_bias_prepare_sig(bias_prepare);
    rnn_bias_finalize_sig(bias_finalize);
    rnn_weights_assign_sig(assign_weights);
    rnn_weights_assign_sig(assign_packed_weights);

    float (*activation_func)(float s, float alpha, float cliping);

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    size_t ws_gates_offset_;
    size_t ws_ht_offset_;
    size_t ws_states_layer_offset_;
    size_t ws_states_iter_offset_;
    size_t ws_states_iter_c_offset_;
    size_t ws_bias_offset_;
    size_t ws_diff_states_layer_offset_;
    size_t ws_diff_states_iter_offset_;
    size_t ws_diff_states_iter_c_offset_;
    size_t ws_grid_comp_offset_;
    size_t scratch_gates_offset_;
    size_t scratch_ht_offset_;
    size_t scratch_diff_ht_offset_;
    size_t scratch_cell_offset_;
    postgemm_t *rnn_postgemm_;

    grid_execution_f grid_computation;
    cell_execution_f cell_func;
    merged_layer_execution_f merged_layer_func;

    bias_prepare_t bias_preparation_func;
    bias_finalize_t bias_finalization_func;
    weights_assign_t weights_layer_assign_func;
    weights_assign_t weights_iter_assign_func;
    weights_assign_t weights_projection_assign_func;

    gemm_t gemm_layer_func;
    gemm_t gemm_iter_func;
    gemm_t gemm_projection_func;
};

template <impl::data_type_t src_type, impl::data_type_t weights_type,
        impl::data_type_t acc_type>
struct _ref_rnn_fwd_t : public _ref_rnn_common_t<prop_kind::forward, src_type,
                                weights_type, acc_type> {
    using base_t = _ref_rnn_common_t<prop_kind::forward, src_type, weights_type,
            acc_type>;
    using src_layer_t = typename base_t::src_layer_t;
    using src_iter_t = typename base_t::src_iter_t;
    using dst_layer_t = typename base_t::dst_layer_t;
    using dst_iter_t = typename base_t::dst_iter_t;
    using weights_t = typename base_t::weights_t;
    using gemm_data_t = typename base_t::gemm_data_t;
    using gemm_acc_t = typename base_t::gemm_acc_t;
    using scratch_t = typename base_t::scratch_t;
    using ht_t = typename base_t::ht_t;
    using gates_t = typename base_t::gates_t;

    using base_t::cell_func;
    using base_t::grid_computation;
    using base_t::merged_layer_func;

    using base_t::bias_finalization_func;
    using base_t::bias_preparation_func;
    using base_t::weights_iter_assign_func;
    using base_t::weights_layer_assign_func;
    using base_t::weights_projection_assign_func;

    using base_t::gemm_iter_func;
    using base_t::gemm_layer_func;
    using base_t::gemm_projection_func;

    using base_t::base_t;

private:
    rnn_gemm_sig(gemm) override;
    rnn_gemm_sig(packed_gemm) override;
    rnn_cell_execution_sig(cell_execution_ref) override;
    rnn_merged_layer_execution_sig(merged_layer_execution_ref) override;
    rnn_cell_execution_sig(cell_execution_brgemm) override;
    rnn_merged_layer_execution_sig(merged_layer_brgemm) override;
    rnn_cell_execution_sig(cell_execution_gru) override;
    rnn_cell_execution_sig(cell_execution_gru_lbr) override;
};

template <impl::data_type_t src_type, impl::data_type_t weights_type,
        impl::data_type_t acc_type>
struct _ref_rnn_bwd_t : public _ref_rnn_common_t<prop_kind::backward, src_type,
                                weights_type, acc_type> {
    using base_t = _ref_rnn_common_t<prop_kind::backward, src_type,
            weights_type, acc_type>;
    using src_layer_t = typename base_t::src_layer_t;
    using src_iter_t = typename base_t::src_iter_t;
    using dst_layer_t = typename base_t::dst_layer_t;
    using dst_iter_t = typename base_t::dst_iter_t;
    using weights_t = typename base_t::weights_t;
    using gemm_data_t = typename base_t::gemm_data_t;
    using gemm_acc_t = typename base_t::gemm_acc_t;
    using scratch_t = typename base_t::scratch_t;
    using ht_t = typename base_t::ht_t;
    using gates_t = typename base_t::gates_t;

    using base_t::cell_func;
    using base_t::grid_computation;
    using base_t::merged_layer_func;

    using base_t::bias_finalization_func;
    using base_t::bias_preparation_func;
    using base_t::weights_iter_assign_func;
    using base_t::weights_layer_assign_func;
    using base_t::weights_projection_assign_func;

    using base_t::gemm_iter_func;
    using base_t::gemm_layer_func;
    using base_t::gemm_projection_func;

    using base_t::base_t;

private:
    rnn_gemm_sig(gemm) override;
    rnn_gemm_sig(packed_gemm) override;
    rnn_cell_execution_sig(cell_execution_ref) override;
    rnn_merged_layer_execution_sig(merged_layer_execution_ref) override;
    rnn_cell_execution_sig(cell_execution_brgemm) override;
    rnn_cell_execution_sig(cell_execution_gru) override;
    rnn_cell_execution_sig(cell_execution_gru_lbr) override;
    rnn_merged_layer_execution_sig(merged_layer_brgemm) override {
        return dnnl_runtime_error;
    };
};

using ref_rnn_common_fwd_f32_t = _ref_rnn_common_t<prop_kind::forward,
        data_type::f32, data_type::f32, data_type::f32>;
using ref_rnn_common_bwd_f32_t = _ref_rnn_common_t<prop_kind::backward,
        data_type::f32, data_type::f32, data_type::f32>;

using ref_rnn_common_fwd_bf16_t = _ref_rnn_common_t<prop_kind::forward,
        data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_common_bwd_bf16_t = _ref_rnn_common_t<prop_kind::backward,
        data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_common_fwd_f16_t = _ref_rnn_common_t<prop_kind::forward,
        data_type::f16, data_type::f16, data_type::f32>;
using ref_rnn_common_bwd_f16_t = _ref_rnn_common_t<prop_kind::backward,
        data_type::f16, data_type::f16, data_type::f32>;
using ref_rnn_common_fwd_u8s8_t = _ref_rnn_common_t<prop_kind::forward,
        data_type::u8, data_type::s8, data_type::s32>;
using ref_rnn_common_fwd_s8s8_t = _ref_rnn_common_t<prop_kind::forward,
        data_type::s8, data_type::s8, data_type::s32>;

using ref_rnn_fwd_f32_t
        = _ref_rnn_fwd_t<data_type::f32, data_type::f32, data_type::f32>;
using ref_rnn_bwd_f32_t
        = _ref_rnn_bwd_t<data_type::f32, data_type::f32, data_type::f32>;

using ref_rnn_fwd_bf16_t
        = _ref_rnn_fwd_t<data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_bwd_bf16_t
        = _ref_rnn_bwd_t<data_type::bf16, data_type::bf16, data_type::f32>;

using ref_rnn_fwd_f16_t
        = _ref_rnn_fwd_t<data_type::f16, data_type::f16, data_type::f32>;
using ref_rnn_bwd_f16_t
        = _ref_rnn_bwd_t<data_type::f16, data_type::f16, data_type::f32>;

using ref_rnn_fwd_u8s8_t
        = _ref_rnn_fwd_t<data_type::u8, data_type::s8, data_type::s32>;
using ref_rnn_fwd_s8s8_t
        = _ref_rnn_fwd_t<data_type::s8, data_type::s8, data_type::s32>;
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
