/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_RNN_REORDERS_HPP
#define CPU_RNN_REORDERS_HPP

#include <assert.h>

#include "bfloat16.hpp"
#include "cpu_reorder_pd.hpp"
#include "dnnl_thread.hpp"
#include "gemm/gemm_pack.hpp"
#include "simple_q10n.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

inline void init_dims(int &L, int &D, int &I, int &G, int &O,
        const memory_desc_wrapper &mdw) {
    auto dims = mdw.dims();
    auto ndims = mdw.ndims();
    L = dims[0];
    D = dims[1];
    I = dims[2];
    G = 0;
    O = 0;
    // weights_layer/weights_iter case
    if (ndims == 5) {
        G = dims[3];
        O = dims[4];
    }
    // projection weights case
    if (ndims == 4) {
        G = 1;
        O = dims[3];
    }
    assert(G != 0 && O != 0);
};

template <data_type_t type_i, data_type_t type_o>
struct rnn_data_reorder_t : public primitive_impl_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_data_reorder", rnn_data_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            using namespace status;
            const memory_desc_wrapper id(src_md), od(dst_md);

            bool args_ok = true;
#define PD_CHECK_ARG(x) args_ok = args_ok && (x)
            PD_CHECK_ARG(id.data_type() == type_i);
            PD_CHECK_ARG(od.data_type() == type_o);
            PD_CHECK_ARG(utils::one_of(id.ndims(), 3, 4));
            auto skip_mask = primitive_attr_t::skip_mask_t::rnn_data_qparams
                    | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                    | primitive_attr_t::skip_mask_t::
                            rnn_weights_projection_qparams;
            PD_CHECK_ARG(attr->has_default_values(skip_mask));
            PD_CHECK_ARG(IMPLICATION(id.ndims() == 3,
                    id.matches_tag(format_tag::tnc)
                            && od.matches_tag(format_tag::tnc)));
            PD_CHECK_ARG(IMPLICATION(id.ndims() == 4,
                    id.matches_tag(format_tag::ldnc)
                            && od.matches_tag(format_tag::ldnc)));
            if (!args_ok) return invalid_arguments;

            auto _pd = new pd_t(
                    engine, attr, src_engine, src_md, dst_engine, dst_md);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) {
                delete _pd;
                return unimplemented;
            }
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
#undef PD_CHECK_ARG
        }
    };

    rnn_data_reorder_t(const pd_t *apd) : primitive_impl_t(apd) {}

private:
    typedef typename prec_traits<type_i>::type in_data_t;
    typedef typename prec_traits<type_o>::type out_data_t;

    bool is_dense() const {
        const memory_desc_wrapper &input_d = pd()->src_md();
        const memory_desc_wrapper &output_d = pd()->dst_md();
        return utils::everyone_is(1,
                input_d.blocking_desc().strides[input_d.ndims() - 1],
                output_d.blocking_desc().strides[output_d.ndims() - 1]);
    }

    /* This function assumes that only the innermost dimension (C) is
       dense (that is to say, stride is 1).  This is enough to have
       good performance and allow non trivial strides on other
       dimensions (to allow an "optimized" path for views for
       example).
     */
    status_t execute_dense(out_data_t *output, const in_data_t *input,
            const float scale, const float shift) const {
        assert(type_i == data_type::f32);
        assert(type_o == data_type::u8);

        const memory_desc_wrapper &input_d = pd()->src_md();
        const memory_desc_wrapper &output_d = pd()->dst_md();
        const dim_t outer_dim
                = utils::array_product(input_d.dims(), input_d.ndims() - 1);
        const dim_t inner_dim = input_d.dims()[input_d.ndims() - 1];

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start {0}, end {0};
            balance211(outer_dim, nthr, ithr, start, end);
            for (int i = start; i < end; ++i) {
                const dim_t off_in = input_d.off_l(i * inner_dim);
                const dim_t off_out = output_d.off_l(i * inner_dim);
                const in_data_t *__restrict i_ = input + off_in;
                out_data_t *__restrict o_ = output + off_out;
                PRAGMA_OMP_SIMD()
                for (int j = 0; j < inner_dim; ++j) {
                    const float in = (float)i_[j] * scale + shift;
                    const float out_l = nstl::max(in, 0.0f);
                    const float out = nstl::min(out_l, 255.0f);
                    o_[j] = (out_data_t)(nearbyintf(out));
                }
            }
        });
        return status::success;
    }

    status_t execute_generic(out_data_t *output, const in_data_t *input,
            float scale, float shift) const {
        assert(type_i == data_type::f32);
        assert(type_o == data_type::u8);

        const memory_desc_wrapper &input_d = pd()->src_md();
        const memory_desc_wrapper &output_d = pd()->dst_md();
        const size_t nelems = input_d.nelems();
        parallel_nd(nelems, [&](size_t i) {
            float in = (float)input[input_d.off_l(i)] * scale + shift;
            float out = nstl::max(nstl::min(in, 255.0f), 0.0f);
            output[output_d.off_l(i)] = (out_data_t)(nearbyintf(out));
        });
        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        auto input = CTX_IN_MEM(const in_data_t *, DNNL_ARG_FROM);
        auto output = CTX_OUT_MEM(out_data_t *, DNNL_ARG_TO);
        const float scale = pd()->attr()->rnn_data_qparams_.scale_;
        const float shift = pd()->attr()->rnn_data_qparams_.shift_;

        if (is_dense())
            return execute_dense(output, input, scale, shift);
        else
            return execute_generic(output, input, scale, shift);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

template <data_type_t type_i>
struct rnn_weights_reorder_s8_t : public primitive_impl_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_weights_reorder_s8", rnn_weights_reorder_s8_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            using namespace status;
            const memory_desc_wrapper id(src_md), od(dst_md);

            bool args_ok = true;
#define PD_CHECK_ARG(x) args_ok = args_ok && (x)
            // Fast checks
            PD_CHECK_ARG(id.data_type() == type_i);
            PD_CHECK_ARG(od.data_type() == data_type::s8);
            PD_CHECK_ARG(od.format_kind() == format_kind::rnn_packed);
            PD_CHECK_ARG(utils::one_of(
                    od.rnn_packed_desc().format, dnnl_ldigo_p, dnnl_ldio_p));
            PD_CHECK_ARG(od.ndims() == id.ndims());
            PD_CHECK_ARG(od.rnn_packed_desc().n_parts == 1);
            // TODO: we have to skip projection qparam even for regular lstm
            // as we use the same attr for regular weights and projection
            auto skip_mask = primitive_attr_t::skip_mask_t::rnn_data_qparams
                    | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                    | primitive_attr_t::skip_mask_t::
                            rnn_weights_projection_qparams;
            PD_CHECK_ARG(attr->has_default_values(skip_mask));
            if (!args_ok) return invalid_arguments;

            // Slower checks
            PD_CHECK_ARG(id.is_dense());
            if (!args_ok) return invalid_arguments;

            format_tag_t itag = id.matches_one_of_tag(format_tag::ldigo,
                    format_tag::ldgoi, format_tag::ldio, format_tag::ldoi);
            if (itag == format_tag::undef) return invalid_arguments;

            // TODO: add support for layer and direction dimensions
            // weights_layer and weights_iter
            if (id.ndims() == 5
                    && !utils::one_of(attr->rnn_weights_qparams_.mask_, 0, 24))
                return unimplemented;
            // weights_projection
            if (id.ndims() == 4
                    && !utils::one_of(
                            attr->rnn_weights_projection_qparams_.mask_, 0, 8))
                return unimplemented;

            auto _pd = new pd_t(
                    engine, attr, src_engine, src_md, dst_engine, dst_md);
            if (_pd == nullptr) return out_of_memory;
            _pd->itag_ = itag;
            if (_pd->init() != success) {
                delete _pd;
                return unimplemented;
            }
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
#undef PD_CHECK_ARG
        }

        status_t init() {
            status_t status = cpu_reorder_pd_t::init();
            if (status != status::success) return status;

            init_scratchpad();

            return status::success;
        }

        format_tag_t itag_ = dnnl_format_tag_undef;
        size_t thr_scratch_comp_sz_ = 0;

    private:
        void init_scratchpad() {
            const memory_desc_wrapper id(src_md());
            const size_t nelems = id.nelems();
            const auto &dims = id.dims();

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            size_t quantization_size = sizeof(int8_t) * nelems;
            // we do not use GO directly, as this can cause false
            // sharing when parallelizing on I (2 threads writing to
            // the same cache line)
            thr_scratch_comp_sz_
                    = itag_ == format_tag::ldigo ? dims[3] * dims[4] : dims[3];
            thr_scratch_comp_sz_ = utils::rnd_up(thr_scratch_comp_sz_, 16);
            size_t reduction_size = 0;
            if (utils::one_of(itag_, format_tag::ldigo, format_tag::ldio))
                reduction_size = dnnl_get_max_threads() * sizeof(int32_t)
                        * thr_scratch_comp_sz_;

            scratchpad.book(
                    key_reorder_rnn_weights_quantization, quantization_size);
            scratchpad.book(key_reorder_rnn_weights_reduction, reduction_size);
        }
    };

    rnn_weights_reorder_s8_t(const pd_t *apd) : primitive_impl_t(apd) {}

private:
    typedef typename prec_traits<type_i>::type in_data_t;

    void quantize_goi(int8_t *scratch_quantized,
            const memory_desc_wrapper &src_d, const float *src) const {
        // TODO: trivial strides assumes here.
        //       Use proper strides where appropriate
        int L, D, I, G, O;
        init_dims(L, D, I, G, O, src_d);

        float *scales = nullptr;
        int mask = 0;
        if (src_d.ndims() == 5) {
            scales = pd()->attr()->rnn_weights_qparams_.scales_;
            mask = pd()->attr()->rnn_weights_qparams_.mask_;
        }
        if (src_d.ndims() == 4) {
            scales = pd()->attr()->rnn_weights_projection_qparams_.scales_;
            mask = pd()->attr()->rnn_weights_projection_qparams_.mask_;
        }
        assert(scales != nullptr);

        parallel_nd(L * D, G * O, [&](int ld, int go) {
            const float s = scales[(mask == 0) ? 0 : go];
            PRAGMA_OMP_SIMD()
            for (int i = 0; i < I; i++) {
                scratch_quantized[ld * I * G * O + i * G * O + go]
                        = qz_b0<in_data_t, int8_t>()(
                                src[ld * G * O * I + go * I + i], s);
            }
        });
    }

    void quantize_igo(int8_t *scratch_quantized,
            const memory_desc_wrapper &src_d, const float *src) const {
        // TODO: trivial strides assumes here.
        //       Use proper strides where appropriate
        int L, D, I, G, O;
        init_dims(L, D, I, G, O, src_d);
        // if projection weights, use weights_projection_qparams
        float *scales = nullptr;
        int mask = 0;
        if (src_d.ndims() == 5) {
            scales = pd()->attr()->rnn_weights_qparams_.scales_;
            mask = pd()->attr()->rnn_weights_qparams_.mask_;
        }
        if (src_d.ndims() == 4) {
            scales = pd()->attr()->rnn_weights_projection_qparams_.scales_;
            mask = pd()->attr()->rnn_weights_projection_qparams_.mask_;
        }
        assert(scales != nullptr);

        parallel(0, [&](const int ithr, const int nthr) {
            int start {0}, end {0};
            balance211(L * D * I, nthr, ithr, start, end);
            for (int ldi = start; ldi < end; ldi++) {
                for (int go = 0; go < G * O; go++) {
                    const float s = scales[(mask == 0) ? 0 : go];
                    scratch_quantized[ldi * G * O + go]
                            = qz_b0<in_data_t, int8_t>()(
                                    src[ldi * G * O + go], s);
                }
            }
        });
    }

    void compensate_goi(float *compensation, const memory_desc_wrapper &src_d,
            int8_t *scratch_quantized) const {
        // TODO: trivial strides assumed here.
        //       Use proper strides where appropriate
        int L, D, I, G, O;
        init_dims(L, D, I, G, O, src_d);

        parallel_nd(L * D, G * O, [&](int ld, int go) {
            int32_t compensation_s32 = 0;
            PRAGMA_OMP_SIMD()
            for (int i = 0; i < I; i++) {
                compensation_s32
                        += scratch_quantized[ld * I * G * O + i * G * O + go];
            }
            // TODO: do not convert to f32 if this compensation is not
            // going to be added to a bias (e.g. like in lstm
            // projection where it is directly added to the s32
            // accumulators)
            compensation[ld * G * O + go]
                    = math::saturate<float>(compensation_s32);
        });
    }

    void compensate_igo(float *compensation, const memory_desc_wrapper &src_d,
            int8_t *scratch_quantized, int32_t *scratch_compensation) const {
        // TODO: trivial strides assumed here.
        //       Use proper strides where appropriate
        int L, D, I, G, O;
        init_dims(L, D, I, G, O, src_d);

        // We parallelize on LD and GO
        // TODO: maybe restrict parallelism as we might have large
        // parallelisation overhead if dimensions are small
        int nthr = dnnl_get_max_threads();
        int LD_nthr = nstl::min(L * D, nthr);
        int GO_nthr = nstl::min(G * O, nthr / LD_nthr);
        parallel(nthr, [&](const int ithr, const int nthr) {
            int LD_ithr = -1, LD_s = -1, LD_e = -1;
            int GO_ithr = -1, GO_s = -1, GO_e = -1;
            if (ithr < LD_nthr * GO_nthr) {
                LD_ithr = ithr % LD_nthr;
                GO_ithr = ithr / LD_nthr;
                balance211(L * D, LD_nthr, LD_ithr, LD_s, LD_e);
                balance211(G * O, GO_nthr, GO_ithr, GO_s, GO_e);
            }
            int32_t *compensation_s32
                    = scratch_compensation + ithr * pd()->thr_scratch_comp_sz_;
            for (int ld = LD_s; ld < LD_e; ld++) {
                if (I == 1) {
                    PRAGMA_OMP_SIMD()
                    for (int go = GO_s; go < GO_e; go++)
                        compensation[ld * G * O + go] = math::saturate<float>(
                                scratch_quantized[ld * I * G * O + go]);
                } else {
                    // We split the loop on I in three to avoid conditionals or zeroing compensation
                    int i = 0;
                    PRAGMA_OMP_SIMD()
                    for (int go = GO_s; go < GO_e; go++)
                        compensation_s32[go] = scratch_quantized[go
                                + G * O * (i + I * (ld))];
                    // 1 <= i < I-1
                    for (i = 1; i < I - 1; i++) {
                        PRAGMA_OMP_SIMD()
                        for (int go = GO_s; go < GO_e; go++)
                            compensation_s32[go] += scratch_quantized[go
                                    + G * O * (i + I * (ld))];
                    }
                    // i = I-1
                    PRAGMA_OMP_SIMD()
                    for (int go = GO_s; go < GO_e; go++)
                        compensation[ld * G * O + go]
                                = math::saturate<float>(compensation_s32[go]
                                        + scratch_quantized[go
                                                + G * O * (i + I * (ld))]);
                }
            }
        });
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        // TODO: trivial strides assumed here.
        //       Use proper strides where appropriate

        auto src = CTX_IN_MEM(const in_data_t *, DNNL_ARG_FROM);
        auto dst = CTX_OUT_MEM(char *, DNNL_ARG_TO);
        const memory_desc_wrapper &src_d = pd()->src_md();
        const memory_desc_wrapper &dst_d = pd()->dst_md();
        if (src_d.has_zero_dim()) {
            assert(dst_d.has_zero_dim());
            return status::success;
        }

        int L, D, I, G, O;
        init_dims(L, D, I, G, O, src_d);

        /* Quantize src & compute compensation */
        auto scratch_quantized
                = (int8_t * __restrict) ctx.get_scratchpad_grantor()
                          .template get<void>(memory_tracking::names::
                                          key_reorder_rnn_weights_quantization);
        auto scratch_compensation
                = (int32_t * __restrict) ctx.get_scratchpad_grantor()
                          .template get<void>(memory_tracking::names::
                                          key_reorder_rnn_weights_reduction);
        float *comp = reinterpret_cast<float *>(
                dst + dst_d.rnn_packed_desc().offset_compensation);

        /* Step 1: we quantize if we need to */
        if (type_i == data_type::f32) {
            switch (pd()->itag_) {
                case format_tag::ldigo:
                case format_tag::ldio:
                    quantize_igo(scratch_quantized, src_d, (float *)src);
                    break;
                case format_tag::ldgoi:
                case format_tag::ldoi:
                    quantize_goi(scratch_quantized, src_d, (float *)src);
                    break;
                default: assert(!"Unsupported reorder");
            }
        } else
            scratch_quantized = (int8_t * __restrict) src;

        /* Step 2: we pre-compute the compensation */
        switch (pd()->itag_) {
            case format_tag::ldigo:
            case format_tag::ldio:
                compensate_igo(
                        comp, src_d, scratch_quantized, scratch_compensation);
                break;
            case format_tag::ldgoi:
            case format_tag::ldoi:
                compensate_goi(comp, src_d, scratch_quantized);
                break;
            default: assert(!"Unsupported reorder");
        }

        /* Step 3: we pack the matrix */
        auto off_igo = [&](int l, int d, int i, int g, int o) {
            return o + O * (g + G * (i + I * (d + D * l)));
        };
        int n_parts = dst_d.rnn_packed_desc().n_parts;
        const size_t *size_packed_cell = dst_d.rnn_packed_desc().part_pack_size;
        const int *parts = dst_d.rnn_packed_desc().parts;
        const int n = dst_d.rnn_packed_desc().n;
        const int ldb = dst_d.rnn_packed_desc().ldb;
        char *to_pack = dst;

        for (int l = 0; l < L; l++) {
            for (int d = 0; d < D; d++) {
                for (int p = 0; p < n_parts; p++) {
                    int g = (p > 0) ? parts[p - 1] : 0;
                    int m_p = parts[p] * O;
                    int k_p = I;
                    int lda = G * O;
                    gemm_s8u8s32_pack("A", "N", "N", &m_p, &n, &k_p, &lda, &ldb,
                            scratch_quantized + off_igo(l, d, 0, g, 0),
                            to_pack);
                    to_pack += size_packed_cell[p];
                }
            }
        }
        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

template <data_type_t type_i, data_type_t type_o>
struct rnn_weights_reorder_t : public primitive_impl_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_weights_reorder", rnn_weights_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            using namespace status;

            const memory_desc_wrapper id(src_md), od(dst_md);
            bool args_ok = true;
#define PD_CHECK_ARG(x) args_ok = args_ok && (x)
            PD_CHECK_ARG(IMPLICATION(
                    type_o == data_type::bf16 || type_i == data_type::bf16,
                    mayiuse(avx512_core)));
            PD_CHECK_ARG(id.data_type() == type_i);
            PD_CHECK_ARG(od.data_type() == type_o);
            PD_CHECK_ARG(od.format_kind() == format_kind::rnn_packed);
            PD_CHECK_ARG(utils::one_of(od.rnn_packed_desc().format,
                    dnnl_ldigo_p, dnnl_ldgoi_p, dnnl_ldio_p));
            PD_CHECK_ARG(attr->has_default_values());
            if (!args_ok) return invalid_arguments;

            format_tag_t itag = id.matches_one_of_tag(format_tag::ldigo,
                    format_tag::ldgoi, format_tag::ldio, format_tag::ldoi);
            if (itag == format_tag::undef) return invalid_arguments;

            auto _pd = new pd_t(
                    engine, attr, src_engine, src_md, dst_engine, dst_md);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) {
                delete _pd;
                return unimplemented;
            }
            _pd->itag_ = itag;
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
#undef PD_CHECK_ARG
        }

        format_tag_t itag_;

        status_t init() {
            status_t status = cpu_reorder_pd_t::init();
            if (status != status::success) return status;

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            const memory_desc_wrapper id(src_md());
            const memory_desc_wrapper od(dst_md());
            const rnn_packed_desc_t &rnn_pdata = od.rnn_packed_desc();

            format_tag_t itag = id.matches_one_of_tag(
                    format_tag::ldigo, format_tag::ldgoi, format_tag::ldio);
            bool layout_cross_case
                    = (itag == format_tag::ldigo
                              && rnn_pdata.format == rnn_packed_format::ldgoi_p)
                    || (itag == format_tag::ldgoi
                            && rnn_pdata.format == rnn_packed_format::ldigo_p)
                    || (itag == format_tag::ldio
                            && rnn_pdata.format == rnn_packed_format::ldio_p),
                    dt_cross_case
                    = type_i == data_type::f32 && type_o == data_type::bf16;
            size_t sz = id.nelems() * sizeof(out_data_t);

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_reorder_rnn_weights_transposition,
                    layout_cross_case ? sz : 0);
            scratchpad.book(
                    key_reorder_rnn_weights_bf16_cvt, dt_cross_case ? sz : 0);
        }
    };

    rnn_weights_reorder_t(const pd_t *apd) : primitive_impl_t(apd) {}

private:
    typedef typename prec_traits<type_i>::type in_data_t;
    typedef typename prec_traits<type_o>::type out_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        // TODO: trivial strides assumed here.
        //       Use proper strides where appropriate

        auto input = CTX_IN_MEM(const in_data_t *, DNNL_ARG_FROM);
        auto output = CTX_OUT_MEM(out_data_t *, DNNL_ARG_TO);
        const memory_desc_wrapper &input_d = pd()->src_md();
        const memory_desc_wrapper &output_d = pd()->dst_md();
        if (input_d.has_zero_dim()) {
            assert(output_d.has_zero_dim());
            return status::success;
        }

        const rnn_packed_desc_t &rnn_pdata = output_d.rnn_packed_desc();
        int L, D, I, G, O;
        init_dims(L, D, I, G, O, input_d);

        /* Pack */
        const bool from_igo = pd()->itag_ == format_tag::ldigo;
        const bool to_igo = rnn_pdata.format == dnnl_ldigo_p;
        int n_parts = rnn_pdata.n_parts;
        const size_t *size_packed_cell = rnn_pdata.part_pack_size;
        const int *parts = rnn_pdata.parts;
        const int n = rnn_pdata.n;

        /* Convert fp32 input to bf16 */
        out_data_t *input_cvt = (out_data_t *)input;
        if (type_i == data_type::f32 && type_o == data_type::bf16) {
            input_cvt
                    = (out_data_t *)ctx.get_scratchpad_grantor()
                              .template get<void>(memory_tracking::names::
                                              key_reorder_rnn_weights_bf16_cvt);
            parallel_nd(L * D, [&](int ld) {
                cvt_float_to_bfloat16((bfloat16_t *)input_cvt + ld * G * O * I,
                        (float *)input + ld * G * O * I, G * O * I);
            });
        }

        /* Transpose weights prior to packing to ensure that packed GEMM
         * algorithm will be dispatched */
        out_data_t *input_tr = input_cvt;
        if (from_igo != to_igo) {
            input_tr
                    = (out_data_t *)ctx.get_scratchpad_grantor().template get<void>(
                            memory_tracking::names::
                                    key_reorder_rnn_weights_transposition);
            const int M = to_igo ? G * O : I;
            const int N = to_igo ? I : G * O;
            parallel_nd(L * D, N, [&](int ld, int i) {
                for (int j = 0; j < M; j++) {
                    input_tr[ld * M * N + i * M + j]
                            = input_cvt[ld * M * N + j * N + i];
                }
            });
        }

        auto off_igo = [&](int l, int d, int i, int g, int o) {
            return l * D * I * G * O + d * I * G * O + i * G * O + g * O + o;
        };
        auto off_goi = [&](int l, int d, int i, int g, int o) {
            return l * D * G * O * I + d * G * O * I + g * O * I + o * I + i;
        };
        const int lda = to_igo ? G * O : I;
        const int ldb = rnn_pdata.ldb;
        for (int l = 0; l < L; l++) {
            for (int d = 0; d < D; d++) {
                for (int p = 0; p < n_parts; p++) {
                    int g = (p > 0) ? parts[p - 1] : 0;
                    int m_p = to_igo ? parts[p] * O : I;
                    int k_p = to_igo ? I : parts[p] * O;
                    dnnl_status_t st;
                    if (type_o == data_type::bf16) {
                        st = gemm_bf16bf16f32_pack("A", "N", "N", &m_p, &n,
                                &k_p, &lda, &ldb,
                                (bfloat16_t *)&input_tr[to_igo
                                                ? off_igo(l, d, 0, g, 0)
                                                : off_goi(l, d, 0, g, 0)],
                                (bfloat16_t *)output);
                    } else {
                        st = sgemm_pack("A", "N", "N", &m_p, &n, &k_p, &lda,
                                &ldb,
                                (float *)&input_tr[to_igo
                                                ? off_igo(l, d, 0, g, 0)
                                                : off_goi(l, d, 0, g, 0)],
                                (float *)output);
                    }
                    assert(st == dnnl_success);
                    MAYBE_UNUSED(st);
                    output += size_packed_cell[p] / sizeof(out_data_t);
                }
            }
        }
        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
