/*******************************************************************************
 * Copyright 2018 Intel Corporation
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

#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "simple_q10n.hpp"
#include "cpu_reorder_pd.hpp"
#include "../gemm/os_blas.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t type_i, data_type_t type_o>
struct rnn_data_reorder_t : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        pd_t(const cpu_memory_pd_t *input_pd, const cpu_memory_pd_t *output_pd,
                const primitive_attr_t *attr)
            : cpu_reorder_pd_t(input_pd, output_pd, attr) {}

        DECLARE_COMMON_PD_T("rnn_data_reorder", rnn_data_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                const memory_pd_t *input_pd, const memory_pd_t *output_pd,
                const primitive_attr_t *attr) {
            using namespace memory_format;
            using namespace data_type;
            assert(input_pd->engine()->kind() == engine_kind::cpu);
            assert(output_pd->engine()->kind() == engine_kind::cpu);

            const memory_desc_wrapper id(input_pd), od(output_pd);
            bool args_ok = true
                    && id.data_type() == type_i
                    && od.data_type() == type_o
                    && utils::one_of(id.format(), tnc, ldsnc)
                    && od.format() == id.format();
            if (!args_ok) return status::invalid_arguments;

            auto _pd = new pd_t((const cpu_memory_pd_t *)input_pd,
                    (const cpu_memory_pd_t *)output_pd, attr);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }
    };

private:
    typedef typename prec_traits<type_i>::type in_data_t;
    typedef typename prec_traits<type_o>::type out_data_t;

    rnn_data_reorder_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    virtual void execute(event_t *e) const {
        auto input = reinterpret_cast<const in_data_t *>(input_memory(0));
        auto output = reinterpret_cast<out_data_t *>(memory());
        const memory_desc_wrapper &input_d = pd()->input_pd();
        const memory_desc_wrapper &output_d = pd()->output_pd();
        const round_mode_t rmode = pd()->attr()->round_mode_;
        const size_t nelems = input_d.nelems();
        const float scale = pd()->attr()->rnn_data_qparams_.scale_;
        const float shift = pd()->attr()->rnn_data_qparams_.shift_;

        parallel_nd(nelems, [&](size_t i) {
            float in = (float)input[input_d.off_l(i)] * scale + shift;
            output[output_d.off_l(i)] = qz_a1b0<float, out_data_t>()(in, rmode);
        });

        e->set_state(event_t::ready);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <data_type_t type_i, data_type_t type_o>
struct rnn_weights_reorder_t : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        pd_t(const cpu_memory_pd_t *input_pd, const cpu_memory_pd_t *output_pd,
                const primitive_attr_t *attr)
            : cpu_reorder_pd_t(input_pd, output_pd, attr) {}

        DECLARE_COMMON_PD_T("rnn_weights_reorder", rnn_weights_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                const memory_pd_t *input_pd, const memory_pd_t *output_pd,
                const primitive_attr_t *attr) {
#if !USE_MKL_PACKED_GEMM
            return status::unimplemented;
#endif
            using namespace memory_format;
            assert(input_pd->engine()->kind() == engine_kind::cpu);
            assert(output_pd->engine()->kind() == engine_kind::cpu);
            const memory_desc_wrapper output_d(output_pd);

            const memory_desc_wrapper id(input_pd), od(output_pd);
            bool args_ok = true
                    && id.data_type() == type_i
                    && od.data_type() == type_o
                    && utils::one_of(id.format(), ldigo, ldgoi)
                    && od.format() == rnn_packed
                    && od.rnn_packed_desc().format
                            == mkldnn_ldigo_p
                    && od.rnn_packed_desc().n_parts == 1
                    && attr != nullptr;
            if (!args_ok) return status::invalid_arguments;

            const int mask = attr->rnn_weights_qparams_.mask_;
            if (!utils::one_of(mask, 0, 3)) return status::unimplemented;

            auto _pd = new pd_t((const cpu_memory_pd_t *)input_pd,
                    (const cpu_memory_pd_t *)output_pd, attr);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        virtual status_t init() override {
            status_t status = cpu_reorder_pd_t::init();
            if (status != status::success) return status;

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            const memory_desc_wrapper id(input_pd());
            const size_t nelems = id.nelems();

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(
                    key_reorder_rnn_weights_space, sizeof(int8_t) * nelems);
        }
    };

private:
    typedef typename prec_traits<type_i>::type in_data_t;
    typedef typename prec_traits<type_o>::type out_data_t;

    rnn_weights_reorder_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    virtual void execute(event_t *e) const {
#if USE_MKL_PACKED_GEMM
        auto input = reinterpret_cast<const in_data_t *>(input_memory(0));
        auto output = reinterpret_cast<char *>(memory());
        const memory_desc_wrapper &input_d = pd()->input_pd();
        const memory_desc_wrapper &output_d = pd()->output_pd();
        const auto &dims = input_d.dims();

        const int L = dims[0];
        const int D = dims[1];
        const int I = dims[2];
        const int G = dims[3];
        const int O = dims[4];

        const bool is_igo = input_d.format() == memory_format::ldigo;
        auto off_igo = [&](int l, int d, int i, int g, int o) {
            return l * D * I * G * O + d * I * G * O + i * G * O + g * O + o;
        };
        auto off_goi = [&](int l, int d, int i, int g, int o) {
            return l * D * G * O * I + d * G * O * I + g * O * I + o * I + i;
        };

        /* Quantize input & compute compensation */
        int8_t *quantized = (int8_t *)scratchpad().template get<void>(
                memory_tracking::names::key_reorder_rnn_weights_space);
        float *comp = reinterpret_cast<float *>(
                output + output_d.rnn_packed_desc().offset_compensation);
        const round_mode_t rmode = pd()->attr()->round_mode_;
        const float *scales = pd()->attr()->rnn_weights_qparams_.scales_;
        const int mask = pd()->attr()->rnn_weights_qparams_.mask_;

        parallel_nd(L, D, G, O, [&](int l, int d, int g, int o) {
            int32_t compensation = 0;
            const float s = scales[(mask == 0) ? 0 : g * O + o];
            for (int i = 0; i < I; i++) {
                int8_t q = qz_b0<in_data_t, out_data_t>()(
                        input[is_igo ? off_igo(l, d, i, g, o) :
                                       off_goi(l, d, i, g, o)],
                        s, rmode);
                compensation += (int32_t)q;
                quantized[off_igo(l, d, i, g, o)] = q;
            }
            comp[l * D * G * O + d * G * O + g * O + o]
                    = saturate<float>(compensation);
        });

        /* Pack */
        int n_parts = output_d.rnn_packed_desc().n_parts;
        const size_t *size_packed_cell
                = output_d.rnn_packed_desc().part_pack_size;
        const int *parts = output_d.rnn_packed_desc().parts;
        const int n = output_d.rnn_packed_desc().n;
        char *to_pack = output;
        for (int l = 0; l < L; l++) {
            for (int d = 0; d < D; d++) {
                for (int p = 0; p < n_parts; p++) {
                    int g = (p > 0) ? parts[p - 1] : 0;
                    int m_p = parts[p] * O;
                    int k_p = I;
                    cblas_gemm_s8u8s32_pack(CblasColMajor, CblasAMatrix,
                            CblasNoTrans, m_p, n, k_p,
                            &quantized[off_igo(l, d, 0, g, 0)], G * O, to_pack);
                    to_pack += size_packed_cell[p];
                }
            }
        }
#endif
        e->set_state(event_t::ready);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <>
struct rnn_weights_reorder_t<data_type::f32, data_type::f32>
        : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        pd_t(const cpu_memory_pd_t *input_pd, const cpu_memory_pd_t *output_pd,
                const primitive_attr_t *attr)
            : cpu_reorder_pd_t(input_pd, output_pd, attr) {}

        DECLARE_COMMON_PD_T("rnn_weights_reorder", rnn_weights_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                const memory_pd_t *input_pd, const memory_pd_t *output_pd,
                const primitive_attr_t *attr) {
#if !USE_MKL_PACKED_GEMM
            return status::unimplemented;
#endif
            using namespace memory_format;
            using namespace data_type;
            assert(input_pd->engine()->kind() == engine_kind::cpu);
            assert(output_pd->engine()->kind() == engine_kind::cpu);
            const memory_desc_wrapper output_d(output_pd);

            const memory_desc_wrapper id(input_pd), od(output_pd);
            bool args_ok = true
                    && id.data_type() == f32
                    && od.data_type() == f32
                    && utils::one_of(id.format(), ldigo, ldgoi)
                    && od.format() == rnn_packed
                    && utils::one_of(od.rnn_packed_desc().format,
                        mkldnn_ldigo_p, mkldnn_ldgoi_p)
                    && attr->has_default_values();
            if (!args_ok) return status::invalid_arguments;

            const int mask = attr->rnn_weights_qparams_.mask_;
            if (!utils::one_of(mask, 0, 3)) return status::unimplemented;

            auto _pd = new pd_t((const cpu_memory_pd_t *)input_pd,
                    (const cpu_memory_pd_t *)output_pd, attr);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }
    };

private:
    rnn_weights_reorder_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    virtual void execute(event_t *e) const {
#if USE_MKL_PACKED_GEMM
        auto input = reinterpret_cast<const float *>(input_memory(0));
        auto output = reinterpret_cast<float *>(memory());
        const memory_desc_wrapper &input_d = pd()->input_pd();
        const memory_desc_wrapper &output_d = pd()->output_pd();
        const auto &dims = input_d.dims();
        const rnn_packed_data_t &rnn_pdata = output_d.rnn_packed_desc();
        const int L = dims[0];
        const int D = dims[1];
        const int I = dims[2];
        const int G = dims[3];
        const int O = dims[4];

        /* Pack */
        bool cross_case = (input_d.format() == memory_format::ldigo
                        && rnn_pdata.format == mkldnn_ldgoi_p)
                || (input_d.format() == memory_format::ldgoi
                        && rnn_pdata.format == mkldnn_ldigo_p);
        auto trans = cross_case ? CblasTrans : CblasNoTrans;
        int n_parts = rnn_pdata.n_parts;
        const size_t *size_packed_cell = rnn_pdata.part_pack_size;
        const int *parts = rnn_pdata.parts;
        const int n = rnn_pdata.n;

        const bool is_igo = input_d.format() == memory_format::ldigo;
        auto off_igo = [&](int l, int d, int i, int g, int o) {
            return l * D * I * G * O + d * I * G * O + i * G * O + g * O + o;
        };
        auto off_goi = [&](int l, int d, int i, int g, int o) {
            return l * D * G * O * I + d * G * O * I + g * O * I + o * I + i;
        };
        for (int l = 0; l < L; l++) {
            for (int d = 0; d < D; d++) {
                for (int p = 0; p < n_parts; p++) {
                    int g = (p > 0) ? parts[p - 1] : 0;
                    int m_p = is_igo ? parts[p] * O : I;
                    int k_p = is_igo ? I : parts[p] * O;
                    int ld = is_igo ? G * O : I;
                    cblas_sgemm_pack(CblasColMajor, CblasAMatrix, trans, m_p, n,
                            k_p, 1.0f, &input[is_igo ? off_igo(l, d, 0, g, 0) :
                                                       off_goi(l, d, 0, g, 0)],
                            ld, output);
                    output += size_packed_cell[p] / sizeof(float);
                }
            }
        }
        e->set_state(event_t::ready);
#endif
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

} // namespace cpu
} // namespace impl
} // namespace mkldnn

#endif
