/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef CPU_JIT_REORDER_HPP
#define CPU_JIT_REORDER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "cpu_reorder_pd.hpp"
#include "cpu_primitive.hpp"

#include "simple_reorder.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace impl_dtype = mkldnn::impl::data_type;
namespace impl_mfmt = mkldnn::impl::memory_format;

template<impl::data_type_t type>
using data_t = typename prec_traits<type>::type;

template<impl::memory_format_t fmt_i, impl::memory_format_t fmt_o,
    impl::data_type_t type_i, impl::data_type_t type_o>
using enable_if_8i8o = typename utils::enable_if<
        ((fmt_i == impl_mfmt::gOIhw8i8o && fmt_o == impl_mfmt::gOIhw8o8i)
         || (fmt_i == impl_mfmt::OIhw8i8o && fmt_o == impl_mfmt::OIhw8o8i))
        && type_i == impl_dtype::f32 && type_o == impl_dtype::f32>::type;

#define JIT_REORDER_TEMPL_DECL \
    impl::data_type_t type_i, impl::memory_format_t fmt_i, \
    impl::data_type_t type_o, impl::memory_format_t fmt_o, bool order_keep
#define JIT_REORDER_TEMPL_INST \
   type_i, fmt_i, type_o, fmt_o, order_keep

template<JIT_REORDER_TEMPL_DECL, typename spec = void>
struct jit_reorder_kernel_f32 {};

template <JIT_REORDER_TEMPL_DECL>
struct jit_reorder_kernel_f32<JIT_REORDER_TEMPL_INST,
    enable_if_8i8o<fmt_i, fmt_o, type_i, type_o>> : public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_reorder_kernel_f32)

    void (*jit_ker_)(const data_t<type_i> *input, data_t<type_o> *output);
    void operator()(const data_t<type_i> *input, data_t<type_o> *output)
    { jit_ker_(input, output); }

    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        return mayiuse(avx2)
            && simple_fmt_check(order_keep, fmt_i, fmt_o, input_d, output_d)
            && simple_attr_check(attr, false);
    }

    jit_reorder_kernel_f32(const cpu_reorder_pd_t &rconf)
        : jit_generator()
    {
        Xbyak::Reg64 reg_input = abi_param1;
        Xbyak::Reg64 reg_output = abi_param2;

        const int blksize = 8;
        const int typesize_i = sizeof(data_t<type_i>);
        const int typesize_o = sizeof(data_t<type_o>);
        const auto dims = rconf.input_pd()->desc()->dims;
        constexpr bool w_grps = fmt_i == impl::memory_format::gOIhw8i8o;

        preamble();

        mov(reg_alpha, float2int(rconf.alpha()));
        mov(reg_beta, float2int(rconf.beta()));
        movq(Xbyak::Xmm(14), reg_alpha);
        movq(Xbyak::Xmm(15), reg_beta);
        vbroadcastss(ymm_al, Xbyak::Xmm(14));
        vbroadcastss(ymm_bt, Xbyak::Xmm(15));

        for (int ih = 0; ih < dims[w_grps + 2]; ++ih) {
            for (int iw = 0; iw < dims[w_grps + 3]; ++iw) {
                for (int i = 0; i < blksize; i++) {
                    vmovups(Ymm(i), ptr[reg_input + i * blksize * typesize_i]);
                }

                if (rconf.alpha() != 1) {
                    for (int i = 0; i < blksize; i++)
                        vmulps(Ymm(i), Ymm(i), ymm_al);
                }

                for (int i = 0; i < blksize / 2; i++) {
                    vunpcklps(Ymm(blksize + i), Ymm(2 * i), Ymm(2 * i + 1));
                    vunpckhps(Ymm(i), Ymm(2 * i), Ymm(2 * i + 1));
                }

                const unsigned int lfloat = 0x44;
                const unsigned int ufloat = 0xee;
                for (int i = 0; i < blksize / 2; i++) {
                    int j = i % 2 == 0 ? blksize + i : i - 1;
                    vshufps(Ymm(blksize / 2 + 2 * i), Ymm(j), Ymm(j + 1),
                            lfloat);
                    vshufps(Ymm(blksize / 2 + 2 * i + 1), Ymm(j), Ymm(j + 1),
                            ufloat);
                }

                const unsigned int lquad = 0x20;
                for (int i = 0; i < blksize / 2; i++)
                    vperm2f128(Ymm(i), Ymm(blksize / 2 + i), Ymm(blksize + i),
                            lquad);

                const unsigned int uquad = 0x31;
                for (int i = blksize / 2; i < blksize; i++)
                    vperm2f128(Ymm(i), Ymm(i), Ymm(blksize / 2 + i), uquad);

                if (rconf.beta() != 0) {
                    for (int i = 0; i < blksize; i++) {
                        vmovups(Ymm(blksize + i),
                                ptr[reg_output + i * blksize * typesize_o]);
                        vmulps(Ymm(blksize + i), Ymm(blksize + i), ymm_bt);
                        vaddps(Ymm(i), Ymm(i), Ymm(blksize + i));
                    }
                }

                for (int i = 0; i < blksize; i++)
                    vmovups(ptr[reg_output + i * blksize * typesize_o], Ymm(i));

                add(reg_input, blksize * blksize * typesize_i);
                add(reg_output, blksize * blksize * typesize_o);
            }
        }
        postamble();
        jit_ker_ = (void (*)(const data_t<type_i>*, data_t<type_o>*))getCode();
    }

private:
    using Ymm = Xbyak::Ymm;

    Xbyak::Reg64 reg_alpha = rbx;
    Xbyak::Reg64 reg_beta = abi_not_param1;

    Ymm ymm_al = Ymm(14);
    Ymm ymm_bt = Ymm(15);
};

/* high level class declaration */
template <JIT_REORDER_TEMPL_DECL, typename spec=void>
struct jit_reorder_t : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        pd_t(const cpu_memory_pd_t *input_pd, const cpu_memory_pd_t *output_pd,
                const primitive_attr_t *attr)
            : cpu_reorder_pd_t(input_pd, output_pd, attr) {}

        DECLARE_COMMON_PD_T("jit:any", jit_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                const memory_pd_t *input_pd, const memory_pd_t *output_pd,
                const primitive_attr_t *attr) {
            assert(input_pd->engine()->kind() == engine_kind::cpu);
            assert(output_pd->engine()->kind() == engine_kind::cpu);

            bool args_ok = true
                && input_pd->desc()->data_type == type_i
                && output_pd->desc()->data_type == type_o
                && jit_reorder_kernel_f32<JIT_REORDER_TEMPL_INST, spec>::
                is_applicable(input_pd->desc(), output_pd->desc(), attr);
            if (!args_ok)
                return impl::status::invalid_arguments;

            auto _pd = new pd_t((const cpu_memory_pd_t *)input_pd,
                    (const cpu_memory_pd_t *)output_pd, attr);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }
    };

    jit_reorder_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
       : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {
           kernel_ = new jit_reorder_kernel_f32<JIT_REORDER_TEMPL_INST, spec>(
                   conf_);
       }

    ~jit_reorder_t() { delete kernel_; }

    enable_if_8i8o<fmt_i, fmt_o, type_i, type_o> execute_reorder(
            const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d,
            const data_t<type_i> *input,
            data_t<type_o> *output) {
        constexpr bool w_grps = fmt_i == impl_mfmt::gOIhw8i8o;
        const auto &dims = input_d.dims();
        constexpr int blksize = 8;

        const int _G = w_grps ? dims[0] : 1;

#       pragma omp parallel for collapse(3) schedule(static)
        for (int g = 0; g < _G; ++g) {
            for (int o = 0; o < dims[w_grps + 0] / blksize; ++o) {
                for (int i = 0; i < dims[w_grps + 1] / blksize; ++i) {
                    auto i_ptr = &input[input_d.blk_off<!w_grps>(g, o, i)];
                    auto o_ptr = &output[output_d.blk_off<!w_grps>(g, o, i)];
                    (*kernel_)(i_ptr, o_ptr);
                }
            }
        }
    }

    virtual void execute(event_t *e) {
        auto input = reinterpret_cast<const data_t<type_i> *>(input_memory(0));
        auto output = reinterpret_cast<data_t<type_o> *>(memory());

        execute_reorder(conf_.input_pd()->desc(), conf_.output_pd()->desc(),
                input, output);

        e->set_state(event_t::ready);
    }

private:
    pd_t conf_;
    jit_reorder_kernel_f32<JIT_REORDER_TEMPL_INST, spec> *kernel_;
};

#undef JIT_REORDER_TEMPL_DECL
#undef JIT_REORDER_TEMPL_INST

}
}
}

#endif
