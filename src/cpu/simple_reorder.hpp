/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef CPU_SIMPLE_REORDER_HPP
#define CPU_SIMPLE_REORDER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "primitive.hpp"
#include "cpu_engine.hpp"

namespace mkldnn { namespace impl { namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::precision;

template<impl::precision_t prec>
    using data_t = typename prec_trait<prec>::type;

namespace fmt_order {
    const bool keep = true;
    const bool reverse = false;
    const bool any = keep;
}

namespace spec {
struct direct_copy {};
struct reference {};
}

#define SIMPLE_REORDER_TEMPL_DECL \
    impl::precision_t prec_i, impl::memory_format_t fmt_i, \
    impl::precision_t prec_o, impl::memory_format_t fmt_o, bool order_keep
#define SIMPLE_REORDER_TEMPL_CALL \
    prec_i, fmt_i, prec_o, fmt_o, order_keep

/* specific reorders: common template */
template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_impl {};

/* specific reorders: implementation */

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == nchw && fmt_o == nChw8c>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const data_t<prec_i> *input,
        data_t<prec_o> *output) {
        const auto &nchw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();

        auto ker = [&](const data_t<prec_i> *i, data_t<prec_o> *o) {
            for (uint32_t w = 0; w < dims[3]; ++w) {
                for (uint32_t c = 0; c < 8; ++c) {
                    const auto nchw_off =
                        c*nchw_d.blocking_desc().strides[0][1] + w;
                    if (order_keep) {
                        o[w*8 + c] = data_t<prec_o>(i[nchw_off]);
                    } else {
                        o[nchw_off] = data_t<prec_o>(i[w*8 + c]);
                    }
                }
            }
        };

#       pragma omp parallel for collapse(3) schedule(static)
        for (uint32_t n = 0; n < dims[0]; ++n) {
            for (uint32_t C = 0; C < dims[1]/8; ++C) {
                for (uint32_t h = 0; h < dims[2]; ++h) {
                    constexpr uint32_t i_c_mult = order_keep ? 8 : 1;
                    constexpr uint32_t o_c_mult = order_keep ? 1 : 8;
                    auto i = &input[input_d.blk_off(n, i_c_mult * C, h)];
                    auto o = &output[output_d.blk_off(n, o_c_mult * C, h)];
                    ker(i, o);
                }
            }
        }

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == nchw && fmt_o == nhwc>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const data_t<prec_i> *input,
        data_t<prec_o> *output) {
        const auto &dims = input_d.dims();

        auto ker = [&](const data_t<prec_i> *i, data_t<prec_o> *o) {
            for (uint32_t w = 0; w < dims[3]; ++w) {
                for (uint32_t c = 0; c < dims[1]; ++c) {
                    const auto &is = input_d.blocking_desc().strides[0];
                    const auto &os = output_d.blocking_desc().strides[0];
                    if (order_keep) {
                        o[w * os[3] + c] = data_t<prec_o>(i[c * is[1] + w]);
                    } else {
                        o[c * os[1] + w] = data_t<prec_o>(i[w * is[3] + c]);
                    }
                }
            }
        };

#       pragma omp parallel for collapse(2) schedule(static)
        for (uint32_t n = 0; n < dims[0]; ++n) {
            for (uint32_t h = 0; h < dims[2]; ++h) {
                auto i = &input[input_d.blk_off(n, 0, h)];
                auto o = &output[output_d.blk_off(n, 0, h)];
                ker(i, o);
            }
        }

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == goihw && fmt_o == gOIhw8i8o)
        || (fmt_i == oihw && fmt_o == OIhw8i8o)
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const data_t<prec_i> *input,
        data_t<prec_o> *output) {
        constexpr bool w_groups = fmt_i == goihw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();

        auto ker = [&](const data_t<prec_i> *i, data_t<prec_o> *o) {
            for (uint32_t ic = 0; ic < 8; ++ic) {
                for (uint32_t oc = 0; oc < 8; ++oc) {
                    const auto _g_oihw_off =
                        oc*_g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic*_g_oihw_d.blocking_desc().strides[0][w_groups + 1];
                    if (order_keep) {
                        o[ic*8 + oc] = data_t<prec_o>(i[_g_oihw_off]);
                    } else {
                        o[_g_oihw_off] = data_t<prec_o>(i[ic*8 + oc]);
                    }
                }
            }
        };

        const uint32_t _G = w_groups ? dims[0] : 1u;

#       pragma omp parallel for collapse(5) schedule(static)
        for (uint32_t g = 0; g < _G; ++g) {
            for (uint32_t O = 0; O < dims[w_groups + 0]/8; ++O) {
                for (uint32_t I = 0; I < dims[w_groups + 1]/8; ++I) {
                    for (uint32_t h = 0; h < dims[w_groups + 2]; ++h) {
                        for (uint32_t w = 0; w < dims[w_groups + 3]; ++w) {
                            constexpr uint32_t i_mult = order_keep ? 8 : 1;
                            constexpr uint32_t o_mult = order_keep ? 1 : 8;
                            auto i = &input[input_d.blk_off<!w_groups>(g,
                                    i_mult * O, i_mult * I, h, w)];
                            auto o = &output[output_d.blk_off<!w_groups>(
                                    g, o_mult * O, o_mult * I, h, w)];
                            ker(i, o);
                        }
                    }
                }
            }
        }

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == any && fmt_o == any && order_keep == fmt_order::any,
    spec::direct_copy>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == output_d.format() && input_d.is_dense();
    }

    static status_t execute(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const data_t<prec_i> *input,
        data_t<prec_o> *output) {
        assert(input_d.is_dense());

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const size_t nelems = input_d.nelems();

#       pragma omp parallel for schedule(static)
        for (size_t e = 0; e < nelems; ++e) {
            output[e] = data_t<prec_o>(input[e]);
        }

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == any && fmt_o == any && order_keep == fmt_order::any,
    spec::reference>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return true;
    }

    static status_t execute(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const data_t<prec_i> *input,
        data_t<prec_o> *output) {
        const size_t nelems = input_d.nelems();

#       pragma omp parallel for schedule(static)
        for (size_t e = 0; e < nelems; ++e) {
            output[output_d.off_l(e)] =
                data_t<prec_o>(input[input_d.off_l(e)]);
        }

        return success;
    }
};


/* high level class declaration */

template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
class simple_reorder: public primitive {
private:
    const impl::reorder_primitive_desc_t &_rpd;

    status_t _execute() {
        const size_t oi = this->input()[0].output_index;
        auto *input = reinterpret_cast<const data_t<prec_i>*>(
                this->input()[0].primitive->output()[oi]->memory_const());
        auto *output = reinterpret_cast<data_t<prec_o>*>(
                this->output()[0]->memory());

        return simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL, spec>::execute(
                this->_rpd.input.memory_desc, this->_rpd.output.memory_desc,
                input, output);
    }

protected:
    status_t execute_impl() { return _execute(); }

public:
    simple_reorder(const reorder_primitive_desc_t &rpd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(rpd, const_cast<impl::engine*>(rpd.base.engine), not_ready)
        , _rpd(_primitive_desc.reorder)
    {
        _input.push_back(inputs[0]);
        _output.push_back(outputs[0]);
    }

    /* static magic */
    static status_t reorder_primitive_desc_init(
            primitive_desc_t *primitive_desc,
            const memory_primitive_desc_t *input,
            const memory_primitive_desc_t *output)
    {
        bool args_ok = true
            && input->memory_desc.precision == prec_i
            && output->memory_desc.precision == prec_o
            && input->base.engine == output->base.engine
            && simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL, spec>::
                    is_applicable(input->memory_desc, output->memory_desc);
        if (!args_ok)
            return invalid_arguments;

        reorder_primitive_desc_t rpd;
        rpd.base.primitive_kind = reorder;
        rpd.base.engine = input->base.engine;
        rpd.base.implementation = reinterpret_cast<const void*>(&implementation);
        rpd.input = *input;
        rpd.output = *output;
        primitive_desc->reorder = rpd;

        return success;
    }

private:
    static status_t create(primitive **aprimitive,
            const primitive_desc_t *primitive_desc,
            const primitive_at_t inputs[], const primitive *outputs[])
    {
        assert(primitive_desc->base.primitive_kind == reorder);
        auto &rpd = primitive_desc->reorder;
        *aprimitive = new simple_reorder(rpd, inputs, outputs);
        return aprimitive ? success : out_of_memory;
    }
    static const primitive_impl implementation;
};

/* XXX: awful style */
template <SIMPLE_REORDER_TEMPL_DECL, typename spec> const primitive_impl
simple_reorder<SIMPLE_REORDER_TEMPL_CALL, spec>::implementation = {
    simple_reorder<SIMPLE_REORDER_TEMPL_CALL, spec>::create,
};

#undef SIMPLE_REORDER_TEMPL_DECL
#undef SIMPLE_REORDER_TEMPL_CALL

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
