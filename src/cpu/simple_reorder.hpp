/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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
#include "cpu_reorder_pd.hpp"
#include "type_helpers.hpp"
#include "cpu_primitive.hpp"
#include "cpu_engine.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::data_type;

template<impl::data_type_t type>
using data_t = typename prec_trait<type>::type;

namespace fmt_order {
    const bool keep = true;
    const bool reverse = false;
    const bool any = keep;
}

namespace spec {
struct direct_copy {};
struct direct_copy_except_dim_0 {};
struct reference {};
}

#define SIMPLE_REORDER_TEMPL_DECL \
    impl::data_type_t type_i, impl::memory_format_t fmt_i, \
    impl::data_type_t type_o, impl::memory_format_t fmt_o, bool order_keep
#define SIMPLE_REORDER_TEMPL_CALL \
    type_i, fmt_i, type_o, fmt_o, order_keep

/* specific reorders: common template */
template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_impl {};

/* specific reorders: implementation */

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == nchw && (fmt_o == nChw8c || fmt_o == nChw16c)
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const data_t<type_i> *input,
        data_t<type_o> *output,
        const double alpha, const double beta) {
        const auto &nchw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize = fmt_o == nChw8c ? 8 : 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int w = 0; w < dims[3]; ++w) {
                    for (int c = 0; c < blksize; ++c) {
                        const auto nchw_off =
                        c*nchw_d.blocking_desc().strides[0][1] + w;
                        if (order_keep) {
                            o[w*blksize + c] = data_t<type_o>(i[nchw_off]);
                        } else {
                            o[nchw_off] = data_t<type_o>(i[w*blksize + c]);
                        }
                    }
                }
            } else {
                for (int w = 0; w < dims[3]; ++w) {
                    for (int c = 0; c < blksize; ++c) {
                        const auto nchw_off =
                        c*nchw_d.blocking_desc().strides[0][1] + w;
                        if (order_keep) {
                            o[w*blksize + c] = alpha*data_t<type_o>(i[nchw_off])
                                        + beta*o[w*blksize + c];
                        } else {
                            o[nchw_off] = alpha*data_t<type_o>(i[w*blksize + c])
                                        + beta*o[nchw_off];
                        }
                    }
                }
            }
        };

#       pragma omp parallel for collapse(3) schedule(static)
        for (int n = 0; n < dims[0]; ++n) {
            for (int C = 0; C < dims[1]/blksize; ++C) {
                for (int h = 0; h < dims[2]; ++h) {
                    constexpr int i_c_mult = order_keep ? blksize : 1;
                    constexpr int o_c_mult = order_keep ? 1 : blksize;
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
        const memory_desc_wrapper &output_d, const data_t<type_i> *input,
        data_t<type_o> *output,
        const double alpha, const double beta) {
        const auto &dims = input_d.dims();

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int w = 0; w < dims[3]; ++w) {
                    for (int c = 0; c < dims[1]; ++c) {
                        const auto &is = input_d.blocking_desc().strides[0];
                        const auto &os = output_d.blocking_desc().strides[0];
                        if (order_keep) {
                            o[w * os[3] + c] = data_t<type_o>(i[c * is[1] + w]);
                        } else {
                            o[c * os[1] + w] = data_t<type_o>(i[w * is[3] + c]);
                        }
                    }
                }
            } else {
                for (int w = 0; w < dims[3]; ++w) {
                    for (int c = 0; c < dims[1]; ++c) {
                        const auto &is = input_d.blocking_desc().strides[0];
                        const auto &os = output_d.blocking_desc().strides[0];
                        if (order_keep) {
                            o[w * os[3] + c] =
                                alpha*data_t<type_o>(i[c * is[1] + w])
                                + beta*o[w * os[3] + c];
                        } else {
                            o[c * os[1] + w] =
                                alpha*data_t<type_o>(i[w * is[3] + c])
                                + beta*o[c * os[1] + w];
                        }
                    }
                }
            }
        };

#       pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < dims[0]; ++n) {
            for (int h = 0; h < dims[2]; ++h) {
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
        (fmt_i == goihw && (fmt_o == gOIhw8i8o || fmt_o == gOIhw16i16o))
        || (fmt_i == oihw && (fmt_o == OIhw8i8o || fmt_o == OIhw16i16o))
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const data_t<type_i> *input,
        data_t<type_o> *output,
        const double alpha, const double beta) {
        constexpr bool w_groups = fmt_i == goihw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize =
            (fmt_o == OIhw8i8o || fmt_o == gOIhw8i8o) ? 8 : 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto _g_oihw_off =
                        oc*_g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic*_g_oihw_d.blocking_desc().strides[0][w_groups + 1];
                    if (order_keep) {
                        o[ic*blksize + oc] = data_t<type_o>(i[_g_oihw_off]);
                    } else {
                        o[_g_oihw_off] = data_t<type_o>(i[ic*blksize + oc]);
                    }
                }
                }
            } else {
                for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto _g_oihw_off =
                        oc*_g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic*_g_oihw_d.blocking_desc().strides[0][w_groups + 1];
                    if (order_keep) {
                        o[ic*blksize + oc] =
                            alpha*data_t<type_o>(i[_g_oihw_off])
                            + beta*o[ic*blksize + oc];
                    } else {
                        o[_g_oihw_off] =
                            alpha*data_t<type_o>(i[ic*blksize + oc])
                            + beta*o[_g_oihw_off];
                    }
                }
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

#       pragma omp parallel for collapse(5) schedule(static)
        for (int g = 0; g < _G; ++g) {
            for (int O = 0; O < dims[w_groups + 0]/blksize; ++O) {
                for (int I = 0; I < dims[w_groups + 1]/blksize; ++I) {
                    for (int h = 0; h < dims[w_groups + 2]; ++h) {
                        for (int w = 0; w < dims[w_groups + 3]; ++w) {
                            constexpr int i_mult = order_keep ? blksize : 1;
                            constexpr int o_mult = order_keep ? 1 : blksize;
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
        (fmt_i == gOIhw8i8o && fmt_o == gOIhw8o8i)
        || (fmt_i == OIhw8i8o && fmt_o == OIhw8o8i)
        || (fmt_i == gOIhw16i16o && fmt_o == gOIhw16o16i)
        || (fmt_i == OIhw16i16o && fmt_o == OIhw16o16i)
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const data_t<type_i> *input,
        data_t<type_o> *output,
        const double alpha, const double beta) {
        constexpr bool w_groups = (fmt_i == gOIhw8i8o || fmt_i == gOIhw16i16o);

        const auto &dims = input_d.dims();
        constexpr int blksize =
            (fmt_i == OIhw8i8o || fmt_i == gOIhw8i8o) ? 8 : 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const int o_idx = ic*blksize + oc;
                    const int i_idx = oc*blksize + ic;
                    o[o_idx] = (alpha == 1.0 && beta == 0.0)
                        ? data_t<type_o>(i[i_idx])
                        : alpha*data_t<type_o>(i[i_idx]) + beta*o[o_idx];
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

#       pragma omp parallel for collapse(5) schedule(static)
        for (int g = 0; g < _G; ++g) {
            for (int o = 0; o < dims[w_groups + 0]/blksize; ++o) {
                for (int i = 0; i < dims[w_groups + 1]/blksize; ++i) {
                    for (int h = 0; h < dims[w_groups + 2]; ++h) {
                        for (int w = 0; w < dims[w_groups + 3]; ++w) {
                            auto i_ptr = &input[input_d.blk_off<!w_groups>(g,
                                    o, i, h, w)];
                            auto o_ptr = &output[output_d.blk_off<!w_groups>(g,
                                    o, i, h, w)];
                            ker(i_ptr, o_ptr);
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
        /* FIXME: is the formule correct? */
        return input_d.format() == output_d.format() && input_d.is_dense()
            && output_d.is_dense();
    }

    static status_t execute(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const data_t<type_i> *input,
        data_t<type_o> *output,
        const double alpha, const double beta) {
        assert(input_d.is_dense());

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const size_t nelems = input_d.nelems();

        if (alpha == 1.0 && beta == 0.0) {
#           pragma omp parallel for schedule(static)
            for (size_t e = 0; e < nelems; ++e) {
                output[e] = data_t<type_o>(input[e]);
            }
        } else {
#           pragma omp parallel for schedule(static)
            for (size_t e = 0; e < nelems; ++e) {
                output[e] = alpha*data_t<type_o>(input[e]) + beta*output[e];
            }
        }

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == any && fmt_o == any && order_keep == fmt_order::any,
    spec::direct_copy_except_dim_0>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        auto is_dense_no_0 = [](const memory_desc_wrapper &data_d) {
            return nelems_no_dim_0(data_d) == _size_no_dim_0(data_d);
        };
        /* FIXME: is the formula correct? */
        return input_d.format() == output_d.format() && is_dense_no_0(input_d)
            && is_dense_no_0(output_d);
    }

    static status_t execute(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const data_t<type_i> *input,
        data_t<type_o> *output,
        const double alpha, const double beta) {

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const int N = input_d.dims()[0];
        const size_t is = input_d.blocking_desc().strides[0][0];
        const size_t os = output_d.blocking_desc().strides[0][0];
        const size_t nelems_no_d0 = nelems_no_dim_0(input_d);

        if (alpha == 1.0 && beta == 0.0) {
#           pragma omp parallel for collapse(2) schedule(static)
            for (int n = 0; n < N; ++n) {
                for (size_t e = 0; e < nelems_no_d0; ++e) {
                    output[os*n + e] = data_t<type_o>(input[is*n + e]);
                }
            }
        } else {
#           pragma omp parallel for collapse(2) schedule(static)
            for (int n = 0; n < N; ++n) {
                for (size_t e = 0; e < nelems_no_d0; ++e) {
                    output[os*n + e] = alpha*data_t<type_o>(input[is*n + e])
                        + beta*output[os*n + e];
                }
            }
        }

        return success;
    }

private:
    static size_t nelems_no_dim_0(const memory_desc_wrapper &data_d) {
        const int ndims = data_d.ndims();
        if (ndims <= 1) return 1;
        return utils::array_product(data_d.dims() + 1, data_d.ndims() - 1);
    }

    static size_t _size_no_dim_0(const memory_desc_wrapper &data_d) {
        size_t max_size = 0;
        auto &blk = data_d.blocking_desc();
        for (int d = 1; d < data_d.ndims(); ++d) {
            auto block = blk.block_dims[d];
            max_size = nstl::max(max_size,
                    size_t(blk.padding_dims[d]/block)*blk.strides[0][d]);
            if (block > 1)
                max_size = nstl::max(max_size,
                        size_t(block*blk.strides[1][d]));
        }
        return max_size;
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
        const memory_desc_wrapper &output_d, const data_t<type_i> *input,
        data_t<type_o> *output,
        const double alpha, const double beta) {
        const size_t nelems = input_d.nelems();

        if (alpha == 1.0 && beta == 0.0) {
#           pragma omp parallel for schedule(static)
            for (size_t e = 0; e < nelems; ++e) {
                output[output_d.off_l(e)] =
                    data_t<type_o>(input[input_d.off_l(e)]);
            }
        } else {
#           pragma omp parallel for schedule(static)
            for (size_t e = 0; e < nelems; ++e) {
                output[output_d.off_l(e)] =
                    alpha*data_t<type_o>(input[input_d.off_l(e)])
                    + beta*output[output_d.off_l(e)];
            }
        }

        return success;
    }
};

/* high level class declaration */

template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_t: public cpu_primitive_t {
    struct pd_t: public cpu_reorder_pd_t {
        pd_t(const cpu_memory_pd_t *input_pd, const cpu_memory_pd_t *output_pd,
                const double alpha, const double beta)
            : cpu_reorder_pd_t(input_pd, output_pd, alpha, beta) {}

        DECLARE_COMMON_PD_T(simple_reorder_t);

        static status_t create(
                reorder_pd_t **reorder_pd,
                const memory_pd_t *input_pd,
                const memory_pd_t *output_pd,
                const double alpha,
                const double beta) {
            assert(input_pd->engine()->kind() == engine_kind::cpu);
            assert(output_pd->engine()->kind() == engine_kind::cpu);
            bool args_ok = true
                && input_pd->desc()->data_type == type_i
                && output_pd->desc()->data_type == type_o
                && simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL, spec>::
                is_applicable(input_pd->desc(), output_pd->desc());
            if (!args_ok)
                return invalid_arguments;

            auto _pd = new pd_t((const cpu_memory_pd_t *)input_pd,
                    (const cpu_memory_pd_t *)output_pd, alpha, beta);
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }
    };

    simple_reorder_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}

    virtual void execute(event_t *e) {
        auto input = reinterpret_cast<const data_t<type_i> *>(
                this->input_memory(0));
        auto output = reinterpret_cast<data_t<type_o> *>(this->memory());
        simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL, spec>::execute(
                conf_.input_pd()->desc(), conf_.output_pd()->desc(),
                input, output, conf_.alpha(), conf_.beta());
        e->set_state(event_t::ready);
    }

private:
    pd_t conf_;
};

#undef SIMPLE_REORDER_TEMPL_DECL
#undef SIMPLE_REORDER_TEMPL_CALL

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
