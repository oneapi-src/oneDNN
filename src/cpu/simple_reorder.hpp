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
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "cpu_reorder_pd.hpp"
#include "cpu_primitive.hpp"

#if (defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1600) || defined(_MSC_VER)
/* Excluding ICC 16.0 from adding simd because it results in accuracy issues.
 * MSC doesn't support simd in _pragma */
#    define pragma_simd
#else
#    define pragma_simd _Pragma("simd")
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::data_type;

using namespace mkldnn::impl::utils;

template<impl::data_type_t type>
using data_t = typename prec_traits<type>::type;

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

#define DECLARE_COMMON_PARAMS() \
        const memory_desc_wrapper &input_d = pd->input_pd(); \
        const memory_desc_wrapper &output_d = pd->output_pd(); \
        const double alpha_ = pd->alpha(); \
        const double beta_ = pd->beta();

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

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &nchw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize = fmt_o == nChw8c ? 8 : 16;

        const float alpha = alpha_, beta = beta_;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int w = 0; w < dims[3]; ++w) {
                    for (int c = 0; c < blksize; ++c) {
                        const auto nchw_off =
                            c * nchw_d.blocking_desc().strides[0][1] + w;
                        if (order_keep) {
                            o[w * blksize + c] = data_t<type_o>(i[nchw_off]);
                        } else {
                            o[nchw_off] = data_t<type_o>(i[w * blksize + c]);
                        }
                    }
                }
            } else {
                for (int w = 0; w < dims[3]; ++w) {
                    for (int c = 0; c < blksize; ++c) {
                        const auto nchw_off =
                            c * nchw_d.blocking_desc().strides[0][1] + w;
                        if (order_keep) {
                            o[w * blksize + c] = data_t<type_o>(
                                alpha * i[nchw_off]
                                + (beta ? beta * o[w * blksize + c] : 0));
                        } else {
                            o[nchw_off] = data_t<type_o>(
                                alpha * i[w * blksize + c]
                                + (beta ? beta * o[nchw_off] : 0));
                        }
                    }
                }
            }
        };

#       pragma omp parallel for collapse(3) schedule(static)
        for (int n = 0; n < dims[0]; ++n) {
            for (int C = 0; C < dims[1] / blksize; ++C) {
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
    typename utils::enable_if<
        fmt_i == nhwc && (fmt_o == nChw8c || fmt_o == nChw16c)
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const float alpha = alpha_, beta = beta_;
        const auto &nchw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize = fmt_o == nChw8c ? 8 : 16;
        const auto is = input_d.blocking_desc().strides[0];
        const auto os = output_d.blocking_desc().strides[0];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
#               pragma omp simd collapse(2)
                for (int C = 0; C < dims[1] / blksize; ++C) {
                    for (int c = 0; c < blksize; ++c) {
                        if (order_keep) {
                            o[C * os[1] + c] = data_t<type_o>(i[C * blksize + c]);
                        } else {
                            o[C * blksize + c] = data_t<type_o>(i[C * is[1] + c]);
                        }
                    }
                }
            } else {
#               pragma omp simd collapse(2)
                for (int C = 0; C < dims[1] / blksize; ++C) {
                    for (int c = 0; c < blksize; ++c) {
                        const auto dst_off = order_keep ? C * os[1] + c :
                                                          C * blksize + c;
                        const auto src_off = order_keep ? C * blksize + c :
                                                          C * is[1] + c;
                        o[dst_off] = data_t<type_o>(alpha * i[src_off]
                                     + (beta ? beta * o[dst_off] : 0));
                    }
                }
            }
        };

#       pragma omp parallel for collapse(3) schedule(static)
        for (int n = 0; n < dims[0]; ++n) {
            for (int h = 0; h < dims[2]; ++h) {
                for (int w = 0; w < dims[3]; ++w) {
                    auto i = &input[input_d.blk_off(n, 0, h, w)];
                    auto o = &output[output_d.blk_off(n, 0, h, w)];
                    ker(i, o);
                }
            }
        }

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == chwn
    && (fmt_o == nChw8c || fmt_o == nChw16c)>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();
        const auto i_st = input_d.blocking_desc().strides[0];
        const auto o_st = output_d.blocking_desc().strides[0];

        const float alpha = alpha_, beta = beta_;
        constexpr int blksize = fmt_o == nChw8c ? 8 : 16;
        constexpr int tsize = 16;

        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;

        const auto ci_mult = order_keep ? i_st[1] : 1;
        const auto co_mult = order_keep ? 1 : o_st[1];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
                const int nsize) {
            if (alpha == 1.0 && beta == 0) {
#               pragma omp simd collapse(2)
                for (int n = 0; n < nsize; n++) {
                    for (int c = 0; c < blksize; ++c) {
                        o[n * o_st[0] + c * co_mult] =
                            data_t<type_o>(i[n * i_st[0] + c * ci_mult]);
                    }
                }
            } else {
#               pragma omp simd collapse(2)
                for (int n = 0; n < nsize; n++) {
                    for (int c = 0; c < blksize; ++c) {
                        o[n * o_st[0] + c * co_mult] = data_t<type_o>(
                            alpha * i[n * i_st[0] + c * ci_mult]
                            + (beta ? beta * o[n * o_st[0] + c * co_mult] : 0));
                    }
                }
            }
        };

#       pragma omp parallel for collapse(4) schedule(static)
        for (int C = 0; C < dims[1] / blksize; ++C) {
            for (int h = 0; h < dims[2]; ++h) {
                for (int n = 0; n < dims[0]; n += tsize) {
                    for (int w = 0; w < dims[3]; ++w) {
                        const int nsize =
                            n + tsize > dims[0] ? dims[0] - n : tsize;
                        auto i = &input[n * i_st[0] + C * i_mult * i_st[1]
                            + h * i_st[2] + w * i_st[3]];
                        auto o = &output[n * o_st[0] + C * o_mult * o_st[1]
                            + h * o_st[2] + w * o_st[3]];
                        ker(i, o, nsize);
                    }
                }
            }
        }

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == nChw8c && fmt_o == nChw16c>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();

        constexpr int blksize_16c = 16;
        constexpr int blksize_8c = 8;
        constexpr int ic_mult = order_keep ? 2 : 1;
        constexpr int oc_mult = order_keep ? 1 : 2;

        const float alpha = alpha_, beta = beta_;
        const auto stride_8c = order_keep ? input_d.blocking_desc().strides[0]
            : output_d.blocking_desc().strides[0];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int blk = 0; blk < 2; ++blk){
                    const int i_blk = order_keep ? blk * (int)stride_8c[1]
                        : blk * blksize_8c;
                    const int o_blk = order_keep ? blk * blksize_8c
                        : blk * (int)stride_8c[1];
                    for (int c = 0; c < blksize_8c; ++c) {
                        o[o_blk + c] = i[i_blk + c];
                    }
                }
            } else {
                for (int blk = 0; blk < 2; ++blk){
                    const int i_blk = order_keep ? blk * (int)stride_8c[1]
                        : blk * blksize_8c;
                    const int o_blk = order_keep ? blk * blksize_8c
                        : blk * (int)stride_8c[1];
                    for (int c = 0; c < blksize_8c; ++c) {
                        o[o_blk + c] = data_t<type_o>(
                            alpha * i[i_blk + c]
                            + (beta ? beta * o[o_blk + c] : 0));
                    }
                }
            }
        };

#       pragma omp parallel for collapse(4) schedule(static)
        for (int n = 0; n < dims[0]; ++n) {
            for (int C = 0; C < dims[1] / blksize_16c; ++C) {
                for (int h = 0; h < dims[2]; ++h) {
                    for (int w = 0; w < dims[3]; ++w) {
                        auto i = &input[input_d.blk_off(n, C * ic_mult, h, w)];
                        auto o = &output[output_d.blk_off(n, C * oc_mult, h, w)];
                        ker(i,o);
                    }
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

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();
        const auto is = input_d.blocking_desc().strides[0];
        const auto os = output_d.blocking_desc().strides[0];

        const float alpha = alpha_, beta = beta_;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int w = 0; w < dims[3]; ++w) {
                    for (int c = 0; c < dims[1]; ++c) {
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
                        if (order_keep) {
                            o[w * os[3] + c] = data_t<type_o>(
                                alpha * i[c * is[1] + w]
                                + (beta ? beta * o[w * os[3] + c] : 0));
                        } else {
                            o[c * os[1] + w] = data_t<type_o>(
                                alpha * i[w * is[3] + c]
                                + (beta ? beta * o[c * os[1] + w] : 0));
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
    typename utils::enable_if<fmt_i == hwio && fmt_o == oihw>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const float alpha = alpha_, beta = beta_;
        const auto &dims = input_d.dims();
        const auto is = input_d.blocking_desc().strides[0];
        const auto os = output_d.blocking_desc().strides[0];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0) {
                for (int oc = 0; oc < dims[0]; ++oc) {
                    for (int kw = 0; kw < dims[3]; ++kw) {
                        if (order_keep) {
                            o[oc * os[0] + kw] = data_t<type_o>(i[kw*is[3]+oc]);
                        } else {
                            o[kw * os[3] + oc] = data_t<type_o>(i[oc*is[0]+kw]);
                        }
                    }
                }
            } else {
                for (int oc = 0; oc < dims[0]; ++oc) {
                    for (int kw = 0; kw < dims[3]; ++kw) {
                        const auto dst_off = order_keep ? oc * os[0] + kw :
                                                          kw * os[3] + oc;
                        const auto src_off = order_keep ? kw * is[3] + oc :
                                                          oc * is[0] + kw;
                        o[dst_off] = data_t<type_o>(alpha * i[src_off]
                                     + (beta ? beta * o[dst_off] : 0));
                    }
                }
            }
        };

#       pragma omp parallel for collapse(2) schedule(static)
        for (int ic = 0; ic < dims[1]; ++ic) {
            for (int kh = 0; kh < dims[2]; ++kh) {
                auto i = &input[input_d.blk_off(0, ic, kh)];
                auto o = &output[output_d.blk_off(0, ic, kh)];
                ker(i, o);
            }
        }

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == nchw && fmt_o == chwn>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();

        constexpr int tsize = 16;

        const auto istrides = input_d.blocking_desc().strides[0];
        const auto ostrides = output_d.blocking_desc().strides[0];
        const auto CHW = dims[1] * dims[2] * dims[3];
        const float alpha = alpha_, beta = beta_;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
                const int nrows, const int ncols) {
            if (alpha == 1.0 && beta == 0) {
#               pragma omp simd collapse(2)
                for (int row = 0; row < nrows; ++row) {
                    for (int col = 0; col < ncols; ++col) {
                        const auto o_idx = row * ostrides[0]
                            + col * ostrides[3];
                        const auto i_idx = row * istrides[0]
                            + col * istrides[3];
                        o[o_idx] = data_t<type_o>(i[i_idx]);
                    }
                }
            } else {
#               pragma omp simd collapse(2)
                for (int row = 0; row < nrows; ++row) {
                    for (int col = 0; col < ncols; ++col) {
                        const auto o_idx = row * ostrides[0]
                            + col * ostrides[3];
                        const auto i_idx = row * istrides[0]
                            + col * istrides[3];
                        o[o_idx] = data_t<type_o>(alpha * i[i_idx]
                            + (beta ? beta * o[o_idx] : 0));
                    }
                }
            }
        };

#       pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < dims[0]; r += tsize) {
            for (int c = 0; c < CHW; c += tsize) {
                const int nrows =
                    r + tsize > dims[0] ? dims[0] - r : tsize;
                const int ncols = c + tsize > CHW ? CHW - c : tsize;
                auto i = &input[r * istrides[0] + c * istrides[3]];
                auto o = &output[r * ostrides[0] + c * ostrides[3]];
                ker(i, o, nrows, ncols);
            }
        }

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == hwio
    && (fmt_o == Ohwi8o || fmt_o == Ohwi16o)>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const float alpha = alpha_, beta = beta_;
        const auto &dims = input_d.dims();
        const auto is = input_d.blocking_desc().strides[0];
        const auto os = output_d.blocking_desc().strides[0];

        constexpr int blksize = fmt_o == Ohwi8o ? 8 : 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0) {
#               pragma omp simd collapse(2)
                for (int O = 0; O < dims[0] / blksize; ++O) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        if (order_keep) {
                            o[O * os[0] + oc] =
                                data_t<type_o>(i[O * blksize + oc]);
                        } else {
                            o[O * blksize + oc] =
                                data_t<type_o>(i[O * is[0] + oc]);
                        }
                    }
                }
            } else {
#               pragma omp simd collapse(2)
                for (int O = 0; O < dims[0] / blksize; ++O) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        const auto dst_off = order_keep ? O * os[0] + oc :
                                                          O * blksize + oc;
                        const auto src_off = order_keep ? O * blksize + oc :
                                                          O * is[0] + oc;
                        o[dst_off] = data_t<type_o>(alpha * i[src_off]
                                     + (beta ? beta * o[dst_off] : 0));
                    }
                }
            }
        };

#       pragma omp parallel for collapse(3) schedule(static)
        for (int h = 0; h < dims[2]; ++h) {
            for (int w = 0; w < dims[3]; ++w) {
                for (int ic = 0; ic < dims[1]; ++ic) {
                    auto i = &input[input_d.blk_off(0, ic, h, w)];
                    auto o = &output[output_d.blk_off(0, ic, h, w)];
                    ker(i, o);
                }
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

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goihw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize =
            (fmt_o == OIhw8i8o || fmt_o == gOIhw8i8o) ? 8 : 16;

        const float alpha = alpha_, beta = beta_;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oihw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[ic * blksize + oc] = data_t<type_o>(i[_g_oihw_off]);
                    } else {
                        o[_g_oihw_off] = data_t<type_o>(i[ic * blksize + oc]);
                    }
                }
                }
            } else {
                for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oihw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[ic * blksize + oc] =
                            data_t<type_o>(alpha * i[_g_oihw_off]
                            + (beta ? beta * o[ic * blksize + oc] : 0));
                    } else {
                        o[_g_oihw_off] =
                            data_t<type_o>(alpha * i[ic * blksize + oc]
                            + (beta ? beta * o[_g_oihw_off] : 0));
                    }
                }
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

#       pragma omp parallel for collapse(5) schedule(static)
        for (int g = 0; g < _G; ++g) {
            for (int O = 0; O < dims[w_groups + 0] / blksize; ++O) {
                for (int I = 0; I < dims[w_groups + 1] / blksize; ++I) {
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
        (fmt_i == goihw && (fmt_o == gOIhw8o8i || fmt_o == gOIhw16o16i))
        || (fmt_i == oihw && (fmt_o == OIhw8o8i || fmt_o == OIhw16o16i))
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goihw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize =
            (fmt_o == OIhw8o8i || fmt_o == gOIhw8o8i) ? 8 : 16;

        const float alpha = alpha_, beta = beta_;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int oc = 0; oc < blksize; ++oc) {
                for (int ic = 0; ic < blksize; ++ic) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oihw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[oc * blksize + ic] = data_t<type_o>(i[_g_oihw_off]);
                    } else {
                        o[_g_oihw_off] = data_t<type_o>(i[oc * blksize + ic]);
                    }
                }
                }
            } else {
                for (int oc = 0; oc < blksize; ++oc) {
                for (int ic = 0; ic < blksize; ++ic) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oihw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[oc * blksize + ic] =
                            data_t<type_o>(alpha * i[_g_oihw_off]
                            + (beta ? beta * o[oc * blksize + ic] : 0));
                    } else {
                        o[_g_oihw_off] =
                            data_t<type_o>(alpha * i[oc * blksize + ic]
                            + (beta ? beta * o[_g_oihw_off] : 0));
                    }
                }
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

#       pragma omp parallel for collapse(5) schedule(static)
        for (int g = 0; g < _G; ++g) {
            for (int O = 0; O < dims[w_groups + 0] / blksize; ++O) {
                for (int I = 0; I < dims[w_groups + 1] / blksize; ++I) {
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
        (fmt_i == goihw && fmt_o == gOihw16o)
        || (fmt_i == oihw && fmt_o == Oihw16o)
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goihw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto strd_oc = _g_oihw_d.blocking_desc().strides[0][w_groups];
        const auto &dims = input_d.dims();
        const int blksize = 16;

        const float alpha = alpha_, beta = beta_;
        const int _G = w_groups ? dims[0] : 1;
        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;

#       pragma omp parallel for collapse(5) schedule(static)
        for (int g = 0; g < _G; ++g) {
            for (int O = 0; O < dims[w_groups + 0] / blksize; ++O) {
                for (int i = 0; i < dims[w_groups + 1]; ++i) {
                    for (int h = 0; h < dims[w_groups + 2]; ++h) {
                        for (int w = 0; w < dims[w_groups + 3]; ++w) {
                            auto inp = &input [input_d.blk_off<!w_groups>(g,
                                    i_mult * O, i, h, w)];
                            auto out = &output[output_d.blk_off<!w_groups>(g,
                                    o_mult * O, i, h, w)];
                            if (alpha == 1.0 && beta == 0.0) {
                                for (int oc = 0; oc < blksize; ++oc) {
                                    const auto off = oc * strd_oc;
                                    if (order_keep) {
                                        out[oc] = data_t<type_o>(inp[off]);
                                    } else {
                                        out[off] = data_t<type_o>(inp[oc]);
                                    }
                                }
                            } else {
                                for (int oc = 0; oc < blksize; ++oc) {
                                    const auto off = oc * strd_oc;
                                    if (order_keep) {
                                        out[oc] = data_t<type_o>(
                                                alpha * inp[off] + (beta
                                                    ? beta * out[oc] : 0));
                                    } else {
                                        out[off] = data_t<type_o>(
                                                alpha * inp[oc] + (beta
                                                    ? beta * out[off] : 0));
                                    }
                                }
                            }
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
        fmt_i == hwio && (fmt_o == OIhw8i8o || fmt_o == OIhw16i16o)
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const float alpha = alpha_, beta = beta_;
        const auto &_hwio_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize = fmt_o == OIhw8i8o ? 8 : 16;
        const auto _hwio_st = _hwio_d.blocking_desc().strides[0];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
#               pragma omp simd collapse(2)
                for (int ic = 0; ic < blksize; ++ic) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        if (order_keep) {
                            o[ic * blksize + oc] =
                                data_t<type_o>(i[oc + ic * _hwio_st[1]]);
                        } else {
                            o[oc + ic * _hwio_st[1]] =
                                data_t<type_o>(i[ic * blksize + oc]);
                        }
                    }
                }
            } else {
#               pragma omp simd collapse(2)
                for (int ic = 0; ic < blksize; ++ic) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        const auto dst_off = order_keep ? ic * blksize + oc :
                                                          ic * _hwio_st[1] + oc;
                        const auto src_off = order_keep ? ic * _hwio_st[1] + oc :
                                                          ic * blksize + oc;
                        o[dst_off] = data_t<type_o>(alpha * i[src_off]
                                     + (beta ? beta * o[dst_off] : 0));
                    }
                }
            }
        };

#       pragma omp parallel for collapse(4) schedule(static)
        for (int h = 0; h < dims[2]; ++h) {
            for (int w = 0; w < dims[3]; ++w) {
                for (int O = 0; O < dims[0] / blksize; ++O) {
                    for (int I = 0; I < dims[1] / blksize; ++I) {
                        constexpr int i_mult = order_keep ? blksize : 1;
                        constexpr int o_mult = order_keep ? 1 : blksize;
                        auto i = &input[input_d.blk_off(
                                i_mult * O, i_mult * I, h, w)];
                        auto o = &output[output_d.blk_off(
                                o_mult * O, o_mult * I, h, w)];
                        ker(i, o);
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
          (fmt_i == goihw && (fmt_o == gOIhw8i16o2i))
          || (fmt_i == oihw && (fmt_o == OIhw8i16o2i))
    >::type>
{
   static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goihw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const int blksize = 16;

        const float alpha = alpha_, beta = beta_;

        auto index = [&](const int ic, const int oc) {
            return ((ic / 2) * blksize * 2 + 2 * oc + ic % 2);
        };

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                      + ic * _g_oihw_d.blocking_desc().strides[0][w_groups + 1];
                    if (order_keep) {
                        o[index(ic, oc)] =
                            data_t<type_o>(i[_g_oihw_off]);
                    } else {
                        o[_g_oihw_off] =
                            data_t<type_o>(i[index(ic, oc)]);
                    }
                }
                }
            } else {
                for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                      + ic * _g_oihw_d.blocking_desc().strides[0][w_groups + 1];
                    if (order_keep) {
                        o[index(ic, oc)] = data_t<type_o>(
                            alpha * i[_g_oihw_off]
                            + (beta ? beta * o[index(ic, oc)] : 0));
                    } else {
                        o[_g_oihw_off] = data_t<type_o>(
                            alpha * i[index(ic, oc)]
                            + (beta ? beta * o[_g_oihw_off] : 0));
                    }
                }
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

#       pragma omp parallel for collapse(5) schedule(static)
        for (int g = 0; g < _G; ++g) {
            for (int O = 0; O < dims[w_groups + 0] / blksize; ++O) {
                for (int I = 0; I < dims[w_groups + 1] / blksize; ++I) {
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
        (fmt_i == gOIhw8i16o2i && fmt_o == gOIhw8o16i2o)
        || (fmt_i == OIhw8i16o2i && fmt_o == OIhw8o16i2o)
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const double alpha = alpha_, beta = beta_;
        constexpr bool w_groups = fmt_i == gOIhw8i16o2i;

        const auto &dims = input_d.dims();
        const int blksize = 16;

        auto index_src = [&](const int ic, const int oc) {
            return ((ic / 2) * blksize * 2 + 2 * oc + ic % 2);
        };
        auto index_dst = [&](const int ic, const int oc) {
            return ((oc / 2) * blksize * 2 + 2 * ic + oc % 2);
        };

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) -> void {
            if (alpha == 1.0 && beta == 0.0) {
                for (int ic = 0; ic < blksize; ++ic) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        o[index_dst(ic,oc)] = data_t<type_o>(i[index_src(ic,oc)]);
                    }
                }
            } else {
                for (int ic = 0; ic < blksize; ++ic) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        o[index_dst(ic,oc)] = data_t<type_o>(
                                alpha * i[index_src(ic,oc)]
                                + (beta ? beta * o[index_dst(ic,oc)] : 0));
                    }
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

#       pragma omp parallel for collapse(5) schedule(static)
        for (int g = 0; g < _G; ++g) {
            for (int o = 0; o < dims[w_groups + 0] / blksize; ++o) {
                for (int i = 0; i < dims[w_groups + 1] / blksize; ++i) {
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

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = (fmt_i == gOIhw8i8o || fmt_i == gOIhw16i16o);

        const auto &dims = input_d.dims();
        constexpr int blksize =
            (fmt_i == OIhw8i8o || fmt_i == gOIhw8i8o) ? 8 : 16;

        const float alpha = alpha_, beta = beta_;
        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const int o_idx = ic * blksize + oc;
                    const int i_idx = oc * blksize + ic;
                    o[o_idx] = (alpha == 1.0 && beta == 0.0)
                        ? data_t<type_o>(i[i_idx])
                        : data_t<type_o>(alpha * i[i_idx]
                            + (beta ? beta * o[o_idx] : 0));
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

#       pragma omp parallel for collapse(5) schedule(static)
        for (int g = 0; g < _G; ++g) {
            for (int o = 0; o < dims[w_groups + 0] / blksize; ++o) {
                for (int i = 0; i < dims[w_groups + 1] / blksize; ++i) {
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
        (fmt_i == Oihw16o && fmt_o == Ohwi16o)
        || (fmt_i == gOihw16o && fmt_o == gOhwi16o)
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        return input_d.format() == (order_keep ? fmt_i : fmt_o)
            && output_d.format() == (order_keep ? fmt_o : fmt_i);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const bool w_groups = fmt_i == gOihw16o;

        const auto &dims = input_d.dims();
        const int blksize = 16;

        const float alpha = alpha_, beta = beta_;
        const int _G = w_groups ? dims[0] : 1;

#       pragma omp parallel for collapse(5) schedule(static)
        for (int g = 0; g < _G; ++g) {
            for (int o = 0; o < dims[w_groups + 0] / blksize; ++o) {
                for (int i = 0; i < dims[w_groups + 1]; ++i) {
                    for (int h = 0; h < dims[w_groups + 2]; ++h) {
                        for (int w = 0; w < dims[w_groups + 3]; ++w) {
                            auto i_ptr = &input[input_d.blk_off<!w_groups>(g,
                                    o, i, h, w)];
                            auto o_ptr = &output[output_d.blk_off<!w_groups>(g,
                                    o, i, h, w)];
                            for (int oc = 0; oc < blksize; ++oc) {
                                o_ptr[oc] = (alpha == 1.0 && beta == 0.0)
                                    ? data_t<type_o>(i_ptr[oc])
                                    : data_t<type_o>(alpha * i_ptr[oc]
                                        + (beta ? beta * o_ptr[oc] : 0));
                            }
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
        /* FIXME: is the formula correct? */
        return input_d.similar_to(output_d, true, false, 0)
            && input_d.is_dense() && output_d.is_dense();
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        assert(input_d.is_dense());

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const size_t nelems = input_d.nelems();

        constexpr int block_size = 16;
        const auto num_blocks = nelems / block_size;
        const auto rem_elems = nelems % block_size;

        const float alpha = alpha_, beta = beta_;

#       pragma omp parallel
        {
            const int ithr = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            size_t start{0}, end{0};
            balance211(num_blocks, nthr, ithr, start, end);
            start = start * block_size;
            end = end * block_size;
            if (alpha == 1.0 && beta == 0.0) {
#               pragma omp simd
                for (size_t e = start; e < end; ++e) {
                    output[e] = data_t<type_o>(input[e]);
                }
            } else{
#               pragma omp simd
                for (size_t e = start; e < end; ++e) {
                    output[e] = data_t<type_o>(alpha * input[e]
                        + (beta ? beta * output[e] : 0));
                }
            }

            if (rem_elems != 0 && ithr == nthr - 1){
                for (int e = nelems - rem_elems; e < nelems; ++e){
                    output[e] = data_t<type_o>((alpha == 1.0 && beta == 0.0) ?
                            input[e]
                            : alpha * input[e] + (beta ? beta * output[e] : 0));
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
    spec::direct_copy_except_dim_0>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        auto is_dense_no_0 = [](const memory_desc_wrapper &data_d) {
            return nelems_no_dim_0(data_d) == _size_no_dim_0(data_d);
        };
        /* FIXME: is the formula correct? */
        return input_d.similar_to(output_d, true, false, 1)
            && is_dense_no_0(input_d) && is_dense_no_0(output_d);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);
        const float alpha = alpha_, beta = beta_;

        const int N = input_d.dims()[0];
        const size_t is = input_d.blocking_desc().strides[0][0];
        const size_t os = output_d.blocking_desc().strides[0][0];
        const size_t nelems_no_d0 = nelems_no_dim_0(input_d);
        const size_t work_amount = N * nelems_no_d0;

        if (alpha == 1.0 && beta == 0.0) {
#           pragma omp parallel
            {
                const int ithr = omp_get_thread_num();
                const int nthr = omp_get_num_threads();
                size_t n{0}, dim1_s{0};
                size_t start{0}, end{0};
                balance211(work_amount, nthr, ithr, start, end);
                nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
                while(start < end) {
                    size_t work_rem = end - start;
                    size_t dim1_e =
                        dim1_s + work_rem > nelems_no_d0 ? nelems_no_d0
                        : dim1_s + work_rem;
#                   pragma omp simd
                    for (size_t e = dim1_s; e < dim1_e; ++e){
                        output[os * n + e] = data_t<type_o>(input[is * n + e]);
                    }
                    nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
                }
            }
        } else {
#           pragma omp parallel
            {
                const int ithr = omp_get_thread_num();
                const int nthr = omp_get_num_threads();
                size_t n{0}, dim1_s{0};
                size_t start{0}, end{0};
                balance211(work_amount, nthr, ithr, start, end);
                nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
                while(start < end) {
                    size_t work_rem = end - start;
                    size_t dim1_e =
                        dim1_s + work_rem > nelems_no_d0 ? nelems_no_d0
                        : dim1_s + work_rem;
#                   pragma omp simd
                    for (size_t e = dim1_s; e < dim1_e; ++e){
                        output[os * n + e] = data_t<type_o>(
                                alpha * input[is * n + e]
                                + beta ? beta * output[os * n + e] : 0);
                    }
                    nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
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
                    size_t(size_t(blk.padding_dims[d] / block)
                        * blk.strides[0][d]));
            if (block > 1)
                max_size = nstl::max(max_size,
                        size_t(block * blk.strides[1][d]));
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

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const size_t nelems = input_d.nelems();

        const float alpha = alpha_, beta = beta_;
        if (alpha == 1.0 && beta == 0.0) {
#           pragma omp parallel for schedule(static)
#           pragma simd
            for (int e = 0; e < nelems; ++e) {
                output[output_d.off_l(e)] =
                    data_t<type_o>(input[input_d.off_l(e)]);
            }
        } else {
#           pragma omp parallel for schedule(static)
#           pragma simd
            for (int e = 0; e < nelems; ++e) {
                output[output_d.off_l(e)] = data_t<type_o>(
                    alpha * input[input_d.off_l(e)]
                    + (beta ? beta * output[output_d.off_l(e)] : 0));
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
                &conf_, input, output);
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
