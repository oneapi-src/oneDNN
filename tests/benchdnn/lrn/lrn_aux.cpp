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

#include <assert.h>
#include <stdlib.h>

#include "lrn/lrn.hpp"

namespace lrn {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(ACROSS);
    CASE(WITHIN);
#undef CASE
    assert(!"unknown algorithm");
    return ACROSS;
}

const char *alg2str(alg_t alg) {
    if (alg == ACROSS) return "ACROSS";
    if (alg == WITHIN) return "WITHIN";
    assert(!"unknown algorithm");
    return "unknown algorithm";
}

dnnl_alg_kind_t alg2alg_kind(alg_t alg) {
    if (alg == ACROSS) return dnnl_lrn_across_channels;
    if (alg == WITHIN) return dnnl_lrn_within_channel;
    assert(!"unknown algorithm");
    return dnnl_alg_kind_undef;
}

int str2desc(desc_t *desc, const char *str) {
    // Canonical form: mbXicXidXihXiwX_lsXalphaYbetaYkY_nS,
    // where
    //     X is integer
    //     Y is float
    //     S is string
    // note: symbol `_` is ignored.
    // Cubic/square shapes are supported by specifying just highest dimension.

    desc_t d {0};
    d.mb = 2;
    d.ls = 5;
    d.alpha = 1.f / 8192; // = 0.000122 ~~ 0.0001, but has exact representation
    d.beta = 0.75f;
    d.k = 1;
    d.ndims = 5;

    const char *s = str;
    assert(s);

    auto mstrtol = [](const char *nptr, char **endptr) {
        return strtol(nptr, endptr, 10);
    };

#define CASE_NN(p, c, cvfunc) \
    do { \
        if (!strncmp(p, s, strlen(p))) { \
            ok = 1; \
            s += strlen(p); \
            char *end_s; \
            d.c = cvfunc(s, &end_s); \
            s += (end_s - s); \
            if (d.c < 0) return FAIL; \
            /* printf("@@@debug: %s: " IFMT "\n", p, d. c); */ \
        } \
    } while (0)
#define CASE_N(c, cvfunc) CASE_NN(#c, c, cvfunc)
    while (*s) {
        int ok = 0;
        CASE_N(mb, mstrtol);
        CASE_N(ic, mstrtol);
        CASE_N(id, mstrtol);
        CASE_N(ih, mstrtol);
        CASE_N(iw, mstrtol);
        CASE_N(ls, mstrtol);
        CASE_N(alpha, strtof);
        CASE_N(beta, strtof);
        CASE_N(k, strtof);
        if (*s == 'n') {
            d.name = s + 1;
            break;
        }
        if (*s == '_') ++s;
        if (!ok) return FAIL;
    }
#undef CASE_NN
#undef CASE_N

    if (d.ic == 0) return FAIL;

    if (d.id == 0) { d.ndims--; }
    if (d.ih == 0) {
        if (d.id == 0) {
            d.ndims--;
        } else { // square shape
            d.ih = d.id;
        }
    }
    if (d.iw == 0) {
        if (d.ih == 0) {
            d.ndims--;
        } else { // square shape
            d.iw = d.ih;
        }
    }

    // to keep logic when treating unspecified dimension as it's of length 1.
    if (d.id == 0) d.id = 1;
    if (d.ih == 0) d.ih = 1;
    if (d.iw == 0) d.iw = 1;

    *desc = d;

    return OK;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    const bool square_form = (d.ih == d.iw);
    const bool cubic_form = square_form && (d.id == d.ih);

    const bool print_d = d.ndims == 5;
    const bool print_h
            = d.ndims == 4 || (d.ndims > 4 && (!cubic_form || canonical));
    const bool print_w
            = d.ndims == 3 || (d.ndims > 3 && (!square_form || canonical));

    if (canonical || d.mb != 2) s << "mb" << d.mb;

    s << "ic" << d.ic;

    if (print_d) s << "id" << d.id;
    if (print_h) s << "ih" << d.ih;
    if (print_w) s << "iw" << d.iw;

    if (canonical || d.ls != 5) s << "ls" << d.ls;
    if (canonical || d.alpha != 1.f / 8192) s << "alpha" << d.alpha;
    if (canonical || d.beta != 0.75f) s << "beta" << d.beta;
    if (canonical || d.k != 1) s << "k" << d.k;

    if (d.name) s << "n" << d.name;

    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    if (canonical || p.dir != FWD_D) s << "--dir=" << dir2str(p.dir) << " ";
    if (canonical || p.dt != dnnl_f32) s << "--dt=" << dt2str(p.dt) << " ";
    if (canonical || p.tag != dnnl_nchw)
        s << "--tag=" << fmt_tag2str(p.tag) << " ";
    if (canonical || p.alg != ACROSS) s << "--alg=" << alg2str(p.alg) << " ";

    s << static_cast<const desc_t &>(p);

    return s;
}

} // namespace lrn
