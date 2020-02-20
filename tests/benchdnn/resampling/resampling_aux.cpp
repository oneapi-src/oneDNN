/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "resampling/resampling.hpp"

namespace resampling {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(nearest);
    CASE(linear);
#undef CASE
    assert(!"unknown algorithm");
    return nearest;
}

const char *alg2str(alg_t alg) {
    if (alg == nearest) return "nearest";
    if (alg == linear) return "linear";
    assert(!"unknown algorithm");
    return "unknown algorithm";
}

dnnl_alg_kind_t alg2alg_kind(alg_t alg) {
    if (alg == nearest) return dnnl_resampling_nearest;
    if (alg == linear) return dnnl_resampling_linear;
    assert(!"unknown algorithm");
    return dnnl_alg_kind_undef;
}

int str2desc(desc_t *desc, const char *str) {
    desc_t d {0};

    /* canonical form:
     * mbXicXidXihXiwXodXohXowXnS
     *
     * where: Y = {fd, fi, bd}, X is number, S - string
     * note: symbol `_` is ignored
     *
     * implicit rules:
     *  - default values:
     *      mb = 2, ih = oh = id = od = 1, S="wip"
     */

    d.mb = 2;
    d.ndims = 5;

    const char *s = str;
    assert(s);

#define CASE_NN(p, c) \
    do { \
        if (!strncmp(p, s, strlen(p))) { \
            ok = 1; \
            s += strlen(p); \
            char *end_s; \
            d.c = strtol(s, &end_s, 10); \
            s += (end_s - s); \
            if (d.c < 0) return FAIL; \
            /* printf("@@@debug: %s: %ld\n", p, d. c); */ \
        } \
    } while (0)
#define CASE_N(c) CASE_NN(#c, c)
    while (*s) {
        int ok = 0;
        CASE_N(mb);
        CASE_N(ic);
        CASE_N(id);
        CASE_N(ih);
        CASE_N(iw);
        CASE_N(od);
        CASE_N(oh);
        CASE_N(ow);
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

    if ((d.id && !d.od) || (!d.id && d.od)) return FAIL;

    if (d.id == 0 && d.od == 0) {
        d.ndims--;
        if ((d.ih && !d.oh) || (!d.ih && d.oh)) return FAIL;
        if (d.ih == 0 && d.oh == 0) {
            d.ndims--;
            if (d.iw == 0 || d.ow == 0) return FAIL;
        }
    }

    // square shape
    if (d.ih == 0 && d.id != 0) d.ih = d.id;
    if (d.iw == 0 && d.ih != 0) d.iw = d.ih;
    if (d.oh == 0 && d.od != 0) d.oh = d.od;
    if (d.ow == 0 && d.oh != 0) d.ow = d.oh;

    // to keep logic when treating unspecified dimension as it's of length 1.
    if (d.id == 0) d.id = 1;
    if (d.ih == 0) d.ih = 1;
    if (d.iw == 0) d.iw = 1;
    if (d.od == 0) d.od = 1;
    if (d.oh == 0) d.oh = 1;
    if (d.ow == 0) d.ow = 1;

    *desc = d;

    return OK;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    const bool square_form = (d.ih == d.iw) && (d.oh == d.ow);
    const bool cubic_form = square_form && (d.id == d.ih) && (d.od == d.oh);

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

    if (print_d) s << "od" << d.od;
    if (print_h) s << "oh" << d.oh;
    if (print_w) s << "ow" << d.ow;

    if (d.name) s << "n" << d.name;

    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    if (canonical || p.dir != FWD_D) s << "--dir=" << dir2str(p.dir) << " ";
    if (canonical || p.dt != dnnl_f32) s << "--dt=" << dt2str(p.dt) << " ";
    if (canonical || p.tag != dnnl_nchw)
        s << "--tag=" << fmt_tag2str(p.tag) << " ";
    if (canonical || p.alg != nearest) s << "--alg=" << alg2str(p.alg) << " ";

    s << static_cast<const desc_t &>(p);

    return s;
}

} // namespace resampling
