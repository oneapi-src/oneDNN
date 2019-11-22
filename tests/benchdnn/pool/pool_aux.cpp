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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "pool/pool.hpp"

namespace pool {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(MAX);
    CASE(AVG_NP);
    CASE(AVG_P);
#undef CASE
    assert(!"unknown algorithm");
    return MAX;
}

const char *alg2str(alg_t alg) {
    if (alg == MAX) return "MAX";
    if (alg == AVG_NP) return "AVG_NP";
    if (alg == AVG_P) return "AVG_P";
    assert(!"unknown algorithm");
    return "unknown algorithm";
}

dnnl_alg_kind_t alg2alg_kind(alg_t alg) {
    if (alg == MAX) return dnnl_pooling_max;
    if (alg == AVG_NP) return dnnl_pooling_avg_exclude_padding;
    if (alg == AVG_P) return dnnl_pooling_avg_include_padding;
    assert(!"unknown algorithm");
    return dnnl_alg_kind_undef;
}

int str2desc(desc_t *desc, const char *str) {
    desc_t d {0};

    /* canonical form:
     * dYmbXicXihXiwXohXowXkhXkwXshXswXphXpwXnS
     *
     * where: Y = {fd, fi, bd}, X is number, S - string
     * note: symbol `_` is ignored
     *
     * implicit rules:
     *  - default values:
     *      mb = 2, d = fd, sh = sw = 1
     *  - if H is undefined => H = W
     *  - if W is undefined => W = H
     *  - if `output` is undefined => compute output
     *  - if padding is undefined => compute trivial padding
     */

    d.mb = 2;
    d.sd = d.sh = d.sw = 1;
    d.pd = d.ph = d.pw = -1;

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
        CASE_N(kd);
        CASE_N(kh);
        CASE_N(kw);
        CASE_N(sd);
        CASE_N(sh);
        CASE_N(sw);
        CASE_N(pd);
        CASE_N(ph);
        CASE_N(pw);
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
    if (d.sd <= 0 || d.sh <= 0 || d.sw <= 0) return FAIL;

    auto compute_out = [](int64_t i, int64_t k, int64_t s, int64_t p) {
        return (i - k + 2 * p) / s + 1;
    };
    auto compute_pad = [](int64_t o, int64_t i, int64_t k, int64_t s) {
        return ((o - 1) * s - i + k) / 2;
    };

    const bool no_d = (d.id | d.kd | d.od) == 0 && d.sd == 1 && d.pd < 1;
    const bool no_h = (d.ih | d.kh | d.oh) == 0 && d.sh == 1 && d.ph < 1;
    const bool no_w = (d.iw | d.kw | d.ow) == 0 && d.sw == 1 && d.pw < 1;
    if (!no_h) {
        if (!d.ih || !d.kh) return FAIL;
        if (!d.oh) {
            if (d.ph < 0) d.ph = 0;
            d.oh = compute_out(d.ih, d.kh, d.sh, d.ph);
        } else if (d.ph < 0)
            d.ph = compute_pad(d.oh, d.ih, d.kh, d.sh);
    }

    if (!no_w) {
        if (!d.iw || !d.kw) return FAIL;
        if (!d.ow) {
            if (d.pw < 0) d.pw = 0;
            d.ow = compute_out(d.iw, d.kw, d.sw, d.pw);
        } else if (d.pw < 0)
            d.pw = compute_pad(d.ow, d.iw, d.kw, d.sw);
    }

    if (!no_d && d.id) {
        if (!d.id || !d.kd) return FAIL;
        if (!d.od) {
            if (d.pd < 0) d.pd = 0;
            d.od = compute_out(d.id, d.kd, d.sd, d.pd);
        } else if (d.pd < 0)
            d.pd = compute_pad(d.od, d.id, d.kd, d.sd);
    }

    if (no_w && no_h && d.id) {
        d.iw = d.ih = d.id;
        d.kw = d.kh = d.kd;
        d.ow = d.oh = d.od;
        d.pw = d.ph = d.pd;
        d.sw = d.sh = d.sd;
    } else if (no_w) {
        d.iw = d.ih;
        d.kw = d.kh;
        d.ow = d.oh;
        d.pw = d.ph;
        d.sw = d.sh;
    } else if (no_h) {
        d.ih = 1;
        d.kh = 1;
        d.oh = 1;
        d.ph = 0;
        d.sh = 1;
    }

    if (d.id < 1) {
        d.id = 1;
        d.kd = 1;
        d.od = 1;
        d.sd = 1;
        d.pd = 0;
    }

    *desc = d;

    return OK;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    const bool canonical = s.flags() & std::ios_base::fixed;

    if (canonical || d.mb != 2) s << "mb" << d.mb;

    const bool print_d = is_3d(&d);
    const bool half_form = (d.ih == d.iw && d.kh == d.kw && d.oh == d.ow
                                   && d.sh == d.sw && d.ph == d.pw)
            && !print_d;

    const bool print_w = canonical || print_d || !half_form;

    auto print_spatial = [&](const char *sd, int64_t vd, const char *sh,
                                 int64_t vh, const char *sw, int64_t vw) {
        if (print_d) s << sd << vd;
        s << sh << vh;
        if (print_w) s << sw << vw;
    };

    s << "ic" << d.ic;
    print_spatial("id", d.id, "ih", d.ih, "iw", d.iw);
    print_spatial("od", d.od, "oh", d.oh, "ow", d.ow);
    print_spatial("kd", d.kd, "kh", d.kh, "kw", d.kw);

    if (canonical || d.sh != 1 || d.sw != 1 || d.sd != 1)
        print_spatial("sd", d.sd, "sh", d.sh, "sw", d.sw);

    print_spatial("pd", d.pd, "ph", d.ph, "pw", d.pw);

    if (d.name) s << "n" << d.name;

    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    if (p.dir != FWD_D) s << "--dir=" << dir2str(p.dir) << " ";
    if (p.cfg != conf_f32) s << "--cfg=" << cfg2str(p.cfg) << " ";
    if (p.tag != dnnl_nchw) s << "--tag=" << fmt_tag2str(p.tag) << " ";
    if (p.alg != MAX) s << "--alg=" << alg2str(p.alg) << " ";

    s << static_cast<const desc_t &>(p);

    return s;
}

} // namespace pool
