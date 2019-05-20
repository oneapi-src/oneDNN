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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "mkldnn.h"

#include "mkldnn_common.hpp"
#include "mkldnn_debug.hpp"
#include "pool/pool.hpp"

namespace pool {

alg_t str2alg(const char *str) {
#define CASE(_alg) if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
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

alg_t alg_kind2alg(mkldnn_alg_kind_t alg) {
    if (alg == mkldnn_pooling_max) return MAX;
    if (alg == mkldnn_pooling_avg) return AVG_NP;
    if (alg == mkldnn_pooling_avg_include_padding) return AVG_P;
    assert(!"unknown algorithm");
    return MAX;
}

int str2desc(desc_t *desc, const char *str) {
    desc_t d{0};

    /* canonical form:
     * dYmbXicXihXiwXohXowXkhXkwXshXswXphXpwXnS
     *
     * where: Y = {fd, fi, bd}, X is number, S - string
     * note: symbol `_` is ignored
     *
     * implicit rules:
     *  - default values:
     *      mb = 2, d = fd, sh = sw = 1, S="wip"
     *  - if H is undefined => H = W
     *  - if W is undefined => W = H
     *  - if `output` is undefined => compute output
     *  - if padding is undefined => compute trivial padding
     */

    d.mb = 2;
    d.sd = d.sh = d.sw = 1;
    d.pd = d.ph = d.pw = -1;
    d.name = "\"wip\"";

    const char *s = str;
    assert(s);

#   define CASE_NN(p, c) do { \
        if (!strncmp(p, s, strlen(p))) { \
            ok = 1; s += strlen(p); \
            char *end_s; d. c = strtol(s, &end_s, 10); s += (end_s - s); \
            /* printf("@@@debug: %s: %ld\n", p, d. c); */ \
        } \
    } while (0)
#   define CASE_N(c) CASE_NN(#c, c)
    while (*s) {
        int ok = 0;
        CASE_N(mb); CASE_N(ic);
        CASE_N(id); CASE_N(ih); CASE_N(iw);
        CASE_N(od); CASE_N(oh); CASE_N(ow);
        CASE_N(kd); CASE_N(kh); CASE_N(kw);
        CASE_N(sd); CASE_N(sh); CASE_N(sw);
        CASE_N(pd); CASE_N(ph); CASE_N(pw);
        if (*s == 'n') { d.name = s + 1; break; }
        if (*s == '_') ++s;
        if (!ok) return FAIL;
    }
#   undef CASE_NN
#   undef CASE_N

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
            d.ph = 0;
            d.oh = compute_out(d.ih, d.kh, d.sh, d.ph);
        } else if (d.ph < 0)
            d.ph = compute_pad(d.oh, d.ih, d.kh, d.sh);
    }

    if (!no_w) {
        if (!d.iw || !d.kw) return FAIL;
        if (!d.ow) {
            d.pw = 0;
            d.ow = compute_out(d.iw, d.kw, d.sw, d.pw);
        } else if (d.pw < 0)
            d.pw = compute_pad(d.ow, d.iw, d.kw, d.sw);
    }

    if (!no_d && d.id) {
        if (!d.id || !d.kd) return FAIL;
        if (!d.od) {
            d.pd = 0;
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

void desc2str(const desc_t *d, char *buffer, bool canonical) {
    int rem_len = max_desc_len;
#   define DPRINT(...) do { \
        int l = snprintf(buffer, rem_len, __VA_ARGS__); \
        buffer += l; rem_len -= l; \
    } while(0)

    if (canonical || d->mb != 2) DPRINT("mb" IFMT "", d->mb);

    const bool half_form = (d->ih == d->iw && d->kh == d->kw && d->oh == d->ow
        && d->sh == d->sw && d->ph == d->pw) && d->id == 1;

    if (!canonical && half_form) {
        DPRINT("ic" IFMT "ih" IFMT "oh" IFMT "kh" IFMT "",
                d->ic, d->ih, d->oh, d->kh);
        if (d->sh != 1) DPRINT("sh" IFMT "", d->sh);
        if (d->ph != 0) DPRINT("ph" IFMT "", d->ph);
    } else {
        if (d->id == 1) {
            DPRINT("ic" IFMT "ih" IFMT "iw" IFMT
                    "oh" IFMT "ow" IFMT "kh" IFMT "kw" IFMT "",
                    d->ic, d->ih, d->iw, d->oh, d->ow, d->kh, d->kw);
            if (canonical || d->sh != 1 || d->sw != 1)
                DPRINT("sh" IFMT "sw" IFMT "", d->sh, d->sw);
            if (canonical || d->ph != 0 || d->pw != 0)
                DPRINT("ph" IFMT "pw" IFMT "", d->ph, d->pw);
        } else {
            DPRINT("ic" IFMT "id" IFMT "ih" IFMT "iw" IFMT "od" IFMT
                    "oh" IFMT "ow" IFMT "kd" IFMT "kh" IFMT "kw" IFMT "",
                    d->ic, d->id, d->ih, d->iw, d->od, d->oh, d->ow,
                    d->kd, d->kh, d->kw);
            if (canonical || d->sh != 1 || d->sw != 1 || d->sd != 1)
                DPRINT("sd" IFMT "sh" IFMT "sw" IFMT "", d->sd, d->sh, d->sw);
            if (canonical || d->ph != 0 || d->pw != 0 || d->pd != 0)
                DPRINT("pd" IFMT "ph" IFMT "pw" IFMT "", d->pd, d->ph, d->pw);
        }
    }

    DPRINT("n%s", d->name);

#   undef DPRINT
}

void prb2str(const prb_t *p, char *buffer, bool canonical) {
    char dir_str[32] = "", cfg_str[32] = "", alg_str[32] = "", tag_str[32] = "",
         desc_buf[max_desc_len] = "";

    if (p->dir != FWD_D)
        snprintf(dir_str, sizeof(dir_str), "--dir=%s ", dir2str(p->dir));
    if (p->cfg != conf_f32)
        snprintf(cfg_str, sizeof(cfg_str), "--cfg=%s ", cfg2str(p->cfg));
    if (p->tag != mkldnn_nchw)
        snprintf(tag_str, sizeof(tag_str), "--tag=%s ", tag2str(p->tag));
    if (p->alg != MAX)
        snprintf(alg_str, sizeof(alg_str), "--alg=%s ", alg2str(p->alg));
    desc2str(p, desc_buf, canonical);

    snprintf(buffer, max_prb_len, "%s%s%s%s%s", dir_str, cfg_str, tag_str,
            alg_str, desc_buf);
}

}
