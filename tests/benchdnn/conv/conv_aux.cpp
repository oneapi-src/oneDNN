/*******************************************************************************
* Copyright 2017 Intel Corporation
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
#include "conv/conv.hpp"

namespace conv {

const char *inp_type2str(int what) {
    switch (what) {
    case SRC: return "SRC";
    case WEI: return "WEI";
    case BIA: return "BIA";
    case DST: return "DST";
    case ACC: return "ACC";
    }
    assert(!"incorrect input type");
    return "incorrect input type";
}

alg_t str2alg(const char *str) {
#define CASE(_alg) if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(DIRECT);
    CASE(WINO);
#undef CASE
    assert(!"unknown algorithm");
    return DIRECT;
}

const char *alg2str(alg_t alg) {
    if (alg == DIRECT) return "direct";
    if (alg == WINO) return "wino";
    assert(!"unknown algorithm");
    return "unknown algorithm";
}

merge_t str2merge(const char *str) {
#define CASE(_mrg) if (!strcasecmp(STRINGIFY(_mrg), str)) return _mrg
    CASE(NONE);
    CASE(RELU);
#undef CASE
    assert(!"unknown merge");
    return NONE;
}

const char *merge2str(merge_t merge) {
    if (merge == NONE) return "none";
    if (merge == RELU) return "relu";
    assert(!"unknown merge");
    return "unknown merge";
}

int str2desc(desc_t *desc, const char *str) {
    desc_t d{0};

    /* canonical form:
     * dYgXmbXicXihXiwXocXohXowXkhXkwXshXswXphXpwXnS
     *
     * where: Y = {fb, fd, bd, bw, bb}, X is number, S - string
     *
     * implicit rules:
     *  - default values: { mb = 2, g = 1, d = fd, sh = sw = 1, S="wip" }
     *  - if H is undefined => H = W
     *  - if W is undefined => W = H
     *  - if `output` is undefined => compute output
     *  - if padding is undefined => compute trivial padding
     */

    d.g = 1; d.mb = 2; d.sh = d.sw = 1; d.name = "\"wip\"";

    const char *s = str;
    assert(s);

#   define CASE_NN(p, c) do { \
        if (!strncmp(p, s, strlen(p))) { \
            ok = 1; s += strlen(p); \
            char *end_s; d. c = strtol(s, &end_s, 10); s += (end_s - s); \
            /* printf("@@@debug: %s: %d\n", p, d. c); */ \
        } \
    } while (0)
#   define CASE_N(c) CASE_NN(#c, c)
    while (*s) {
        int ok = 0;
        CASE_N(g); CASE_N(mb);
        CASE_N(ic); CASE_N(ih); CASE_N(iw);
        CASE_N(oc); CASE_N(oh); CASE_N(ow);
        CASE_N(kh); CASE_N(kw);
        CASE_N(sh); CASE_N(sw);
        CASE_N(ph); CASE_N(pw);
        if (*s == 'n') { d.name = s + 1; break; }
        if (!ok) return FAIL;
    }
#   undef CASE_NN
#   undef CASE_N

    if (d.ih * d.iw == 0) d.ih = d.iw = MAX2(d.ih, d.iw);
    if (d.kh * d.kw == 0) d.kh = d.kw = MAX2(d.kh, d.kw);
    if (d.ph * d.pw == 0) d.ph = d.pw = MAX2(d.ph, d.pw);

    if (d.oh == 0 && d.ow != 0) d.oh = d.ow;
    if (d.ow == 0 && d.oh != 0) d.ow = d.oh;

    if (d.ih == 0 || d.kh == 0) return FAIL;
    if (d.ic == 0 || d.oc == 0) return FAIL;

    auto compute_out = [](int i, int k, int s, int p) {
        return (i - k + 2 * p) / s + 1;
    };
    auto compute_pad = [](int o, int i, int k, int s) {
        return ((o - 1) * s - i + k) / 2; /* XXX: is it oK? */
    };

    if (d.oh == 0) d.oh = compute_out(d.ih, d.kh, d.sh, d.ph);
    else if (d.ph == 0 && d.oh != compute_out(d.ih, d.kh, d.sh, d.ph)) {
        d.ph = compute_pad(d.oh, d.ih, d.kh, d.ph);
    }

    if (d.ow == 0) d.ow = compute_out(d.iw, d.kw, d.sw, d.pw);
    else if (d.pw == 0 && d.ow != compute_out(d.iw, d.kw, d.sw, d.pw)) {
        d.pw = compute_pad(d.ow, d.iw, d.kw, d.pw);
    }

    *desc = d;

    return OK;
}

void desc2str(const desc_t *d, char *buffer, bool canonical) {
    if (d->ih == d->iw && d->oh == d->ow && d->kh == d->kw && d->sh == d->sw
            && d->sh == 1 && d->ph == d->pw && !canonical) {
        if (d->g == 1) {
            if (d->mb == 2) {
                snprintf(buffer, max_desc_len, "ic%dih%doc%doh%dkh%dph%dn%s",
                        d->ic, d->ih, d->oc, d->oh, d->kh, d->ph, d->name);
            } else {
                snprintf(buffer, max_desc_len,
                        "mb%dic%dih%doc%doh%dkh%dph%dn%s", d->mb, d->ic, d->ih,
                        d->oc, d->oh, d->kh, d->ph, d->name);
            }
            return;
        }
        snprintf(buffer, max_desc_len, "g%dmb%dic%dih%doc%doh%dkh%dph%dn%s",
                d->g, d->mb, d->ic, d->ih, d->oc, d->oh, d->kh, d->ph, d->name
                );
        return;
    }
    snprintf(buffer, max_desc_len,
            "g%dmb%dic%dih%diw%doc%doh%dow%dkh%dkw%dsh%dsw%dph%dpw%dn%s",
            d->g, d->mb, d->ic, d->ih, d->iw, d->oc, d->oh, d->ow,
            d->kh, d->kw, d->sh, d->sw, d->ph, d->pw, d->name);
}

const dt_conf_t *str2cfg(const char *str) {
#define CASE(name, cfg) if (!strcasecmp(name, str)) return cfg
    CASE("f32", conf_f32);
    CASE("f32_wino", conf_f32_wino);
    CASE("s16s32", conf_s16s32);
    CASE("s8s32", conf_s8s32);
#undef CASE
    []() { SAFE(FAIL, CRIT); return 0; }();
    return (const dt_conf_t *)1;
}

const char *cfg2str(const dt_conf_t *cfg) {
#define CASE(name, _cfg) if (cfg == _cfg) return name
    CASE("f32", conf_f32);
    CASE("f32_wino", conf_f32_wino);
    CASE("s16s32", conf_s16s32);
    CASE("s8s32", conf_s8s32);
#undef CASE
    []() { SAFE(FAIL, CRIT); return 0; }();
    return NULL;
}

void prb2str(const prb_t *p, char *buffer, bool canonical) {
    char desc_buf[max_desc_len];
    char dir_str[32] = {0}, cfg_str[32] = {0}, alg_str[32] = {0},
         merge_str[32] = {0};
    desc2str(p, desc_buf, canonical);
    snprintf(dir_str, sizeof(dir_str), "--dir=%s ", dir2str(p->dir));
    snprintf(cfg_str, sizeof(cfg_str), "--cfg=%s ", cfg2str(p->cfg));
    snprintf(alg_str, sizeof(alg_str), "--alg=%s ", alg2str(p->alg));
    snprintf(merge_str, sizeof(merge_str), "--merge=%s ", merge2str(p->merge));
    snprintf(buffer, max_prb_len, "%s%s%s%s%s",
            p->dir == FWD_B ? "" : dir_str,
            p->cfg == conf_f32 ? "" : cfg_str,
            p->alg == DIRECT ? "" : alg_str,
            p->merge == NONE ? "" : merge_str,
            desc_buf);
}

bool maybe_skip(const char *impl_str) {
    if (skip_impl == NULL || *skip_impl == '\0')
        return false;

    const size_t max_len = 128;
    char what[max_len] = {0};

    const char *s_start = skip_impl;
    while (1) {
        if (*s_start == '"' || *s_start == '\'')
            ++s_start;

        const char *s_end = strchr(s_start, ':');
        size_t len = s_end ? s_end - s_start : strlen(s_start);

        if (s_start[len - 1] == '"' || s_start[len - 1] == '\'')
            --len;

        SAFE(len < max_len ? OK : FAIL, CRIT);
        len = MIN2(len, max_len - 1);
        strncpy(what, s_start, len);
        what[len] = '\0';

        if (strstr(impl_str, what))
            return true;

        if (s_end == NULL)
            break;

        s_start = s_end + 1;
        if (*s_start == '\0')
            break;
    }

    return false;
}

}
