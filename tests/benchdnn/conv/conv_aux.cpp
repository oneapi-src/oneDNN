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

attr_t::round_mode_t attr_t::str2rmode(const char *str) {
#define CASE(_rmd) if (!strncasecmp(STRINGIFY(_rmd), str, \
            strlen(STRINGIFY(_rmd)))) return _rmd
    CASE(NEAREST);
    CASE(DOWN);
#undef CASE
    assert(!"unknown attr::round_mode");
    return NEAREST;
}

const char *attr_t::rmode2str(attr_t::round_mode_t rmode) {
    if (rmode == NEAREST) return "nearest";
    if (rmode == DOWN) return "down";
    assert(!"unknown attr::round_mode");
    return "unknown attr::round_mode";
}

attr_t::scale_t::policy_t attr_t::scale_t::str2policy(const char *str) {
#define CASE(_plc) if (!strcasecmp(STRINGIFY(_plc), str)) return _plc
    CASE(NONE);
    CASE(COMMON);
    CASE(PER_OC);
#undef CASE
    assert(!"unknown attr::scale::policy");
    return NONE;
}

const char *attr_t::scale_t::policy2str(attr_t::scale_t::policy_t policy) {
    if (policy == NONE) return "none";
    if (policy == COMMON) return "common";
    if (policy == PER_OC) return "per_oc";
    assert(!"unknown attr::scale::policy");
    return "unknown attr::scale::policy";
}

int attr_t::scale_t::str2scale(const char *str, const char **end_s) {
    if (str == NULL) return FAIL;

    const char *s_;
    const char * &s = end_s ? *end_s : s_;
    s = str;

    for (policy_t p = NONE; true; p = (policy_t)((int)p + 1)) {
        if (p == POLICY_TOTAL) return FAIL;

        const char *ps = policy2str(p);
        if (!strncasecmp(ps, s, strlen(ps))) {
            this->policy = p;
            s += strlen(ps);
            break;
        }
    }

    if (*s != ':') return OK;
    s++;

    char *end;
    this->scale = strtof(s, &end);
    if (this->scale <= 0 || end == s) return FAIL;

    s = end;
    assert(*s == '\0' || *s == ';');

    return OK;
}

void attr_t::scale_t::scale2str(char *buffer, char **end_b) const {
    assert(buffer);
    int len = sprintf(buffer, "%s:%g", policy2str(this->policy), this->scale);
    if (end_b) *end_b = buffer + len;
}

bool attr_t::is_def() const {
    return true
        && irmode == round_mode_t::NEAREST
        && oscale.is_def();
}

int attr_t::mkldnn_attr_recreate() {
    if (mkldnn_attr) mkldnn_primitive_attr_destroy(mkldnn_attr);
    DNN_SAFE(mkldnn_primitive_attr_create(&mkldnn_attr), CRIT);

    if (irmode != round_mode_t::NEAREST)
        DNN_SAFE(mkldnn_primitive_attr_set_int_output_round_mode(mkldnn_attr,
                    (mkldnn_round_mode_t)irmode), CRIT);

    if (!oscale.is_def()) {
        int count = oscale.policy == scale_t::policy_t::COMMON ? 1 : 4096;
        int mask = oscale.policy == scale_t::policy_t::PER_OC ? 1 << 1 : 0;
        float *scales = (float *)zmalloc(count * sizeof(float), 64);
        SAFE(scales != NULL ? OK : FAIL, CRIT);
        for (int i = 0; i < count; ++i)
            scales[i] = oscale.scale; /* TODO: extend for many cases */
        DNN_SAFE(mkldnn_primitive_attr_set_output_scales(mkldnn_attr, count,
                    mask, scales), CRIT);
        zfree(scales);
    }

    return OK;
}

int str2attr(attr_t *attr, const char *str) {
    if (attr == NULL || str == NULL) return FAIL;
    *attr = attr_t();

    const char *s = str;

    while (*s != '\0') {
        int rc = FAIL;
        const char *param;

        param = "irmode=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            attr->irmode = attr_t::str2rmode(s);
            s += strlen(attr_t::rmode2str(attr->irmode));
            rc = OK;
        }

        param = "oscale=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            rc = attr->oscale.str2scale(s, &s);
            if (rc != OK) return rc;
        }

        if (rc != OK) return FAIL;
        if (*s == ';') ++s;
    }

    attr->mkldnn_attr_recreate();

    return OK;
}

void attr2str(const attr_t *attr, char *buffer) {
    buffer += sprintf(buffer, "irmode=%s", attr_t::rmode2str(attr->irmode));
    buffer += sprintf(buffer, ";oscale=");
    attr->oscale.scale2str(buffer, NULL);
}

int str2desc(desc_t *desc, const char *str) {
    desc_t d{0};

    /* canonical form:
     * dYgXmbXicXihXiwXocXohXowXkhXkwXshXswXphXpwXdhXdwXnS
     *
     * where: Y = {fb, fd, bd, bw, bb}, X is number, S - string
     * note: symbol `_` is ignored
     *
     * implicit rules:
     *  - default values:
     *      mb = 2, g = 1, d = fd, sh = sw = 1, dh = dw = 0, S="wip"
     *  - if H is undefined => H = W
     *  - if W is undefined => W = H
     *  - if `output` is undefined => compute output
     *  - if padding is undefined => compute trivial padding
     */

    d.g = 1; d.mb = 2; d.sh = d.sw = 1; d.dh = d.dw = 0; d.name = "\"wip\"";

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
        CASE_N(dh); CASE_N(dw);
        if (*s == 'n') { d.name = s + 1; break; }
        if (*s == '_') ++s;
        if (!ok) return FAIL;
    }
#   undef CASE_NN
#   undef CASE_N

    if (d.ic == 0 || d.oc == 0) return FAIL;

    auto compute_out = [](int i, int k, int s, int p, int d) {
        return (i - ((k - 1) * (d + 1) + 1) + 2 * p) / s + 1;
    };
    auto compute_pad = [](int o, int i, int k, int s, int d) {
        /* XXX: is it oK? */
        return ((o - 1) * s - i + ((k - 1) * (d + 1) + 1)) / 2;
    };

    const bool no_h = (d.ih | d.kh | d.oh | d.ph | d.dh) == 0 && d.sh == 1;
    const bool no_w = (d.iw | d.kw | d.ow | d.pw | d.dw) == 0 && d.sw == 1;

    if (!no_h) {
        if (!d.ih || !d.kh) return FAIL;

        if (!d.oh) d.oh = compute_out(d.ih, d.kh, d.sh, d.ph, d.dh);
        else if (!d.ph && d.oh != compute_out(d.ih, d.kh, d.sh, d.ph, d.dh))
            d.ph = compute_pad(d.oh, d.ih, d.kh, d.ph, d.dh);
    }

    if (!no_w) {
        if (!d.iw || !d.kw) return FAIL;

        if (!d.ow) d.ow = compute_out(d.iw, d.kw, d.sw, d.pw, d.dw);
        else if (!d.pw && d.ow != compute_out(d.iw, d.kw, d.sw, d.pw, d.dw))
            d.pw = compute_pad(d.ow, d.iw, d.kw, d.pw, d.dw);
    }

    if (no_w) {
        d.iw = d.ih;
        d.kw = d.kh;
        d.ow = d.oh;
        d.pw = d.ph;
        d.sw = d.sh;
        d.dw = d.dh;
    } else if (no_h) {
        d.ih = d.iw;
        d.kh = d.kw;
        d.oh = d.ow;
        d.ph = d.pw;
        d.sh = d.sw;
        d.dh = d.dw;
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

    if (canonical || d->g != 1) DPRINT("g%d", d->g);
    if (canonical || d->mb != 2) DPRINT("mb%d", d->mb);

    const bool half_form = d->ih == d->iw && d->kh == d->kw && d->oh == d->ow
        && d->sh == d->sw && d->ph == d->pw && d->dh == d->dw;

    if (!canonical && half_form) {
        DPRINT("ic%dih%doc%doh%dkh%d", d->ic, d->ih, d->oc, d->oh, d->kh);
        if (d->sh != 1) DPRINT("sh%d", d->sh);
        if (d->ph != 0) DPRINT("ph%d", d->ph);
        if (d->dh != 0) DPRINT("dh%d", d->dh);
    } else {
        DPRINT("ic%dih%diw%doc%doh%dow%dkh%dkw%d",
                d->ic, d->ih, d->iw, d->oc, d->oh, d->ow, d->kh, d->kw);
        if (canonical || d->sh != 1 || d->sw != 1)
            DPRINT("sh%dsw%d", d->sh, d->sw);
        if (canonical || d->ph != 0 || d->sh != 0)
            DPRINT("ph%dpw%d", d->ph, d->pw);
        if (canonical || d->dh != 0 || d->dw != 0)
            DPRINT("dh%ddw%d", d->dh, d->dw);
    }

    DPRINT("n%s", d->name);

#   undef DPRINT
}

void prb_t::count_ops() {
    if (ops > 0) return;

    double sp_ops = 0;
    for (int oh = 0; oh < this->oh; ++oh) {
    for (int ow = 0; ow < this->ow; ++ow) {
        for (int kh = 0; kh < this->kh; ++kh) {
            const int ih = oh * this->sh - this->ph + kh * (this->dh + 1);
            if (ih < 0 || ih >= this->ih) continue;
            for (int kw = 0; kw < this->kw; ++kw) {
                const int iw = ow * this->sw - this->pw + kw * (this->dw + 1);
                if (iw < 0 || iw >= this->iw) continue;
                sp_ops += 1;
            }
        }
    }
    }

    ops = 2 * this->mb * this->oc * this->ic / this->g * sp_ops;
}

void prb2str(const prb_t *p, char *buffer, bool canonical) {
    char desc_buf[max_desc_len], attr_buf[max_attr_len];
    char dir_str[32] = {0}, cfg_str[32] = {0}, alg_str[32] = {0},
         merge_str[32] = {0};
    desc2str(p, desc_buf, canonical);
    snprintf(dir_str, sizeof(dir_str), "--dir=%s ", dir2str(p->dir));
    snprintf(cfg_str, sizeof(cfg_str), "--cfg=%s ", cfg2str(p->cfg));
    snprintf(alg_str, sizeof(alg_str), "--alg=%s ", alg2str(p->alg));
    snprintf(merge_str, sizeof(merge_str), "--merge=%s ", merge2str(p->merge));
    bool is_attr_def = p->attr.is_def();
    if (!is_attr_def) {
        int len = snprintf(attr_buf, max_attr_len, "--attr=\"");
        attr2str(&p->attr, attr_buf + len);
        len = strnlen(attr_buf, max_attr_len);
        snprintf(attr_buf + len, max_attr_len - len, "\" ");
    }
    snprintf(buffer, max_prb_len, "%s%s%s%s%s%s",
            p->dir == FWD_B ? "" : dir_str,
            p->cfg == conf_f32 ? "" : cfg_str,
            p->alg == DIRECT ? "" : alg_str,
            p->merge == NONE ? "" : merge_str,
            is_attr_def ? "" : attr_buf,
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
