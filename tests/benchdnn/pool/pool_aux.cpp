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
    /* canonical form:
     * mbXicX_odXihXiwX_odXohXowX_kdXkhXkwX_sdXshXswX_pdXphXpwX_nS
     *
     * where X is number, S - string
     * note: symbol `_` is ignored
     *
     * implicit rules:
     *  - if smaller dimensions are not specified => square or cubic form;
     *  - if output is undefined => compute output
     *  - if padding is undefined => compute trivial padding
     */

    desc_t d {0};
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

    if (!no_d) {
        if (!d.id || !d.kd) return FAIL;
        if (!d.od) {
            if (d.pd < 0) d.pd = 0;
            d.od = compute_out(d.id, d.kd, d.sd, d.pd);
        } else if (d.pd < 0)
            d.pd = compute_pad(d.od, d.id, d.kd, d.sd);
    }

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

    if (sanitize_desc(d.ndims, {d.od, d.id, d.kd, d.sd, d.pd},
                {d.oh, d.ih, d.kh, d.sh, d.ph}, {d.ow, d.iw, d.kw, d.sw, d.pw},
                {1, 1, 1, 1, 0}, true)
            != OK)
        return FAIL;

    *desc = d;

    return OK;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    bool print_d = true, print_h = true, print_w = true;
    print_dhw(print_d, print_h, print_w, d.ndims,
            {d.od, d.id, d.kd, d.sd, d.pd}, {d.oh, d.ih, d.kh, d.sh, d.ph},
            {d.ow, d.iw, d.kw, d.sw, d.pw});

    auto print_spatial
            = [&](const char *d_str, int64_t d_val, const char *h_str,
                      int64_t h_val, const char *w_str, int64_t w_val) {
                  if (print_d) s << d_str << d_val;
                  if (print_h) s << h_str << h_val;
                  if (print_w) s << w_str << w_val;
              };

    if (canonical || d.mb != 2) s << "mb" << d.mb;
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
    settings_t def;

    if (canonical || p.dir != def.dir[0]) s << "--dir=" << p.dir << " ";
    if (canonical || p.cfg != def.cfg[0]) s << "--cfg=" << p.cfg << " ";
    if (canonical || p.tag != def.tag[0]) s << "--tag=" << p.tag << " ";
    if (canonical || p.alg != def.alg[0])
        s << "--alg=" << alg2str(p.alg) << " ";

    s << static_cast<const desc_t &>(p);

    return s;
}

bool prb_t::maybe_skip_nvidia() const {
    const auto dat_tag = convert_tag(this->tag, this->ndims);

    if (!cudnn_supported_tag_plain(dat_tag)
            && !cudnn_supported_tag_blocking(dat_tag)) {
        return true;
    }

    if (cfg[SRC].dt == dnnl_u8 || cfg[SRC].dt == dnnl_s32) { return true; }

    const bool is_4b = dat_tag == dnnl_aBc4b || dat_tag == dnnl_aBcd4b
            || dat_tag == dnnl_aBcde4b;

    if (cfg[SRC].dt != dnnl_s8 && is_4b) return true;
    if (cfg[SRC].dt == dnnl_s8 && is_4b && ic % 4 != 0) return true;
    if (cfg[SRC].dt == dnnl_f16 && dir != FWD_I) { return true; }

    auto bph = [](int64_t ih, int64_t oh, int64_t kh, int64_t sh, int64_t ph) {
        return (oh - 1) * sh - ih + kh - ph;
    };

    dnnl_dim_t padding_l_nd[] = {pd, ph, pw};
    dnnl_dim_t padding_r_nd[] = {bph(id, od, kd, sd, pd),
            bph(ih, oh, kh, sh, ph), bph(iw, ow, kw, sw, pw)};
    dnnl_dim_t *padding_l = padding_l_nd + (5 - ndims);
    dnnl_dim_t *padding_r = padding_r_nd + (5 - ndims);

    const auto padFront = ndims >= 5 ? padding_l[ndims - 5] : 0;
    const auto padT = ndims >= 4 ? padding_l[ndims - 4] : 0;
    const auto padL = padding_l[ndims - 3];

    const auto padBack = ndims >= 5 ? padding_r[ndims - 5] : 0;
    const auto padB = ndims >= 4 ? padding_r[ndims - 4] : 0;
    const auto padR = padding_r[ndims - 3];

    // The following cases are skipped due to a padding limitation of cuDNN
    if (alg == alg_t::AVG_P
            && (padL < padR || padT < padB || padFront < padBack)) {
        return true;
    }

    return false;
}

} // namespace pool
