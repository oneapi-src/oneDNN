/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "dnn_types.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_debug.hpp"
#include "conv/conv.hpp"

namespace conv {

alg_t str2alg(const char *str) {
#define CASE(_alg) if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(AUTO);
    CASE(DIRECT);
    CASE(WINO);
#undef CASE
    assert(!"unknown algorithm");
    return DIRECT;
}

const char *alg2str(alg_t alg) {
    if (alg == AUTO) return "auto";
    if (alg == DIRECT) return "direct";
    if (alg == WINO) return "wino";
    assert(!"unknown algorithm");
    return "unknown algorithm";
}

alg_t alg_kind2alg(mkldnn_alg_kind_t alg) {
    if (alg == mkldnn_convolution_auto) return AUTO;
    if (alg == mkldnn_convolution_direct) return DIRECT;
    if (alg == mkldnn_convolution_winograd) return WINO;
    assert(!"unknown algorithm");
    return DIRECT;
}

int str2desc(desc_t *desc, const char *str, bool is_deconv) {
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

    d.g = 1; d.mb = 2; d.sd = d.sh = d.sw = 1; d.dd = d.dh = d.dw = 0;
    d.has_groups = false, d.name = "\"wip\"";
    d.pw = -1; d.ph = -1; d.pd = -1;

    const char *s = str;
    assert(s);

#   define CASE_NN(p, c) do { \
        if (!strncmp(p, s, strlen(p))) { \
            ok = 1; s += strlen(p); \
            char *end_s; d. c = strtol(s, &end_s, 10); s += (end_s - s); \
            if (!strncmp(p, "g", 1)) d.has_groups = true; \
            if (d. c < 0) return FAIL; \
            /* printf("@@@debug: %s: %d\n", p, d. c); */ \
        } \
    } while (0)
#   define CASE_N(c) CASE_NN(#c, c)
    while (*s) {
        int ok = 0;
        CASE_N(g); CASE_N(mb);
        CASE_N(ic); CASE_N(id); CASE_N(ih); CASE_N(iw);
        CASE_N(oc); CASE_N(od); CASE_N(oh); CASE_N(ow);
        CASE_N(kd); CASE_N(kh); CASE_N(kw);
        CASE_N(sd); CASE_N(sh); CASE_N(sw);
        CASE_N(pd); CASE_N(ph); CASE_N(pw);
        CASE_N(dd); CASE_N(dh); CASE_N(dw);
        if (*s == 'n') { d.name = s + 1; break; }
        if (*s == '_') ++s;
        if (!ok) return FAIL;
    }
#   undef CASE_NN
#   undef CASE_N

    if (d.ic == 0 || d.oc == 0) return FAIL;
    if (d.sd <= 0 || d.sh <= 0 || d.sw <= 0) return FAIL;

    auto compute_out = [](bool is_deconv, int64_t i, int64_t k, int64_t s,
            int64_t p, int64_t d) {
        if (is_deconv)
            return (i - 1) * s + (k - 1) * (d + 1) + 2 * p + 1;
        else
            return (i - ((k - 1) * (d + 1) + 1) + 2 * p) / s + 1;
    };
    auto compute_pad = [](bool is_deconv, int64_t o, int64_t i, int64_t k,
            int64_t s, int64_t d) {
        if (is_deconv)
            return ((i - 1) * s - o + ((k - 1) * (d + 1) + 1)) / 2;
        else
            return ((o - 1) * s - i + ((k - 1) * (d + 1) + 1)) / 2;
    };

    const bool no_d = (d.id | d.kd | d.od | d.dd) == 0 && d.sd == 1 && d.pd < 1;
    const bool no_h = (d.ih | d.kh | d.oh | d.dh) == 0 && d.sh == 1 && d.ph < 1;
    const bool no_w = (d.iw | d.kw | d.ow | d.dw) == 0 && d.sw == 1 && d.pw < 1;
    if (no_d && no_h && no_w) return FAIL;

    if (!no_d) {
        if (!d.id || !d.kd) return FAIL;
        if (!d.od) {
            d.pd = 0;
            d.od = compute_out(is_deconv, d.id, d.kd, d.sd, d.pd, d.dd);
        } else if (d.pd < 0)
            d.pd = compute_pad(is_deconv, d.od, d.id, d.kd, d.sd, d.dd);
    }

    if (!no_h) {
        if (!d.ih || !d.kh) return FAIL;
        if (!d.oh) {
            d.ph = 0;
            d.oh = compute_out(is_deconv, d.ih, d.kh, d.sh, d.ph, d.dh);
        } else if (d.ph < 0)
            d.ph = compute_pad(is_deconv, d.oh, d.ih, d.kh, d.sh, d.dh);
    }

    if (!no_w) {
        if (!d.iw || !d.kw) return FAIL;
        if (!d.ow) {
            d.pw = 0;
            d.ow = compute_out(is_deconv, d.iw, d.kw, d.sw, d.pw, d.dw);
        } else if (d.pw < 0)
            d.pw = compute_pad(is_deconv, d.ow, d.iw, d.kw, d.sw, d.dw);
    }

    if (no_w && no_h && d.id > 0) {
        // User specified values for the d dimesion but not values h and w
        // dimensions. Propagate *d values to h and w dimesions.
        d.iw = d.ih = d.id;
        d.kw = d.kh = d.kd;
        d.ow = d.oh = d.od;
        d.pw = d.ph = d.pd;
        d.sw = d.sh = d.sd;
        d.dw = d.dh = d.dd;
    } else if (no_w && d.ih > 0) {
        /* user specified values for h dimesion but not w
         * dimension. propagate *h values to the w dimension. */
        d.iw = d.ih;
        d.kw = d.kh;
        d.ow = d.oh;
        d.pw = d.ph;
        d.sw = d.sh;
        d.dw = d.dh;
    } else if (no_h && !no_w) {
        /* this is a 1D convolution. set default values for the h dimension  */
        d.ih = 1;
        d.kh = 1;
        d.oh = 1;
        d.ph = 0;
        d.sh = 1;
        d.dh = 0;
    }

    if (no_d) {
        /* user did not specify values for the d dimension. set them
         * to defaults. this also happens for 1D convolutions. */
        d.id = 1;
        d.kd = 1;
        d.od = 1;
        d.sd = 1;
        d.pd = 0;
        d.dd = 0;
    }

    *desc = d;

    return OK;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    const bool canonical = s.flags() & std::ios_base::fixed;

    if (canonical || d.has_groups) s << "g" << d.g;
    if (canonical || d.mb != 2) s << "mb" << d.mb;

    const bool half_form = (d.ih == d.iw && d.kh == d.kw && d.oh == d.ow
        && d.sh == d.sw && d.ph == d.pw && d.dh == d.dw) && d.id == 1;

    const bool print_d = d.id > 1;
    const bool print_w = canonical || print_d || !half_form;

    auto print_spatial = [&](
            const char *sd, int64_t vd,
            const char *sh, int64_t vh,
            const char *sw, int64_t vw) {
        if (print_d) s << sd << vd;
        s << sh << vh;
        if (print_w) s << sw << vw;
    };

    s << "ic" << d.ic;
    print_spatial("id", d.id, "ih", d.ih, "iw", d.iw);
    s << "oc" << d.oc;
    print_spatial("od", d.od, "oh", d.oh, "ow", d.ow);
    print_spatial("kd", d.kd, "kh", d.kh, "kw", d.kw);

    if (canonical || d.sh != 1 || d.sw != 1 || d.sd != 1)
        print_spatial("sd", d.sd, "sh", d.sh, "sw", d.sw);

    if (canonical || d.ph != 0 || d.pw != 0 || d.pd != 0)
        print_spatial("pd", d.pd, "ph", d.ph, "pw", d.pw);

    if (canonical || d.dh != 0 || d.dw != 0 || d.dd != 0)
        print_spatial("dd", d.dd, "dh", d.dh, "dw", d.dw);

    s << "n" << d.name;

    return s;
}

void prb_t::count_ops() {
    if (ops > 0) return;

    int64_t od_t = is_deconv ? this->id : this->od;
    int64_t oh_t = is_deconv ? this->ih : this->oh;
    int64_t ow_t = is_deconv ? this->iw : this->ow;
    int64_t id_t = is_deconv ? this->od : this->id;
    int64_t ih_t = is_deconv ? this->oh : this->ih;
    int64_t iw_t = is_deconv ? this->ow : this->iw;
    double sp_ops = 0;
    for (int64_t od = 0; od < od_t; ++od) {
    for (int64_t oh = 0; oh < oh_t; ++oh) {
    for (int64_t ow = 0; ow < ow_t; ++ow) {
        for (int64_t kd = 0; kd < this->kd; ++kd) {
            const int64_t id = od * this->sd - this->pd + kd * (this->dd + 1);
            if (id < 0 || id >= id_t) continue;
            for (int64_t kh = 0; kh < this->kh; ++kh) {
                const int64_t ih = oh * this->sh - this->ph + kh * (this->dh + 1);
                if (ih < 0 || ih >= ih_t) continue;
                for (int64_t kw = 0; kw < this->kw; ++kw) {
                    const int64_t iw = ow * this->sw - this->pw + kw * (this->dw + 1);
                    if (iw < 0 || iw >= iw_t) continue;
                    sp_ops += 1;
                }
            }
        }
    }
    }
    }

    ops = 2 * this->mb * this->oc * this->ic / this->g * sp_ops;
}

void prb_t::generate_oscales() {
    if (attr.oscale.policy != attr_t::scale_t::policy_t::PER_OC) return;

    scales = (float *)zmalloc(sizeof(float) * oc, 64);
    SAFE_V(scales != NULL ? OK : FAIL);

    const float K = 32;
    /* scale in [1/K .. K], with starting point at oscale.scale */
    float s[2] = {attr.oscale.scale, attr.oscale.scale/2};
    for (int64_t i = 0; i < oc; ++i) {
        int64_t si = i % 2; // 0 -> left, 1 -> right
        scales[i] = s[si];
        if (si == 0) {
            s[si] /= 2.;
            if (s[si] < 1./K) s[si] *= K*K; // turn around to become ~K
        } else {
            s[si] *= 2.;
            if (s[si] > K) s[si] /= K*K; // turn around to become ~K
        }
    }
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    if (p.dir != FWD_B)
        s << "--dir=" << dir2str(p.dir) << " ";
    if (p.cfg != conf_f32)
        s << "--cfg=" << cfg2str(p.cfg) << " ";
    if (p.alg != DIRECT)
        s << "--alg=" << alg2str(p.alg) << " ";
    if (!p.attr.is_def())
        s << "--attr=\"" << p.attr << "\" ";

    s << static_cast<const desc_t &>(p);

    return s;
}

}
