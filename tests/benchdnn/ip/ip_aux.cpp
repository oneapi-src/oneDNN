/*******************************************************************************
 * Copyright 2018-2020 Intel Corporation
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

#include <algorithm>
#include <cctype>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"

#include "ip/ip.hpp"

namespace ip {

void prb_t::generate_oscales() {
    if (attr.oscale.policy != attr_t::scale_t::policy_t::PER_OC) return;

    scales = (float *)zmalloc(sizeof(float) * oc, 64);
    SAFE_V(scales != NULL ? OK : FAIL);

    const float K = 32;
    /* scale in [1/K .. K], with starting point at oscale.scale */
    float s[2] = {attr.oscale.scale, attr.oscale.scale / 2};
    for (int64_t i = 0; i < oc; ++i) {
        int64_t si = i % 2; // 0 -> left, 1 -> right
        scales[i] = s[si];
        if (si == 0) {
            s[si] /= 2.;
            if (s[si] < 1. / K) s[si] *= K * K; // turn around to become ~K
        } else {
            s[si] *= 2.;
            if (s[si] > K) s[si] /= K * K; // turn around to become ~K
        }
    }
}

bool prb_t::maybe_skip_nvidia() const {
    if (!(cfg == conf_f32 || cfg == ip::conf_f16 || cfg == ip::conf_s8s8f32
                || cfg == ip::conf_s8s8s8)) {
        return true;
    }

    // Otherwise fail for unsupported combinations of tags.
    auto input_tag = convert_tag(stag, ndims);
    auto weights_tag = convert_tag(wtag, ndims);
    auto dest_tag = convert_tag(dtag, ndims);

    // If a digit is in the string, blocking is used.
    bool source_uses_blocking
            = std::find_if(stag.begin(), stag.end(), ::isdigit) != stag.end();
    bool weights_uses_blocking
            = std::find_if(wtag.begin(), wtag.end(), ::isdigit) != wtag.end();
    bool blocking_used = source_uses_blocking || weights_uses_blocking;
    if (blocking_used) {
        auto &known = source_uses_blocking ? input_tag : weights_tag;
        auto &unknown = source_uses_blocking ? weights_tag : input_tag;
        if (known == dnnl_nChw4c || known == dnnl_nCdhw4c) {
            return !(known == unknown || unknown == dnnl_format_tag_any);
        } else {
            return true;
        }
    }

    bool src_ok = input_tag == dnnl_ncdhw || input_tag == dnnl_ndhwc
            || input_tag == dnnl_nchw || input_tag == dnnl_nhwc
            || input_tag == dnnl_ncw || input_tag == dnnl_nwc
            || input_tag == dnnl_nc || input_tag == dnnl_format_tag_any;
    bool wei_ok = weights_tag == dnnl_oidhw || weights_tag == dnnl_odhwi
            || weights_tag == dnnl_dhwio || weights_tag == dnnl_oihw
            || weights_tag == dnnl_ohwi || weights_tag == dnnl_hwio
            || weights_tag == dnnl_oiw || weights_tag == dnnl_owi
            || weights_tag == dnnl_wio || weights_tag == dnnl_io
            || weights_tag == dnnl_oi || weights_tag == dnnl_format_tag_any;
    bool dst_ok = dest_tag == dnnl_format_tag_any || dest_tag == dnnl_nc;

    bool format_tags_ok = src_ok && wei_ok && dst_ok;

    if (!format_tags_ok) { return true; }

    // Check that output scales are applied uniformly, and not per output
    // channel, if present.
    if (!(this->attr.oscale.is_def())) {
        if (!(this->attr.oscale.policy == attr_t::scale_t::policy_t::COMMON)) {
            return true;
        }
    }

    // Check for eltwise post-op restrictions.
    const auto &p = this->attr.post_ops;
    auto idx = p.find(attr_t::post_ops_t::kind_t::LINEAR);
    if (idx != -1) { return true; }
    idx = p.find(attr_t::post_ops_t::kind_t::TANH);
    if (idx != -1 && p.entry[idx].eltwise.beta != 0.f) { return true; }
    idx = p.find(attr_t::post_ops_t::kind_t::ELU);
    if (idx != -1 && p.entry[idx].eltwise.beta != 0.f) { return true; }
    idx = p.find(attr_t::post_ops_t::kind_t::RELU);
    if (idx != -1 && p.entry[idx].eltwise.alpha != 0.f) { return true; }
    if (idx != -1 && p.entry[idx].eltwise.beta != 0.f) { return true; }
    idx = p.find(attr_t::post_ops_t::kind_t::LOGISTIC);
    if (idx != -1 && p.entry[idx].eltwise.beta != 0.f) { return true; }
    idx = p.find(attr_t::post_ops_t::kind_t::BRELU);
    if (idx != -1 && p.entry[idx].eltwise.beta != 0.f) { return true; }

    if (p.len == 2) {
        return !(p.entry[0].kind == attr_t::post_ops_t::kind_t::SUM
                && (p.entry[1].kind == attr_t::post_ops_t::kind_t::TANH
                        || p.entry[1].kind == attr_t::post_ops_t::kind_t::ELU
                        || p.entry[1].kind == attr_t::post_ops_t::kind_t::RELU
                        || p.entry[1].kind
                                == attr_t::post_ops_t::kind_t::LOGISTIC
                        || p.entry[1].kind
                                == attr_t::post_ops_t::kind_t::BRELU));
    }
    return false;
}

int str2desc(desc_t *desc, const char *str) {
    // Canonical form: mbXicXidXihXiwXocXnS,
    // where
    //     X is integer
    //     S is string
    // note: symbol `_` is ignored.
    // Cubic/square shapes are supported by specifying just highest dimension.

    desc_t d {0};
    d.mb = 2;

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
            /* printf("@@@debug: %s: %d\n", p, d. c); */ \
        } \
    } while (0)
#define CASE_N(c) CASE_NN(#c, c)
    while (*s) {
        int ok = 0;
        CASE_N(mb);
        CASE_N(ic);
        CASE_N(ih);
        CASE_N(iw);
        CASE_N(id);
        CASE_N(oc);
        if (*s == 'n') {
            d.name = s + 1;
            break;
        }
        if (*s == '_') ++s;
        if (!ok) return FAIL;
    }
#undef CASE_NN
#undef CASE_N

    if (d.ic == 0 || d.oc == 0) return FAIL;

    if (sanitize_desc(d.ndims, {d.id}, {d.ih}, {d.iw}, {1}) != OK) return FAIL;

    *desc = d;

    return OK;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    bool print_d = true, print_h = true, print_w = true;
    print_dhw(print_d, print_h, print_w, d.ndims, {d.id}, {d.ih}, {d.iw});

    if (canonical || d.mb != 2) s << "mb" << d.mb;

    s << "ic" << d.ic;

    if (print_d) s << "id" << d.id;
    if (print_h) s << "ih" << d.ih;
    if (print_w) s << "iw" << d.iw;

    s << "oc" << d.oc;

    if (d.name) s << "n" << d.name;

    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);
    settings_t def;

    if (canonical || p.dir != def.dir[0]) s << "--dir=" << p.dir << " ";
    if (canonical || p.cfg != def.cfg[0]) s << "--cfg=" << p.cfg << " ";
    if (canonical || p.stag != def.stag[0]) s << "--stag=" << p.stag << " ";
    if (canonical || p.wtag != def.wtag[0]) s << "--wtag=" << p.wtag << " ";
    if (canonical || p.dtag != def.dtag[0]) s << "--dtag=" << p.dtag << " ";
    if (canonical || !p.attr.is_def()) s << "--attr=\"" << p.attr << "\" ";

    s << static_cast<const desc_t &>(p);

    return s;
}

} // namespace ip
