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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

void prb_t::generate_oscales() {
    if (attr.oscale.is_def()) return;

    if (attr.oscale.policy == attr_t::scale_t::policy_t::COMMON) {
        scales = (float *)zmalloc(sizeof(float), 4);
        SAFE_V(scales != NULL ? OK : FAIL);
        scales[0] = attr.oscale.scale;
        return;
    }

    assert(attr.oscale.policy == attr_t::scale_t::policy_t::PER_OC);

    scales = (float *)zmalloc(sizeof(float) * n, 64);
    SAFE_V(scales != NULL ? OK : FAIL);

    const float K = 32;
    /* scale in [1/K .. K], with starting point at oscale.scale */
    float s[2] = {attr.oscale.scale, attr.oscale.scale / 2};
    for (int64_t i = 0; i < n; ++i) {
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

int str2desc(desc_t *desc, const char *str) {
    desc_t d {0};

    /* canonical form:
     * mbXmXnXkXnS
     *
     * where:
     * - X is number,
     * - S - string,
     *
     * note: symbol `_` is ignored
     *
     * note: n describes both 1) n - dimension and 2) n - name.
     *       The name is assumed to start with not a number symbol.
     *
     * default values:
     *      mb = 0, S="wip"
     */

    d.mb = 0;

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
        // order is important: check for name before n-dim
        if (*s == 'n' && !isdigit(*(s + 1))) {
            d.name = s + 1;
            break;
        }
        CASE_N(mb);
        CASE_N(m);
        CASE_N(n);
        CASE_N(k);
        if (*s == '_') ++s;
        if (!ok) return FAIL;
    }
#undef CASE_NN
#undef CASE_N

    if (d.mb < 0 || d.m < 0 || d.n < 0 || d.k < 0) return FAIL;
    if (d.m * d.n * d.k == 0) return FAIL;

    d.ndims = 2 + (d.mb != 0);
    if (d.ndims == 2) d.mb = 1;

    *desc = d;

    return OK;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    if (d.ndims == 3) s << "mb" << d.mb;
    s << "m" << d.m << "n" << d.n << "k" << d.k;

    if (d.name) s << "_n" << d.name;

    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);
    settings_t def;

    if (canonical || p.cfg != def.cfg[0]) s << "--cfg=" << p.cfg << " ";
    if (canonical || p.stag != def.stag[0]) s << "--stag=" << p.stag << " ";
    if (canonical || p.wtag != def.wtag[0]) s << "--wtag=" << p.wtag << " ";
    if (canonical || p.dtag != def.dtag[0]) s << "--dtag=" << p.dtag << " ";

    // TODO: switch me on when run-time leading dimensions will be supported
    // if (canonical || p.ld_src != defaults::ld)
    //     s << "--ld_src=" << p.ld_src << " ";
    // if (canonical || p.ld_wei != defaults::ld)
    //     s << "--ld_wei=" << p.ld_wei << " ";
    // if (canonical || p.ld_dst != defaults::ld)
    //     s << "--ld_dst=" << p.ld_dst << " ";

    if (canonical || p.runtime_mb != def.runtime_mb[0])
        s << "--runtime_mb=" << p.runtime_mb << " ";
    if (canonical || p.runtime_m != def.runtime_m[0])
        s << "--runtime_m=" << p.runtime_m << " ";
    if (canonical || p.runtime_n != def.runtime_n[0])
        s << "--runtime_n=" << p.runtime_n << " ";
    if (canonical || p.runtime_k != def.runtime_k[0])
        s << "--runtime_k=" << p.runtime_k << " ";

    if (canonical || p.bia_dt != def.bia_dt[0]) {
        s << "--bia_dt=" << p.bia_dt << " ";

        if (canonical || p.bia_mask != def.bia_mask[0])
            s << "--bia_mask=" << p.bia_mask << " ";
    }

    if (canonical || !p.attr.is_def()) s << "--attr=\"" << p.attr << "\" ";

    s << static_cast<const desc_t &>(p);

    return s;
}

bool prb_t::maybe_skip_nvidia() const {
    // Only support f32, f16, s8s8s8, and s8s8f32. Bias must be f32, or
    // optionally f16 in f16 config.
    if (!(cfg == matmul::conf_f32 || cfg == matmul::conf_f16
                || cfg == matmul::conf_s8s8f32 || cfg == matmul::conf_s8s8s8)) {
        return true;
    }

    if (!(attr.oscale.is_def())) {
        if (!(attr.oscale.policy == attr_t::scale_t::policy_t::COMMON)) {
            return true;
        }
    }

    if (!(attr.zero_points.is_def())) { return true; }

    if (bia_dt != dnnl_data_type_undef
            && !(bia_dt == dnnl_s8 || bia_dt == dnnl_f16
                    || bia_dt == dnnl_f32)) {
        return true;
    }

    // If 2 post-ops then must be sum->eltwise. Only sum and eltwise post-ops
    // supported.
    const auto &p = attr.post_ops;
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

} // namespace matmul
