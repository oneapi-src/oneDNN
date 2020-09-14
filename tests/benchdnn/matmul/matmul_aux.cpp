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

    if (attr.oscale.policy == policy_t::COMMON) {
        scales = (float *)zmalloc(sizeof(float), 4);
        SAFE_V(scales != nullptr ? OK : FAIL);
        scales[0] = attr.oscale.scale;
        return;
    }

    assert(attr.oscale.policy == policy_t::PER_OC);

    scales = (float *)zmalloc(sizeof(float) * n, 64);
    SAFE_V(scales != nullptr ? OK : FAIL);

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

#define CASE_NN(prb, c) \
    do { \
        if (!strncmp(prb, s, strlen(prb))) { \
            ok = 1; \
            s += strlen(prb); \
            char *end_s; \
            d.c = strtol(s, &end_s, 10); \
            s += (end_s - s); \
            if (d.c < 0) return FAIL; \
            /* printf("@@@debug: %s: %d\n", prb, d. c); */ \
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

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    dump_global_params(s);
    settings_t def;

    if (canonical || prb.cfg != def.cfg[0]) s << "--cfg=" << prb.cfg << " ";
    if (canonical || prb.stag != def.stag[0]) s << "--stag=" << prb.stag << " ";
    if (canonical || prb.wtag != def.wtag[0]) s << "--wtag=" << prb.wtag << " ";
    if (canonical || prb.dtag != def.dtag[0]) s << "--dtag=" << prb.dtag << " ";

    // TODO: switch me on when run-time leading dimensions will be supported
    // if (canonical || prb.ld_src != defaults::ld)
    //     s << "--ld_src=" << prb.ld_src << " ";
    // if (canonical || prb.ld_wei != defaults::ld)
    //     s << "--ld_wei=" << prb.ld_wei << " ";
    // if (canonical || prb.ld_dst != defaults::ld)
    //     s << "--ld_dst=" << prb.ld_dst << " ";

    if (canonical || prb.runtime_mb != def.runtime_mb[0])
        s << "--runtime_mb=" << prb.runtime_mb << " ";
    if (canonical || prb.runtime_m != def.runtime_m[0])
        s << "--runtime_m=" << prb.runtime_m << " ";
    if (canonical || prb.runtime_n != def.runtime_n[0])
        s << "--runtime_n=" << prb.runtime_n << " ";
    if (canonical || prb.runtime_k != def.runtime_k[0])
        s << "--runtime_k=" << prb.runtime_k << " ";

    if (canonical || prb.bia_dt != def.bia_dt[0]) {
        s << "--bia_dt=" << prb.bia_dt << " ";

        if (canonical || prb.bia_mask != def.bia_mask[0])
            s << "--bia_mask=" << prb.bia_mask << " ";
    }

    s << prb.attr;
    s << static_cast<const desc_t &>(prb);

    return s;
}

} // namespace matmul
