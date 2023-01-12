/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"

#include "ip/ip.hpp"

namespace ip {

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

    if (!d.name.empty()) s << "n" << d.name;

    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    dump_global_params(s);
    settings_t def;

    if (canonical || prb.dir != def.dir[0]) s << "--dir=" << prb.dir << " ";
    if (canonical || prb.cfg != def.cfg[0]) s << "--cfg=" << prb.cfg << " ";
    if (canonical || prb.stag != def.stag[0]) s << "--stag=" << prb.stag << " ";
    if (canonical || prb.wtag != def.wtag[0]) s << "--wtag=" << prb.wtag << " ";
    if (canonical || prb.dtag != def.dtag[0]) s << "--dtag=" << prb.dtag << " ";

    s << prb.attr;
    if (canonical || prb.ctx_init != def.ctx_init[0])
        s << "--ctx-init=" << prb.ctx_init << " ";
    if (canonical || prb.ctx_exe != def.ctx_exe[0])
        s << "--ctx-exe=" << prb.ctx_exe << " ";

    s << static_cast<const desc_t &>(prb);

    return s;
}

dims_t desc_t::src_dims() const {
    dims_t src_dims {mb, ic, id, ih, iw};
    for (int d = 0; d < 5 - ndims; ++d) {
        src_dims.erase(src_dims.begin() + 2);
    }

    return src_dims;
}

dims_t desc_t::wei_dims() const {
    dims_t wei_dims {oc, ic, id, ih, iw};
    for (int d = 0; d < 5 - ndims; ++d) {
        wei_dims.erase(wei_dims.begin() + 2);
    }

    return wei_dims;
}

dims_t desc_t::bia_dims() const {
    dims_t bia_dims {oc};
    return bia_dims;
}

dims_t desc_t::dst_dims() const {
    dims_t dst_dims {mb, oc};
    return dst_dims;
}

int64_t desc_t::desc_nelems(int arg, int mask) const {
    dims_t dims;
    switch (arg) {
        case DNNL_ARG_SRC: dims = src_dims(); break;
        case DNNL_ARG_WEIGHTS: dims = wei_dims(); break;
        case DNNL_ARG_DST: dims = dst_dims(); break;
        default: assert(!"unsupported arg");
    }

    int64_t nelems = 1;
    for (int d = 0; d < ndims; d++) {
        nelems *= (mask & (1 << d)) ? dims[d] : 1;
    }
    return nelems;
}

} // namespace ip
