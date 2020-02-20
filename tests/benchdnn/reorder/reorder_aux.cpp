/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "dnnl_debug.hpp"

#include "reorder/reorder.hpp"

namespace reorder {

alg_t str2alg(const char *str) {
    if (!strcasecmp("bootstrap", str)) return ALG_BOOT;
    if (!strcasecmp("reference", str)) return ALG_REF;
    assert(!"unknown algorithm");
    return ALG_REF;
}

const char *alg2str(alg_t alg) {
    switch (alg) {
        case ALG_REF: return "reference";
        case ALG_BOOT: return "bootstrap";
        default: assert(!"unknown algorithm"); return "unknown algorithm";
    }
}

flag_t str2flag(const char *str) {
    if (!strcasecmp("conv_s8s8", str))
        return FLAG_CONV_S8S8;
    else if (!strcasecmp("gconv_s8s8", str))
        return FLAG_GCONV_S8S8;
    assert(!"unknown flag");
    return FLAG_NONE;
}

const char *flag2str(flag_t flag) {
    switch (flag) {
        case FLAG_NONE: return "";
        case FLAG_CONV_S8S8: return "conv_s8s8";
        case FLAG_GCONV_S8S8: return "gconv_s8s8";
        default: assert(!"Invalid flag"); return "";
    }
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    s << "--sdt=" << dt2str(cfg2dt(p.conf_in)) << " ";
    s << "--ddt=" << dt2str(cfg2dt(p.conf_out)) << " ";
    s << "--stag=" << fmt_tag2str(p.reorder.tag_in) << " ";
    s << "--dtag=" << fmt_tag2str(p.reorder.tag_out) << " ";

    if (canonical || p.alg != ALG_REF) s << "--alg=" << alg2str(p.alg) << " ";
    if (canonical || p.oflag != FLAG_NONE)
        s << "--oflag=" << flag2str(p.oflag) << " ";
    if (canonical || p.runtime_dim_mask != 0)
        s << "--runtime-dim-mask=" << p.runtime_dim_mask << " ";
    if (canonical || !p.attr.is_def()) s << "--attr=\"" << p.attr << "\" ";

    s << p.reorder.dims;

    return s;
}

} // namespace reorder
