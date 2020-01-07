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

#include <assert.h>
#include <stdlib.h>
#include "lnorm/lnorm.hpp"

namespace lnorm {
check_alg_t str2check_alg(const char *str) {
    if (!strcasecmp("alg_0", str)) return ALG_0;
    if (!strcasecmp("alg_1", str)) return ALG_1;
    return ALG_AUTO;
}

const char *check_alg2str(check_alg_t alg) {
    switch (alg) {
        case ALG_0: return "alg_0";
        case ALG_1: return "alg_1";
        case ALG_AUTO: return "alg_auto";
    }
    return "alg_auto";
}

flags_t str2flags(const char *str) {
    flags_t flags = (flags_t)0;
    while (str && *str) {
        if (*str == 'G') flags |= GLOB_STATS;
        if (*str == 'S') flags |= USE_SCALESHIFT;
        str++;
    }
    return flags;
}

const char *flags2str(flags_t flags) {
    if (flags & GLOB_STATS) {
        if (flags & USE_SCALESHIFT) return "GS";
        return "G";
    }

    if (flags & USE_SCALESHIFT) return "S";

    return "";
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    if (p.dir != FWD_D) s << "--dir=" << dir2str(p.dir) << " ";
    if (p.tag != dnnl_tnc) s << "--tag=" << fmt_tag2str(p.tag) << " ";
    if (p.stat_tag != dnnl_format_tag_any)
        s << "--stat_tag=" << fmt_tag2str(p.stat_tag) << " ";
    if (p.dt != dnnl_f32) s << "--dt=" << dt2str(p.dt) << " ";
    if (p.flags != (flags_t)0) s << "--flags=" << flags2str(p.flags) << " ";
    if (!p.attr.is_def()) s << "--attr=\"" << p.attr << "\" ";
    if (p.inplace != true) s << "--inplace=" << bool2str(p.inplace) << " ";

    s << p.dims;

    return s;
}
} // namespace lnorm
