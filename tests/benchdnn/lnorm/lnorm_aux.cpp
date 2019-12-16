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

flags_t str2flags(const char *str) {
    flags_t flags = bnorm::str2flags(str);
    assert(flags <= (GLOB_STATS | USE_SCALESHIFT));
    return flags;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    if (canonical || p.dir != FWD_D) s << "--dir=" << dir2str(p.dir) << " ";
    if (canonical || p.tag != dnnl_tnc)
        s << "--tag=" << fmt_tag2str(p.tag) << " ";
    if (canonical || p.stat_tag != dnnl_tn)
        s << "--stat_tag=" << fmt_tag2str(p.stat_tag) << " ";
    if (canonical || p.dt != dnnl_f32) s << "--dt=" << dt2str(p.dt) << " ";
    if (canonical || p.flags != (flags_t)0)
        s << "--flags=" << flags2str(p.flags) << " ";
    if (canonical || !p.attr.is_def()) s << "--attr=\"" << p.attr << "\" ";
    if (canonical || p.inplace != true)
        s << "--inplace=" << bool2str(p.inplace) << " ";

    s << p.dims;

    return s;
}
} // namespace lnorm
