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
    settings_t def;

    if (canonical || p.dir != def.dir[0]) s << "--dir=" << p.dir << " ";
    if (canonical || p.dt != def.dt[0]) s << "--dt=" << p.dt << " ";
    if (canonical || p.tag != def.tag[0]) s << "--tag=" << p.tag << " ";
    if (canonical || p.stat_tag != def.stat_tag[0])
        s << "--stat_tag=" << p.stat_tag << " ";
    if (canonical || p.flags != def.flags[0])
        s << "--flags=" << flags2str(p.flags) << " ";
    if (canonical || p.inplace != def.inplace[0])
        s << "--inplace=" << bool2str(p.inplace) << " ";

    s << p.dims;

    return s;
}
} // namespace lnorm
