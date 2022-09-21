/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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
    assert(flags <= (GLOB_STATS | USE_SCALE | USE_SHIFT));
    return flags;
}

void prb_t::generate_oscales() {
    if (attr.oscale.is_def()) return;

    assert(attr.oscale.policy == policy_t::COMMON);

    if (attr.oscale.policy == policy_t::COMMON) {
        scales = (float *)zmalloc(sizeof(float), 4);
        SAFE_V(scales != nullptr ? OK : FAIL);
        scales[0] = attr.oscale.scale;
        return;
    }
}

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    dump_global_params(s);
    settings_t def;

    bool has_default_dts = true;
    for (const auto &i_dt : prb.dt)
        has_default_dts = has_default_dts && i_dt == dnnl_f32;

    if (canonical || prb.dir != def.dir[0]) s << "--dir=" << prb.dir << " ";
    if (canonical || !has_default_dts) s << "--dt=" << prb.dt << " ";
    if (canonical || prb.tag != def.tag[0]) {
        s << "--tag=";
        if (prb.tag[1] != def.tag[0][1])
            s << prb.tag[0] << ":" << prb.tag[1] << " ";
        else
            s << prb.tag[0] << " ";
    }
    if (canonical || prb.stat_tag != def.stat_tag[0])
        s << "--stat_tag=" << prb.stat_tag << " ";
    if (canonical || prb.flags != def.flags[0])
        s << "--flags=" << flags2str(prb.flags) << " ";
    if (canonical || prb.inplace != def.inplace[0])
        s << "--inplace=" << bool2str(prb.inplace) << " ";

    s << prb.attr;
    if (canonical || prb.ctx_init != def.ctx_init[0])
        s << "--ctx-init=" << prb.ctx_init << " ";
    if (canonical || prb.ctx_exe != def.ctx_exe[0])
        s << "--ctx-exe=" << prb.ctx_exe << " ";

    s << static_cast<prb_dims_t>(prb);

    return s;
}
} // namespace lnorm
