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

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"

#include "softmax/softmax.hpp"

namespace softmax {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(SOFTMAX);
    CASE(softmax_accurate);
    CASE(LOGSOFTMAX);
    CASE(softmax_log);
#undef CASE
    assert(!"unknown algorithm");
    return UNDEF;
}

const char *alg2str(alg_t alg) {
    if (alg == SOFTMAX) return "SOFTMAX";
    if (alg == LOGSOFTMAX) return "LOGSOFTMAX";
    assert(!"unknown algorithm");
    return "UNDEF";
}

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    dump_global_params(s);
    settings_t def;

    if (canonical || prb.dir != def.dir[0]) s << "--dir=" << prb.dir << " ";
    if (canonical || prb.sdt != def.sdt[0]) s << "--sdt=" << prb.sdt << " ";
    if (canonical || prb.ddt != def.ddt[0]) s << "--ddt=" << prb.ddt << " ";
    if (canonical || prb.stag != def.stag[0]) s << "--stag=" << prb.stag << " ";
    if (canonical || prb.dtag != def.dtag[0]) s << "--dtag=" << prb.dtag << " ";
    if (canonical || prb.alg != def.alg[0])
        s << "--alg=" << alg2str(prb.alg) << " ";
    if (canonical || prb.axis != def.axis[0]) s << "--axis=" << prb.axis << " ";
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

} // namespace softmax
