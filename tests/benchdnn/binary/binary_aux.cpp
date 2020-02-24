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

#include "dnnl_debug.hpp"

#include "binary/binary.hpp"

namespace binary {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(ADD);
    CASE(MUL);
    CASE(MAX);
    CASE(MIN);
#undef CASE
    assert(!"unknown algorithm");
    return ADD;
}

const char *alg2str(alg_t alg) {
    if (alg == ADD) return "ADD";
    if (alg == MUL) return "MUL";
    if (alg == MAX) return "MAX";
    if (alg == MIN) return "MIN";
    assert(!"unknown algorithm");
    return "unknown algorithm";
}

dnnl_alg_kind_t alg2alg_kind(alg_t alg) {
    if (alg == ADD) return dnnl_binary_add;
    if (alg == MUL) return dnnl_binary_mul;
    if (alg == MAX) return dnnl_binary_max;
    if (alg == MIN) return dnnl_binary_min;
    assert(!"unknown algorithm");
    return dnnl_alg_kind_undef;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    using ::operator<<;

    dump_global_params(s);

    if (canonical || !(p.sdt[0] == dnnl_f32 && p.sdt[1] == dnnl_f32))
        s << "--sdt=" << p.sdt << " ";
    if (canonical || p.ddt != dnnl_f32) s << "--ddt=" << dt2str(p.ddt) << " ";
    if (canonical || !(p.stag[0] == tag::nchw && p.stag[1] == tag::nchw))
        s << "--stag=" << p.stag << " ";
    if (canonical || p.alg != ADD) s << "--alg=" << alg2str(p.alg) << " ";
    if (canonical || p.inplace != true)
        s << "--inplace=" << bool2str(p.inplace) << " ";
    if (canonical || !p.attr.is_def()) s << "--attr=\"" << p.attr << "\" ";

    s << p.sdims;

    return s;
}

} // namespace binary
