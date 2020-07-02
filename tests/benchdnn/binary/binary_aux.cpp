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
    settings_t def;

    if (canonical || p.sdt != def.sdt[0]) s << "--sdt=" << p.sdt << " ";
    if (canonical || p.ddt != def.ddt[0]) s << "--ddt=" << p.ddt << " ";
    if (canonical || p.stag != def.stag[0]) s << "--stag=" << p.stag << " ";
    if (canonical || p.alg != def.alg[0])
        s << "--alg=" << alg2str(p.alg) << " ";
    if (canonical || p.inplace != def.inplace[0])
        s << "--inplace=" << bool2str(p.inplace) << " ";
    if (canonical || !p.attr.is_def()) s << "--attr=\"" << p.attr << "\" ";

    s << p.sdims;

    return s;
}

bool prb_t::maybe_skip_nvidia() const {
    if (!this->attr.post_ops.is_def()) return true;
    bool dt_ok = ddt == dnnl_s8 || ddt == dnnl_f16 || ddt == dnnl_f32;
    if (!dt_ok) return true;
    auto extra_supported_plain_tags = [](dnnl_format_tag_t tag) {
        return tag == dnnl_acbde || tag == dnnl_acbdef || tag == dnnl_ba
                || tag == dnnl_bac || tag == dnnl_bacd || tag == dnnl_bacde
                || tag == dnnl_bca || tag == dnnl_bcda || tag == dnnl_cba
                || tag == dnnl_cdba || tag == dnnl_cdeba || tag == dnnl_oiw
                || tag == dnnl_oihw || tag == dnnl_oidhw || tag == dnnl_goiw
                || tag == dnnl_goihw || tag == dnnl_goidhw;
    };
    for (auto i = 0; i < this->n_inputs(); i++) {
        // Check data type
        auto dt = sdt[i];
        bool dt_ok = dt == dnnl_s8 || dt == dnnl_f16 || dt == dnnl_f32;
        if (!dt_ok) return true;

        // Check for supported plain tags
        auto tag = convert_tag(this->stag[i], this->ndims[i]);
        auto plain_tag_ok = cudnn_supported_tag_plain(tag)
                || extra_supported_plain_tags(tag);
        if (!plain_tag_ok) return true;
    }
    return false;
}

} // namespace binary
