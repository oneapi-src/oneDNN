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
    settings_t def;

    s << "--sdt=" << cfg2dt(p.conf_in) << " ";
    s << "--ddt=" << cfg2dt(p.conf_out) << " ";
    s << "--stag=" << p.reorder.tag_in << " ";
    s << "--dtag=" << p.reorder.tag_out << " ";

    if (canonical || p.alg != def.alg[0])
        s << "--alg=" << alg2str(p.alg) << " ";
    if (canonical || p.oflag != def.oflag[0])
        s << "--oflag=" << flag2str(p.oflag) << " ";
    if (canonical || p.runtime_dim_mask != def.runtime_dim_mask[0])
        s << "--runtime-dim-mask=" << p.runtime_dim_mask << " ";
    if (canonical || !p.attr.is_def()) s << "--attr=\"" << p.attr << "\" ";

    s << p.reorder.dims;

    return s;
}

bool prb_t::maybe_skip_nvidia() const {
    if (this->attr.oscale.policy != attr_t::scale_t::COMMON) return true;
    if (this->attr.oscale.runtime) return true;
    auto extra_supported_plain_tags = [](dnnl_format_tag_t tag) {
        return tag == dnnl_acbde || tag == dnnl_acbdef || tag == dnnl_ba
                || tag == dnnl_bac || tag == dnnl_bacd || tag == dnnl_bacde
                || tag == dnnl_bca || tag == dnnl_bcda || tag == dnnl_cba
                || tag == dnnl_cdba || tag == dnnl_cdeba || tag == dnnl_oiw
                || tag == dnnl_oihw || tag == dnnl_oidhw || tag == dnnl_goiw
                || tag == dnnl_goihw || tag == dnnl_goidhw;
    };
    auto extra_supported_blocking_tags
            = [](dnnl_format_tag_t tag) { return tag == dnnl_ABc16a16b; };
    for (auto i = 0; i < 2; i++) {
        // Check data type
        auto dt = i ? this->conf_in->dt : this->conf_out->dt;
        bool dt_ok = dt == dnnl_s8 || dt == dnnl_f16 || dt == dnnl_f32;
        // if (i && (dt == dnnl_s32 || dt == dnnl_u8)) dt_ok = true;
        if (!dt_ok) return true;

        // Check for supported plain tags
        auto tag = convert_tag(
                i ? this->reorder.tag_in : this->reorder.tag_out, this->ndims);
        auto plain_tag_ok = cudnn_supported_tag_plain(tag)
                || extra_supported_plain_tags(tag);

        // dst tag is allowed to be undef
        if (i && tag == dnnl_format_tag_undef) plain_tag_ok = true;

        // If using unconventional formats then tag must be undef
        if (!IMPLICATION((dt == dnnl_s32 || dt == dnnl_u8),
                    tag == dnnl_format_tag_undef))
            return true;

        // Check for supported blocking tags
        auto block_tag_ok = cudnn_supported_tag_blocking(tag)
                || extra_supported_blocking_tags(tag);
        if (!(plain_tag_ok || block_tag_ok)) return true;

        // If blocking check that data type is s8
        auto s8_tag_ok = IMPLICATION(block_tag_ok, dt == dnnl_s8);
        if (!s8_tag_ok) return true;

        // If using blocking check that channel dimension is divisible by 4
        auto channel_div_ok
                = IMPLICATION(block_tag_ok, this->reorder.dims[1] % 4 == 0);
        if (!channel_div_ok) return true;
    }
    return false;
}

} // namespace reorder
