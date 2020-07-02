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
#include "sum/sum.hpp"

namespace sum {

std::ostream &operator<<(std::ostream &s, const std::vector<float> &scales) {
    bool has_single_scale = true;
    for (size_t d = 0; d < scales.size() - 1; ++d)
        has_single_scale = has_single_scale && scales[d] == scales[d + 1];

    s << scales[0];
    if (!has_single_scale)
        for (size_t d = 1; d < scales.size(); ++d)
            s << ":" << scales[d];
    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    using ::operator<<;
    using sum::operator<<;

    dump_global_params(s);
    settings_t def;

    bool has_default_tags = true;
    for (const auto &i_stag : p.stag)
        has_default_tags = has_default_tags && i_stag == tag::abx;

    if (canonical || p.sdt != def.sdt[0]) s << "--sdt=" << p.sdt << " ";
    if (canonical || p.ddt != def.ddt[0]) s << "--ddt=" << p.ddt << " ";
    if (canonical || !has_default_tags) s << "--stag=" << p.stag << " ";
    if (canonical || p.dtag != def.dtag[0]) s << "--dtag=" << p.dtag << " ";
    s << "--scales=" << p.scales << " ";

    s << p.dims;

    return s;
}
bool prb_t::maybe_skip_nvidia() const {
    for (auto i = -1; i < this->n_inputs(); i++) {
        // Check data type
        auto dt = i == -1 ? ddt : sdt[i];
        bool dt_ok = dt == dnnl_s8 || dt == dnnl_f16 || dt == dnnl_f32;
        // if (i == -1 && (dt == dnnl_s32 || dt == dnnl_u8)) dt_ok = true;
        if (!dt_ok) return true;

        // Check for supported plain tags
        auto tag = convert_tag(i == -1 ? dtag : this->stag[i], this->ndims);
        auto plain_tag_ok = cudnn_supported_tag_plain(tag);

        // dst tag is allowed to be undef
        if (i == -1 && tag == dnnl_format_tag_undef) plain_tag_ok = true;

        // If using unconventional formats then tag must be undef
        if (!IMPLICATION((dt == dnnl_s32 || dt == dnnl_u8),
                    tag == dnnl_format_tag_undef))
            return true;

        // Check for supported blocking tags
        auto block_tag_ok = cudnn_supported_tag_blocking(tag);
        if (!(plain_tag_ok || block_tag_ok)) return true;

        // If blocking check that data type is s8
        auto s8_tag_ok = IMPLICATION(block_tag_ok, dt == dnnl_s8);
        if (!s8_tag_ok) return true;

        // If using blocking check that channel dimension is divisible by 4
        auto channel_div_ok = IMPLICATION(block_tag_ok, this->dims[1] % 4 == 0);
        if (!channel_div_ok) return true;
    }
    return false;
}

} // namespace sum
