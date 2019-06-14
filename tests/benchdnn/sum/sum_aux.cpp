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

#include "mkldnn_debug.hpp"
#include "sum/sum.hpp"

namespace sum {

std::ostream &operator<<(std::ostream &s, const std::vector<float> &scales) {
    s << scales[0];
    for (int i = 1; i < (int)scales.size(); ++i)
        s << ":" << scales[i];
    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    if (!(p.n_inputs() == 2 && p.idt[0] == mkldnn_f32 && p.idt[1] == mkldnn_f32))
        s << "--idt=" << p.idt << " ";
    if (p.odt != mkldnn_f32)
        s << "--odt=" << dt2str(p.odt) << " ";
    if (!(p.n_inputs() == 2 && p.itag[0] == mkldnn_nchw
                && p.itag[1] == mkldnn_nchw))
        s << "--itag=" << p.itag << " ";
    if (p.otag != mkldnn_format_tag_undef)
        s << "--otag=" << fmt_tag2str(p.otag) << " ";
    s << "--scales=" << p.scales << " ";

    s << p.dims;

    return s;
}

}
