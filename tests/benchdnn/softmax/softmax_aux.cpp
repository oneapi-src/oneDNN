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

#include "mkldnn_common.hpp"
#include "mkldnn_debug.hpp"

#include "softmax/softmax.hpp"

namespace softmax {

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    if (p.dir != FWD_D)
        s << "--dir=" << dir2str(p.dir) << " ";
    if (p.dt != mkldnn_f32)
        s << "--dt=" << dt2str(p.dt) << " ";
    if (p.tag != mkldnn_nchw)
        s << "--tag=" << fmt_tag2str(p.tag) << " ";
    if (p.axis != 1)
        s << "--axis=" << p.axis << " ";
    if (p.inplace != true)
        s << "--inplace=" << bool2str(p.inplace) << " ";

    s << p.dims;

    return s;
}

}
