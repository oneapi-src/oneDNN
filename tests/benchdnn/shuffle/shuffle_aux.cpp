/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#include "shuffle/shuffle.hpp"

namespace shuffle {

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    if (canonical || p.dir != FWD_D) s << "--dir=" << dir2str(p.dir) << " ";
    if (canonical || p.dt != dnnl_f32) s << "--dt=" << dt2str(p.dt) << " ";
    if (canonical || p.tag != dnnl_nchw)
        s << "--tag=" << fmt_tag2str(p.tag) << " ";
    if (canonical || p.group != 1) s << "--group=" << p.group << " ";
    if (canonical || p.axis != 1) s << "--axis=" << p.axis << " ";

    s << p.dims;

    return s;
}

} // namespace shuffle
