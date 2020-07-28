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

#include "concat/concat.hpp"
#include "dnnl_debug.hpp"

namespace concat {

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    using ::operator<<;

    dump_global_params(s);
    settings_t def;

    bool has_default_tags = true;
    for (const auto &i_stag : p.stag)
        has_default_tags = has_default_tags && i_stag == tag::abx;

    if (canonical || p.sdt != def.sdt[0]) s << "--sdt=" << p.sdt << " ";
    if (canonical || (p.dtag != def.dtag[0] && p.ddt != def.ddt[0]))
        s << "--ddt=" << p.ddt << " ";
    if (canonical || !has_default_tags) s << "--stag=" << p.stag << " ";
    if (canonical || p.dtag != def.dtag[0]) s << "--dtag=" << p.dtag << " ";
    if (canonical || p.axis != def.axis[0]) s << "--axis=" << p.axis << " ";

    s << p.attr;
    s << p.sdims;

    return s;
}

} // namespace concat
