/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include <sstream>

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"

#include "prelu/prelu.hpp"

namespace prelu {

std::string prb_t::set_repro_line() {
    std::stringstream s;
    using ::operator<<;

    dump_global_params(s);
    settings_t def;

    if (canonical || dir != def.dir[0]) s << "--dir=" << dir << " ";
    if (canonical || sdt != def.sdt[0]) s << "--sdt=" << sdt << " ";
    if (canonical || stag != def.stag[0]) s << "--stag=" << stag << " ";

    s << attr;
    if (canonical || ctx_init != def.ctx_init[0])
        s << "--ctx-init=" << ctx_init << " ";
    if (canonical || ctx_exe != def.ctx_exe[0])
        s << "--ctx-exe=" << ctx_exe << " ";

    s << static_cast<prb_vdims_t>(*this);

    return s.str();
}

} // namespace prelu
