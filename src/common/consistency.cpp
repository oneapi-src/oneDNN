/*******************************************************************************
* Copyright 2020 NEC Labs America
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
#include "consistency.hpp"

namespace dnnl {
namespace impl {

void Consistency::show(CondLocV const &cl) const {
#if DNNL_VERBOSE_EXTRA /* this function will never be called by operator&& */
    if (dnnl_get_verbose() >= 3) {
        printf(" %s [%s:%d] %s\n", pfx, cl.file, cl.line, cl.cond_msg);
        fflush(stdout);
    }
#endif
}

void Consistency::show(CondLocVV const &cl) const {
    if (dnnl_get_verbose() >= 3) {
        printf(" %s [%s:%d] %s\n", pfx, cl.file, cl.line, cl.cond_msg);
        fflush(stdout);
    }
}

} // namespace impl
} // namespace dnnl
// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s
