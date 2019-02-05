/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_PRIMITIVE_HPP
#define CPU_PRIMITIVE_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "primitive.hpp"
#include "scratchpad.hpp"

#define CTX_IN_MEM(type, arg) static_cast<type>(ctx.input(arg))
#define CTX_OUT_MEM(type, arg) static_cast<type>(ctx.output(arg))

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_memory_t;

struct cpu_primitive_t: public primitive_t {
    cpu_primitive_t(const primitive_desc_t *pd)
        : primitive_t(pd) { }

    virtual ~cpu_primitive_t() { }

protected:
    memory_tracking::grantor_t scratchpad(const exec_ctx_t &ctx) const {
        return pd()->scratchpad_registry().grantor(
                CTX_OUT_MEM(void *, MKLDNN_ARG_SCRATCHPAD));
    }

private:
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
