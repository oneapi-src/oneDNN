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

#include <memory>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "memory_storage.hpp"
#include "memory_tracking.hpp"
#include "primitive.hpp"
#include "primitive_exec_types.hpp"
#include "scratchpad.hpp"

#include <type_traits>

#define ARG_TYPE(t) \
    typename std::remove_cv<typename std::remove_pointer<t>::type>::type

#define CTX_IN_MEM(type, arg) \
    static_cast<const ARG_TYPE(type) *>(ctx.host_ptr(arg))

#define CTX_OUT_MEM(type, arg) static_cast<ARG_TYPE(type) *>(ctx.host_ptr(arg))

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_primitive_t: public primitive_t {
    cpu_primitive_t(const primitive_desc_t *pd,
            bool use_global_scratchpad = false)
        : primitive_t(pd)
    {
        const size_t scratchpad_size =
            this->pd()->scratchpad_size(scratchpad_mode::library);

        if (scratchpad_size) {
            if (use_global_scratchpad) {
                auto *scratchpad_ptr
                        = create_scratchpad(engine(), scratchpad_size);
                global_scratchpad_.reset(scratchpad_ptr);
            } else {
                auto *mem_storage_ptr = create_scratchpad_memory_storage(
                        engine(), scratchpad_size, 64);
                scratchpad_buffer_.reset(mem_storage_ptr);
            }
        }
    }

    const memory_storage_t *scratchpad_memory_storage(
            const exec_ctx_t &ctx) const {
        if (pd()->attr()->scratchpad_mode_ == scratchpad_mode::user)
            return ctx.output(MKLDNN_ARG_SCRATCHPAD)->memory_storage();

        return global_scratchpad_ ? global_scratchpad_->get_memory_storage()
                                  : scratchpad_buffer_.get();
    }

    memory_tracking::grantor_t scratchpad(const exec_ctx_t &ctx) const {
        return pd()->scratchpad_registry().grantor(
                scratchpad_memory_storage(ctx), ctx);
    }

private:
    std::unique_ptr<memory_storage_t> scratchpad_buffer_;
    std::unique_ptr<scratchpad_t> global_scratchpad_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
