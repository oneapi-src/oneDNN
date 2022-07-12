/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_JIT_REORDER_CONFIG_HPP
#define GPU_JIT_REORDER_CONFIG_HPP

#include <iostream>
#include <sstream>

#include "common/memory_desc_wrapper.hpp"
#include "common/reorder_pd.hpp"
#include "gpu/jit/ir/hw_config.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Parameters for kernel generation.
struct reorder_config_t {
    std::string str() const {
        std::ostringstream ss;
        ss << src_layout.str() << " -> " << dst_layout.str();
        return ss.str();
    }

    layout_t src_layout;
    layout_t dst_layout;

    exec_config_t exec_cfg;

    reorder_config_t(engine_t *engine, const memory_desc_t *src_md,
            const memory_desc_t *dst_md)
        : src_layout(memory_desc_wrapper(src_md), /*do_normalize=*/false)
        , dst_layout(memory_desc_wrapper(dst_md), /*do_normalize=*/false)
        , exec_cfg(engine) {}
};

inline std::ostream &operator<<(
        std::ostream &out, const reorder_config_t &cfg) {
    out << cfg.str();
    return out;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
