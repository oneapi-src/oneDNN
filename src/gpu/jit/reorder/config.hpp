/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "gpu/jit/ir/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Parameters for kernel generation.
class reorder_config_t : public prim_config_t {
public:
    std::string str() const override {
        std::ostringstream ss;
        ss << src_layout().user().str() << " -> " << dst_layout().user().str();
        return ss.str();
    }

    reorder_config_t(
            const exec_config_t &ec, const layout_t &src, const layout_t &dst) {
        src_layout().set_user(src);
        dst_layout().set_user(dst);
        set_exec_cfg(ec);
    }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
