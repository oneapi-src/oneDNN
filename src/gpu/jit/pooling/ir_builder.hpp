/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_JIT_POOLING_IR_BUILDER_HPP
#define GPU_JIT_POOLING_IR_BUILDER_HPP

#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/ir_builder.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/pooling/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class pooling_ir_builder_t : public ir_builder_t {
public:
    pooling_ir_builder_t(pooling_config_t &cfg, const kernel_info_t &ki,
            const primitive_desc_t &pd)
        : ir_builder_t(ki) {
        while ((stmt_ = try_build(*this, kernel_info_, cfg, pd)).is_empty()) {
            ir_warning() << "loop too large: cut and retry!" << std::endl;
            const bool cut_ok = cfg.cut();
            if (!cut_ok) ir_error_not_expected() << "minimal loop too large!";
        }
    }

private:
    void build() override {}
    static stmt_t try_build(pooling_ir_builder_t &pb, const kernel_info_t &ki,
            const pooling_config_t &cfg, const primitive_desc_t &pd);
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
