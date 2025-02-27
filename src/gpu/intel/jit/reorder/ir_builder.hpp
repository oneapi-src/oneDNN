/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_REORDER_IR_BUILDER_HPP
#define GPU_INTEL_JIT_REORDER_IR_BUILDER_HPP

#include <array>

#include "common/convolution_pd.hpp"
#include "gpu/intel/jit/ir/gemm_schedule.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/ir_builder.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/reorder/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class reorder_ir_builder_t : public ir_builder_t {
public:
    reorder_ir_builder_t(const reorder_config_t &cfg,
            const kernel_info_t &kernel_info, const primitive_attr_t *attr,
            const memory_desc_t *dst_md)
        : cfg_(cfg), kernel_info_(kernel_info), attr_(attr), dst_md_(dst_md) {
        build();
    }

    const grid_info_t &kernel_grid() const { return cfg_.kernel_grid(); }

private:
    void build() override;
    bool try_build(const pvar_tile_t &iter_tile, const pvar_tile_t &loop_tile);

    const reorder_config_t &cfg_;
    const kernel_info_t &kernel_info_;
    const primitive_attr_t *attr_;
    const memory_desc_t *dst_md_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
