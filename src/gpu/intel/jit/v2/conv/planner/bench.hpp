/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_PLANNER_BENCH_HPP
#define GPU_INTEL_JIT_V2_CONV_PLANNER_BENCH_HPP

#include "gpu/intel/jit/v2/conv/bench_data.hpp"
#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

class bench_manager_t {
public:
    bench_manager_t() : engine_(engine::kind::gpu, 0) {}
    const engine &get_engine() const { return engine_; }
    ~bench_manager_t();

private:
    engine engine_;
};

bench_data_t bench(
        const bench_manager_t &bench_mger, const kernel_desc_t &kernel_desc);

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
