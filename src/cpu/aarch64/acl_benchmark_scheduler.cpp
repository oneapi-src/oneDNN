/*******************************************************************************
* Copyright 2023 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_benchmark_scheduler.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
using namespace arm_compute;

BenchmarkScheduler::BenchmarkScheduler(IScheduler &real_scheduler)
    : _real_scheduler(real_scheduler) {}

BenchmarkScheduler::~BenchmarkScheduler() = default;

void BenchmarkScheduler::set_num_threads(unsigned int num_threads) {
    _real_scheduler.set_num_threads(num_threads);
}

void BenchmarkScheduler::set_num_threads_with_affinity(
        unsigned int num_threads, BindFunc func) {
    _real_scheduler.set_num_threads_with_affinity(num_threads, func);
}

unsigned int BenchmarkScheduler::num_threads() const {
    return _real_scheduler.num_threads();
}

void BenchmarkScheduler::schedule(ICPPKernel *kernel, const Hints &hints) {
    double start_ms = get_msec();
    _real_scheduler.schedule(kernel, hints);
    double duration_ms = get_msec() - start_ms;
    const char *name = kernel->name();
    VPROF(start_ms, primitive, exec, VERBOSE_external, name, duration_ms);
}

void BenchmarkScheduler::schedule_op(ICPPKernel *kernel, const Hints &hints,
        const Window &window, ITensorPack &tensors) {
    double start_ms = get_msec();
    _real_scheduler.schedule_op(kernel, hints, window, tensors);
    double duration_ms = get_msec() - start_ms;
    const char *name = kernel->name();
    VPROF(start_ms, primitive, exec, VERBOSE_external, name, duration_ms);
}

void BenchmarkScheduler::run_tagged_workloads(
        std::vector<Workload> &workloads, const char *tag) {
    double start_ms = get_msec();
    _real_scheduler.run_tagged_workloads(workloads, tag);
    double duration_ms = get_msec() - start_ms;
    const char *name = tag != nullptr ? tag : "Unknown";
    VPROF(start_ms, primitive, exec, VERBOSE_external, name, duration_ms);
}

void BenchmarkScheduler::run_workloads(std::vector<Workload> &workloads) {
    ARM_COMPUTE_UNUSED(workloads);
    ARM_COMPUTE_ERROR("Can't be reached");
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl