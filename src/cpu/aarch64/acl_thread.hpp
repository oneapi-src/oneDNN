/*******************************************************************************
* Copyright 2022 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_THREAD_HPP
#define CPU_AARCH64_ACL_THREAD_HPP

#include "common/dnnl_thread.hpp"

#include "arm_compute/runtime/Scheduler.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_thread_utils {

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
void acl_thread_bind();
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
// Retrieve threadpool size during primitive execution and set ThreadpoolScheduler num_threads
void acl_set_custom_scheduler();
void acl_set_threadpool_num_threads();
#endif

} // namespace acl_thread_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_THREAD_HPP