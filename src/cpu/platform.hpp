/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_PLATFORM_HPP
#define CPU_PLATFORM_HPP

#include "dnnl_config.h"

#include "common/c_types_map.hpp"

// Possible architectures:
// - DNNL_X64
// - DNNL_AARCH64
// - DNNL_ARCH_GENERIC
// All macros are defined to either 0 or 1. Only one macro is defined as 1,
// and indicates the target architecture. By default the architecture is
// auto-detected, but user can override it by passing `-DDNNL_${ARCH}=1`.

#if defined(DNNL_X64) + defined(DNNL_AARCH64) + defined(DNNL_ARCH_GENERIC) == 0
#if defined(__x86_64__) || defined(_M_X64)
#define DNNL_X64 1
#elif defined(__aarch64__)
#define DNNL_AARCH64 1
#else
#define DNNL_ARCH_GENERIC 1
#endif
#endif // defined(DNNL_X64) + ... == 0

#if defined(DNNL_X64) + defined(DNNL_AARCH64) + defined(DNNL_ARCH_GENERIC) != 1
#error One and only one architecure should be defined at a time
#endif

#if !defined(DNNL_X64)
#define DNNL_X64 0
#endif
#if !defined(DNNL_AARCH64)
#define DNNL_AARCH64 0
#endif
#if !defined(DNNL_ARCH_GENERIC)
#define DNNL_ARCH_GENERIC 0
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace platform {

const char *get_isa_info();

bool DNNL_API has_data_type_support(data_type_t data_type);

unsigned get_per_core_cache_size(int level);
unsigned get_num_cores();

int get_vector_register_size();

} // namespace platform

// XXX: find a better place for these values?
enum {
    PAGE_4K = 4096,
    PAGE_2M = 2097152,
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
