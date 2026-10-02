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

#ifndef COMMON_ITTNOTIFY_HPP
#define COMMON_ITTNOTIFY_HPP

#include "c_types_map.hpp"
#include "dnnl.h"

#if defined(DNNL_ENABLE_ITT_TASKS)
#include "dnnl_debug.h"
#include "ittnotify/ittnotify.h"
#else
// Forward declaration of the ITT id type defined in ittnotify/ittnotify.h.
struct ___itt_id; // NOLINT
typedef struct ___itt_id __itt_id; // NOLINT
#endif

namespace dnnl {
namespace impl {
namespace itt {

// GCC treats using and typedef differently for enums and structs
// https://stackoverflow.com/questions/48613758
typedef enum { // NOLINT(modernize-use-using)
    __itt_task_level_none = 0,
    __itt_task_level_low,
    __itt_task_level_high
} __itt_task_level;

struct itt_task_level_t {
    int level;
};

#if defined(DNNL_ENABLE_ITT_TASKS)
__itt_id make_itt_id(const char *tname, double stamp);
#endif

// Conditional definitions below are used since dnnl_thread.hpp is included in
// test_thread.hpp and gets pulled into most (all) test sources, which are
// built without DNNL_ENABLE_ITT_TASKS.
// Strictly follow this style when adding new entry points.

// Returns `true` if requested @p level is less or equal to default or specified
// one by env variable. Even when `true`, ITT calls on the returned domain are
// no-ops unless a collector is attached at runtime.
#if defined(DNNL_ENABLE_ITT_TASKS)
bool get_itt(__itt_task_level level);
#else
static inline bool get_itt(__itt_task_level) {
    return bool();
}
#endif

#if defined(DNNL_ENABLE_ITT_TASKS)
void primitive_task_start(primitive_kind_t kind, const char *log_kind);
#else
static inline void primitive_task_start(primitive_kind_t, const char *) {}
#endif

#if defined(DNNL_ENABLE_ITT_TASKS)
void primitive_add_metadata_and_id(
        const char *pd_info, const char *log_kind, const __itt_id *task_id);
#else
static inline void primitive_add_metadata_and_id(
        const char *, const char *, const __itt_id *) {}
#endif

#if defined(DNNL_ENABLE_ITT_TASKS)
primitive_kind_t primitive_task_get_current_kind();
#else
static inline primitive_kind_t primitive_task_get_current_kind() {
    return primitive_kind_t();
}
#endif

#if defined(DNNL_ENABLE_ITT_TASKS)
void primitive_task_end(const char *log_kind);
#else
static inline void primitive_task_end(const char *) {}
#endif

#if defined(DNNL_ENABLE_ITT_TASKS)
const char *primitive_task_get_current_info();
#else
static inline const char *primitive_task_get_current_info() {
    return nullptr;
}
#endif

#if defined(DNNL_ENABLE_ITT_TASKS)
const char *primitive_task_get_current_log_kind();
#else
static inline const char *primitive_task_get_current_log_kind() {
    return nullptr;
}
#endif

#if defined(DNNL_ENABLE_ITT_TASKS)
const __itt_id *primitive_task_get_itt_id();
#else
static inline const __itt_id *primitive_task_get_itt_id() {
    return nullptr;
}
#endif
} // namespace itt
} // namespace impl
} // namespace dnnl

#endif
