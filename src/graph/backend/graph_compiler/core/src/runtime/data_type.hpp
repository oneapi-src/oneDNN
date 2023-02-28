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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DATA_TYPE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DATA_TYPE_HPP

#include <stdint.h> // uint64_t, uint32_t

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// The basic data types for scalars and pointers. The lower 8-bits represent
// the "base type". And the basic type is a pointer if the 9-th bit is 1.
// For example, `sc_data_etype::POINTER || sc_data_etype::F32` means the pointer
// type to float32. A sc_data_etype which equals to POINTER is a `void*` type.
enum class sc_data_etype : uint32_t {
    /// Undefined data type.
    UNDEF = 0,
    /// 16-bit/half-precision floating point.
    F16 = 1,
    /// non-standard 16-bit floating point with 7-bit mantissa.
    BF16 = 2,
    /// 16-bit unsigned integer.
    U16 = 3,
    /// 32-bit/single-precision floating point.
    F32 = 4,
    /// 32-bit signed integer.
    S32 = 5,
    /// 32-bit unsigned integer.
    U32 = 6,
    /// 8-bit signed integer.
    S8 = 7,
    /// 8-bit unsigned integer.
    U8 = 8,
    /// data type used for indexing.
    INDEX = 9,
    /// generic_val type, a union type for all supported scalar types
    GENERIC = 10,
    /// boolean
    BOOLEAN = 11,
    /// void type
    VOID_T = 12,
    /// the max enum value + 1
    MAX_VALUE = 13,
    /// general pointer type, also used as a pointer bit mask
    /// void* type. The opaque pointer type. Any pointers (including tensor /
    /// tensor ptr) can be auto-cast to a pointer value. But casting back
    /// is not allowed
    POINTER = 0x100,
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
