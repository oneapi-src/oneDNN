/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#include <compiler/jit/xbyak/x86_64/native_types.hpp>

#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

const cpu_data_type_table &get_cpu_data_types() {
    using avk = abi_value_kind;

    // clang-format off
    static const cpu_data_type_table local_cpu_data_types(std::vector<cpu_data_type_table::row> {                                    // NOLINT
    //  +-----------------------------+------+-----------+-----------+-------------+-----------+-------------+-------------------+   // NOLINT
    //  | type                        | size | natural   | strictest | abi precall | abi stack | local value | ABI value         |   // NOLINT
    //  |                             |      | alignment | alignment | stack       | slot size | stack slot  | kind              |   // NOLINT
    //  |                             |      |           |           | alignment   |           | size        |                   |   // NOLINT
    //  +-----------------------------+------+-----------+-----------+-------------+-----------+-------------+-------------------+   // NOLINT
        { cpu_data_type::uint_8       ,    1 ,         1 ,         1 ,          16 ,         8 ,           8 , avk::INTEGER      },  // NOLINT
        { cpu_data_type::uint_8_x8    ,   16 ,        16 ,        16 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_8_x16   ,   16 ,        16 ,        16 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_8_x32   ,   32 ,        32 ,        32 ,          32 ,        32 ,          32 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_8_x64   ,   64 ,        64 ,        64 ,          64 ,        64 ,          64 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::sint_8       ,    1 ,         1 ,         1 ,          16 ,         8 ,           8 , avk::INTEGER      },  // NOLINT
        { cpu_data_type::sint_8_x8    ,   16 ,        16 ,        16 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::sint_8_x16   ,   16 ,        16 ,        16 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::sint_8_x32   ,   32 ,        32 ,        32 ,          32 ,        32 ,          32 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::sint_8_x64   ,   64 ,        64 ,        64 ,          64 ,        64 ,          64 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_16      ,    2 ,         2 ,         2 ,          16 ,         8 ,           8 , avk::INTEGER      },  // NOLINT
        { cpu_data_type::uint_16_x4   ,    8 ,         8 ,         8 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_16_x8   ,   16 ,        16 ,        16 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_16_x16  ,   32 ,        32 ,        32 ,          32 ,        32 ,          32 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_16_x32  ,   64 ,        64 ,        64 ,          64 ,        64 ,          64 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_32      ,    4 ,         4 ,         1 ,          16 ,         8 ,           8 , avk::INTEGER      },  // NOLINT
        { cpu_data_type::uint_32_x2   ,    8 ,         8 ,         8 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_32_x4   ,   16 ,        16 ,        16 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_32_x8   ,   32 ,        32 ,        32 ,          32 ,        32 ,          32 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_32_x16  ,   64 ,        64 ,        64 ,          64 ,        64 ,          64 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::sint_32      ,    4 ,         4 ,         1 ,          16 ,         8 ,           8 , avk::INTEGER      },  // NOLINT
        { cpu_data_type::sint_32_x2   ,    8 ,         8 ,         8 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::sint_32_x4   ,   16 ,        16 ,        16 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::sint_32_x8   ,   32 ,        32 ,        32 ,          32 ,        32 ,          32 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::sint_32_x16  ,   64 ,        64 ,        64 ,          64 ,        64 ,          64 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_64      ,    8 ,         8 ,         1 ,          16 ,         8 ,           8 , avk::INTEGER      },  // NOLINT
        { cpu_data_type::uint_64_x2   ,   16 ,        16 ,        16 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_64_x4   ,   32 ,        32 ,        32 ,          32 ,        32 ,          32 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::uint_64_x8   ,   64 ,        64 ,        64 ,          64 ,        64 ,          64 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::float_16     ,    2 ,         2 ,         2 ,          16 ,         8 ,           8 , avk::SSE          },  // NOLINT
        { cpu_data_type::float_16_x4   ,   8 ,         8 ,         8 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::float_16_x8   ,  16 ,        16 ,        16 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::float_16_x16  ,  32 ,        32 ,        32 ,          32 ,        32 ,          32 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::float_16_x32  ,  64 ,        64 ,        64 ,          64 ,        64 ,          64 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::float_32     ,    4 ,         4 ,         1 ,          16 ,         8 ,           8 , avk::SSE          },  // NOLINT
        { cpu_data_type::float_32_x2  ,    8 ,         8 ,         8 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::float_32_x4  ,   16 ,        16 ,        16 ,          16 ,        16 ,          16 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::float_32_x8  ,   32 ,        32 ,        32 ,          32 ,        32 ,          32 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::float_32_x16 ,   64 ,        64 ,        64 ,          64 ,        64 ,          64 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::mask_x4      ,    1 ,         1 ,         1 ,          16 ,         8 ,           8 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::mask_x8      ,    1 ,         1 ,         1 ,          16 ,         8 ,           8 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::mask_x16     ,    2 ,         2 ,         1 ,          16 ,         8 ,           8 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::mask_x32     ,    4 ,         4 ,         1 ,          16 ,         8 ,           8 , avk::SSEUPx15_SSE },  // NOLINT
        { cpu_data_type::mask_x64     ,    8 ,         8 ,         1 ,          16 ,         8 ,           8 , avk::SSEUPx15_SSE },  // NOLINT
    //  +-----------------------------+------+-----------+-----------+-------------+-----------+-------------+-------------------+   // NOLINT
    });

    // clang-format on
    return local_cpu_data_types;
}

std::ostream &operator<<(std::ostream &os, const cpu_data_type t) {
    switch (t) {
#define HANDLE_CASE(X) \
    case xbyak::x86_64::cpu_data_type::X: \
        os << "xbyak::x86_64::cpu_data_type::" #X; \
        break;

        HANDLE_CASE(uint_8)
        HANDLE_CASE(uint_8_x8)
        HANDLE_CASE(uint_8_x16)
        HANDLE_CASE(uint_8_x32)
        HANDLE_CASE(uint_8_x64)
        HANDLE_CASE(sint_8)
        HANDLE_CASE(sint_8_x8)
        HANDLE_CASE(sint_8_x16)
        HANDLE_CASE(sint_8_x32)
        HANDLE_CASE(sint_8_x64)
        HANDLE_CASE(uint_16)
        HANDLE_CASE(uint_16_x4)
        HANDLE_CASE(uint_16_x8)
        HANDLE_CASE(uint_16_x16)
        HANDLE_CASE(uint_16_x32)
        HANDLE_CASE(uint_32)
        HANDLE_CASE(uint_32_x2)
        HANDLE_CASE(uint_32_x4)
        HANDLE_CASE(uint_32_x8)
        HANDLE_CASE(uint_32_x16)
        HANDLE_CASE(sint_32)
        HANDLE_CASE(sint_32_x2)
        HANDLE_CASE(sint_32_x4)
        HANDLE_CASE(sint_32_x8)
        HANDLE_CASE(sint_32_x16)
        HANDLE_CASE(uint_64)
        HANDLE_CASE(float_16)
        HANDLE_CASE(float_16_x4)
        HANDLE_CASE(float_16_x8)
        HANDLE_CASE(float_16_x16)
        HANDLE_CASE(float_32)
        HANDLE_CASE(float_32_x2)
        HANDLE_CASE(float_32_x4)
        HANDLE_CASE(float_32_x8)
        HANDLE_CASE(float_32_x16)
        HANDLE_CASE(mask_x4)
        HANDLE_CASE(mask_x8)
        HANDLE_CASE(mask_x16)
        HANDLE_CASE(mask_x32)
        HANDLE_CASE(mask_x64)
        HANDLE_CASE(void_t)
#undef HANDLE_CASE
        default: os << "(unrecognized cpu_data_type value)"; break;
    }
    return os;
}

cpu_data_type_table::row::row(cpu_data_type type, size_t size_in_bytes,
        size_t cpu_natural_alignment, size_t cpu_strictest_alignment,
        size_t abi_precall_stack_alignment, size_t abi_stack_slot_size,
        size_t local_value_stack_slot_size, abi_value_kind abi_initial_val_kind)
    : type_(type)
    , size_in_bytes_(size_in_bytes)
    , cpu_natural_alignment_(cpu_natural_alignment)
    , cpu_strictest_alignment_(cpu_strictest_alignment)
    , abi_precall_stack_alignment_(abi_precall_stack_alignment)
    , abi_stack_slot_size_(abi_stack_slot_size)
    , local_value_stack_slot_size_(local_value_stack_slot_size)
    , abi_initial_val_kind_(abi_initial_val_kind) {}

cpu_data_type_table::cpu_data_type_table(const std::vector<row> &content)
    : content_(content) {}

const cpu_data_type_table::row &cpu_data_type_table::lookup(
        cpu_data_type t) const {
    for (const auto &r : content_) {
        if (r.type_ == t) { return r; }
    }

    COMPILE_ASSERT(false, "No matching row for " << t);
}

size_t get_local_value_stack_slot_size(cpu_data_type t) {
    return get_cpu_data_types().lookup(t).local_value_stack_slot_size_;
}

size_t get_size_in_bytes(cpu_data_type t) {
    return get_cpu_data_types().lookup(t).size_in_bytes_;
}

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
