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

#include <compiler/jit/xbyak/x86_64/type_mapping.hpp>

#include <runtime/generic_val.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

cpu_data_type get_cpu_data_type(sc_data_type_t t) {
    const sc_data_etype e = t.type_code_;

    if (t.is_tile()) {
        // cpu_data_type related operations for tile dtype are not supported
        return cpu_data_type::void_t;
    }

    if (t.lanes_ == 1) {
        if (etypes::is_pointer(e)) { return cpu_data_type::uint_64; }

        switch (e) {
            case sc_data_etype::BOOLEAN: return cpu_data_type::uint_8;
            case sc_data_etype::U8: return cpu_data_type::uint_8;
            case sc_data_etype::S8: return cpu_data_type::sint_8;
            case sc_data_etype::U16: return cpu_data_type::uint_16;
            case sc_data_etype::U32: return cpu_data_type::uint_32;
            case sc_data_etype::S32: return cpu_data_type::sint_32;
            case sc_data_etype::BF16: return cpu_data_type::uint_16;
            case sc_data_etype::F16: return cpu_data_type::float_16;
            case sc_data_etype::F32: return cpu_data_type::float_32;
            case sc_data_etype::INDEX: return cpu_data_type::uint_64;

            // We use this lowering simply because it has the same size
            // as a generic_val union.
            // Which asm operations we actually perform on values of this
            // type will continue to be guided by the IR, which is fully
            // aware that this value holds a generic_val.
            case sc_data_etype::GENERIC: return cpu_data_type::uint_64;
            case sc_data_etype::VOID_T: return cpu_data_type::void_t;

            default: // to prevent compiler warnings
                break;
        }
    } else if (t.lanes_ == 2) {
        switch (e) {
            case sc_data_etype::S32: return cpu_data_type::sint_32_x2;
            case sc_data_etype::U32: return cpu_data_type::uint_32_x2;
            case sc_data_etype::F32: return cpu_data_type::float_32_x2;
            case sc_data_etype::INDEX: return cpu_data_type::uint_64_x2;
            default: // to prevent compiler warnings
                break;
        }
    } else if (t.lanes_ == 4) {
        switch (e) {
            case sc_data_etype::BOOLEAN: return cpu_data_type::mask_x4;
            case sc_data_etype::BF16: return cpu_data_type::uint_16_x4;
            case sc_data_etype::U16: return cpu_data_type::uint_16_x4;
            case sc_data_etype::U32: return cpu_data_type::uint_32_x4;
            case sc_data_etype::S32: return cpu_data_type::sint_32_x4;
            case sc_data_etype::F16: return cpu_data_type::float_16_x4;
            case sc_data_etype::F32: return cpu_data_type::float_32_x4;
            case sc_data_etype::INDEX: return cpu_data_type::uint_64_x4;
            default: // to prevent compiler warnings
                break;
        }
    } else if (t.lanes_ == 8) {
        switch (e) {
            case sc_data_etype::BOOLEAN: return cpu_data_type::mask_x8;
            case sc_data_etype::U8: return cpu_data_type::uint_8_x8;
            case sc_data_etype::S8: return cpu_data_type::sint_8_x8;
            case sc_data_etype::BF16: return cpu_data_type::uint_16_x8;
            case sc_data_etype::U16: return cpu_data_type::uint_16_x8;
            case sc_data_etype::U32: return cpu_data_type::uint_32_x8;
            case sc_data_etype::S32: return cpu_data_type::sint_32_x8;
            case sc_data_etype::F16: return cpu_data_type::float_16_x8;
            case sc_data_etype::F32: return cpu_data_type::float_32_x8;
            case sc_data_etype::INDEX: return cpu_data_type::uint_64_x8;
            default: // to prevent compiler warnings
                break;
        }
    } else if (t.lanes_ == 16) {
        switch (e) {
            case sc_data_etype::BOOLEAN: return cpu_data_type::mask_x16;
            case sc_data_etype::U8: return cpu_data_type::uint_8_x16;
            case sc_data_etype::S8: return cpu_data_type::sint_8_x16;
            case sc_data_etype::U16: return cpu_data_type::uint_16_x16;
            case sc_data_etype::U32: return cpu_data_type::uint_32_x16;
            case sc_data_etype::S32: return cpu_data_type::sint_32_x16;
            case sc_data_etype::BF16: return cpu_data_type::uint_16_x16;
            case sc_data_etype::F16: return cpu_data_type::float_16_x16;
            case sc_data_etype::F32: return cpu_data_type::float_32_x16;
            default: // to prevent compiler warnings
                break;
        }
    } else if (t.lanes_ == 32) {
        switch (e) {
            case sc_data_etype::BOOLEAN: return cpu_data_type::mask_x32;
            case sc_data_etype::BF16: return cpu_data_type::uint_16_x32;
            case sc_data_etype::F16: return cpu_data_type::float_16_x32;
            case sc_data_etype::U16: return cpu_data_type::uint_16_x32;
            case sc_data_etype::U8: return cpu_data_type::uint_8_x32;
            case sc_data_etype::S8: return cpu_data_type::sint_8_x32;
            default: // to prevent compiler warnings
                break;
        }
    } else if (t.lanes_ == 64) {
        switch (e) {
            case sc_data_etype::BOOLEAN: return cpu_data_type::mask_x64;
            case sc_data_etype::U8: return cpu_data_type::uint_8_x64;
            case sc_data_etype::S8: return cpu_data_type::sint_8_x64;
            default: // to prevent compiler warnings
                break;
        }
    }

    COMPILE_ASSERT(false, "Unhandled type: " << t);
}

const cpu_data_type_table::row &get_cpu_data_type_row(sc_data_type_t t) {
    return get_cpu_data_types().lookup(get_cpu_data_type(t));
}

void get_stack_allocated_tensor_buffer_lowering_info(
        sc_data_type_t element_type, size_t num_elements, size_t &buffer_size) {
    COMPILE_ASSERT(num_elements > 0, "cannot allocate zero-element tensors");

    const cpu_data_type_table::row &r = get_cpu_data_type_row(element_type);

    // TODO(xxx): We're supposed to ensure that *every* element of the tensor
    // buffer will meet the "natural" alignment standard. We might want an
    // assert to verify that, instead of just assuming that no intra-element
    // padding is needed to get that outcome.

    buffer_size = r.size_in_bytes_ * num_elements;

    if (const size_t excess = buffer_size % 8) { buffer_size += (8 - excess); }
}

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
