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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_NATIVE_TYPES_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_NATIVE_TYPES_HPP

/**
 * @file
 * @brief Defines some data types that are native to various x86-64 ISAs.
 *
 * x86-64 types can be defined according using a number of reasonable
 * taxonomies. The taxonomy we define here is guided by the following goals:
 *
 * - Standard. The mapping between our type taxonomy, and the taxonomies used
 *   in certain standard documentation is intuitive. The documentation guiding
 *   our taxonomy is:
 *
 *   - "Intel(R) 64 and IA-32 Architectures Software Developer's Manual"
 *     (especially Volume 1 Section 4). Abbreviated "SDM".
 *
 *   - (maintained here: https://gitlab.com/x86-psABIs/x86-64-ABI).
 *     Abbreviated "psABI".
 *
 * - Simple. Our taxonomy only covers that data types required by the
 *   Graphcompiler's Xbyak JIT engine. This means:
 *
 *   - There are some datatypes defined in the standard documentation
 *     (e.g. 80-bit x87 floats) that are simply omitted from our taxonomy.
 *
 *   - There may be some datatypes that other documentation treats as distinct,
 *     but our taxonomy treats as a single type. E.g., {__m128i, __m128u} vs.
 *     simply {__m128}.
 *
 * - Flat / denormalized. The Manual defines a
 *   type hierarchy with several levels: numeric data type, fundamental data
 *   type, etc. In cases where such distinctions are needed by our taxonomy,
 *   we add columns as needed to our one and only type table, rather than
 *   adding more tables.
 *
 * - Combined CPU + ABI. The data types, and the data-type table, contains
 *   all of our tabular information for x86-64 ISA *and* psABI.
 *
 *   (If/when we support additional ABIs for this ISA, it may be reasonable to
 *   separate the ABI-specific information into separate, per-ABI tables.)
 */

#include <cstddef> // size_t
#include <ostream>
#include <vector>
#include <compiler/jit/xbyak/x86_64/abi_common.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

/// Terminology notes:
///
/// From the Intel SDM:
/// "Fundamental data types" are units of data expressed as some number of
/// contiguously grouped bits, without consideration to the interpretation
/// of those bits. The SDM defines these fundamental data types:
///
/// - In Volume 1, Section 4.1:
///    - "byte" : 8 bits
///    - "word" : 2 bytes
///    - "doubleword" : 4 bytes
///    - "quadword" : 8 bytes
///    - "double quadword" : 16 bytes

enum class cpu_data_type {
    uint_8,
    uint_8_x8,
    uint_8_x16,
    uint_8_x32,
    uint_8_x64,
    sint_8,
    sint_8_x8,
    sint_8_x16,
    sint_8_x32,
    sint_8_x64,
    uint_16,
    uint_16_x4,
    uint_16_x8,
    uint_16_x16,
    uint_16_x32,
    uint_32,
    uint_32_x2,
    uint_32_x4,
    uint_32_x8,
    uint_32_x16,
    sint_32,
    sint_32_x2,
    sint_32_x4,
    sint_32_x8,
    sint_32_x16,
    uint_64,
    uint_64_x2,
    uint_64_x4,
    uint_64_x8,
    float_16,
    float_16_x4,
    float_16_x8,
    float_16_x16,
    float_16_x32,
    float_32,
    float_32_x2,
    float_32_x4,
    float_32_x8,
    float_32_x16,
    mask_x4,
    mask_x8,
    mask_x16,
    mask_x32,
    mask_x64,
    void_t,
};

std::ostream &operator<<(std::ostream &os, const cpu_data_type t);

class cpu_data_type_table {
public:
    struct row {
        cpu_data_type type_;

        /// The number of bytes needed to store this data type in memory.
        size_t size_in_bytes_;

        /// For CPU memory operands having formal type \c type_,
        /// this is the minimum memory-address alignment required for the CPU
        /// to efficiently access the value. See SDM 4.1.1.
        ///
        /// The actual requirement is:
        /// (runtime memory address) % (cpu_natural_alignment_) == 0
        size_t cpu_natural_alignment_;

        /// Some CPU instructions have a hard requirement for the memory-address
        /// alignment of their operands. This field is the numerically-greatest
        /// alignment requirement across all CPU instructions that operate on
        /// this type of value. If there is no such requirements for this \c
        /// type_, this field's value is 1. See SDM vol. 1, sections 4.1
        /// and 15.7.
        ///
        /// The actual requirement is:
        /// (runtime memory address) % (cpu_strictest_alignment_) == 0
        size_t cpu_strictest_alignment_;

        /// The psABI (section 3.2.2) requires that %rsp meets certain alignment
        /// requirements upon entry to any callee function.
        /// This gives the minimum alignment required by the ABI for
        /// stack-passed arguments of type \c type_.
        ///
        /// This value should NOT be confused with the "Alignment (bytes)"
        /// column from psABI Figure 3.1. That column seems to be commentary
        /// on the alignment needed for efficient memory access on certain
        /// x86-64 microarchitectures.
        size_t abi_precall_stack_alignment_;

        /// Indicates the size of the stack slot used for function-call
        /// parameters of this type, as specified by the psABI.
        /// This will always be a multiple of 8.
        size_t abi_stack_slot_size_;

        /// Prescribes the amount of stack-memory used to store local variables
        /// and temporary r-values of this type.
        /// This value is a design choice for the Xback-JIT codegen.
        /// It's not (directly) constrained by the psABI.
        size_t local_value_stack_slot_size_;

        /// How the psABI (section 3.2.3) initially classifies this kind of
        /// value when determining how function parameters and return values are
        /// passed.
        abi_value_kind abi_initial_val_kind_;

        row(cpu_data_type type, size_t size_in_bytes,
                size_t cpu_natural_alignment, size_t cpu_strictest_alignment,
                size_t abi_precall_stack_alignment, size_t abi_stack_slot_size,
                size_t local_value_stack_slot_size,
                abi_value_kind abi_initial_val_kind);
    };

    /// Populate the table with the specified content.
    /// It is an error for two or more rows to have the same
    /// cpu_data_type value.
    cpu_data_type_table(const std::vector<row> &content);

    /// Return the row with the specified data type.
    /// It is an error for \p t to not be a member of the table.
    const row &lookup(cpu_data_type t) const;

private:
    const std::vector<row> content_;
};
const cpu_data_type_table &get_cpu_data_types();

// Convenience functions...
size_t get_local_value_stack_slot_size(cpu_data_type t);
size_t get_size_in_bytes(cpu_data_type t);

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
