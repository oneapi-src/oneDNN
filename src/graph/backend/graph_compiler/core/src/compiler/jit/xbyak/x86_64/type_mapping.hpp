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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_TYPE_MAPPING_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_TYPE_MAPPING_HPP

#include <compiler/ir/sc_data_type.hpp>
#include <compiler/jit/xbyak/x86_64/native_types.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

/**
 * @brief Defines the mapping we'll use for IR types --> CPU native types.
 *
 * Returns the cpu data type prescribed by our time-mapping policy, or
 * fails a COMPILE_ASSERT if the mapping is undefined for \p t.
 *
 * Our type-mapping scheme is based on the following chain:
 *
 * (Graphcompiler IR data type)
 * --> (C/C++ type)             // prescribed by graphcompiler's design
 * --> (CPU data type)          // prescribed by psABI Table 3.1.
 *
 * Note: The type-mapping implementation will generally be incomplete.
 * We add to it on an as-needed basis.
 *
 * TL;DR:
 *
 * Our type-mapping is constrained by the following requirements:
 *
 *    - Function call arguments / return values must be psABI compliant:
 *
 *        - Graphcompiler's IR type system assumes a particular mapping from
 *          \c sc_data_type_t to (C/C++ data type).
 *
 *        - And psABI (as well as the compiler)  prescribe a particular
 *          mapping from (C/C++ data type) to (CPU data type).
 *
 *    - Computations within the body of a function less constrainted, but...
 *
 *        Strictly speaking, the Xbyak JIT engine is free to use alternative
 *        data types within the body of a function, because the psABI's
 *        constraints don't apply.
 *
 *        However, at the moment we have no particular reason to use an
 *        alternative type-mapping within the body of JIT-generated functions.
 *
 * For these reasons, we'll (currently) use the just one type-mapping scheme
 * for function arguments / return values AND within each function body.
 */
cpu_data_type get_cpu_data_type(sc_data_type_t t);

/// A convenience function that combines \c get_cpu_data_type() and
/// \c cpu_data_type_table::lookup().
const cpu_data_type_table::row &get_cpu_data_type_row(sc_data_type_t t);

/**
 * Computes the details needed for allocating a tensor buffer onto the
 * stack.
 *
 * Assumptions:
 *   - The tensor elements will be packed, i.e., that no padding
 *     will be used between adjacent tensor elements.
 *
 *   - The %rsp is already 8-byte aligned before allocating this buffer,
 *     and that %rsp must *also* be 8-byte aligned after allocating
 *     this buffer.
 *
 *     (This is a policy decision. It may waste a little stack space, but
 *     it simplifies our logic for meeting psABI-required stack-alignment
 *     requirements.)
 *
 * \param[in] element_type
 * \param[in] num_elements Must be positive.
 * \param[out] buffer_size The minimum required size for the tensor buffer,
 *      in bytes. The buffer will be oversized as needed to meet the
 *      alignment requirements stated above.
 */
void get_stack_allocated_tensor_buffer_lowering_info(
        sc_data_type_t element_type, size_t num_elements, size_t &buffer_size);

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
