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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_ABI_VALUE_LOCATION_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_X86_64_ABI_VALUE_LOCATION_HPP

#include <compiler/jit/xbyak/configured_xbyak.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

class abi_value_location {
public:
    enum class tag_type {
        /// A placeholder that indicates a particular role (e.g., void return
        /// value) is not needed and therefore has no location.
        NONE,

        /// The value is stored in a register.
        REGISTER,

        /// The value is stored on the stack.
        STACK,
    };

    abi_value_location();
    abi_value_location(Xbyak::Reg reg);
    abi_value_location(int rsp_offset);

    void set_to_none();
    void set_to_register(Xbyak::Reg reg);
    void set_to_rsp_offset(int rsp_offset);

    tag_type get_type() const;
    Xbyak::Reg get_register() const;
    int get_rsp_offset() const;

private:
    tag_type tag_ = tag_type::NONE;

    union {
        /// Indicate's the memory address of this argument as a byte offset
        /// relative to the value of %rsp when control first reaches the
        /// callee.
        int rsp_offset_;

        /// The register containing the value.
        /// NOTE: This works for integer-like values, but we'll need to
        /// refactor if/when we support xmm/ymm/zmm registers.
        Xbyak::Reg reg_ = Xbyak::Reg();
    } val_;
};

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
