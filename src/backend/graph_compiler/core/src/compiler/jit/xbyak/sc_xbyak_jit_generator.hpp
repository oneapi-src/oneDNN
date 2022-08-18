/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_SC_XBYAK_JIT_GENERATOR_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_SC_XBYAK_JIT_GENERATOR_HPP

// We need these headers to be included in the specified order...
// clang-format off
#include <compiler/jit/xbyak/configured_xbyak.hpp>
// clang-format on

#include <map>
#include <string>

namespace sc {
namespace sc_xbyak {
/**
 * @class sc_xbyak_jit_generator
 *
 * \brief Provides JIT translation services during translation, and owns
 * the memory containing the resulting code and data.
 *
 * The class ::xbyak::CodeGenerator provides two distinct services, which
 * (ideally) we'd prefer to keep separate:
 *
 * 1) ownership of the memory in which the JIT'ed code and data reside, and
 *
 * 2) methods and member variables that are used <i>during</i> JIT translation
 *    and are irrelevant once translation is complete.
 *
 * This class extends ::Xbyak::CodeGenerator, and so we're stuck
 * with this class also serving both of those roles.
 *
 * Once the act of JIT translation is complete, the most of the class members
 * that this class inherets from its ancestor classes are no longer relevant.
 */

class sc_xbyak_jit_generator : public ::Xbyak::CodeGenerator {
public:
    sc_xbyak_jit_generator();
    virtual ~sc_xbyak_jit_generator() = default;

    // The entry-point address of the specified JIT'ed function, or null if
    // none has that name.
    void *get_func_address(const std::string &func_name) const;

private:
    friend class location_manager;
    friend class xbyak_lowering_viewer;

    std::map<std::string, void *> func_name_to_address_;
};

} // namespace sc_xbyak
} // namespace sc

#endif
