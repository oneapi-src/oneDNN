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

#include <compiler/jit/xbyak/x86_64/abi_value_location.hpp>

#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

abi_value_location::abi_value_location() {
    set_to_none();
}

abi_value_location::abi_value_location(Xbyak::Reg reg) {
    set_to_register(reg);
}

abi_value_location::abi_value_location(int rsp_offset) {
    set_to_rsp_offset(rsp_offset);
}

void abi_value_location::set_to_none() {
    tag_ = tag_type::NONE;
}

void abi_value_location::set_to_register(Xbyak::Reg reg) {
    tag_ = tag_type::REGISTER;
    val_.reg_ = reg;
}

void abi_value_location::set_to_rsp_offset(int rsp_offset) {
    tag_ = tag_type::STACK;
    val_.rsp_offset_ = rsp_offset;
}

abi_value_location::tag_type abi_value_location::get_type() const {
    return tag_;
}

Xbyak::Reg abi_value_location::get_register() const {
    COMPILE_ASSERT(tag_ == tag_type::REGISTER, "wrong tag type");
    return val_.reg_;
}

int abi_value_location::get_rsp_offset() const {
    COMPILE_ASSERT(tag_ == tag_type::STACK, "wrong tag type");
    return val_.rsp_offset_;
}

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
