/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_GENERIC_VAL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_GENERIC_VAL_HPP

#include <stdint.h>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

union generic_val {
    uint16_t v_uint16_t;
    float v_float;
    int32_t v_int32_t;
    int8_t v_int8_t;
    uint8_t v_uint8_t;
    uint64_t v_uint64_t;
    void *v_ptr;
    generic_val() = default;
    generic_val(uint16_t v) : v_uint16_t(v) {}
    generic_val(float v) : v_float(v) {}
    generic_val(int32_t v) : v_int32_t(v) {}
    generic_val(int8_t v) : v_int8_t(v) {}
    generic_val(uint8_t v) : v_uint8_t(v) {}
    generic_val(uint64_t v) : v_uint64_t(v) {}
    generic_val(void *v) : v_ptr(v) {}
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
