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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_GENERIC_VAL_PACK_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_GENERIC_VAL_PACK_HPP

#include <util/def.hpp>

namespace sc {

union generic_val;

struct generic_val_pack {
    virtual generic_val *get() = 0;
    virtual void flush_cache() const = 0;
    virtual SC_INTERNAL_API ~generic_val_pack() = default;
};

} // namespace sc

#endif
