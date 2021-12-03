/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_CONFIG_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_CONFIG_HPP
#include <stdint.h>
#include <string>
#include <util/def.hpp>

namespace sc {

struct SC_INTERNAL_API runtime_config_t {
    // if in muti-instance simulation, the number of threads per instance.
    int threads_per_instance_;
    bool amx_exclusive_;
    std::string trace_out_path_;
    int trace_initial_cap_;
    bool execution_verbose_;
    static runtime_config_t &get();

private:
    runtime_config_t();
};

} // namespace sc
#endif
