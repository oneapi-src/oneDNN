/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace ir_utils {

thread_local int ir_check_log_level_t::level_ = LOG_CHECK_DEFAULT;

} // namespace ir_utils

void stringify_to_cpp_file(const std::string &file_name,
        const std::string &var_name, const std::vector<std::string> &namespaces,
        const std::vector<std::string> &lines) {
    std::ofstream out(file_name);
    for (auto &ns : namespaces)
        out << "namespace " << ns << " {\n";
    out << "\n// clang-format off\n";
    out << "const char** get_" << var_name << "() {\n";
    out << "    static const char *entries[] = {\n";
    for (auto &l : lines) {
        out << "        \"" << l << "\",\n";
    }
    out << "        nullptr,\n";
    out << "\n    };";
    out << "\n    return entries;";
    out << "\n};";
    out << "\n// clang-format on\n\n";
    for (auto it = namespaces.rbegin(); it != namespaces.rend(); it++)
        out << "} // namespace " << *it << "\n";
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
