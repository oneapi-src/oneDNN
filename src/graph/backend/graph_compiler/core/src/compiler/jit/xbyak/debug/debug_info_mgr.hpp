/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_DEBUG_DEBUG_INFO_MGR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_DEBUG_DEBUG_INFO_MGR_HPP

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

struct func_symbol {
    std::string name_;
    void *start_;
    void *end_;
};
struct debug_line {
    void *start_;
    int line_;
    int pos_;
};

struct debug_info_mgr {
    void *base_;
    size_t size_;
    std::string src_path_;
    debug_info_mgr(void *base, size_t size, const std::string &src_path)
        : base_ {base}, size_ {size}, src_path_ {src_path} {}

    virtual ~debug_info_mgr() = default;
};

extern std::mutex debug_info_lock;

#ifdef SC_PROFILING
extern std::unique_ptr<debug_info_mgr> create_vtune_debug_info(void *base,
        size_t size, const std::string &src_path,
        const std::vector<func_symbol> &funcs,
        const std::vector<debug_line> &lines);
#endif

inline std::vector<std::unique_ptr<debug_info_mgr>> create_debug_info_mgr(
        void *base, size_t size, const std::string &src_path,
        const std::vector<func_symbol> &funcs,
        const std::vector<debug_line> &lines) {
    std::vector<std::unique_ptr<debug_info_mgr>> ret;
#ifdef SC_PROFILING
    ret.emplace_back(
            create_vtune_debug_info(base, size, src_path, funcs, lines));
#endif
    return ret;
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
