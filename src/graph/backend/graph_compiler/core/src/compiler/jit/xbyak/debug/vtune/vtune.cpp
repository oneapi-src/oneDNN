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

#include <map>
#include <memory>
#include <common/ittnotify/jitprofiling.h>
#include <compiler/jit/xbyak/debug/debug_info_mgr.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

struct vtune_debug_info_mgr_t : debug_info_mgr {
    vtune_debug_info_mgr_t(void *base, size_t size, const std::string &src_path,
            const std::vector<func_symbol> &funcs,
            const std::vector<debug_line> &lines)
        : debug_info_mgr(base, size, src_path) {
        if (iJIT_IsProfilingActive() != iJIT_SAMPLING_ON) { return; }

        std::map<void *, size_t> addr_func_map;
        for (size_t i = 0; i < funcs.size(); i++) {
            addr_func_map.insert(std::make_pair(funcs[i].start_, i));
        }
        std::vector<std::vector<const debug_line *>> lines_by_func;
        lines_by_func.resize(funcs.size());
        if (!addr_func_map.empty()) {
            for (auto &line : lines) {
                // find the first func addr > line.start_
                auto itr = addr_func_map.upper_bound(line.start_);
                if (itr == addr_func_map.end()) {
                    lines_by_func.back().emplace_back(&line);
                } else if (itr == addr_func_map.begin()) {
                    lines_by_func.front().emplace_back(&line);
                } else {
                    --itr;
                    lines_by_func[itr->second].emplace_back(&line);
                }
            }
        }

        for (size_t i = 0; i < funcs.size(); i++) {
            auto jmethod = iJIT_Method_Load();
            jmethod.method_id = iJIT_GetNewMethodID();
            jmethod.method_name = (char *)funcs[i].name_.c_str();
            jmethod.class_file_name = nullptr;
            jmethod.source_file_name = (char *)src_path.c_str();
            jmethod.method_load_address = (void *)funcs[i].start_;
            jmethod.method_size = (unsigned int)((uint8_t *)funcs[i].end_
                    - (uint8_t *)funcs[i].start_);
            jmethod.line_number_size = lines_by_func[i].size();
            std::vector<LineNumberInfo> lineinfo;
            lineinfo.reserve(lines_by_func[i].size());
            for (auto &line : lines_by_func[i]) {
                LineNumberInfo v {(unsigned int)((uint8_t *)line->start_
                                          - (uint8_t *)funcs[i].start_),
                        (unsigned int)line->line_};
                lineinfo.emplace_back(v);
            }
            jmethod.line_number_table = lineinfo.data();
            iJIT_NotifyEvent(
                    iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED, (void *)&jmethod);
        }
    }

    ~vtune_debug_info_mgr_t() override = default;
};

std::unique_ptr<debug_info_mgr> create_vtune_debug_info(void *base, size_t size,
        const std::string &src_path, const std::vector<func_symbol> &funcs,
        const std::vector<debug_line> &lines) {
    return utils::make_unique<vtune_debug_info_mgr_t>(
            base, size, src_path, funcs, lines);
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
