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

#include <utility>

#include "graph/utils/any.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/layout_id_mgr.hpp"

#include "oneapi/dnnl/dnnl.hpp"
#ifdef DNNL_GRAPH_LAYOUT_DEBUG
#include "oneapi/dnnl/dnnl_debug.h"
static const size_t LAST_TAG
        = static_cast<size_t>(dnnl::memory::format_tag::format_tag_last);
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

graph::utils::optional_t<memory::desc> dnnl_layout_id_manager_t::get_mem_desc(
        size_t layout_id) const {
    std::lock_guard<std::mutex> lock(mem_descs_.m_);
#ifdef DNNL_GRAPH_LAYOUT_DEBUG
    layout_id -= LAST_TAG;
    if (layout_id >= mem_descs_.data_.size())
        return graph::utils::nullopt;
    else
        return mem_descs_.data_[layout_id];
#else
    if (layout_id >= mem_descs_.data_.size())
        return graph::utils::nullopt;
    else
        return mem_descs_.data_[layout_id];
#endif
}

graph::utils::optional_t<size_t> dnnl_layout_id_manager_t::set_mem_desc(
        const memory::desc &md) {
    std::lock_guard<std::mutex> lock(mem_descs_.m_);
    size_t layout_id = 0;
    auto pos = std::find_if(mem_descs_.data_.begin(), mem_descs_.data_.end(),
            [&](const memory::desc &val) -> bool { return val == md; });

#ifdef DNNL_GRAPH_LAYOUT_DEBUG
    if (pos != mem_descs_.data_.end()) {
        // if the md is already in the manager, it means the layout of the md
        // cannot be determined by the format tag value. For example, the md may
        // contain compensation values. For this case, we still use the position
        // as the layout id and use LAST_TAG as the offset to distinguish format
        // tag based layout id from position based layout id.
        layout_id = static_cast<size_t>(
                            std::distance(mem_descs_.data_.begin(), pos))
                + LAST_TAG;
    } else {
        // the md is not in the manager. If md is trivial and can be determined
        // simply by a format tag, we will not save the md and return the format
        // tag value directly as the layout id. Otherwise, the md will be saved
        // in the manager and a position based layout id will be returned.
        if (md.get_format_kind() != memory::format_kind::blocked) {
            mem_descs_.data_.emplace_back(md);
            layout_id = mem_descs_.data_.size() - 1 + LAST_TAG;
        } else { // blocked format
            const size_t format_tag = static_cast<size_t>(get_format_tag(md));
            if (format_tag == dnnl_format_tag_undef
                    || format_tag >= dnnl_format_tag_last) {
                // for format tag not supported by api, it's non-trivial and md
                // cannot be determined by it.
                mem_descs_.data_.emplace_back(md);
                layout_id = static_cast<size_t>(mem_descs_.data_.size() - 1)
                        + LAST_TAG;
            } else {
                // Check if md has extra flags. Note that since onednn didn't provide
                // api to check extra flags, here we construct a temp md without extra
                // flag, and then compare it with the origin md. If they are not equal,
                // the origin md may has extra flags. Only using shape, data type and
                // format tag can't describe the md anymore, so we must cache it to
                // layout id manager.
                const auto &dims = md.get_dims();
                const auto &dtype = md.get_data_type();
                memory::desc temp_md(dims, dtype,
                        static_cast<memory::format_tag>(format_tag));
                if (md != temp_md) {
                    mem_descs_.data_.emplace_back(md);
                    layout_id = static_cast<size_t>(mem_descs_.data_.size() - 1)
                            + LAST_TAG;
                } else {
                    // finally, the md is trivial and can be determined by the
                    // format tag, we return the format tag as the layout id and
                    // do not need to save the md.
                    layout_id = format_tag;
                }
            }
        }
    }
#else
    if (pos != mem_descs_.data_.end()) {
        // the md is already in the manager and the position is returned.
        layout_id = static_cast<size_t>(
                std::distance(mem_descs_.data_.begin(), pos));
    } else {
        // store the md in the manager and the position is returned.
        mem_descs_.data_.emplace_back(md);
        layout_id = static_cast<size_t>(mem_descs_.data_.size() - 1);
    }

#endif
    return layout_id;
}
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
