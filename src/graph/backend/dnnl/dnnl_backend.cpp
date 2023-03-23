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

#include <utility>

#include "graph/utils/any.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/dnnl_backend.hpp"
#include "graph/backend/dnnl/dnnl_opset.hpp"
#include "graph/backend/dnnl/kernels/kernels.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"

#ifdef DNNL_GRAPH_LAYOUT_DEBUG
#include "oneapi/dnnl/dnnl_debug.h"

static const size_t LAST_TAG
        = static_cast<size_t>(dnnl::memory::format_tag::format_tag_last);
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

bool dnnl_layout_id_manager_t::is_mem_desc_equal(
        const graph::utils::any_t &mem_desc1,
        const graph::utils::any_t &mem_desc2) const {
    auto &md1 = graph::utils::any_cast<const memory::desc &>(mem_desc1);
    auto &md2 = graph::utils::any_cast<const memory::desc &>(mem_desc2);
    return md1 == md2;
}

dnnl_backend::dnnl_backend(const std::string &name, float priority)
    : backend(name, priority) {
    register_op_schemas();
    register_passes();
}

bool dnnl_backend::register_op_schemas() {
    register_dnnl_opset_schema();
    return true;
}

bool dnnl_backend::register_passes() {
#define DNNL_BACKEND_REGISTER_PATTERN_CALL(pattern_class_, pattern_registry_) \
    pattern::register_##pattern_class_(pattern_registry_);

    DNNL_BACKEND_REGISTER_PATTERN_CALL(binary_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(bn_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(concat_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(conv_block_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(conv_post_ops_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(convtranspose_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(matmul_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(single_op_pass, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(pool_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(eltwise_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(quantize_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(interpolate_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(softmax_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(layernorm_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(sum_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(reorder_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(shuffle_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(reduction_fusion, pass_registry_);
    pass_registry_.sort_passes();

#undef DNNL_BACKEND_REGISTER_PATTERN_CALL

    return true;
}

size_t dnnl_backend::get_mem_size(const logical_tensor_t &lt) const {
    auto md = make_dnnl_memory_desc(lt);
    return md.get_size();
}

bool dnnl_backend::compare_logical_tensor(
        const logical_tensor_t &lhs, const logical_tensor_t &rhs) const {
    auto md1 = make_dnnl_memory_desc(lhs);
    auto md2 = make_dnnl_memory_desc(rhs);
    return md1 == md2;
}

graph::utils::optional_t<size_t> dnnl_backend::set_mem_desc(
        const graph::utils::any_t &mem_desc) {
    return layout_id_manager_.set_mem_desc(mem_desc);
}

graph::utils::optional_t<graph::utils::any_t> dnnl_backend::get_mem_desc(
        const size_t &layout_id) const {
    return layout_id_manager_.get_mem_desc(layout_id);
}

kernel_ptr large_partition_kernel_creator() {
    return std::make_shared<larger_partition_kernel_t>();
}

graph::utils::optional_t<graph::utils::any_t>
dnnl_layout_id_manager_t::get_mem_desc(size_t layout_id) const {
    std::lock_guard<std::mutex> lock(mem_descs_.m_);
#ifdef DNNL_GRAPH_LAYOUT_DEBUG
    layout_id -= LAST_TAG;
    if (layout_id >= mem_descs_.data_.size()) return graph::utils::nullopt;
    return mem_descs_.data_[layout_id];
#else
    if (layout_id >= mem_descs_.data_.size()) return graph::utils::nullopt;
    return mem_descs_.data_[layout_id];
#endif
}

graph::utils::optional_t<size_t> dnnl_layout_id_manager_t::set_mem_desc(
        const graph::utils::any_t &mem_desc) {
    std::lock_guard<std::mutex> lock(mem_descs_.m_);
    size_t layout_id = 0;
    auto pos = std::find_if(mem_descs_.data_.begin(), mem_descs_.data_.end(),
            [&](const graph::utils::any_t &m) -> bool {
                return is_mem_desc_equal(m, mem_desc);
            });

#ifdef DNNL_GRAPH_LAYOUT_DEBUG
    auto &md = graph::utils::any_cast<const memory::desc &>(mem_desc);

    if (pos != mem_descs_.data_.end()) {
        // ????
        layout_id = static_cast<size_t>(
                            std::distance(mem_descs_.data_.begin(), pos))
                + LAST_TAG;
    } else if (md.get_format_kind() != format_kind::blocked) {
        mem_descs_.data_.emplace_back(mem_desc);
        layout_id = mem_descs_.data_.size() - 1 + LAST_TAG;
    }

    if (md.get_format_kind() == format_kind::blocked) {
        size_t format_tag = static_cast<size_t>(get_format_tag(md));
        for (size_t tag = 0; tag < dnnl_format_tag_last; ++tag) {
            if (tag == format_tag) {
                layout_id = tag;
                break;
            }
        }

        if (!(format_tag > 0 && format_tag < dnnl_format_tag_last)) {
            mem_descs_.data_.emplace_back(mem_desc);
            layout_id = static_cast<size_t>(mem_descs_.data_.size() - 1);
            return layout_id + LAST_TAG;
        }

        // Check if md has extra flags. Note that since onednn didn't
        // provide api to check extra flags, here we construct a temp md
        // without extra flag, and then compare it with the origin md. If
        // they are not equal, the origin md may has extra flags. Only using
        // shape, data type and format tag can't describe the md anymore, so
        // we must cache it to layout id manager.
        const auto &dims = md.get_dims();
        const auto &dtype = md.get_data_type();
        memory::desc temp_md(
                dims, dtype, static_cast<memory::format_tag>(format_tag));
        if (md != temp_md) {
            mem_descs_.data_.emplace_back(mem_desc);
            layout_id = static_cast<size_t>(mem_descs_.data_.size() - 1);
            return layout_id + LAST_TAG;
        }
    }

    return layout_id;
#else
    if (pos != mem_descs_.data_.end()) {
        layout_id = static_cast<size_t>(
                std::distance(mem_descs_.data_.begin(), pos));
    } else {
        mem_descs_.data_.emplace_back(mem_desc);
        layout_id = static_cast<size_t>(mem_descs_.data_.size() - 1);
    }

    return layout_id;
#endif
}
} // namespace dnnl_impl

// This function should be called by backend_registry_t
void register_dnnl_backend() {
    backend_registry_t::get_singleton().register_backend(
            &dnnl_impl::dnnl_backend::get_singleton());
}

} // namespace graph
} // namespace impl
} // namespace dnnl
