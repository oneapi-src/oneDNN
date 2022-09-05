/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef INTERFACE_PARTITION_HPP
#define INTERFACE_PARTITION_HPP

#include <cstring>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "interface/c_types_map.hpp"
#include "interface/engine.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/op.hpp"
#include "interface/partition_impl.hpp"
#include "interface/stream.hpp"

#include "utils/compatible.hpp"
#include "utils/id.hpp"
#include "utils/utils.hpp"
#include "utils/verbose.hpp"

namespace impl = dnnl::graph::impl;

namespace dnnl {
namespace graph {
namespace impl {
class backend;
} // namespace impl
} // namespace graph
} // namespace dnnl

struct dnnl_graph_compilation_context {
public:
    dnnl_graph_compilation_context() = default;

    dnnl_graph_compilation_context(const dnnl_graph_compilation_context &other)
            = default;

    void set_tensor_data_handle(size_t id, void *handle) {
        tensor_data_handle_map_[id] = handle;
    }

    void *get_tensor_data_handle(size_t id) const {
        return tensor_data_handle_map_.at(id);
    }

    std::unordered_set<size_t> get_ids() const {
        std::unordered_set<size_t> ids {};
        ids.reserve(tensor_data_handle_map_.size());
        for (const auto &e : tensor_data_handle_map_)
            ids.insert(e.first);
        return ids;
    }

private:
    /// logical tensor id -> tensor data handle
    std::unordered_map<size_t, void *> tensor_data_handle_map_;
};

struct dnnl_graph_partition : public impl::utils::id_t {
public:
    friend struct dnnl_graph_compiled_partition;
    friend struct impl::utils::partition_info_t;

    dnnl_graph_partition() = default;

    // deep copy
    dnnl_graph_partition(const dnnl_graph_partition &other) = default;

    // disable assign
    dnnl_graph_partition &operator=(const dnnl_graph_partition &other) = delete;

    ~dnnl_graph_partition() = default;

    void init(const std::shared_ptr<impl::partition_impl_t> &pimpl) {
        pimpl_ = pimpl;
        const_cast<impl::partition_impl_t *>(pimpl_.get())->set_id(id());
    }

    bool is_initialized() const {
        return (pimpl_ != nullptr) && pimpl_->is_initialized();
    }

    bool is_supported() const;

    const impl::partition_impl_t *get_pimpl() const { return pimpl_.get(); }

    const impl::backend *get_assigned_backend() const {
        return pimpl_->get_assigned_backend();
    }

    impl::engine_kind_t get_engine_kind() const {
        return pimpl_->get_engine_kind();
    }

    impl::fpmath_mode_t get_fpmath_mode() const {
        return pimpl_->get_fpmath_mode();
    }

    std::unordered_set<size_t> get_input_index_having_context() const {
        return pimpl_->get_input_index_having_context();
    }

    impl::partition_kind_t get_kind() const { return pimpl_->get_kind(); }

    const std::vector<std::shared_ptr<impl::op_t>> &get_ops() const {
        return pimpl_->get_ops();
    }

    size_t num_ops() const { return pimpl_->get_ops().size(); }

    std::vector<size_t> get_op_ids() const {
        std::vector<size_t> ids;
        auto ops = pimpl_->get_ops();
        ids.reserve(ops.size());
        for (auto &op : ops) {
            ids.emplace_back(op->get_id());
        }
        return ids;
    }

    const std::vector<impl::logical_tensor_t> &get_inputs() const {
        return pimpl_->get_inputs();
    }

    const std::vector<impl::logical_tensor_t> &get_outputs() const {
        return pimpl_->get_outputs();
    }

    size_t get_inputs_num() const { return pimpl_->get_inputs().size(); }

    size_t get_outputs_num() const { return pimpl_->get_outputs().size(); }

    impl::status_t compile(impl::compiled_partition_t *compiled_partition,
            std::vector<const impl::logical_tensor_t *> &inputs,
            std::vector<const impl::logical_tensor_t *> &outputs,
            const impl::engine_t *e = nullptr,
            const impl::compilation_context_t *acompilation_context
            = nullptr) const;

    impl::status_t compile(
            std::pair<impl::compiled_partition_t *, bool> &compiled_partition,
            std::vector<const impl::logical_tensor_t *> &inputs,
            std::vector<const impl::logical_tensor_t *> &outputs,
            const impl::engine_t *aengine,
            const impl::compilation_context_t *acompilation_context
            = nullptr) const;

    impl::status_t infer_shape(
            std::vector<const impl::logical_tensor_t *> &inputs,
            std::vector<impl::logical_tensor_t *> &outputs);

private:
    std::shared_ptr<const impl::partition_impl_t> pimpl_;
};

///
/// \brief dnnl_graph_compiled_partition_t
///
struct dnnl_graph_compiled_partition : public impl::utils::id_t {
public:
    friend struct dnnl_graph_partition;
    friend struct impl::utils::partition_info_t;

    dnnl_graph_compiled_partition(const impl::partition_t &src_partition)
        : src_partition_ {src_partition} {}

    ~dnnl_graph_compiled_partition() = default;

    const impl::partition_t &src_partition() const { return src_partition_; }

    void init(const std::shared_ptr<impl::compiled_partition_impl_t> &pimpl) {
        pimpl_ = pimpl;
    }

    bool is_initialized() const { return pimpl_ != nullptr; }

    const impl::compiled_partition_impl_t *get_pimpl() const {
        return pimpl_.get();
    }

    const std::vector<impl::inplace_pair_t> &get_inplace_pairs() const {
        return pimpl_->get_inplace_pairs();
    }

    impl::status_t execute(const impl::stream_t *astream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) const;

#ifdef DNNL_GRAPH_WITH_SYCL
    impl::status_t execute_sycl(const impl::stream_t *astream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) const;
#endif

    impl::status_t query_logical_tensor(
            size_t tid, impl::logical_tensor_t *lt) const {
        if (!pimpl_) {
            std::memset(lt, 0, sizeof(impl::logical_tensor_t));
            return impl::status::success;
        }
        return pimpl_->query_logical_tensor(tid, lt);
    }

    const impl::engine_t &get_engine() const { return pimpl_->get_engine(); }

    std::vector<impl::logical_tensor_t> &get_mutable_inputs() {
        return pimpl_->get_mutable_inputs();
    }

    std::vector<impl::logical_tensor_t> &get_mutable_outputs() {
        return pimpl_->get_mutable_outputs();
    }

    const std::vector<impl::logical_tensor_t> &get_inputs() const {
        return pimpl_->get_inputs();
    }

    const std::vector<impl::logical_tensor_t> &get_outputs() const {
        return pimpl_->get_outputs();
    }

    const char *info() const {
        auto &eng = pimpl_->get_engine();
        if (!info_.is_initialized()) info_.init(&eng, this);
        return info_.c_str();
    }

private:
    std::shared_ptr<impl::compiled_partition_impl_t> pimpl_;

    const impl::partition_t src_partition_;

    // Partition information
    mutable impl::utils::partition_info_t info_;
};

#endif
