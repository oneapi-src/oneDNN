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

#ifndef BACKEND_DNNL_DNNL_BACKEND_HPP
#define BACKEND_DNNL_DNNL_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "interface/backend.hpp"
#include "interface/c_types_map.hpp"
#include "interface/logical_tensor.hpp"

#include "utils/compatible.hpp"
#include "utils/pm/pass_manager.hpp"

#include "common.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

class dnnl_partition_impl_t;

class layout_id_manager {
public:
    layout_id_manager() {};
    virtual ~layout_id_manager() {}

    /*! \brief Set a backend memory descriptor to manager and get a
    * corresponding layout id
    * \param mem_desc The backend's memory descriptor, it can
    * be both plain or opaque
    * \return a cache index, will be used as layout id
    * \note This function should be invoked in every where we want to
    * convert a md to layout id
    */
    virtual impl::utils::optional<size_t> set_mem_desc(
            const impl::utils::any &mem_desc) {
        std::lock_guard<std::mutex> lock(mem_descs_.m_);

        auto pos = std::find_if(mem_descs_.data_.begin(),
                mem_descs_.data_.end(), [&](const impl::utils::any &m) -> bool {
                    return is_mem_desc_equal(m, mem_desc);
                });

        size_t layout_id;
        if (pos != mem_descs_.data_.end()) {
            layout_id = static_cast<size_t>(
                    std::distance(mem_descs_.data_.begin(), pos));
        } else {
            mem_descs_.data_.emplace_back(mem_desc);
            layout_id = static_cast<size_t>(mem_descs_.data_.size() - 1);
        }

        return layout_id;
    }

    /*! \brief Get a backend memory descriptor from manager by using a
    * layout id
    * \param layout_id The layout id, which is generated and managed
    * by backends
    * \return When the input is a valid cache index, the return value
    * is a cached memory descriptor; otherwise, the return value will
    * be a utils::nullopt
    */
    virtual impl::utils::optional<impl::utils::any> get_mem_desc(
            const size_t &layout_id) const {
        std::lock_guard<std::mutex> lock(mem_descs_.m_);
        if (layout_id >= mem_descs_.data_.size()) return {};
        return mem_descs_.data_[layout_id];
    }

private:
    /*! \brief compare two backend mem desc
    * \param mem_desc1
    * \param mem_desc2
    * \return bool
    */
    virtual bool is_mem_desc_equal(const impl::utils::any &mem_desc1,
            const impl::utils::any &mem_desc2) const = 0;

    mutable struct {
        std::vector<impl::utils::any> data_;
        mutable std::mutex m_;
    } mem_descs_;
};

class dnnl_layout_id_manager : public layout_id_manager {
    friend class dnnl_backend;

    // private, only can be created in dnnl_backend
    dnnl_layout_id_manager() {}

    bool is_mem_desc_equal(const impl::utils::any &mem_desc1,
            const impl::utils::any &mem_desc2) const override;
};

// gcc4.8.5 can 't support enum class as key
struct enum_hash {
    template <typename T>
    size_t operator()(const T &t) const {
        return static_cast<size_t>(t);
    }
};

struct kernel_base {
    virtual ~kernel_base() {}

    template <typename T>
    impl::status_t compile(const T *op_or_part, const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) {
        auto ret = compile_impl(op_or_part, aengine, inputs, outputs);
        if (ret != impl::status::success) return ret;
        return prepare_inplace_pairs_impl(aengine, inputs, outputs);
    }

    template <typename T>
    impl::status_t execute(const T *op_or_part, const impl::stream_t *astream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) {
        return execute_impl(op_or_part, astream, inputs, outputs);
    }

    // This function is used in fused op mode
    virtual impl::status_t compile_impl(const impl::op_t *aop,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) {
        UNUSED(aop);
        UNUSED(aengine);
        UNUSED(inputs);
        UNUSED(outputs);
        return impl::status::success;
    }

    // This function is used in fused op mode
    virtual impl::status_t execute_impl(const impl::op_t *aop,
            const impl::stream_t *astream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) {
        UNUSED(aop);
        UNUSED(astream);
        UNUSED(inputs);
        UNUSED(outputs);
        return impl::status::success;
    }

    // This function is used in subgraph mode
    virtual impl::status_t compile_impl(const dnnl_partition_impl_t *part,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) {
        UNUSED(part);
        UNUSED(aengine);
        UNUSED(inputs);
        UNUSED(outputs);
        return status::success;
    }

    // This function is used in subgraph mode
    virtual impl::status_t execute_impl(const dnnl_partition_impl_t *part,
            const impl::stream_t *astream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) {
        UNUSED(astream);
        UNUSED(inputs);
        UNUSED(outputs);
        return impl::status::success;
    }

    virtual impl::status_t prepare_inplace_pairs_impl(
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) {
        UNUSED(aengine);
        UNUSED(inputs);
        UNUSED(outputs);
        return impl::status::success;
    };

    std::vector<impl::inplace_pair_t> inplace_pairs_;
};

using kernel_ptr = std::shared_ptr<kernel_base>;

class kernel_registry {
public:
    using kernel_creator_f = kernel_ptr (*)();
    using ptr = std::shared_ptr<kernel_registry>;

    kernel_registry() = default;
    virtual ~kernel_registry() {}

    template <typename kernel_type>
    static kernel_ptr create_kernel() {
        return std::make_shared<kernel_type>();
    }

    /*!
     * \brief register a backend kernel's creator for a op_kind
     */
    bool register_kernel(impl::op_kind_t op_kind, kernel_creator_f fn) {
        std::lock_guard<std::mutex> lock(kernel_creator_f_map_.m_);
        kernel_creator_f_map_.data_.insert({op_kind, fn});
        return true;
    }

    /*!
     * \brief create an kernel instance for a op
     */
    kernel_ptr create_kernel(const impl::op_t &aop) {
        auto op_kind = aop.get_kind();
        std::lock_guard<std::mutex> lock(kernel_creator_f_map_.m_);

        auto pos = kernel_creator_f_map_.data_.find(op_kind);
        if (pos == kernel_creator_f_map_.data_.end()) return {};

        auto create_fn = pos->second;
        return create_fn();
    }

    /*!
     * \brief get registered kernel number
     */
    size_t get_register_kernels_num() const {
        std::lock_guard<std::mutex> lock(kernel_creator_f_map_.m_);
        return kernel_creator_f_map_.data_.size();
    }

private:
    // Disable assignment and copy
    kernel_registry(const kernel_registry &) = delete;
    kernel_registry(kernel_registry &&) = delete;
    kernel_registry &operator=(const kernel_registry &) = delete;
    kernel_registry &operator=(kernel_registry &&) = delete;

    mutable struct {
        std::unordered_map<impl::op_kind_t, kernel_creator_f, enum_hash> data_;
        mutable std::mutex m_;
    } kernel_creator_f_map_;
};

class dnnl_backend : public backend {
    friend class dnnl_partition_impl_t;

public:
    static dnnl_backend &get_singleton() {
        static dnnl_backend ins("dnnl_backend", /*priority*/ 1.f);
        return ins;
    }

    // Used by DNNL backend to cache memory descriptor and get layout id
    impl::utils::optional<size_t> set_mem_desc(
            const impl::utils::any &mem_desc);
    impl::utils::optional<impl::utils::any> get_mem_desc(
            const size_t &layout_id) const;

    kernel_registry &get_kernel_registry() { return kernel_registry_; }

    impl::pass::pass_registry &get_pass_registry() { return pass_registry_; }

    dnnl_layout_id_manager &get_layout_id_manager() {
        return layout_id_manager_;
    }

    kernel_ptr create_kernel(const impl::op_t &aop) {
        return kernel_registry_.create_kernel(aop);
    }

    virtual size_t get_mem_size(
            const impl::logical_tensor_t &lt) const override;

    virtual bool compare_logical_tensor(const impl::logical_tensor_t &lhs,
            const impl::logical_tensor_t &rhs) const override;

    virtual status_t get_partitions(impl::graph_t &agraph,
            impl::partition_policy_t policy
            = impl::partition_policy::fusion) override {
        impl::pass::pass_manager pm(get_pass_registry());
        pm.run_passes(agraph, "", policy);
        return status::success;
    }

private:
    dnnl_backend(const std::string &name, float priority);

    bool register_passes();
    bool register_kernels();

    dnnl_layout_id_manager layout_id_manager_;
    kernel_registry kernel_registry_;
    impl::pass::pass_registry pass_registry_;
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
