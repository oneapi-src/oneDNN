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

#ifndef BACKEND_DNNL_BACKEND_HPP
#define BACKEND_DNNL_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>

#include "interface/backend.hpp"
#include "interface/logical_tensor.hpp"

#include "common.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

class dnnl_layout_id_manager : public layout_id_manager {
    friend class dnnl_backend;

    // private, only can be created in dnnl_backend
    dnnl_layout_id_manager(backend *owner_backend)
        : layout_id_manager(owner_backend) {}

    bool is_mem_desc_equal(const impl::utils::any &mem_desc1,
            const impl::utils::any &mem_desc2) const override;
};

class dnnl_executable : public executable {
    friend class dnnl_backend;

    dnnl_executable(kernel_base::ptr &kernel) : kernel_(kernel) {}

    // Workaround, because current dnnl operator's execute need op.
    // So, we store the op ptr and use it in execute
    dnnl_executable(kernel_base::ptr &kernel, const impl::op_t *op)
        : kernel_(kernel), op_(op) {}

public:
    virtual impl::status_t execute(const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override;

    virtual const std::vector<impl::inplace_pair_t> &
    get_inplace_pairs() const override;

private:
    kernel_base::ptr kernel_;
    const impl::op_t *op_;
};

class dnnl_backend : public backend {
    friend class impl::backend_manager;

public:
    // Used by DNNL backend to cache memory descriptor and get layout id
    impl::utils::optional<size_t> set_mem_desc(
            const impl::utils::any &mem_desc);
    impl::utils::optional<impl::utils::any> get_mem_desc(
            const size_t &layout_id) const;

    kernel_registry &get_kernel_registry() { return kernel_registry_; }
    dnnl_layout_id_manager &get_layout_id_manager() {
        return layout_id_manager_;
    }

private:
    dnnl_backend(std::string name, size_t id);

    virtual size_t get_mem_size_impl(const impl::logical_tensor_t &lt) override;

    virtual executable::ptr compile_impl(const partition_t *p,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    virtual bool to_public_impl(const impl::tensor_t &input,
            impl::tensor_t &output, impl::engine_t &aengine,
            impl::stream_t &g_stream) override;

    virtual bool is_similar_impl(const impl::logical_tensor_t &lhs,
            const impl::logical_tensor_t &rhs) override;

    dnnl_layout_id_manager layout_id_manager_;
    kernel_registry kernel_registry_;
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
