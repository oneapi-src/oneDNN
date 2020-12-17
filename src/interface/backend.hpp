/*******************************************************************************
 * Copyright 2020 Intel Corporation
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

#ifndef LLGA_INTERFACE_BACKEND_HPP
#define LLGA_INTERFACE_BACKEND_HPP

#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "interface/common.hpp"
#include "interface/engine.hpp"
#include "interface/partition.hpp"
#include "interface/stream.hpp"

#define BACKEND_ID_LENGTH 4
#define MAX_BACKEND_NUMS (1 << BACKEND_ID_LENGTH)
#define RESERVED_BACKEND_ID 0 // reserved but not used now

namespace llga {
namespace impl {
struct kernel_base {
    using ptr = std::shared_ptr<kernel_base>;

    virtual ~kernel_base() {}

    virtual status_t compile_impl(const node_t *anode, const engine_t *aengine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs)
            = 0;
    virtual status_t execute_impl(const node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs)
            = 0;
    virtual status_t prepare_inplace_pairs_impl(const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) {
        UNUSED(aengine);
        UNUSED(inputs);
        UNUSED(outputs);
        return status::success;
    };

    status_t compile(const node_t *anode, const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) {
        auto ret = compile_impl(anode, aengine, inputs, outputs);
        if (ret != status::success) return ret;
        return prepare_inplace_pairs_impl(aengine, inputs, outputs);
    }

    status_t execute(const node_t *anode, const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) {
        return execute_impl(anode, astream, inputs, outputs);
    }

    std::vector<impl::inplace_pair_t> inplace_pairs_;
};

// gcc4.8.5 can 't support enum class as key
struct enum_hash {
    template <typename T>
    size_t operator()(const T &t) const {
        return static_cast<size_t>(t);
    }
};

class kernel_registry {
public:
    using kernel_creator_f = kernel_base::ptr (*)();
    using ptr = std::shared_ptr<kernel_registry>;

    kernel_registry() = default;
    virtual ~kernel_registry() {}

    template <typename kernel_type>
    static kernel_base::ptr create_kernel() {
        return std::make_shared<kernel_type>();
    }

    /*! 
     * \brief register a backend kernel's creator for a op_kind
     */
    bool register_kernel(op_kind_t op_kind, kernel_creator_f fn) {
        std::lock_guard<std::mutex> lock(kernel_creator_f_map_.m_);
        kernel_creator_f_map_.data_.insert({op_kind, fn});
        return true;
    }

    /*! 
     * \brief create an kernel instance for a node
     */
    kernel_base::ptr create_kernel(const impl::node_t &anode) {
        auto op_kind = anode.get_op_kind();
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
        std::unordered_map<op_kind_t, kernel_creator_f, enum_hash> data_;
        mutable std::mutex m_;
    } kernel_creator_f_map_;
};

class executable {
public:
    using ptr = std::shared_ptr<executable>;
    executable() = default;
    virtual ~executable() {};

    virtual status_t execute(const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs)
            = 0;

    virtual const std::vector<impl::inplace_pair_t> &
    get_inplace_pairs() const = 0;
};

namespace {
std::pair<size_t, size_t> decode_layout_id(size_t layout_id) {
    size_t backend_id = layout_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1);
    size_t layout_idx = layout_id >> BACKEND_ID_LENGTH;
    return {layout_idx, backend_id};
}

size_t encode_layout_id(size_t layout_idx, size_t backend_id) {
    size_t layout_id = (layout_idx << BACKEND_ID_LENGTH)
            | (backend_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1));
    return layout_id;
}
} // namespace

class backend {
public:
    using ptr = std::shared_ptr<backend>;

    backend(std::string name, size_t id) : name_(name), id_(id) {}

    virtual ~backend() {}

    std::string get_name() const { return name_; };
    size_t get_id() const { return id_; }

    virtual size_t get_mem_size(const impl::logical_tensor_t &lt) {
        if (!is_interpretable(lt)) return 0;
        return get_mem_size_impl(lt);
    };

    virtual executable::ptr compile(const impl::partition *p,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) {
        return compile_impl(p, aengine, inputs, outputs);
    }

    // convert a tensor in private layout to public layout
    virtual bool to_public(const impl::tensor &input, impl::tensor &output,
            impl::engine_t &aengine) {
        if (!is_interpretable(input)) return false;
        return to_public_impl(input, output, aengine);
    }

    // check wheather two logical tensor is similar (similar means two
    // logical tensors can be converted to same backend md)
    virtual bool is_similar(const impl::logical_tensor_t &lhs,
            const impl::logical_tensor_t &rhs) {
        if (!(is_interpretable(lhs) && is_interpretable(rhs))) return false;
        return is_similar_impl(lhs, rhs);
    }

private:
    virtual size_t get_mem_size_impl(const impl::logical_tensor_t &lt) = 0;

    virtual executable::ptr compile_impl(const impl::partition *p,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs)
            = 0;

    virtual bool to_public_impl(const impl::tensor &input, impl::tensor &output,
            impl::engine_t &aengine)
            = 0;

    // This is a default impl. The default impl regards two logical tensors
    // are similar if they are equal bit by bit except their ids
    // User-defined backend can override this method to provide
    // specific check
    virtual bool is_similar_impl(const impl::logical_tensor_t &lhs,
            const impl::logical_tensor_t &rhs) {
        bool equal = true;
        equal = equal && (lhs.ndims == rhs.ndims)
                && (lhs.data_type == rhs.data_type)
                && (lhs.layout_type == rhs.layout_type);

        if (!equal) return false;
        if (lhs.ndims == 0 || lhs.ndims == -1) return true;

        // check dims
        equal = std::equal(std::begin(lhs.dims),
                std::begin(lhs.dims) + lhs.ndims, std::begin(rhs.dims));
        if (!equal) return false;

        // check layout information
        if (lhs.layout_type == layout_type::strided) {
            return std::equal(std::begin(lhs.layout.strides),
                    std::begin(lhs.layout.strides) + lhs.ndims,
                    std::begin(rhs.layout.strides));
        } else if (lhs.layout_type == layout_type::opaque) {
            return lhs.layout.layout_id == rhs.layout.layout_id;
        } else {
            return true;
        }
    }

    // a logical tensor is interpretable if it's strided or its layout id
    // is generated by this backend
    virtual bool is_interpretable(const impl::logical_tensor_t &lt) {
        auto ltw = impl::logical_tensor_wrapper(lt);
        return ltw.is_strided()
                || (ltw.is_opaque()
                        && (decode_layout_id(
                                    static_cast<size_t>(ltw.layout_id()))
                                        .second
                                == id_));
    }

    // a tensor tensor is interpretable if it's logical tesnor is interpretable
    virtual bool is_interpretable(const impl::tensor &in) {
        return is_interpretable(in.get_logical_tensor());
    }

    std::string name_;
    size_t id_;
};

class layout_id_manager {
public:
    using ptr = std::shared_ptr<layout_id_manager>;

    layout_id_manager(backend *owner_backend)
        : owning_backend_(owner_backend) {};
    virtual ~layout_id_manager() {}

    /*! \brief Set a backend memory descriptor to manager and get a 
    * corresponding llga layout id
    * \param mem_desc The backend's memory descriptor, it can
    * be both plain or opaque
    * \return a cache index, will be used as layout id
    * \note This function should be invoked in every where we want to
    * convert a md to layout id
    */
    virtual utils::optional<size_t> set_mem_desc(const utils::any &mem_desc) {
        std::lock_guard<std::mutex> lock(mem_descs_.m_);

        auto pos = std::find_if(mem_descs_.data_.begin(),
                mem_descs_.data_.end(), [&](const utils::any &m) -> bool {
                    return is_mem_desc_equal(m, mem_desc);
                });

        size_t layout_idx;
        if (pos != mem_descs_.data_.end()) {
            layout_idx = static_cast<size_t>(
                    std::distance(mem_descs_.data_.begin(), pos));
        } else {
            mem_descs_.data_.emplace_back(mem_desc);
            layout_idx = static_cast<size_t>(mem_descs_.data_.size() - 1);
        }

        return encode_layout_id(layout_idx, owning_backend_->get_id());
    }

    /*! \brief Get a backend memory descriptor from manager by using a 
    * layout id
    * \param layout_id The layout id, which is generated and managed 
    * by backends
    * \return When the input is a valid cache index, the return value
    * is a cached memory descriptor; otherwise, the return value will
    * be a utils::nullopt
    */
    virtual utils::optional<utils::any> get_mem_desc(
            const size_t &layout_id) const {
        // get out the backend_id and real idx from layout id
        auto decoded_layout_id = decode_layout_id(layout_id);
        size_t layout_idx = decoded_layout_id.first;
        size_t backend_id = decoded_layout_id.second;

        if (backend_id != owning_backend_->get_id()) return utils::nullopt;

        std::lock_guard<std::mutex> lock(mem_descs_.m_);

        if (layout_idx >= mem_descs_.data_.size()) return utils::nullopt;

        return mem_descs_.data_[layout_idx];
    }

private:
    /*! \brief compare two backend mem desc 
    * \param mem_desc1 
    * \param mem_desc2 
    * \return bool
    */
    virtual bool is_mem_desc_equal(
            const utils::any &mem_desc1, const utils::any &mem_desc2) const = 0;

    backend *owning_backend_;

    mutable struct {
        std::vector<utils::any> data_;
        mutable std::mutex m_;
    } mem_descs_;
};

class backend_manager {
    using backend_creator_f = backend::ptr (*)(std::string, size_t);

public:
    template <typename backend_class>
    static backend::ptr create_backend(std::string name, size_t backend_id) {
        return std::shared_ptr<backend_class>(
                new backend_class(name, backend_id));
    }

    static bool register_backend(std::string name, backend_creator_f fn) {
        return backend_manager::get()->register_backend_impl(name, fn);
    }

    static backend::ptr get_backend(std::string name) {
        return backend_manager::get()->get_backend_impl(name);
    }

    static backend::ptr get_backend(size_t layout_id) {
        return backend_manager::get()->get_backend_impl(layout_id);
    }

private:
    backend_manager() = default;
    backend_manager(const backend_manager &) = delete;
    backend_manager(backend_manager &&) = delete;
    backend_manager &operator=(const backend_manager &) = delete;
    backend_manager &operator=(backend_manager &&) = delete;

    static backend_manager *get() {
        static backend_manager inst;
        return &inst;
    }

    bool register_backend_impl(std::string name, backend_creator_f fn) {
        std::lock_guard<std::mutex> lock(m_);
        auto pos = backend_creators_.find(name);
        if (pos != backend_creators_.end()) return true; // have registered

        backend_creators_.insert({name, fn});
        return true;
    }

    backend::ptr get_backend_impl(std::string name) {
        std::lock_guard<std::mutex> lock(m_);

        // return created backend according to name
        auto backend_pos = names_to_backends_.find(name);
        if (backend_pos != names_to_backends_.end()) return backend_pos->second;

        // If there is no created backend according to this name,
        // create and store it
        auto creator_pos = backend_creators_.find(name);
        if (creator_pos != backend_creators_.end()) {
            if (backend_counter_ >= MAX_BACKEND_NUMS) {
                assertm(false,
                        "created backends number is greater than  "
                        "MAX_BACKEND_NUMS");
            }

            backend::ptr backend
                    = (creator_pos->second)(name, backend_counter_);
            names_to_backends_.insert({name, backend});
            ids_to_names_.insert({backend_counter_, name});
            backend_counter_++;

            return backend;
        }

        assertm(false, "backend name is not registered");
        return {};
    }

    backend::ptr get_backend_impl(size_t layout_id) {
        size_t backend_id = decode_layout_id(layout_id).second;

        std::lock_guard<std::mutex> lock(m_);

        // find the backend name according to backend_id
        auto backend_id_pos = ids_to_names_.find(backend_id);
        if (backend_id_pos == ids_to_names_.end()) return {};

        // return created backend according to name
        std::string name = backend_id_pos->second;
        auto backend_pos = names_to_backends_.find(name);
        if (backend_pos == names_to_backends_.end()) return {};

        return backend_pos->second;
    }

    std::mutex m_;
    size_t backend_counter_ {RESERVED_BACKEND_ID + 1};
    std::unordered_map<std::string, backend_creator_f> backend_creators_;
    std::unordered_map<std::string, backend::ptr> names_to_backends_;
    std::unordered_map<size_t, std::string> ids_to_names_;
};

// This macro is used to register a backend creator to backend manager
// The backend creator is a static template function
#define DNNL_GRAPH_REGISTER_BACKEND(name_, backend_class_) \
    static auto _flag_##name_##_ = backend_manager::register_backend( \
            #name_, &backend_manager::create_backend<backend_class_>)

} // namespace impl
} // namespace llga
#endif
