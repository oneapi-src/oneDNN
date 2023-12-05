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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_STATICS_TABLE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_STATICS_TABLE_HPP
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <util/def.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace runtime {
struct engine_t;
}

struct cached_const_graph_tensor;

// Manages the ownership of the internal data buffer
struct SC_API aligned_buffer_t {
    void *data_;
    // the size of the data in bytes
    size_t size_;
    runtime::engine_t *engine_;

    // creates a buffer with given size in bytes
    aligned_buffer_t(size_t size, runtime::engine_t *engine);

    aligned_buffer_t(aligned_buffer_t &&other);
    aligned_buffer_t() : data_(nullptr), size_(0), engine_(nullptr) {}
    ~aligned_buffer_t();
};

// The aligned buffer used in compiler using default runtime stream. It is a
// segment of statically allocated data for a module (e.g. for a single
// initialized global tensor). In contrast, stack/FILO memory pool are
// dynamically allocated.
struct SC_API static_data_t : public aligned_buffer_t {
    using aligned_buffer_t::aligned_buffer_t;

    /**
     * Creates a buffer and copies an existing buffer to it
     * @param indata the existing buffer. Does not take the ownership of it
     * @param size the size of the new buffer and the existing buffer
     * */
    static_data_t(const void *indata, size_t size);
    template <typename T,
            typename Dummy
            = typename std::enable_if<!std::is_same<T, bool>::value>>
    static_data_t(const std::vector<T> &indata)
        : static_data_t((void *)indata.data(), indata.size() * sizeof(T)) {}
};

// the table to hold the statically allocated data for a jit_module/ir_module,
// or for multiple linked modules
// memory layout:
// ------------ start of the buffer
// | global vars            |
// ------------
// | initialized tensors    |
// ------------ end of initialized section
// | uninitialized tensors  |
// ------------ end of the buffer
struct statics_table_t {
    std::unordered_map<std::string, size_t> impl_;
    aligned_buffer_t data_;
    // the size of initialized section
    size_t initialized_size_;
    std::vector<std::shared_ptr<cached_const_graph_tensor>> shared_tensors_;
    std::vector<std::shared_ptr<void>> device_kernels_;
    // gets the data by name, will throw an exception if name not found
    void *get(const std::string &name) const;
    // gets the data by name, returns null if name not found instead of throwing
    void *get_or_null(const std::string &name) const;
    // adds and copies the static_data_t to the table
    void add(const std::string &name, size_t offset);
    statics_table_t(aligned_buffer_t &&data)
        : data_(std::move(data)), initialized_size_(0) {}
    statics_table_t(statics_table_t &&other) = default;
    statics_table_t() : initialized_size_(0) {}
    // save the buffer size and the contents of initialized section
    SC_INTERNAL_API void save_to_file(const std::string &path) const;
    // load the saved statics_table from file
    SC_INTERNAL_API static statics_table_t load_from_file(
            const std::string &path);
    statics_table_t copy() const;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
