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

#ifndef LLGA_BACKEND_PASS_PASS_BACKEND_HPP
#define LLGA_BACKEND_PASS_PASS_BACKEND_HPP

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

namespace dnnl {
namespace graph {
namespace impl {
namespace pass {

/*!
 * \brief pass_backend provides backend for passes,
 *        each pass should have a valid backend.
 */
class pass_backend {
public:
    explicit pass_backend(const std::string &backend_name)
        : backend_name_(backend_name) {}
    virtual ~pass_backend() {}

    std::string get_backend_name() { return backend_name_; }

private:
    std::string backend_name_;
};

using pass_backend_ptr = std::shared_ptr<pass_backend>;

/*!
 * \brief pass_backend_registry is a registry class that
 *        is responsible for registering pass backends
 */
class pass_backend_registry {
public:
    // get a static pass_backend_registry instance
    static pass_backend_registry *get() {
        static pass_backend_registry reg_inst;
        return &reg_inst;
    }

    // register a backend
    pass_backend_ptr &register_backend(const std::string &name) {
        backend_map_[name] = std::make_shared<pass_backend>(name);
        return backend_map_[name];
    }

    // find whether a backend exists
    bool has_backend(const std::string &name) {
        return backend_map_.find(name) != backend_map_.end();
    }

    pass_backend_registry() = default;
    pass_backend_registry(const pass_backend_registry &) = delete;
    pass_backend_registry(pass_backend_registry &&) = delete;
    pass_backend_registry &operator=(const pass_backend_registry &) = delete;

private:
    std::unordered_map<std::string, pass_backend_ptr> backend_map_;
};

} // namespace pass
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
