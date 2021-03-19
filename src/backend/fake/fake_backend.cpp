/*******************************************************************************
 * Copyright 2021 Intel Corporation
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

#include "utils/compatible.hpp"

#include "fake_backend.hpp"
#include "single_node_pass.hpp"
#include "transformation_pass.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace fake_impl {

fake_backend::fake_backend(const std::string &name, float priority)
    : backend(name, priority) {
    bool ret = register_passes();
    if (!ret) { throw std::runtime_error(name + " initialize failed"); }
}

bool fake_backend::register_passes() {
    FAKE_BACKEND_REGISTER_PASSES_CALL(single_node_pass, pass_registry_);
    return true;
}

DNNL_GRAPH_REGISTER_BACKEND(fake_backend::get_singleton())

} // namespace fake_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
