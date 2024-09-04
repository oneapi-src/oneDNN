/*******************************************************************************
 * Copyright 2021-2024 Intel Corporation
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

#include "graph/backend/fake/fake_backend.hpp"
#include "graph/backend/fake/single_op_pass.hpp"
#include "graph/backend/fake/transformation_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace fake_impl {

graph::pass::pass_registry_t fake_backend_t::register_passes() {
    graph::pass::pass_registry_t pass_registry;
    FAKE_BACKEND_REGISTER_PASSES_CALL(single_op_pass, pass_registry);
    return pass_registry;
}

graph::pass::pass_registry_t fake_backend_t::pass_registry_
        = fake_backend_t::register_passes();

} // namespace fake_impl

// This function must be called by backend_registry_t
status_t register_fake_backend() {
    const status_t ret = backend_registry_t::get_singleton().register_backend(
            &fake_impl::fake_backend_t::get_singleton());
    return ret;
}

} // namespace graph
} // namespace impl
} // namespace dnnl
