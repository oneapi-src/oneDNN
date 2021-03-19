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
#ifndef BACKEND_FAKE_SINGLE_NODE_PASS_HPP
#define BACKEND_FAKE_SINGLE_NODE_PASS_HPP

#include <string>

#include "backend/fake/transformation_pass.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace fake_impl {
namespace pass {

using pattern = impl::pass::pattern;
using FCreatePattern = impl::pass::FCreatePattern;
using FCreateOptPattern = impl::pass::FCreateOptPattern;

FAKE_BACKEND_REGISTER_PASSES_DEF_BEGIN(single_node_pass)

#define FAKE_BACKEND_SINGLE_NODE_TRANSFORM(name, backend, p) \
    FAKE_BACKEND_REGISTER_TRANSFORMATION_PASS(backend, name).set_priority(p);

// register a wildward matched pass
FAKE_BACKEND_SINGLE_NODE_TRANSFORM(wildcard_match_pass, fake, 1.f)

#undef FAKE_BACKEND_SINGLE_NODE_TRANSFORM

FAKE_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace fake_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
