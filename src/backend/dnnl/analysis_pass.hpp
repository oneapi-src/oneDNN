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

#ifndef BACKEND_DNNL_ANALYSIS_PASS_HPP
#define BACKEND_DNNL_ANALYSIS_PASS_HPP

#include <string>
#include <utility>

#include "interface/pass_base.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

/*!
 * \brief analysis_pass provides analysis on a given graph,
 *        e.g. data type deduction, memory planning.
 */
class analysis_pass : public impl::pass::pass_base {
public:
    explicit analysis_pass(std::string pbackend, std::string pname)
        : impl::pass::pass_base(impl::pass::pass_type::kAnalysis,
                std::move(pbackend), std::move(pname)) {}
};

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
