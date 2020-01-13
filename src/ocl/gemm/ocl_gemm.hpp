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

#ifndef OCL_GEMM_HPP
#define OCL_GEMM_HPP

#include "common/c_types_map.hpp"
#include "ocl/gemm/ocl_gemm_exec_types.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct ocl_gemm_t : public primitive_impl_t {
    using primitive_impl_t::primitive_impl_t;
    virtual status_t execute(const gemm_exec_ctx_t &ctx) const = 0;
};

// XXX: Use this function with caution. There is no guarantee that prim_impl
// will be alive once primitive is destoyed.
template <typename T>
status_t get_primitive_impl(T **prim_impl, const primitive_t *p) {
    *prim_impl = dynamic_cast<T *>(p->get_primitive_impl().get());
    if (!*prim_impl) return status::runtime_error;
    return status::success;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif
