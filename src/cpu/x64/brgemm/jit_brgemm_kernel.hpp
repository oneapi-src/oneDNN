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

#ifndef JIT_BRGEMM_KERNEL_HPP
#define JIT_BRGEMM_KERNEL_HPP

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise_injector.hpp"

#include "cpu/x64/brgemm/brgemm_amx.hpp"
#include "cpu/x64/brgemm/brgemm_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_brgemm_kernel_base_t;

struct jit_brgemm_kernel_t {
    jit_brgemm_kernel_t(const brgemm_conf_t abrd);
    ~jit_brgemm_kernel_t();

    status_t create_kernel();
    void operator()(brgemm_kernel_params_t *) const;

    jit_brgemm_kernel_base_t *brgemm_kernel_ = nullptr;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_brgemm_kernel_t);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
