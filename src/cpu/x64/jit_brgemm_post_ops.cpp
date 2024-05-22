/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "cpu/x64/jit_brgemm_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_brgemm_kernel_post_ops_base_t *jit_brgemm_kernel_post_ops_base_t::create(
        cpu_isa_t isa, const brgemm_desc_t &abrg,
        const primitive_attr_t &aattr) {
    if (utils::one_of(isa, avx2, avx2_vnni, avx2_vnni_2)) {
        return new jit_brgemm_kernel_post_ops_t<Xbyak::Ymm>(abrg, aattr);
    } else {
        return new jit_brgemm_kernel_post_ops_t<Xbyak::Zmm>(abrg, aattr);
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
