/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_DEPTHWISE_INJECTOR_HPP
#define CPU_X64_JIT_UNI_DEPTHWISE_INJECTOR_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct jit_uni_depthwise_injector_f32 {
    jit_uni_depthwise_injector_f32(jit_generator* host, alg_kind_t depthwise_alg_, Xbyak::Opmask k_mask_ = Xbyak::Opmask(1))
        : h(host), depthwise_alg(depthwise_alg_), k_mask(k_mask_) {
        assert(utils::one_of(depthwise_alg, alg_kind::depthwise_scale_shift, alg_kind::depthwise_prelu));
    }

    void compute_vector_range(int start_idx, int end_idx, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias, bool is_broadcast = false);

private:
    jit_generator* h;

    using Vmm = typename utils::conditional3<isa == sse41, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    size_t vlen = cpu_isa_traits<isa>::vlen;

    alg_kind_t depthwise_alg;

    Vmm vmm_mask;
    Vmm vmm_aux0;

    Xbyak::Opmask k_mask;

    const static size_t preserved_vecs_max = 5;
    size_t vecs_to_preserve = 0;
    size_t vecs_count = isa == avx512_common ? 32 : 16;
    size_t preserved_vecs_count = 0;
    size_t preserved_vec_idxs[preserved_vecs_max] = {0};
    size_t start_idx_tail = 0;

    int aux_vecs_count(alg_kind_t elt_alg, bool is_broadcast);

    void compute_body(size_t start_idx, size_t end_idx, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias, bool is_broadcast = false);
    void injector_preamble(size_t start_idx, size_t end_idx, bool is_broadcast = false);
    void injector_preamble_tail(size_t start_idx, size_t end_idx);
    void injector_postamble();
    void assign_regs();

    void scale_shift_compute_vector(const Vmm &vmm_src, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias, bool is_broadcast = false);
    void prelu_compute_vector(const Vmm &vmm_src, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias, bool is_broadcast = false);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
