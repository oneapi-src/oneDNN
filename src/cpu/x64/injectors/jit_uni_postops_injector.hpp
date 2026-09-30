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
#ifndef CPU_X64_INJECTORS_JIT_UNI_POSTOPS_INJECTOR_HPP
#define CPU_X64_INJECTORS_JIT_UNI_POSTOPS_INJECTOR_HPP

#include <map>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_sum_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include <initializer_list>

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector {

int aux_vec_count(const post_ops_t &post_ops, cpu_isa_t isa, bool is_fwd);

// The main mechanism of handling various post-ops types. It utilizes internally
// specialized injectors to generate post-ops code to host primitive. Random
// order of post-ops is supported.
//
// The ISA the code is generated for is taken from the host generator, see
// `jit_generator_t::max_cpu_isa()`. `Vmm` remains a template parameter to keep
// the vector width the injector operates on statically typed.
template <typename Vmm>
class jit_uni_postops_injector_t {
public:
    /*
     * @param host <required> - user primitive where post-ops generated code is
     * injected
     * @param post_ops <required> - struct representing requested post-ops chain
     * @binary_static_params <reguired> - static params needed for binary_injector.
     * see: jit_uni_binary_injector.hpp for more info.
     * @param eltwise_static_params <optional> - allows user specify non default
     * params for eltwise_injector
     * @param inject_sum <required> - whether the injector applies the sum
     * post-op. Whether the chain holds a sum entry is visible in `post_ops`,
     * but whether this kernel wants the injector to act on it is not. Some
     * kernels apply sum manually and pass the entry along only so the rest of
     * the chain is handled. Those must pass `false`, otherwise the value is
     * summed twice. The argument is mandatory so that the intent is stated at
     * every call site. When set, the injector reads the previous value from the
     * address supplied per accumulator in `rhs_arg_dynamic_params_t`, so the
     * caller must provide it, as it does for binary post-ops.
     */
    jit_uni_postops_injector_t(jit_generator_t *host,
            const post_ops_t &post_ops,
            const binary_injector::static_params_t &binary_static_params,
            bool inject_sum);
    // This is the delegation target for the other constructor, since it owns
    // all construction state.
    jit_uni_postops_injector_t(jit_generator_t *host,
            const post_ops_t &post_ops,
            const binary_injector::static_params_t &binary_static_params,
            const eltwise_injector::static_params_t &eltwise_static_params,
            bool inject_sum);

    // Generates code of post_ops chain injected to host primitive. Applied to
    // ordered set of vector registers' indexes.
    // @rhs_arg_params: see jit_uni_binary_injector description
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params);
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs);

    // Generates code of post_ops chain injected to host primitive. Applied to
    // range <start_idx, end_idx) of vector registers' indexes.
    // @rhs_arg_params: see jit_uni_binary_injector description
    void compute_vector_range(int start_idx, int end_idx,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params);
    void compute_vector_range(int start_idx, int end_idx);

    // Generates code of post_ops chain injected to host primitive. Applied to
    // a single vector register index.
    // @rhs_arg_params: see jit_uni_binary_injector description
    void compute_vector(int idx,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params);
    void compute_vector(int idx);

    // Loads the scalar RHS of eligible binary post-ops once, as f32 in all
    // lanes, into registers taken in order from `vmm_idxs`. Passing the result
    // in rhs_arg_dynamic_params_t::preloaded_rhs makes compute_vector_range()
    // use these registers instead of memory. The registers in `vmm_idxs` must
    // be free.
    binary_injector::preloaded_rhs_t preload_scalar_vector_range(
            const injector_utils::vmm_index_set_t &vmm_idxs) const;

    // Thin wrapper for eltwise injector specific function
    void prepare_table(bool gen_table);

private:
    post_ops_t post_ops_;
    jit_generator_t *host_;
    // Key is a numerical order of a post-op in attributes.
    std::map<int, jit_uni_eltwise_injector_t<Vmm>> alg_to_eltwise_injector_;
    std::unique_ptr<binary_injector::jit_uni_binary_injector_t<Vmm>>
            binary_injector_;
    // Native sum injectors, one per sum post-op, keyed by post-op index same as
    // `alg_to_eltwise_injector_`. They contain a `binary_injector_` pointer but
    // do not own it. Keep `idx_to_sum_injector_` after the `binary_injector_`
    // field, just in case.
    std::map<int, jit_uni_sum_injector_t<Vmm>> idx_to_sum_injector_;
    // When unset, a sum entry is skipped because the kernel applies it itself.
    const bool inject_sum_;
};

enum post_op_type { sum = 0, eltwise, binary, prelu };

struct post_ops_ok_args_t {
    post_ops_ok_args_t(const cpu_isa_t isa,
            const std::vector<post_op_type> &accepted_post_op_types,
            const post_ops_t &post_ops, const memory_desc_wrapper *dst_d,
            const bool sum_at_pos_0_only, const bool sum_requires_scale_one,
            const bool sum_requires_zp_zero = true,
            const bool sum_requires_same_params = true,
            const bcast_set_t &enabled_bcast_strategy = default_strategies());

    const cpu_isa_t isa;
    const std::vector<post_op_type> &accepted_post_op_types;
    const post_ops_t &post_ops;
    const memory_desc_wrapper *dst_d;
    const bool sum_at_pos_0_only;
    const bool sum_requires_scale_one;
    const bool sum_requires_zp_zero;
    const bool sum_requires_same_params;
    const bcast_set_t enabled_bcast_strategy;
};

bool post_ops_ok(const post_ops_ok_args_t &args);

} // namespace injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
