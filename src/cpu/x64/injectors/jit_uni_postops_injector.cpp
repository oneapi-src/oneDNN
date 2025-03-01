/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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
#include <cassert>
#include "common/verbose.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector {

#define VCHECK_PO_INJ_BOOL(cond, msg) \
    VCONDCHECK(primitive, create, check, postops_injector, cond, false, msg);

size_t aux_vec_count(const post_ops_t &post_ops, cpu_isa_t isa, bool is_fwd) {
    size_t res = 0;
#define CASE_ELTWISE_SUPERSET(_isa) \
    if (is_superset(isa, _isa)) { \
        res = nstl::max(res, \
                jit_uni_eltwise_injector<_isa>::aux_vecs_count( \
                        post_op.eltwise.alg, is_fwd, post_op.eltwise.alpha)); \
        continue; \
    }

    for (int i = 0; i < post_ops.len(); i++) {
        const auto &post_op = post_ops.entry_[i];
        if (post_op.is_eltwise()) {
            CASE_ELTWISE_SUPERSET(avx512_core);
            CASE_ELTWISE_SUPERSET(avx2);
            CASE_ELTWISE_SUPERSET(sse41);
        }
        // TODO: add support for other post-ops types. For now we assume that
        // other post operations do not use vectors implicitly.
    }
#undef CASE_ELTWISE_SUPERSET

    return res;
}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(
        jit_generator *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params,
        const lambda_jit_injectors_t &lambda_jit_injectors)
    : post_ops_(post_ops)
    , host_(host)
    , binary_injector_(nullptr)
    , lambda_jit_injectors_(lambda_jit_injectors) {

    const auto &esp = eltwise_static_params;
    bool is_like_binary = false;
    bool is_eltwise = false;

    for (int i = 0; i < post_ops.len(); i++) {
        const auto &post_op = post_ops.entry_[i];
        if (post_op.is_eltwise()) {
            is_eltwise = true;
            // Note: `dt` argument for eltwise injector is not propagated from
            // the top-level constructor due to lack of use cases till this
            // moment. Once the use case show up, add the argument to the
            // top-level ctor and propagate its value.
            alg_to_eltwise_injector_.emplace(i,
                    jit_uni_eltwise_injector<isa, Vmm>(host_, post_op.eltwise,
                            data_type::f32, esp.save_state, esp.p_table_,
                            esp.k_mask_, esp.is_fwd, esp.use_dst,
                            esp.preserve_vmm, esp.preserve_p_table));
        } else if (post_op.is_like_binary()) {
            is_like_binary = true;
        }
    }

    if (is_superset(isa, avx512_core) && is_eltwise && is_like_binary
            && binary_static_params.rhs_arg_static_params.tail_size)
        assert(eltwise_static_params.k_mask_
                != binary_static_params.rhs_arg_static_params.tail_opmask &&
                "Binary and prelu tail opmask should be different than eltwise \
                injector opmask. Otherwise eltwise injector will overwrite \
                binary tail opmask.");

    if (is_like_binary)
        binary_injector_ = utils::make_unique<
                binary_injector::jit_uni_binary_injector_t<isa, Vmm>>(
                host, binary_static_params);
}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(
        jit_generator *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
            eltwise_injector::static_params_t(), lambda_jit_injectors_t()) {}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(
        jit_generator *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const lambda_jit_injectors_t &lambda_jit_injectors)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
            eltwise_injector::static_params_t(), lambda_jit_injectors) {}

template <cpu_isa_t isa, typename Vmm>
jit_uni_postops_injector_t<isa, Vmm>::jit_uni_postops_injector_t(
        jit_generator *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
            eltwise_static_params, lambda_jit_injectors_t()) {}

// Specialization instantiations are needed to avoid instantiating ISA with
// Vmm that don't make any sense like sse41 + Zmm.
template <>
jit_uni_postops_injector_base_t<Xbyak::Zmm> *
jit_uni_postops_injector_base_t<Xbyak::Zmm>::create(jit_generator *host,
        cpu_isa_t isa, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params) {

// Exact match case goes first and required to force `isa` passed by user.
#define CASE_EXACT_MATCH(_isa) \
    if (isa == (_isa)) \
        return new jit_uni_postops_injector_t<_isa, Xbyak::Zmm>( \
                host, post_ops, binary_static_params, eltwise_static_params);

    CASE_EXACT_MATCH(avx512_core_fp16);
    CASE_EXACT_MATCH(avx512_core_bf16);
    CASE_EXACT_MATCH(avx512_core);

#undef CASE_EXACT_MATCH

// When there's no exact match, pick up what's allowed through mayiuse since
// not every ISA has instances in post-ops injector.
#define CASE_MAYIUSE(_isa) \
    if (mayiuse(_isa)) \
        return new jit_uni_postops_injector_t<_isa, Xbyak::Zmm>( \
                host, post_ops, binary_static_params, eltwise_static_params);

    CASE_MAYIUSE(avx512_core_fp16);
    CASE_MAYIUSE(avx512_core_bf16);
    CASE_MAYIUSE(avx512_core);

#undef CASE_MAYIUSE

    assert(!"Kernel is empty!");
    return nullptr;
}

template <>
jit_uni_postops_injector_base_t<Xbyak::Ymm> *
jit_uni_postops_injector_base_t<Xbyak::Ymm>::create(jit_generator *host,
        cpu_isa_t isa, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params) {

// Exact match case goes first and required to force `isa` passed by user.
#define CASE_EXACT_MATCH(_isa) \
    if (isa == (_isa)) \
        return new jit_uni_postops_injector_t<_isa, Xbyak::Ymm>( \
                host, post_ops, binary_static_params, eltwise_static_params);

    CASE_EXACT_MATCH(avx512_core_fp16);
    CASE_EXACT_MATCH(avx512_core);
    CASE_EXACT_MATCH(avx2_vnni_2);
    CASE_EXACT_MATCH(avx2);
    CASE_EXACT_MATCH(avx);

#undef CASE_EXACT_MATCH

// When there's no exact match, pick up what's allowed through mayiuse since
// not every ISA has instances in post-ops injector.
#define CASE_MAYIUSE(_isa) \
    if (mayiuse(_isa)) \
        return new jit_uni_postops_injector_t<_isa, Xbyak::Ymm>( \
                host, post_ops, binary_static_params, eltwise_static_params);

    CASE_MAYIUSE(avx512_core_fp16);
    CASE_MAYIUSE(avx512_core);
    CASE_MAYIUSE(avx2_vnni_2);
    CASE_MAYIUSE(avx2);
    CASE_MAYIUSE(avx);

#undef CASE_MAYIUSE

    assert(!"Kernel is empty!");
    return nullptr;
}

template <>
jit_uni_postops_injector_base_t<Xbyak::Xmm> *
jit_uni_postops_injector_base_t<Xbyak::Xmm>::create(jit_generator *host,
        cpu_isa_t isa, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params) {

// Exact match case goes first and required to force `isa` passed by user.
#define CASE_EXACT_MATCH(_isa) \
    if (isa == (_isa)) \
        return new jit_uni_postops_injector_t<_isa, Xbyak::Xmm>( \
                host, post_ops, binary_static_params, eltwise_static_params);

    CASE_EXACT_MATCH(avx512_core_fp16);
    CASE_EXACT_MATCH(avx512_core);
    CASE_EXACT_MATCH(avx2_vnni_2);
    CASE_EXACT_MATCH(avx2);
    CASE_EXACT_MATCH(avx);
    CASE_EXACT_MATCH(sse41);

#undef CASE_EXACT_MATCH

// When there's no exact match, pick up what's allowed through mayiuse since
// not every ISA has instances in post-ops injector.
#define CASE_MAYIUSE(_isa) \
    if (mayiuse(_isa)) \
        return new jit_uni_postops_injector_t<_isa, Xbyak::Xmm>( \
                host, post_ops, binary_static_params, eltwise_static_params);

    CASE_MAYIUSE(avx512_core_fp16);
    CASE_MAYIUSE(avx512_core);
    CASE_MAYIUSE(avx2_vnni_2);
    CASE_MAYIUSE(avx2);
    CASE_MAYIUSE(avx);
    CASE_MAYIUSE(sse41);

#undef CASE_MAYIUSE

    assert(!"Kernel is empty!");
    return nullptr;
}

template <typename Vmm>
jit_uni_postops_injector_base_t<Vmm> *
jit_uni_postops_injector_base_t<Vmm>::create(jit_generator *host, cpu_isa_t isa,
        const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params) {
    const eltwise_injector::static_params_t eltwise_static_params;
    return create(
            host, isa, post_ops, binary_static_params, eltwise_static_params);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector_range(
        size_t start_idx, size_t end_idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {

    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_params);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    compute_vector_range(
            start_idx, end_idx, binary_injector::rhs_arg_dynamic_params_t());
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {

    std::size_t rhs_arg_idx = 0;
    for (int i = 0; i < post_ops_.len(); i++) {
        const auto &post_op = post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            alg_to_eltwise_injector_.at(i).compute_vector_range(vmm_idxs);
        } else if (post_op.is_like_binary()) {
            binary_injector_->compute_vector_range(
                    vmm_idxs, rhs_arg_idx, post_op, rhs_arg_params);
            ++rhs_arg_idx;
        } else {
            const auto lam = lambda_jit_injectors_.find(post_op.kind);
            if (lam != lambda_jit_injectors_.end()) lam->second();
        }
    }
}
template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    compute_vector_range(vmm_idxs, binary_injector::rhs_arg_dynamic_params_t());
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::prepare_table(bool gen_table) {
    for (auto &alg_elt_inject : alg_to_eltwise_injector_)
        alg_elt_inject.second.prepare_table(gen_table);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector(size_t idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {
    compute_vector_range({idx}, rhs_arg_params);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::compute_vector(size_t idx) {
    compute_vector_range({idx});
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_postops_injector_t<isa, Vmm>::set_lambda_injector(
        dnnl_primitive_kind_t kind, const std::function<void()> &jit_injector) {
    lambda_jit_injectors_[kind] = jit_injector;
}

post_ops_ok_args_t::post_ops_ok_args_t(const cpu_isa_t isa,
        const std::vector<post_op_type> &accepted_post_op_types,
        const post_ops_t &post_ops, const memory_desc_wrapper *dst_d,
        const bool sum_at_pos_0_only, const bool sum_requires_scale_one,
        const bool sum_requires_zp_zero, const bool sum_requires_same_params,
        const bcast_set_t &enabled_bcast_strategy)
    : isa(isa)
    , accepted_post_op_types(accepted_post_op_types)
    , post_ops(post_ops)
    , dst_d(dst_d)
    , sum_at_pos_0_only(sum_at_pos_0_only)
    , sum_requires_scale_one(sum_requires_scale_one)
    , sum_requires_zp_zero(sum_requires_zp_zero)
    , sum_requires_same_params(sum_requires_same_params)
    , enabled_bcast_strategy(enabled_bcast_strategy) {};

bool post_ops_ok(const post_ops_ok_args_t &post_ops_ok_args) {
    const cpu_isa_t isa = post_ops_ok_args.isa;
    const std::vector<post_op_type> &accepted_post_op_types
            = post_ops_ok_args.accepted_post_op_types;
    const post_ops_t &post_ops = post_ops_ok_args.post_ops;
    const memory_desc_wrapper *dst_d = post_ops_ok_args.dst_d;
    const bool sum_at_pos_0_only = post_ops_ok_args.sum_at_pos_0_only;
    const bool sum_requires_scale_one = post_ops_ok_args.sum_requires_scale_one;
    const bool sum_requires_zp_zero = post_ops_ok_args.sum_requires_zp_zero;
    const bool sum_requires_same_params
            = post_ops_ok_args.sum_requires_same_params;
    const auto &enabled_bcast_strategy
            = post_ops_ok_args.enabled_bcast_strategy;

    VCHECK_PO_INJ_BOOL(dst_d && dst_d->md_->format_kind != dnnl_format_kind_any,
            VERBOSE_UNSUPPORTED_FORMAT_KIND);

    // Save scale and zero point of first sum postop in order to check that any
    // subsequent sum postops have the same values. This check is necessary
    // because there is only one lambda injector.
    const auto sum_idx = post_ops.find(primitive_kind::sum);
    const bool with_sum = sum_idx != -1;
    const auto &entry
            = with_sum ? post_ops.entry_[sum_idx] : dnnl_post_ops::entry_t();
    const auto sum_scale = with_sum ? entry.sum.scale : 0;
    const auto sum_zero_point = with_sum ? entry.sum.zero_point : 0;

    const auto is_accepted_postop = [&](const int idx) {
        for (const auto &post_op : accepted_post_op_types) {
            const auto &entry = post_ops.entry_[idx];
            // Note: check for post-op kinds is needed as `post_op` value
            // represents all supported but not only passed kinds.
            switch (post_op) {
                case sum:
                    if (!entry.is_sum(false, false)) continue;
                    if (sum_requires_same_params) {
                        VCHECK_PO_INJ_BOOL(entry.sum.scale == sum_scale,
                                "Unsupported sum scale value");
                        VCHECK_PO_INJ_BOOL(
                                entry.sum.zero_point == sum_zero_point,
                                "Unsupported sum zero-point value");
                    }
                    if (sum_requires_scale_one) {
                        VCHECK_PO_INJ_BOOL(entry.sum.scale == 1.f,
                                "Unsupported sum scale value");
                    }
                    if (sum_requires_zp_zero) {
                        VCHECK_PO_INJ_BOOL(entry.sum.zero_point == 0,
                                "Unsupported sum zero-point value");
                    }
                    VCHECK_PO_INJ_BOOL(IMPLICATION(sum_at_pos_0_only, idx == 0),
                            "Unsupported sum position in post-ops");
                    return true;
                case eltwise:
                    if (!entry.is_eltwise()) continue;
                    return eltwise_injector::is_supported(
                            isa, entry.eltwise.alg, data_type::f32);
                case binary:
                case prelu:
                    if (!entry.is_like_binary()) continue;
                    assert(dst_d && "dst_d is null");
                    return binary_injector::is_supported(isa,
                            binary_injector::get_src1_desc(entry, *dst_d),
                            *dst_d, enabled_bcast_strategy);
                default: assert(!"Unhandled post_op type");
            }
        }
        return false;
    };

    for (int i = 0; i < post_ops.len(); i++) {
        if (!is_accepted_postop(i)) return false;
    }

    return true;
}

template class jit_uni_postops_injector_t<avx512_core_fp16, Xbyak::Zmm>;
template class jit_uni_postops_injector_t<avx512_core_fp16, Xbyak::Ymm>;
template class jit_uni_postops_injector_t<avx512_core_fp16, Xbyak::Xmm>;
template class jit_uni_postops_injector_t<avx512_core_bf16, Xbyak::Zmm>;
template class jit_uni_postops_injector_t<avx512_core, Xbyak::Zmm>;
template class jit_uni_postops_injector_t<avx512_core, Xbyak::Ymm>;
template class jit_uni_postops_injector_t<avx512_core, Xbyak::Xmm>;
template class jit_uni_postops_injector_t<avx2_vnni_2, Xbyak::Ymm>;
template class jit_uni_postops_injector_t<avx2_vnni_2, Xbyak::Xmm>;
template class jit_uni_postops_injector_t<avx2, Xbyak::Ymm>;
template class jit_uni_postops_injector_t<avx2, Xbyak::Xmm>;
template class jit_uni_postops_injector_t<avx, Xbyak::Ymm>;
template class jit_uni_postops_injector_t<avx, Xbyak::Xmm>;
template class jit_uni_postops_injector_t<sse41, Xbyak::Xmm>;

template class jit_uni_postops_injector_base_t<Xbyak::Zmm>;
template class jit_uni_postops_injector_base_t<Xbyak::Ymm>;
template class jit_uni_postops_injector_base_t<Xbyak::Xmm>;

#undef VCHECK_PO_INJ_BOOL

} // namespace injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
