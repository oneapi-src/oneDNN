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
#include <cassert>
#include <vector>
#include "common/verbose.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector {

#define VCHECK_PO_INJ_BOOL(cond, msg) \
    VCONDCHECK(primitive, create, check, postops_injector, cond, false, msg);

int aux_vec_count(const post_ops_t &post_ops, cpu_isa_t isa, bool is_fwd) {
    int res = 0;
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &post_op = post_ops.entry_[i];
        if (post_op.is_eltwise()) {
            // The count doesn't depend on the vector width, thus, the `Vmm`
            // argument is arbitrary.
            res = nstl::max(res,
                    jit_uni_eltwise_injector_t<Xbyak::Zmm>::aux_vecs_count(isa,
                            post_op.eltwise.alg, is_fwd,
                            post_op.eltwise.alpha));
        }
        // TODO: add support for other post-ops types. For now we assume that
        // other post operations do not use vectors implicitly.
    }
    return res;
}

template <typename Vmm>
jit_uni_postops_injector_t<Vmm>::jit_uni_postops_injector_t(
        jit_generator_t *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params,
        bool inject_sum)
    : post_ops_(post_ops)
    , host_(host)
    , binary_injector_(nullptr)
    , inject_sum_(inject_sum) {

    const auto &esp = eltwise_static_params;
    bool is_like_binary = false;
    bool is_eltwise = false;
    bool is_sum = false;

    for (int i = 0; i < post_ops.len(); i++) {
        const auto &post_op = post_ops.entry_[i];
        if (post_op.is_eltwise()) {
            is_eltwise = true;
            // Note: `dt` argument for eltwise injector is not propagated from
            // the top-level constructor due to lack of use cases till this
            // moment. Once the use case show up, add the argument to the
            // top-level ctor and propagate its value.
            alg_to_eltwise_injector_.emplace(i,
                    jit_uni_eltwise_injector_t<Vmm>(host_, post_op.eltwise,
                            data_type::f32, esp.save_state, esp.p_table_,
                            esp.k_mask_, esp.is_fwd, esp.use_dst,
                            esp.preserve_vmm, esp.preserve_p_table));
        } else if (post_op.is_like_binary()) {
            is_like_binary = true;
        } else if (post_op.is_sum(false, false)) {
            is_sum = true;
        }
    }

    if (is_superset(host->max_cpu_isa(), avx512_core) && is_eltwise
            && is_like_binary
            && binary_static_params.rhs_arg_static_params.tail_size)
        assert(eltwise_static_params.k_mask_
                != binary_static_params.rhs_arg_static_params.tail_opmask &&
                "Binary and prelu tail opmask should be different than eltwise \
                injector opmask. Otherwise eltwise injector will overwrite \
                binary tail opmask.");

    // The sum injector reads the previous value through the binary injector's
    // load path so the binary injector is created for a sum-only chain as well.
    if (is_like_binary || is_sum)
        binary_injector_ = utils::make_unique<
                binary_injector::jit_uni_binary_injector_t<Vmm>>(
                host, binary_static_params);

    // Build one sum injector per sum post-op when the caller asked for it.
    if (is_sum && inject_sum_) {
        const auto &rhs = binary_static_params.rhs_arg_static_params;
        const auto dst_dt = rhs.dst_d.data_type();
        for (int i = 0; i < post_ops.len(); i++) {
            const auto &post_op = post_ops.entry_[i];
            if (!post_op.is_sum(false, false)) continue;

            idx_to_sum_injector_.emplace(i,
                    jit_uni_sum_injector_t<Vmm>(host_, post_op.sum, dst_dt,
                            binary_injector_.get(), rhs.rhs_dt_helper_vmm_idx,
                            rhs.preserve_vmm_helper));
        }
    }
}

template <typename Vmm>
jit_uni_postops_injector_t<Vmm>::jit_uni_postops_injector_t(
        jit_generator_t *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        bool inject_sum)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
              eltwise_injector::static_params_t(), inject_sum) {}

template <typename Vmm>
void jit_uni_postops_injector_t<Vmm>::compute_vector_range(int start_idx,
        int end_idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {

    injector_utils::vmm_index_set_t vmm_idxs;
    for (int i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_params);
}

template <typename Vmm>
void jit_uni_postops_injector_t<Vmm>::compute_vector_range(
        int start_idx, int end_idx) {
    compute_vector_range(
            start_idx, end_idx, binary_injector::rhs_arg_dynamic_params_t());
}

template <typename Vmm>
void jit_uni_postops_injector_t<Vmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {

    int rhs_arg_idx = 0;
    for (int i = 0; i < post_ops_.len(); i++) {
        const auto &post_op = post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            alg_to_eltwise_injector_.at(i).compute_vector_range(vmm_idxs);
        } else if (post_op.is_like_binary()) {
            binary_injector_->compute_vector_range(
                    vmm_idxs, rhs_arg_idx, post_op, rhs_arg_params);
            ++rhs_arg_idx;
            // Ternary op handles two arguments at the same time, thus,
            // skipping one more.
            if (post_op.is_binary_with_ternary_op()) ++rhs_arg_idx;
        } else if (inject_sum_ && post_op.is_sum(false, false)) {
            idx_to_sum_injector_.at(i).compute_vector_range(
                    vmm_idxs, rhs_arg_params);
        }
        // A sum entry falls through when `inject_sum_` is unset. The kernel
        // applies it itself in that case.
    }
}
template <typename Vmm>
void jit_uni_postops_injector_t<Vmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    compute_vector_range(vmm_idxs, binary_injector::rhs_arg_dynamic_params_t());
}

template <typename Vmm>
void jit_uni_postops_injector_t<Vmm>::prepare_table(bool gen_table) {
    for (auto &alg_elt_inject : alg_to_eltwise_injector_)
        alg_elt_inject.second.prepare_table(gen_table);

    // Sum injectors emit their scale/zero-point constants here, similar to the
    // eltwise loop above.
    if (inject_sum_)
        for (auto &kv : idx_to_sum_injector_)
            kv.second.prepare_table(gen_table);
}

template <typename Vmm>
void jit_uni_postops_injector_t<Vmm>::compute_vector(int idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {
    compute_vector_range({idx}, rhs_arg_params);
}

template <typename Vmm>
void jit_uni_postops_injector_t<Vmm>::compute_vector(int idx) {
    compute_vector_range({idx});
}

template <typename Vmm>
binary_injector::preloaded_rhs_t
jit_uni_postops_injector_t<Vmm>::preload_scalar_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) const {
    binary_injector::preloaded_rhs_t preloaded_rhs;
    if (!binary_injector_) return preloaded_rhs;

    auto vmm_it = vmm_idxs.cbegin();
    int rhs_arg_idx = 0;
    for (int i = 0; i < post_ops_.len() && vmm_it != vmm_idxs.cend(); i++) {
        const auto &post_op = post_ops_.entry_[i];
        if (!post_op.is_like_binary()) continue;
        if (binary_injector_->preload_scalar_rhs(*vmm_it, rhs_arg_idx, post_op))
            preloaded_rhs.emplace(rhs_arg_idx, *vmm_it++);
        // Counts arguments the same way as compute_vector_range().
        rhs_arg_idx += post_op.is_binary_with_ternary_op() ? 2 : 1;
    }
    return preloaded_rhs;
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

    // Save scale and zero point of first sum postop in order to check that any
    // subsequent sum postops have the same values. Callers that apply sum with
    // a single set of constants ask for this check through
    // `sum_requires_same_params`.
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
                    // The sum post-op reads the destination back.
                    VCHECK_PO_INJ_BOOL(dst_d, VERBOSE_UNSUPPORTED_FORMAT_KIND);
                    VCHECK_PO_INJ_BOOL(
                            binary_injector::is_data_supported(isa,
                                    post_ops.get_sum_dt(
                                            dst_d->data_type(), idx)),
                            VERBOSE_ISA_DT_MISMATCH);
                    return true;
                case eltwise:
                    if (!entry.is_eltwise()) continue;
                    return eltwise_injector::is_supported(
                            isa, entry.eltwise.alg, data_type::f32);
                case binary:
                case prelu:
                    if (entry.is_like_binary()) {
                        VCHECK_PO_INJ_BOOL(
                                dst_d, VERBOSE_UNSUPPORTED_FORMAT_KIND);
                        VCHECK_PO_INJ_BOOL(
                                dst_d->md_->format_kind != format_kind::any,
                                VERBOSE_UNSUPPORTED_FORMAT_KIND);

                        bool ok = binary_injector::is_supported(isa,
                                binary_injector::get_src1_desc(entry, *dst_d),
                                *dst_d, enabled_bcast_strategy);
                        if (entry.is_binary_with_ternary_op()) {
                            const auto src2_d = binary_injector::get_src2_desc(
                                    entry, *dst_d);
                            VCHECK_PO_INJ_BOOL(
                                    binary_injector::is_data_supported(
                                            isa, src2_d.data_type),
                                    VERBOSE_ISA_DT_MISMATCH);
                            VCHECK_PO_INJ_BOOL(dst_d->is_dense(),
                                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
                            VCHECK_PO_INJ_BOOL(
                                    binary_injector::is_ternary_bcast_supported(
                                            src2_d, *dst_d),
                                    VERBOSE_UNSUPPORTED_POSTOP);
                        }
                        return ok;
                    }
                    break;
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

template class jit_uni_postops_injector_t<Xbyak::Zmm>;
template class jit_uni_postops_injector_t<Xbyak::Ymm>;

#undef VCHECK_PO_INJ_BOOL

} // namespace injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
