/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include <algorithm>
#include <cmath>

#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace binary_injector {

bool is_data_supported(cpu_isa_t isa, data_type_t data_type) {
    return IMPLICATION(data_type == data_type::bf16,
            utils::one_of(isa, avx512_core_bf16, avx512_core));
}

bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
            src1_desc, dst_d, supported_strategy_set);
    return bcast_type != broadcasting_strategy_t::unsupported;
}

bool is_supported(cpu_isa_t isa, const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    return is_data_supported(isa, src1_desc.data_type)
            && is_bcast_supported(src1_desc, dst_d, supported_strategy_set);
}

bool binary_args_broadcast_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d,
                            supported_strategy_set);
                    return bcast_type == broadcasting_strategy_t::unsupported;
                }
                return false;
            });
}

bool binary_args_tail_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d, int vlen,
        const bcast_set_t &supported_strategy_set) {
    const auto channels = dst_d.dims()[1];
    const int vmm_l_len = vlen / 4;

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    return utils::one_of(bcast_type,
                                   broadcasting_strategy_t::per_oc,
                                   broadcasting_strategy_t::per_oc_spatial)
                            && (channels % vmm_l_len != 0);
                }
                return false;
            });
}

bool binary_args_matches_tag(format_tag_t tag, const post_ops_t &post_ops) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) {
                if (entry.is_binary()) {
                    const memory_desc_wrapper rhs_arg_d(entry.binary.src1_desc);
                    return rhs_arg_d.matches_tag(tag);
                }
                return true;
            });
}

bool any_binary_postop_rhs_per_oc_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    return bcast_type == broadcasting_strategy_t::per_oc
                            || bcast_type
                            == broadcasting_strategy_t::per_oc_spatial;
                }
                return false;
            });
}

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const std::function<bool(const memory_desc_wrapper &)> predicate) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    if (bcast_type == broadcasting_strategy_t::per_oc
                            || bcast_type
                                    == broadcasting_strategy_t::per_oc_spatial)
                        return predicate(
                                memory_desc_wrapper(entry.binary.src1_desc));
                }
                return true;
            });
}

static_params_t::static_params_t(const Xbyak::Reg64 &param1,
        const bcast_set_t &supported_strategy_set,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : param1(param1)
    , supported_strategy_set(supported_strategy_set)
    , rhs_arg_static_params(rhs_arg_static_params) {}

static_params_t::static_params_t(const Xbyak::Reg64 &param1,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : static_params_t(param1,
            bcast_set_t {broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::per_oc_spatial,
                    broadcasting_strategy_t::no_broadcast},
            rhs_arg_static_params) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx, const Xbyak::Reg64 &rhs_addr_reg,
        const Xbyak::Reg64 &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
            rhs_helper_reg, preserve_gpr_helpers, preserve_vmm_helper,
            abi_param_offset, dst_d, tail_size, Xbyak::Opmask(2),
            use_exact_tail_scalar_bcast, false /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx, const Xbyak::Reg64 &rhs_addr_reg,
        const Xbyak::Reg64 &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak::Opmask &tail_opmask, bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
            rhs_helper_reg, preserve_gpr_helpers, preserve_vmm_helper,
            abi_param_offset, dst_d, tail_size, tail_opmask,
            use_exact_tail_scalar_bcast, true /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx, const Xbyak::Reg64 &rhs_addr_reg,
        const Xbyak::Reg64 &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak::Opmask &tail_opmask, bool use_exact_tail_scalar_bcast,
        bool is_opmask_set)
    : rhs_dt_helper_vmm_idx(rhs_dt_helper_vmm_idx)
    , rhs_addr_reg(rhs_addr_reg)
    , rhs_helper_reg(rhs_helper_reg)
    , preserve_gpr_helpers(preserve_gpr_helpers)
    , preserve_vmm_helper(preserve_vmm_helper)
    , abi_param_offset(abi_param_offset)
    , dst_d(dst_d)
    , tail_size(tail_size)
    , tail_opmask(tail_opmask)
    , use_exact_tail_scalar_bcast(use_exact_tail_scalar_bcast)
    , is_opmask_set_(is_opmask_set) {}

template <cpu_isa_t isa>
jit_uni_binary_injector_t<isa>::jit_uni_binary_injector_t(
        jit_generator *host, const static_params_t &static_params)
    : host_(host)
    , rhs_arg_static_params_(static_params.rhs_arg_static_params)
    , param1_(static_params.param1)
    , supported_strategy_set_(static_params.supported_strategy_set) {}

template <typename ParamsMap>
static bool params_differ(ParamsMap &params,
        const typename ParamsMap::key_type key1,
        const typename ParamsMap::key_type key2) {
    const auto &it1 = params.find(key1);
    const auto &it2 = params.find(key2);
    if (utils::one_of(params.end(), it1, it2)) return it1 != it2;
    return it1->second != it2->second;
}

static bool rhs_arg_params_differ(size_t vmm_idx1, size_t vmm_idx2,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        broadcasting_strategy_t rhs_broadcasting_strategy) {

    const auto &out_elem_off_addr = rhs_arg_params.vmm_idx_to_out_elem_off_addr;
    const auto &out_elem_off_val = rhs_arg_params.vmm_idx_to_out_elem_off_val;
    const auto &out_off_oprnd = rhs_arg_params.vmm_idx_to_out_off_oprnd;
    const auto &oc_off_addr = rhs_arg_params.vmm_idx_to_oc_elem_off_addr;
    const auto &oc_off_val = rhs_arg_params.vmm_idx_to_oc_elem_off_val;
    const auto &oc_off_oprnd = rhs_arg_params.vmm_idx_to_oc_off_oprnd;

    if (rhs_broadcasting_strategy == broadcasting_strategy_t::scalar) {
        return false;
    } else if (rhs_broadcasting_strategy
            == broadcasting_strategy_t::no_broadcast) {
        return params_differ(out_elem_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_val, vmm_idx1, vmm_idx2)
                || params_differ(out_off_oprnd, vmm_idx1, vmm_idx2);
    } else if (rhs_broadcasting_strategy == broadcasting_strategy_t::per_oc
            || rhs_broadcasting_strategy
                    == broadcasting_strategy_t::per_oc_spatial) {
        return params_differ(oc_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_val, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_oprnd, vmm_idx1, vmm_idx2);
    }
    return true;
}

template <cpu_isa_t isa>
int jit_uni_binary_injector_t<isa>::adjust_temp_vmm_hint(
        int user_hint, int start_idx, int end_idx, int max_vmm_idx) const {
    const bool user_hint_in_vector_range
            = user_hint >= start_idx && user_hint <= end_idx;
    const bool user_hint_exceeded_limit = user_hint > max_vmm_idx;
    const bool user_hint_invalid
            = user_hint_in_vector_range || user_hint_exceeded_limit;

    if (user_hint_invalid) {
        const bool max_vmm_idx_in_vector_range
                = max_vmm_idx >= start_idx && max_vmm_idx <= end_idx;

        if (max_vmm_idx_in_vector_range || user_hint_exceeded_limit
                || user_hint == max_vmm_idx)
            return 0;
        else
            return max_vmm_idx;
    }

    return user_hint;
}

template <typename Vmm>
static void push_vmm(jit_generator *host, const Vmm &vmm) {
    host->sub(host->rsp, injector_utils::vmm_size_t<Vmm>::bytes);
    host->uni_vmovups(host->ptr[host->rsp], vmm);
}

template <typename Vmm>
static void pop_vmm(jit_generator *host, const Vmm &vmm) {
    host->uni_vmovups(vmm, host->ptr[host->rsp]);
    host->add(host->rsp, injector_utils::vmm_size_t<Vmm>::bytes);
}

static void push_opmask(jit_generator *host, const Xbyak::Opmask &k) {
    static constexpr int k_mask_size = 8;
    host->sub(host->rsp, k_mask_size);
    if (mayiuse(avx512_core))
        host->kmovq(host->ptr[host->rsp], k);
    else
        host->kmovw(host->ptr[host->rsp], k);
}

static void pop_opmask(jit_generator *host, const Xbyak::Opmask &k) {
    static constexpr int k_mask_size = 8;
    if (mayiuse(avx512_core))
        host->kmovq(k, host->ptr[host->rsp]);
    else
        host->kmovw(k, host->ptr[host->rsp]);
    host->add(host->rsp, k_mask_size);
}

template <typename Vmm>
static void restore_stack(jit_generator *host, const Vmm &vmm) {
    host->add(host->rsp, injector_utils::vmm_size_t<Vmm>::bytes);
}

template <cpu_isa_t isa>
std::pair<bool, int> jit_uni_binary_injector_t<isa>::should_preserve_vmm(
        int curr_idx, int vmm_hint, int max_vmm_idx,
        bool dt_helper_vmm_needed) const {
    if (dt_helper_vmm_needed && vmm_hint == curr_idx) {
        if (curr_idx == 0)
            return std::make_pair(true, max_vmm_idx);
        else
            return std::make_pair(true, 0);
    }
    return std::make_pair(false, vmm_hint);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_idx, post_op, rhs_arg_params);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {

    if (vmm_idxs.empty()) return;
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin());

    // Phase 1 Validate temporary vmm user hint
    static constexpr int max_vmm_idx = cpu_isa_traits<isa>::n_vregs - 1;
    auto &vmm_hint = rhs_arg_static_params_.rhs_dt_helper_vmm_idx;
    vmm_hint = adjust_temp_vmm_hint(vmm_hint, start_idx, end_idx, max_vmm_idx);

    const auto rhs_broadcasting_strategy
            = get_rhs_arg_broadcasting_strategy(post_op.binary.src1_desc,
                    rhs_arg_static_params_.dst_d, supported_strategy_set_);
    const auto rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    const auto &vmm_tail_idx = rhs_arg_params.vmm_tail_idx_;
    const bool tail_exists_in_range = !vmm_tail_idx.empty();
    const bool bcast_f32_non_avx512 = !is_avx512_
            && utils::one_of(rhs_broadcasting_strategy,
                    broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::per_oc_spatial)
            && rhs_arg_data_type == data_type::f32;
    const bool should_preserve_vmm_tail = tail_exists_in_range
            && (!is_avx512_
                    || !utils::one_of(rhs_broadcasting_strategy,
                            broadcasting_strategy_t::scalar,
                            broadcasting_strategy_t::per_oc_spatial)
                    || rhs_arg_data_type != data_type::f32);
    const bool dt_helper_vmm_needed
            = !binary_op_with_unaligned_mem_operand_allowed_
            || rhs_arg_data_type != data_type::f32 || bcast_f32_non_avx512
            || should_preserve_vmm_tail;

    // Phase 2 Protect temporary registers content.
    const injector_utils::register_preserve_guard_t register_guard {host_,
            (rhs_arg_static_params_.preserve_gpr_helpers
                            ? std::initializer_list<Xbyak::Reg64>(
                                    {rhs_arg_static_params_.rhs_addr_reg,
                                            rhs_arg_static_params_
                                                    .rhs_helper_reg})
                            : std::initializer_list<Xbyak::Reg64>()),
            (rhs_arg_static_params_.preserve_vmm_helper && dt_helper_vmm_needed
                            ? std::initializer_list<Xbyak::Xmm>({Vmm(vmm_hint)})
                            : std::initializer_list<Xbyak::Xmm>())};

    bool vmm0_was_preserved = false;
    static const Vmm zero_vmm(0);

    Xbyak::Address rhs_arg_addr(0);

    // Phase 3 Apply binary post-op over all vmms.
    for (const auto vmm_idx : vmm_idxs) {
        if (vmm_idx == start_idx
                || rhs_arg_params_differ(vmm_idx, vmm_idx - 1, rhs_arg_params,
                        rhs_broadcasting_strategy)) {
            rhs_arg_addr = prepare_rhs_arg_addr(vmm_idx, rhs_arg_idx, post_op,
                    rhs_arg_params, rhs_broadcasting_strategy);
        }

        const auto local_vmm_preservation = should_preserve_vmm(
                vmm_idx, vmm_hint, max_vmm_idx, dt_helper_vmm_needed);
        const bool &vmm_preservation_needed = local_vmm_preservation.first;
        const Vmm dst_vmm(vmm_idx);
        const bool with_tail = rhs_arg_static_params_.tail_size
                && vmm_tail_idx.find(vmm_idx) != vmm_tail_idx.cend()
                && IMPLICATION(rhs_broadcasting_strategy
                                == broadcasting_strategy_t::scalar,
                        rhs_arg_static_params_.use_exact_tail_scalar_bcast);

        if (vmm_preservation_needed) {
            const Vmm vmm_to_preserve(local_vmm_preservation.second);
            push_vmm(host_, vmm_to_preserve);
            inject_binary(post_op, dst_vmm, rhs_arg_addr, with_tail);
            pop_vmm(host_, vmm_to_preserve);
            // in case all Vmm are occupied, Vmm(0) is chosen for tmp by default,
            // so it's content needs to be preserved...

            push_vmm(host_, zero_vmm);
            vmm0_was_preserved = true;
        } else
            inject_binary(post_op, dst_vmm, rhs_arg_addr, with_tail);
    }
    // ...and restored afterwards
    if (vmm0_was_preserved) pop_vmm(host_, zero_vmm);
}

template <cpu_isa_t isa>
Xbyak::Address jit_uni_binary_injector_t<isa>::prepare_rhs_arg_addr(
        std::size_t vmm_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        const broadcasting_strategy_t rhs_broadcasting_strategy) const {

    static constexpr auto rhs_arg_ptr_size = sizeof(const void *);
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    const auto &abi_param_offset = rhs_arg_static_params_.abi_param_offset;
    const auto &rhs_helper_reg = rhs_arg_static_params_.rhs_helper_reg;
    const auto rhs_arg_elem_size
            = types::data_type_size(post_op.binary.src1_desc.data_type);

    host_->mov(rhs_addr_reg, host_->ptr[param1_ + abi_param_offset]);
    host_->mov(rhs_addr_reg,
            host_->ptr[rhs_addr_reg + rhs_arg_idx * rhs_arg_ptr_size]);

    switch (rhs_broadcasting_strategy) {
        case broadcasting_strategy_t::scalar: return host_->ptr_b[rhs_addr_reg];
        case broadcasting_strategy_t::no_broadcast: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_out_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_out_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_out_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);

            return host_->ptr[rhs_addr_reg];
        }
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_oc_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_oc_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_oc_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);

            return rhs_broadcasting_strategy
                            == broadcasting_strategy_t::per_oc_spatial
                    ? host_->ptr_b[rhs_addr_reg]
                    : host_->ptr[rhs_addr_reg];
        }
        default: assert(false && "Broadcasting type not supported");
    }

    return host_->ptr[rhs_addr_reg];
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_from_operand(
        const std::map<int, Xbyak::Operand> &vmm_idx_to_elem_operand_off,
        int vmm_idx, const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_operand_off = vmm_idx_to_elem_operand_off.find(vmm_idx);
    if (it_operand_off != vmm_idx_to_elem_operand_off.end()) {
        if (elem_size_bytes == 1) {
            host_->add(addr_reg, it_operand_off->second);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->mov(tmp_reg, it_operand_off->second);
            host_->sal(tmp_reg, shift_val);
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_under_mem_addr(
        const std::map<int, Xbyak::Address> &vmm_idx_to_elem_addr_off,
        int vmm_idx, const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_off_addr = vmm_idx_to_elem_addr_off.find(vmm_idx);
    if (it_off_addr != vmm_idx_to_elem_addr_off.end()) {
        if (elem_size_bytes == 1) {
            host_->add(addr_reg, it_off_addr->second);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->mov(tmp_reg, it_off_addr->second);
            host_->sal(tmp_reg, shift_val);
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_value_offset(
        const std::map<int, int> &vmm_idx_to_elem_val_off, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, std::size_t elem_size_bytes) const {

    const auto it_off_val = vmm_idx_to_elem_val_off.find(vmm_idx);
    if (it_off_val != vmm_idx_to_elem_val_off.end())
        host_->add(addr_reg, it_off_val->second * elem_size_bytes);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::inject_binary(
        const dnnl_post_ops::entry_t &post_op, Vmm dst,
        const Xbyak::Address &rhs_addr, bool with_tail) const {

    const auto &alg = post_op.binary.alg;
    const bool cmp_op = utils::one_of(alg, alg_kind::binary_ge,
            alg_kind::binary_gt, alg_kind::binary_le, alg_kind::binary_lt,
            alg_kind::binary_eq, alg_kind::binary_ne);
    const auto &rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    const bool scalar_f32
            = rhs_addr.isBroadcast() && rhs_arg_data_type == data_type::f32;
    const bool with_tail_not_fusable_to_binary_op
            = with_tail && !(scalar_f32 && is_avx512_);
    const bool process_rhs_arg_using_tmp_vmm
            = rhs_arg_data_type != data_type::f32 || (scalar_f32 && !is_avx512_)
            || with_tail_not_fusable_to_binary_op
            || !binary_op_with_unaligned_mem_operand_allowed_
            || (cmp_op && !is_avx512_);

    if (process_rhs_arg_using_tmp_vmm) {

        const Vmm tmp_vmm = Vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);

        if (rhs_addr.isBroadcast())
            execute_broadcast(rhs_arg_data_type, tmp_vmm,
                    remove_bcast_bit(rhs_addr), with_tail);
        else
            load_rhs(rhs_arg_data_type, tmp_vmm, rhs_addr, with_tail);

        if (rhs_arg_data_type != data_type::bf16
                && rhs_arg_data_type != data_type::f32)
            cvt_to_f32(tmp_vmm);

        execute_binary(alg, dst, dst, tmp_vmm);
    } else {
        const auto lhs = dst;
        const bool with_tail_fusable_to_binary_op
                = with_tail && scalar_f32 && is_avx512_;
        if (with_tail_fusable_to_binary_op) {
            assert(rhs_arg_static_params_.is_opmask_set()
                    && "Opmask is not set for tail loading avx512");
            const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;
            dst = dst | tail_opmask | host_->T_z;
        }

        execute_binary(alg, dst, lhs, rhs_addr);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast(
        const dnnl_data_type_t &data_type, const Vmm &tmp_reg,
        const Xbyak::Address &rhs_addr, bool with_tail) const {
    if (with_tail)
        execute_broadcast_tail(data_type, tmp_reg, rhs_addr);
    else
        execute_broadcast_no_tail(data_type, tmp_reg, rhs_addr);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs(const dnnl_data_type_t &data_type,
        const Vmm &tmp_reg, const Xbyak::Address &rhs_addr,
        bool with_tail) const {
    if (with_tail)
        load_rhs_tail(data_type, tmp_reg, rhs_addr);
    else
        load_rhs_no_tail(data_type, tmp_reg, rhs_addr);
}

template <cpu_isa_t isa>
Xbyak::Address jit_uni_binary_injector_t<isa>::remove_bcast_bit(
        const Xbyak::Address &rhs_addr) const {
    return Xbyak::Address(rhs_addr.getBit(), false, rhs_addr.getRegExp());
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::cvt_to_f32(const Vmm &tmp_vmm) const {
    host_->vcvtdq2ps(tmp_vmm, tmp_vmm);
}

template <>
void jit_uni_binary_injector_t<sse41>::cvt_to_f32(const Vmm &tmp_vmm) const {
    static_assert(
            std::is_same<Vmm, Xbyak::Xmm>::value, "Vmm type should match Xmm");
    host_->cvtdq2ps(tmp_vmm, tmp_vmm);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast_no_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    switch (data_type) {
        case data_type::f32: host_->uni_vbroadcastss(tmp_vmm, rhs_addr); break;
        case data_type::s32: host_->uni_vpbroadcastd(tmp_vmm, rhs_addr); break;
        case data_type::s8:
        case data_type::u8:
            execute_broadcast_s8u8_no_tail(data_type, tmp_vmm, rhs_addr);
            break;
        case data_type::bf16:
            if (is_avx512_
                    && utils::one_of(isa, avx512_core_bf16, avx512_core)) {
                host_->vpbroadcastw(tmp_vmm, rhs_addr);
                host_->vpslld(tmp_vmm, tmp_vmm, 0x10);
                break;
            }
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    const Xbyak::Xmm xmm(tmp_vmm.getIdx());
    switch (data_type) {
        case data_type::s8:
            host_->vpbroadcastb(xmm, rhs_addr);
            host_->vpmovsxbd(tmp_vmm, xmm);
            break;
        case data_type::u8:
            host_->vpbroadcastb(xmm, rhs_addr);
            host_->vpmovzxbd(tmp_vmm, xmm);
            break;
        default: assert(!"unsupported data type");
    }
}

template <>
void jit_uni_binary_injector_t<avx>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {

    if (data_type == data_type::s8 || data_type == data_type::u8) {
        const auto tmp_reg64_idx
                = rhs_arg_static_params_.rhs_helper_reg.getIdx();
        const Xbyak::Reg8 tmp_reg8 = Xbyak::Reg8(tmp_reg64_idx);
        const Xbyak::Reg32 tmp_reg32 = Xbyak::Reg32(tmp_reg64_idx);
        const auto tmp_xmm = Xbyak::Xmm(tmp_vmm.getIdx());
        host_->mov(tmp_reg8, rhs_addr);
        host_->vmovd(tmp_xmm, tmp_reg32);
        host_->vpunpcklbw(tmp_xmm, tmp_xmm, tmp_xmm);
        host_->vpshuflw(tmp_xmm, tmp_xmm, 0);
        if (data_type == data_type::s8)
            host_->vpmovsxbd(tmp_xmm, tmp_xmm);
        else
            host_->vpmovzxbd(tmp_xmm, tmp_xmm);

        host_->vinsertf128(tmp_vmm, tmp_vmm, tmp_xmm, 1);
    } else
        assert(!"unsupported data type");
}

template <>
void jit_uni_binary_injector_t<sse41>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {

    if (data_type == data_type::s8 || data_type == data_type::u8) {
        const auto tmp_reg64_idx
                = rhs_arg_static_params_.rhs_helper_reg.getIdx();
        const Xbyak::Reg8 tmp_reg8 = Xbyak::Reg8(tmp_reg64_idx);
        host_->mov(tmp_reg8, rhs_addr);
        const Xbyak::Reg32 tmp_reg32 = Xbyak::Reg32(tmp_reg64_idx);
        host_->movd(tmp_vmm, tmp_reg32);
        host_->punpcklbw(tmp_vmm, tmp_vmm);
        host_->pshuflw(tmp_vmm, tmp_vmm, 0);
        if (data_type == data_type::s8)
            host_->pmovsxbd(tmp_vmm, tmp_vmm);
        else
            host_->pmovzxbd(tmp_vmm, tmp_vmm);
    } else
        assert(!"unsupported data type");
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {

    assert(rhs_arg_static_params_.is_opmask_set()
            && "Opmask is not set for tail loading avx512");
    const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;

    host_->uni_vxorps(tmp_vmm, tmp_vmm, tmp_vmm);
    switch (data_type) {
        case data_type::f32:
            host_->vbroadcastss(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::s32:
            host_->vpbroadcastd(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::s8:
            host_->vpbroadcastb(tmp_vmm | tail_opmask | host_->T_z,
                    rhs_addr); // broadcast to tmm_xmm should be enough ?
            host_->vpmovsxbd(tmp_vmm | tail_opmask | host_->T_z, tmp_vmm);
            break;
        case data_type::u8:
            host_->vpbroadcastb(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            host_->vpmovzxbd(tmp_vmm | tail_opmask | host_->T_z, tmp_vmm);
            break;
        case data_type::bf16:
            if (is_avx512_
                    && utils::one_of(isa, avx512_core_bf16, avx512_core)) {
                host_->vpbroadcastw(tmp_vmm, rhs_addr);
                host_->vpslld(
                        tmp_vmm | tail_opmask | host_->T_z, tmp_vmm, 0x10);
                break;
            }
        default: assert(!"unsupported data type");
    }
}

static constexpr int xmm_size_elem = 4;

static void load_tail_avx(jit_generator *host, std::size_t ymm_idx,
        std::size_t tail_size, const std::function<void()> &init_op,
        const std::function<void(int, bool)> &ymm_upper_half_op,
        const std::function<void(int)> &ymm_lower_half_op) {

    if (init_op) init_op();

    const auto res = std::div(tail_size, xmm_size_elem);
    const auto &ymm_upper_half_op_data_size = res.rem;
    const bool should_load_lower_half = res.quot;

    if (ymm_upper_half_op_data_size && ymm_upper_half_op)
        ymm_upper_half_op(ymm_upper_half_op_data_size, should_load_lower_half);

    if (should_load_lower_half) {
        const auto tmp_xmm = Xbyak::Xmm(ymm_idx);

        if (ymm_upper_half_op_data_size) push_vmm(host, tmp_xmm);

        if (ymm_lower_half_op) ymm_lower_half_op(ymm_upper_half_op_data_size);

        if (ymm_upper_half_op_data_size) {
            const auto tmp_ymm = Xbyak::Ymm(ymm_idx);
            host->vinsertf128(tmp_ymm, tmp_ymm, host->ptr[host->rsp], 1);
            restore_stack(host, tmp_xmm);
        }
    }
}

static void load_tail_avx(jit_generator *host, std::size_t ymm_idx,
        std::size_t tail_size,
        const std::function<void(int, bool)> &ymm_upper_half_op,
        const std::function<void(int)> &ymm_lower_half_op) {
    load_tail_avx(host, ymm_idx, tail_size, nullptr, ymm_upper_half_op,
            ymm_lower_half_op);
}

static Xbyak::uint8 MM_SHUFFLE(
        Xbyak::uint8 z, Xbyak::uint8 y, Xbyak::uint8 x, Xbyak::uint8 w) {
    return (((z) << 6) | ((y) << 4) | ((x) << 2) | (w));
}

static void execute_broadcast_f32_tail_avx(jit_generator *host,
        std::size_t ymm_idx, const Xbyak::Address &rhs_addr,
        std::size_t tail_size) {

    const auto tmp_xmm = Xbyak::Xmm(ymm_idx);
    static const std::array<Xbyak::uint8, 2> imms {
            {MM_SHUFFLE(3, 2, 0, 0), MM_SHUFFLE(3, 0, 0, 0)}};

    const auto init_op = [&] { host->vmovss(tmp_xmm, rhs_addr); };
    const auto upper_half_op
            = [&](int upper_half_data_size, bool should_load_lower_half) {
                  // one element is already loaded
                  if (upper_half_data_size > 1)
                      host->vshufps(tmp_xmm, tmp_xmm, tmp_xmm,
                              imms.at(upper_half_data_size - 2));
              };
    const auto lower_half_op = [&](int upper_half_data_size) {
        host->vshufps(tmp_xmm, tmp_xmm, tmp_xmm, 0);
    };

    load_tail_avx(
            host, ymm_idx, tail_size, init_op, upper_half_op, lower_half_op);
}

template <>
void jit_uni_binary_injector_t<avx2>::execute_broadcast_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    const auto &tail_size = rhs_arg_static_params_.tail_size;
    host_->uni_vxorps(tmp_vmm, tmp_vmm, tmp_vmm);

    if (data_type == data_type::f32 || data_type == data_type::s32) {
        execute_broadcast_f32_tail_avx(
                host_, tmp_vmm.getIdx(), rhs_addr, tail_size);
    } else if (data_type == data_type::u8 || data_type == data_type::s8) {
        const auto tmp_xmm = Xbyak::Xmm(tmp_vmm.getIdx());
        for (std::size_t i = 0; i < tail_size; i++)
            host_->vpinsrb(tmp_xmm, tmp_xmm, rhs_addr, i);

        if (data_type == data_type::s8)
            host_->vpmovsxbd(tmp_vmm, tmp_xmm);
        else
            host_->vpmovzxbd(tmp_vmm, tmp_xmm);
    } else
        assert(!"unsupported data type");
}

template <>
void jit_uni_binary_injector_t<avx>::execute_broadcast_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {

    const auto &tail_size = rhs_arg_static_params_.tail_size;
    host_->uni_vxorps(tmp_vmm, tmp_vmm, tmp_vmm);

    if (data_type == data_type::f32 || data_type == data_type::s32) {
        execute_broadcast_f32_tail_avx(
                host_, tmp_vmm.getIdx(), rhs_addr, tail_size);
    } else if (data_type == data_type::u8 || data_type == data_type::s8) {
        const auto ymm_idx = tmp_vmm.getIdx();
        const auto tmp_xmm = Xbyak::Xmm(ymm_idx);
        const auto cvt_to_dword = [&] {
            if (data_type == data_type::s8)
                host_->vpmovsxbd(tmp_xmm, tmp_xmm);
            else
                host_->vpmovzxbd(tmp_xmm, tmp_xmm);
        };

        const auto init_op
                = [&] { host_->vpinsrb(tmp_xmm, tmp_xmm, rhs_addr, 0); };

        const auto upper_half_op
                = [&](int upper_half_data_size, bool should_load_lower_half) {
                      for (int i = 1; i < upper_half_data_size; i++)
                          host_->vpinsrb(tmp_xmm, tmp_xmm, rhs_addr, i);

                      cvt_to_dword();
                  };

        const auto lower_half_op = [&](int upper_half_data_size) {
            if (upper_half_data_size) {
                const auto dword_to_copy = xmm_size_elem - upper_half_data_size;
                for (int i = upper_half_data_size; i < dword_to_copy; i++)
                    host_->vpinsrd(tmp_xmm, tmp_xmm, tmp_xmm, i);
            } else {
                for (int i = 1; i < xmm_size_elem; i++)
                    host_->vpinsrb(tmp_xmm, tmp_xmm, rhs_addr, i);

                cvt_to_dword();
            }
        };

        load_tail_avx(host_, ymm_idx, tail_size, init_op, upper_half_op,
                lower_half_op);
    } else
        assert(!"unsupported data type");
}

template <>
void jit_uni_binary_injector_t<sse41>::execute_broadcast_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {

    host_->uni_vxorps(tmp_vmm, tmp_vmm, tmp_vmm);
    const auto &tail_size = rhs_arg_static_params_.tail_size;
    if (data_type == data_type::f32 || data_type == data_type::s32) {
        static const std::array<Xbyak::uint8, 2> imms {
                {MM_SHUFFLE(3, 2, 0, 0), MM_SHUFFLE(3, 0, 0, 0)}};

        host_->movss(tmp_vmm, rhs_addr);
        if (tail_size > 1) host_->shufps(tmp_vmm, tmp_vmm, imms[tail_size - 1]);

    } else if (data_type == data_type::u8 || data_type == data_type::s8) {
        for (std::size_t i = 0; i < tail_size; i++)
            host_->pinsrb(tmp_vmm, rhs_addr, i);

        if (data_type == data_type::s8)
            host_->pmovsxbd(tmp_vmm, tmp_vmm);
        else
            host_->pmovzxbd(tmp_vmm, tmp_vmm);
    } else
        assert(!"unsupported data type");
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs_no_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    switch (data_type) {
        case data_type::f32:
        case data_type::s32: host_->uni_vmovups(tmp_vmm, rhs_addr); break;
        case data_type::s8: host_->uni_vpmovsxbd(tmp_vmm, rhs_addr); break;
        case data_type::u8: host_->uni_vpmovzxbd(tmp_vmm, rhs_addr); break;
        case data_type::bf16:
            if (is_avx512_
                    && utils::one_of(isa, avx512_core_bf16, avx512_core)) {
                host_->vpmovzxwd(tmp_vmm, rhs_addr);
                host_->vpslld(tmp_vmm, tmp_vmm, 0x10);
                break;
            }
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {

    assert(rhs_arg_static_params_.is_opmask_set()
            && "Opmask is not set for tail loading avx512");

    const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;
    host_->uni_vxorps(tmp_vmm, tmp_vmm, tmp_vmm);

    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
            host_->vmovups(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::s8:
            host_->vpmovsxbd(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::u8:
            host_->vpmovzxbd(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::bf16:
            if (is_avx512_
                    && utils::one_of(isa, avx512_core_bf16, avx512_core)) {
                host_->vpmovzxwd(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
                host_->vpslld(
                        tmp_vmm | tail_opmask | host_->T_z, tmp_vmm, 0x10);
                break;
            }
        default: assert(!"unsupported data type");
    }
}

template <>
void jit_uni_binary_injector_t<avx2>::load_rhs_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    if (!utils::one_of(data_type, data_type::f32, data_type::s32, data_type::s8,
                data_type::u8))
        assert(!"unsupported data type");

    const auto &tail_size = rhs_arg_static_params_.tail_size;
    host_->uni_vxorps(tmp_vmm, tmp_vmm, tmp_vmm);
    host_->load_data(data_type, tmp_vmm, rhs_arg_static_params_.rhs_addr_reg, 0,
            tail_size);
}

template <>
void jit_uni_binary_injector_t<avx>::load_rhs_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    host_->uni_vxorps(tmp_vmm, tmp_vmm, tmp_vmm);
    static constexpr int xmm_size_elem = 4;
    const auto &tail_size = rhs_arg_static_params_.tail_size;
    const auto res = std::div(tail_size, xmm_size_elem);
    const auto ymm_idx = tmp_vmm.getIdx();
    const auto tmp_xmm = Xbyak::Xmm(ymm_idx);

    if (data_type == data_type::f32 || data_type == data_type::s32) {

        const auto upper_half_op
                = [&](int upper_half_data_size, bool should_load_lower_half) {
                      const int offset = should_load_lower_half
                              ? xmm_size_elem * sizeof(float)
                              : 0;
                      for (int i = 0; i < res.rem; i++)
                          host_->vpinsrd(tmp_xmm, tmp_xmm,
                                  host_->ptr[rhs_arg_static_params_.rhs_addr_reg
                                          + offset + i * sizeof(float)],
                                  i);
                  };

        const auto lower_half_op = [&](int upper_half_data_size) {
            host_->vmovups(tmp_xmm, rhs_addr);
        };
        load_tail_avx(host_, ymm_idx, tail_size, upper_half_op, lower_half_op);

    } else if (data_type == data_type::u8 || data_type == data_type::s8) {
        const auto cvt_to_dword = [&](const Xbyak::Operand &operand) {
            if (data_type == data_type::s8)
                host_->vpmovsxbd(tmp_vmm, operand);
            else
                host_->vpmovzxbd(tmp_xmm, operand);
        };

        const auto upper_half_op = [&](int upper_half_data_size,
                                           bool should_load_lower_half) {
            const int offset = should_load_lower_half ? xmm_size_elem : 0;
            for (int i = 0; i < upper_half_data_size; i++)
                host_->vpinsrb(tmp_xmm, tmp_xmm,
                        host_->ptr[rhs_arg_static_params_.rhs_addr_reg + offset
                                + i * sizeof(float)],
                        i);
            cvt_to_dword(tmp_xmm);
        };

        const auto lower_half_op
                = [&](int upper_half_data_size) { cvt_to_dword(rhs_addr); };

        load_tail_avx(host_, ymm_idx, tail_size, upper_half_op, lower_half_op);

    } else
        assert(!"unsupported data type");
}

template <>
void jit_uni_binary_injector_t<sse41>::load_rhs_tail(
        const dnnl_data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    if (!utils::one_of(data_type, data_type::f32, data_type::s32, data_type::s8,
                data_type::u8))
        assert(!"unsupported data type");

    const auto &tail_size = rhs_arg_static_params_.tail_size;
    host_->uni_vxorps(tmp_vmm, tmp_vmm, tmp_vmm);
    host_->load_data(data_type, tmp_vmm, rhs_arg_static_params_.rhs_addr_reg, 0,
            tail_size);
}

// Support compare with Address param only when isa is avx512.
// AVX512 implementation
template <cpu_isa_t isa>
template <typename T>
typename std::enable_if<std::is_same<T, Xbyak::Zmm>::value
        || std::is_same<T, Xbyak::Address>::value>::type
jit_uni_binary_injector_t<isa>::execute_cmp_binary(const Vmm &dst,
        const Vmm &lhs, const T &rhs, const unsigned int cmp_predicate) const {
    // For GreaterEqual op, replace 0xFFFFFFFF by 1
    // which was returned by vcmpps.
    const auto &cmp_mask = rhs_arg_static_params_.tail_opmask;
    const Xbyak::Xmm xreg_one
            = Xbyak::Xmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);
    const Xbyak::Reg64 reg_tmp = rhs_arg_static_params_.rhs_helper_reg;

    push_opmask(host_, cmp_mask);
    host_->vcmpps(cmp_mask, lhs, rhs, cmp_predicate);
    host_->mov(reg_tmp, float2int(1));
    host_->uni_vmovq(xreg_one, reg_tmp);
    // broadcast 1.0f with mask
    host_->vbroadcastss(dst | cmp_mask | host_->T_z, xreg_one);
    // pop tail mask from stack
    pop_opmask(host_, cmp_mask);
}

// SSE4.1., AVX and AVX2 implementation
template <cpu_isa_t isa>
template <typename T>
typename std::enable_if<!(std::is_same<T, Xbyak::Zmm>::value
        || std::is_same<T, Xbyak::Address>::value)>::type
jit_uni_binary_injector_t<isa>::execute_cmp_binary(const Vmm &dst,
        const Vmm &lhs, const T &rhs, const unsigned int cmp_predicate) const {
    const int vmm_idx = rhs_arg_static_params_.rhs_dt_helper_vmm_idx;
    const Vmm vreg_one = Vmm(vmm_idx);
    const Xbyak::Xmm xreg_one = Xbyak::Xmm(vmm_idx);
    const Xbyak::Reg64 reg_tmp = rhs_arg_static_params_.rhs_helper_reg;

    host_->uni_vcmpps(dst, lhs, rhs, cmp_predicate);
    host_->mov(reg_tmp, float2int(1));
    host_->uni_vmovq(xreg_one, reg_tmp);
    host_->uni_vbroadcastss(vreg_one, xreg_one);
    host_->uni_vminps(dst, dst, vreg_one);
}

template <cpu_isa_t isa>
template <typename T>
void jit_uni_binary_injector_t<isa>::execute_binary(alg_kind_t binary_alg,
        const Vmm &dst, const Vmm &lhs, const T &rhs) const {
    switch (binary_alg) {
        case alg_kind::binary_add: host_->uni_vaddps(dst, lhs, rhs); break;
        case alg_kind::binary_mul: host_->uni_vmulps(dst, lhs, rhs); break;
        case alg_kind::binary_max: host_->uni_vmaxps(dst, lhs, rhs); break;
        case alg_kind::binary_min: host_->uni_vminps(dst, lhs, rhs); break;
        case alg_kind::binary_div: host_->uni_vdivps(dst, lhs, rhs); break;
        case alg_kind::binary_sub: host_->uni_vsubps(dst, lhs, rhs); break;
        case alg_kind::binary_ge:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_nlt_us);
            break;
        case alg_kind::binary_gt:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_nle_us);
            break;
        case alg_kind::binary_le:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_le_os);
            break;
        case alg_kind::binary_lt:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_lt_os);
            break;
        case alg_kind::binary_eq:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_eq_oq);
            break;
        case alg_kind::binary_ne:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_neq_uq);
            break;
        default: assert(!"unsupported algorithm");
    }
}

template <>
template <typename T>
void jit_uni_binary_injector_t<avx>::execute_binary(alg_kind_t binary_alg,
        const Vmm &dst, const Vmm &lhs, const T &rhs) const {
    switch (binary_alg) {
        case alg_kind::binary_add: host_->vaddps(dst, lhs, rhs); break;
        case alg_kind::binary_mul: host_->vmulps(dst, lhs, rhs); break;
        case alg_kind::binary_max: host_->vmaxps(dst, lhs, rhs); break;
        case alg_kind::binary_min: host_->vminps(dst, lhs, rhs); break;
        case alg_kind::binary_div: host_->vdivps(dst, lhs, rhs); break;
        case alg_kind::binary_sub: host_->vsubps(dst, lhs, rhs); break;
        case alg_kind::binary_ge:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_nlt_us);
            break;
        case alg_kind::binary_gt:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_nle_us);
            break;
        case alg_kind::binary_le:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_le_os);
            break;
        case alg_kind::binary_lt:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_lt_os);
            break;
        case alg_kind::binary_eq:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_eq_oq);
            break;
        case alg_kind::binary_ne:
            execute_cmp_binary(dst, lhs, rhs, jit_generator::_cmp_neq_uq);
            break;
        default: assert(!"unsupported algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector(size_t idx,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    compute_vector_range({idx}, rhs_arg_idx, post_op, rhs_arg_params);
}

template class jit_uni_binary_injector_t<avx512_core_bf16>;
template class jit_uni_binary_injector_t<avx512_core>;
template class jit_uni_binary_injector_t<avx512_common>;
template class jit_uni_binary_injector_t<avx2>;
template class jit_uni_binary_injector_t<avx>;
template class jit_uni_binary_injector_t<sse41>;

} // namespace binary_injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
