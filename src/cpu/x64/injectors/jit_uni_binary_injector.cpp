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
#include <algorithm>
#include <cmath>
#include <limits>

#include "common/math_utils.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace binary_injector {

#define VCHECK_BIN_INJ_BOOL(cond, msg) \
    VCONDCHECK(primitive, create, check, binary_injector, cond, false, msg);

bcast_set_t get_all_strategies_supported_by_injector() {
    return bcast_set_t {broadcasting_strategy_t::scalar,
            broadcasting_strategy_t::per_oc, broadcasting_strategy_t::per_oc_d,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::per_mb,
            broadcasting_strategy_t::per_mb_spatial,
            broadcasting_strategy_t::per_mb_w, broadcasting_strategy_t::per_w,
            broadcasting_strategy_t::per_hw, broadcasting_strategy_t::batch,
            broadcasting_strategy_t::spatial,
            broadcasting_strategy_t::no_broadcast};
}

bool is_data_supported(cpu_isa_t isa, data_type_t data_type) {
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
        case data_type::s8:
        case data_type::u8: return true;
        case data_type::bf16:
            return is_superset(isa, avx512_core)
                    || is_superset(isa, avx2_vnni_2);
        case data_type::f16:
            return is_superset(isa, avx512_core_fp16)
                    || is_superset(isa, avx2_vnni_2);
        case data_type::f8_e5m2:
        case data_type::f8_e4m3: return is_superset(isa, avx512_core_fp16);
        case data_type::f4_e2m1:
        case data_type::e8m0:
        case data_type::f64:
        case data_type::s4:
        case data_type::u4:
        case data_type::boolean: return false;
        case data_type::undef:
        default: assert(!"unknown data_type"); return false;
    }
}

static bool src1_desc_layout_same_as_dst_d(
        const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d) {
    if (dst_d.md_ == nullptr) return false;
    const auto &lhs = src1_desc;
    const auto &rhs = *(dst_d.md_);

    using namespace dnnl::impl::utils;
    const bool is_format_any
            = one_of(format_kind::any, lhs.format_kind, rhs.format_kind);

    return lhs.ndims == rhs.ndims
            && (is_format_any
                    || (lhs.format_kind == rhs.format_kind
                            && array_cmp(lhs.format_desc.blocking.strides,
                                    rhs.format_desc.blocking.strides,
                                    lhs.ndims)))
            && array_cmp(lhs.dims, rhs.dims, lhs.ndims)
            && array_cmp(lhs.padded_dims, rhs.padded_dims, lhs.ndims)
            && array_cmp(lhs.padded_offsets, rhs.padded_offsets, lhs.ndims)
            && lhs.offset0 == rhs.offset0;
}

bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
            src1_desc, dst_d, supported_strategy_set);

    if (bcast_type == broadcasting_strategy_t::no_broadcast) {
        // in case of no broadcast data layout of dst and src1 have to be the same
        VCHECK_BIN_INJ_BOOL(src1_desc_layout_same_as_dst_d(src1_desc, dst_d),
                "Dst and src1 layout are not the same");
        return true;
    }

    VCHECK_BIN_INJ_BOOL(bcast_type != broadcasting_strategy_t::unsupported,
            "Unsupported broadcast type");
    return true;
}

bool is_supported(cpu_isa_t isa, const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    VCHECK_BIN_INJ_BOOL(is_data_supported(isa, src1_desc.data_type),
            VERBOSE_ISA_DT_MISMATCH);
    const memory_desc_wrapper rhs_d(src1_desc);
    const bool rhs_nontrivial_strides = !rhs_d.is_dense(true);
    if (rhs_nontrivial_strides) {
        VCHECK_BIN_INJ_BOOL(rhs_d.is_plain(), VERBOSE_NONTRIVIAL_STRIDE);
        const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                src1_desc, dst_d, supported_strategy_set);
        VCHECK_BIN_INJ_BOOL(
                bcast_type == broadcasting_strategy_t::per_mb_spatial,
                VERBOSE_NONTRIVIAL_STRIDE);
        VCHECK_BIN_INJ_BOOL(dst_d.is_plain()
                        && injector_utils::get_layout_type(dst_d)
                                == injector_utils::layout_t::ncsp,
                VERBOSE_NONTRIVIAL_STRIDE);
    }
    return is_bcast_supported(src1_desc, dst_d, supported_strategy_set);
}

static dim_t rhs_offset_per_mb_spatial(const memory_desc_wrapper &dst_d,
        const memory_desc_wrapper &rhs_d, dim_t dst_offset_bytes) {
    // dst_off = n*dst_stride_n + c*dst_stride_c + d*dst_stride_d + h*dst_stride_h + w*dst_stride_w
    // rhs_off = n*rhs_stride_n + d*rhs_stride_d + h*rhs_stride_h + w*rhs_stride_w
    // per_mb_spatial: channel c is broadcast, so rem = (dst_off % dst_stride_n) % dst_stride_c.
    const auto &dst_strides = dst_d.blocking_desc().strides;
    const auto &rhs_strides = rhs_d.blocking_desc().strides;
    const int ndims = dst_d.ndims();
    const int dst_dt_shift
            = math::ilog2q(types::data_type_size(dst_d.data_type()));
    const dim_t dst_offset_elems = dst_offset_bytes >> dst_dt_shift;

    const dim_t n = dst_offset_elems / dst_strides[0];
    dim_t rem = dst_offset_elems % dst_strides[0];
    if (dst_strides[1] != 0) rem %= dst_strides[1];

    dim_t d = 0;
    dim_t h = 0;
    dim_t w = 0;
    if (ndims >= 5) {
        d = rem / dst_strides[ndims - 3];
        rem %= dst_strides[ndims - 3];
    }
    if (ndims >= 4) {
        h = rem / dst_strides[ndims - 2];
        rem %= dst_strides[ndims - 2];
    }
    if (ndims >= 3) w = rem / dst_strides[ndims - 1];

    // output = rhs_off (in elements)
    dim_t rhs_offset = n * rhs_strides[0];
    if (ndims >= 5) rhs_offset += d * rhs_strides[ndims - 3];
    if (ndims >= 4) rhs_offset += h * rhs_strides[ndims - 2];
    if (ndims >= 3) rhs_offset += w * rhs_strides[ndims - 1];
    return rhs_offset;
}

bool binary_args_broadcast_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_like_binary()) {
            const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                    get_src1_desc(entry, dst_d), dst_d, supported_strategy_set);
            return bcast_type == broadcasting_strategy_t::unsupported;
        }
        return false;
    });
}

bool any_binary_postop_rhs_non_scalar_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_like_binary()) {
            const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                    get_src1_desc(entry, dst_d), dst_d,
                    get_all_strategies_supported_by_injector());
            return !utils::one_of(bcast_type, broadcasting_strategy_t::scalar,
                           broadcasting_strategy_t::unsupported)
                    || entry.is_binary_with_ternary_op();
        }
        return false;
    });
}

bool any_binary_postop_rhs_with_ternary_scalar_bcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_like_binary()) {
            const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                    get_src1_desc(entry, dst_d), dst_d,
                    get_all_strategies_supported_by_injector());
            return bcast_type == broadcasting_strategy_t::scalar
                    && entry.is_binary_with_ternary_op();
        }
        return false;
    });
}

bool is_ternary_bcast_supported(
        const memory_desc_t &src2_md, const memory_desc_wrapper &dst_d) {
    if (src2_md.ndims != dst_d.ndims()) return false;
    return utils::array_cmp(src2_md.dims, dst_d.dims(), dst_d.ndims());
}

bool any_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_like_binary()) {
            const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                    get_src1_desc(entry, dst_d), dst_d, supported_strategy_set);
            return bcast_type == broadcasting_strategy_t::per_oc
                    || bcast_type == broadcasting_strategy_t::per_oc_spatial;
        }
        return false;
    });
}

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set,
        const std::function<bool(const memory_desc_wrapper &)> &predicate) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_like_binary()) {
            const auto src1_desc = get_src1_desc(entry, dst_d);
            const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                    src1_desc, dst_d, supported_strategy_set);
            if (bcast_type == broadcasting_strategy_t::per_oc
                    || bcast_type == broadcasting_strategy_t::per_oc_spatial)
                return predicate(memory_desc_wrapper(src1_desc));
        }
        return true;
    });
}

bool any_binary_postop_rhs_per_w_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_like_binary()) {
            const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                    get_src1_desc(entry, dst_d), dst_d, supported_strategy_set);
            return bcast_type == broadcasting_strategy_t::per_w
                    || bcast_type == broadcasting_strategy_t::per_mb_w;
        }
        return false;
    });
}

void extend_binary_args_per_w(const post_ops_t &post_ops,
        const std::vector<const void *> &orig_post_ops_binary_rhs_arg_vec,
        std::vector<const void *> &post_ops_binary_rhs_arg_vec,
        uint8_t *expanded_rhs, const std::vector<dim_t> &expanded_elems_len) {
    dim_t offset = 0;
    const int po_len = post_ops.len();
    auto binary_post_op_idx = 0;

    for (int i = 0; i < po_len; ++i) {
        if (!post_ops.entry_[i].is_binary()) { continue; }
        if (expanded_elems_len[i] > 0) {
            const auto &rhs_md = post_ops.entry_[i].binary.src1_desc;
            const memory_desc_wrapper rhs_md_wrap(&rhs_md);
            const dim_t rhs_len = rhs_md_wrap.nelems();
            const data_type_t dt = rhs_md.data_type;
            const auto dt_size = types::data_type_size(dt);

            const uint8_t *src = reinterpret_cast<const uint8_t *>(
                    orig_post_ops_binary_rhs_arg_vec[binary_post_op_idx]);
            auto *dst = expanded_rhs + offset;

            for (dim_t j = 0; j < expanded_elems_len[i]; ++j) {
                const dim_t src_idx = j % rhs_len;
                memcpy(dst + j * dt_size, src + src_idx * dt_size, dt_size);
            }

            post_ops_binary_rhs_arg_vec[binary_post_op_idx] = dst;
            offset += expanded_elems_len[binary_post_op_idx] * dt_size;
        } else {
            post_ops_binary_rhs_arg_vec[binary_post_op_idx]
                    = orig_post_ops_binary_rhs_arg_vec[binary_post_op_idx];
        }
        binary_post_op_idx++;
    }
}

static_params_t::static_params_t(const Xbyak::Reg64 &param1,
        const bcast_set_t &supported_strategy_set,
        const rhs_arg_static_params_t &rhs_arg_static_params,
        fp8_conversion_e5m2_t *f8_e5m2_cvt, fp8_conversion_e4m3_t *f8_e4m3_cvt)
    : param1(param1)
    , supported_strategy_set(supported_strategy_set)
    , rhs_arg_static_params(rhs_arg_static_params)
    , f8_e5m2_cvt_(f8_e5m2_cvt)
    , f8_e4m3_cvt_(f8_e4m3_cvt) {}

static_params_t::static_params_t(const Xbyak::Reg64 &param1,
        const bcast_set_t &supported_strategy_set,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : static_params_t(param1, supported_strategy_set, rhs_arg_static_params,
              nullptr, nullptr) {}

static_params_t::static_params_t(const Xbyak::Reg64 &param1,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : static_params_t(param1, get_all_strategies_supported_by_injector(),
              rhs_arg_static_params) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(int rhs_dt_helper_vmm_idx,
        const Xbyak::Reg64 &rhs_addr_reg, const Xbyak::Reg64 &rhs_helper_reg,
        const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, int abi_param_offset, dim_t dst_orig_offset,
        const memory_desc_wrapper &dst_d, int tail_size,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, dst_orig_offset, dst_d,
              tail_size, Xbyak::Opmask(2), use_exact_tail_scalar_bcast,
              rhs_helper_reg, false /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(int rhs_dt_helper_vmm_idx,
        const Xbyak::Reg64 &rhs_addr_reg, const Xbyak::Reg64 &rhs_helper_reg,
        const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, int abi_param_offset, dim_t dst_orig_offset,
        const memory_desc_wrapper &dst_d, int tail_size,
        const Xbyak::Opmask &tail_opmask, bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, dst_orig_offset, dst_d,
              tail_size, tail_opmask, use_exact_tail_scalar_bcast,
              rhs_helper_reg, true /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(int rhs_dt_helper_vmm_idx,
        const Xbyak::Reg64 &rhs_addr_reg, const Xbyak::Reg64 &rhs_helper_reg,
        const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, int abi_param_offset, dim_t dst_orig_offset,
        const memory_desc_wrapper &dst_d, int tail_size,
        const Xbyak::Opmask &tail_opmask, const Xbyak::Reg64 &reg_tail_size,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, dst_orig_offset, dst_d,
              tail_size, tail_opmask, use_exact_tail_scalar_bcast,
              reg_tail_size, true /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(int rhs_dt_helper_vmm_idx,
        const Xbyak::Reg64 &rhs_addr_reg, const Xbyak::Reg64 &rhs_helper_reg,
        const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, int abi_param_offset, dim_t dst_orig_offset,
        const memory_desc_wrapper &dst_d, int tail_size,
        const Xbyak::Opmask &tail_opmask, bool use_exact_tail_scalar_bcast,
        const Xbyak::Reg64 &reg_tail_size, bool is_opmask_set)
    : rhs_dt_helper_vmm_idx(rhs_dt_helper_vmm_idx)
    , rhs_addr_reg(rhs_addr_reg)
    , rhs_helper_reg(rhs_helper_reg)
    , rhs_addr_cache_reg(rhs_addr_cache_reg)
    , preserve_gpr_helpers(preserve_gpr_helpers)
    , preserve_vmm_helper(preserve_vmm_helper)
    , abi_param_offset(abi_param_offset)
    , dst_orig_offset(dst_orig_offset)
    , dst_d(dst_d)
    , tail_size(tail_size)
    , tail_opmask(tail_opmask)
    , use_exact_tail_scalar_bcast(use_exact_tail_scalar_bcast)
    , reg_tail_size(reg_tail_size)
    , is_tail(tail_size)
    , is_opmask_set_(is_opmask_set) {}

template <typename Vmm>
jit_uni_binary_injector_t<Vmm>::jit_uni_binary_injector_t(
        jit_generator_t *host, const static_params_t &static_params)
    : host_(host)
    , f8_e5m2_cvt_(static_params.f8_e5m2_cvt_)
    , f8_e4m3_cvt_(static_params.f8_e4m3_cvt_)
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

static bool rhs_arg_params_differ(int vmm_idx1, int vmm_idx2,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        broadcasting_strategy_t rhs_broadcasting_strategy) {

    const auto &out_addr = rhs_arg_params.vmm_idx_to_out_addr;
    const auto &out_reg = rhs_arg_params.vmm_idx_to_out_reg;
    const auto &out_elem_off_val = rhs_arg_params.vmm_idx_to_out_elem_off_val;

    if (rhs_broadcasting_strategy != broadcasting_strategy_t::scalar) {
        return params_differ(out_addr, vmm_idx1, vmm_idx2)
                || params_differ(out_reg, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_val, vmm_idx1, vmm_idx2);
    }
    return false;
}

template <typename Vmm>
int jit_uni_binary_injector_t<Vmm>::adjust_temp_vmm_hint(
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
static void push_vmm(jit_generator_t *host, const Vmm &vmm) {
    host->sub(host->rsp, vreg_traits_t<Vmm>::vlen);
    host->uni_vmovups(host->ptr[host->rsp], vmm);
}

template <typename Vmm>
static void pop_vmm(jit_generator_t *host, const Vmm &vmm) {
    host->uni_vmovups(vmm, host->ptr[host->rsp]);
    host->add(host->rsp, vreg_traits_t<Vmm>::vlen);
}

static void push_opmask(jit_generator_t *host, const Xbyak::Opmask &k) {
    static constexpr int k_mask_size = 8;
    host->sub(host->rsp, k_mask_size);
    if (mayiuse(avx512_core))
        host->kmovq(host->ptr[host->rsp], k);
    else
        host->kmovw(host->ptr[host->rsp], k);
}

static void pop_opmask(jit_generator_t *host, const Xbyak::Opmask &k) {
    static constexpr int k_mask_size = 8;
    if (mayiuse(avx512_core))
        host->kmovq(k, host->ptr[host->rsp]);
    else
        host->kmovw(k, host->ptr[host->rsp]);
    host->add(host->rsp, k_mask_size);
}

template <typename Vmm>
static void restore_stack(jit_generator_t *host, const Vmm &vmm) {
    host->add(host->rsp, vreg_traits_t<Vmm>::vlen);
}

template <typename Vmm>
std::pair<bool, int> jit_uni_binary_injector_t<Vmm>::should_preserve_vmm(
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

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::compute_vector_range(int start_idx,
        int end_idx, int rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (int i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_idx, post_op, rhs_arg_params);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs, int rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {

    if (vmm_idxs.empty()) return;
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin());

    // Phase 1 Validate temporary vmm user hint
    const int max_vmm_idx = isa_num_vregs(isa_) - 1;
    auto &vmm_hint = rhs_arg_static_params_.rhs_dt_helper_vmm_idx;
    vmm_hint = adjust_temp_vmm_hint(vmm_hint, start_idx, end_idx, max_vmm_idx);

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto src1_desc = get_src1_desc(post_op, dst_d);
    const auto rhs_broadcasting_strategy = get_rhs_arg_broadcasting_strategy(
            src1_desc, rhs_arg_static_params_.dst_d, supported_strategy_set_);
    const auto rhs_arg_data_type = src1_desc.data_type;
    const auto needs_ternary_input = post_op.is_binary_with_ternary_op();
    const auto &vmm_tail_idx = rhs_arg_params.vmm_tail_idx_;
    const bool tail_exists_in_range = !vmm_tail_idx.empty();
    const bool bcast_f32_non_avx512 = !has_avx512_core_
            && utils::one_of(rhs_broadcasting_strategy,
                    broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::per_oc_spatial)
            && rhs_arg_data_type == data_type::f32;
    const bool should_preserve_vmm_tail = tail_exists_in_range
            && (!has_avx512_core_
                    || !utils::one_of(rhs_broadcasting_strategy,
                            broadcasting_strategy_t::scalar,
                            broadcasting_strategy_t::per_oc_spatial)
                    || rhs_arg_data_type != data_type::f32);
    const bool dt_helper_vmm_needed
            = !binary_op_with_unaligned_mem_operand_allowed_
            || rhs_arg_data_type != data_type::f32 || bcast_f32_non_avx512
            || should_preserve_vmm_tail || post_op.is_prelu();
    const auto tail_load_mode = rhs_arg_params.tail_load_mode;
    const int simd_w = static_cast<int>(
            isa_max_vlen(isa_) / types::data_type_size(dst_d.data_type()));
    const int blk_size = static_cast<int>(dst_d.blocking_desc().inner_blks[0]);
    const bool use_offset_conversions
            = (!rhs_arg_params.vmm_idx_to_out_addr.empty()
                    || !rhs_arg_params.vmm_idx_to_out_reg.empty());
    const bool should_preserve_oc_offset_conversion_regs
            = use_offset_conversions
            && utils::one_of(rhs_broadcasting_strategy,
                    broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::per_oc_spatial)
            && blk_size > simd_w;
    const bool shoud_preserve_oc_d_offset_conversion_regs
            = use_offset_conversions
            && rhs_broadcasting_strategy == broadcasting_strategy_t::per_oc_d;
    const bool should_preserve_mb_sp_offset_conversion_regs
            = use_offset_conversions
            && utils::one_of(rhs_broadcasting_strategy,
                    broadcasting_strategy_t::per_mb_spatial,
                    broadcasting_strategy_t::per_mb_w,
                    broadcasting_strategy_t::per_mb,
                    broadcasting_strategy_t::per_hw,
                    broadcasting_strategy_t::batch);
    const bool should_preserve_w_offset_conversion_regs = use_offset_conversions
            && rhs_broadcasting_strategy == broadcasting_strategy_t::per_w;
    const bool should_preserve_spatial_offset_conversion_regs
            = use_offset_conversions
            && rhs_broadcasting_strategy == broadcasting_strategy_t::spatial;
    const bool should_preserve_w_or_oc_offset_conversion_regs
            = should_preserve_oc_offset_conversion_regs
            || should_preserve_w_offset_conversion_regs
            || should_preserve_spatial_offset_conversion_regs;
    const bool scalar_arithmetic = has_avx512_core_ && post_op.is_binary()
            && !needs_ternary_input
            && rhs_broadcasting_strategy == broadcasting_strategy_t::scalar
            && utils::one_of(post_op.binary.alg, alg_kind::binary_add,
                    alg_kind::binary_sub, alg_kind::binary_mul,
                    alg_kind::binary_div, alg_kind::binary_min,
                    alg_kind::binary_max);
    const bool scalar_f32_memory_operand = scalar_arithmetic
            && rhs_arg_data_type == data_type::f32 && !dt_helper_vmm_needed
            && (!rhs_arg_static_params_.is_tail
                    || rhs_arg_static_params_.is_opmask_set());
    const bool scalar_s32_broadcast
            = scalar_arithmetic && rhs_arg_data_type == data_type::s32;
    const bool scalar_bf16_broadcast_once = scalar_arithmetic
            && rhs_arg_data_type == data_type::bf16
            && vmm_idxs.count(vmm_hint) == 0
            && !(rhs_arg_static_params_.is_tail && tail_exists_in_range
                    && rhs_arg_static_params_.use_exact_tail_scalar_bcast);

    // Phase 2 Protect temporary registers content.
    // These scalar paths only modify the existing RHS address GPR helper.
    // Neither computes destination offsets or needs other GPR helpers.
    const injector_utils::register_preserve_guard_t register_guard {host_,
            ((scalar_f32_memory_operand || scalar_s32_broadcast
                     || scalar_bf16_broadcast_once)
                            ? (rhs_arg_static_params_.preserve_gpr_helpers
                                              ? std::initializer_list<
                                                        Xbyak::Reg64>(
                                                        {rhs_arg_static_params_
                                                                        .rhs_addr_reg})
                                              : std::initializer_list<
                                                        Xbyak::Reg64>())
                            : rhs_arg_static_params_.preserve_gpr_helpers
                                    && should_preserve_w_or_oc_offset_conversion_regs
                            ? std::initializer_list<
                                      Xbyak::Reg64>({rhs_arg_static_params_
                                                             .rhs_addr_reg,
                                      rhs_arg_static_params_.rhs_helper_reg,
                                      rhs_arg_static_params_.rhs_addr_cache_reg,
                                      host_->rax, host_->rdx, host_->r8})
                            : rhs_arg_static_params_.preserve_gpr_helpers
                                    && should_preserve_mb_sp_offset_conversion_regs
                            ? std::initializer_list<Xbyak::Reg64>(
                                      {rhs_arg_static_params_.rhs_addr_reg,
                                              rhs_arg_static_params_
                                                      .rhs_helper_reg,
                                              rhs_arg_static_params_
                                                      .rhs_addr_cache_reg,
                                              host_->rax, host_->rdx, host_->r8,
                                              host_->r9})
                            : rhs_arg_static_params_.preserve_gpr_helpers
                            ? std::initializer_list<Xbyak::Reg64>(
                                      {rhs_arg_static_params_.rhs_addr_reg,
                                              rhs_arg_static_params_
                                                      .rhs_helper_reg,
                                              rhs_arg_static_params_
                                                      .rhs_addr_cache_reg,
                                              host_->rax, host_->rdx})
                            : should_preserve_w_or_oc_offset_conversion_regs
                            ? std::initializer_list<Xbyak::Reg64>(
                                      {rhs_arg_static_params_
                                                      .rhs_addr_cache_reg,
                                              host_->rax, host_->rdx,
                                              host_->r8})
                            : shoud_preserve_oc_d_offset_conversion_regs
                            ? std::initializer_list<Xbyak::Reg64>(
                                      {rhs_arg_static_params_
                                                      .rhs_addr_cache_reg,
                                              host_->rax, host_->rdx})
                            : should_preserve_mb_sp_offset_conversion_regs
                            ? std::initializer_list<Xbyak::Reg64>(
                                      {rhs_arg_static_params_
                                                      .rhs_addr_cache_reg,
                                              host_->rax, host_->rdx, host_->r8,
                                              host_->r9})
                            : use_offset_conversions
                            ? std::initializer_list<Xbyak::Reg64>(
                                      {rhs_arg_static_params_
                                                      .rhs_addr_cache_reg,
                                              host_->rax, host_->rdx})
                            : std::initializer_list<Xbyak::Reg64>()),
            (rhs_arg_static_params_.preserve_vmm_helper && dt_helper_vmm_needed
                            ? std::initializer_list<Xbyak::Xmm>({Vmm(vmm_hint)})
                            : std::initializer_list<Xbyak::Xmm>())};

    if (scalar_bf16_broadcast_once) {
        // Reuse the converted scalar only within this invocation. Exact tails
        // and an overlapping helper retain the per-vector path below.
        const Vmm rhs_vmm(vmm_hint);
        const auto rhs_addr
                = prepare_rhs_arg_addr(start_idx, rhs_arg_idx, post_op,
                        rhs_arg_params, rhs_broadcasting_strategy, true, false);
        execute_broadcast_no_tail(
                rhs_arg_data_type, rhs_vmm, remove_bcast_bit(rhs_addr));
        for (const auto vmm_idx : vmm_idxs) {
            const Vmm dst_vmm(vmm_idx);
            execute_binary(post_op.binary.alg, dst_vmm, dst_vmm, rhs_vmm);
        }
        return;
    }

    bool vmm0_was_preserved = false;
    static const Vmm zero_vmm(0);
    if (post_op.is_prelu() && has_avx512_core_)
        push_opmask(host_, get_aux_kmask());

    Xbyak::Address rhs1_arg_addr {};
    Xbyak::Address rhs2_arg_addr {};

    // Phase 3 Apply binary post-op over all vmms.
    for (const auto vmm_idx : vmm_idxs) {
        const bool is_start_idx = vmm_idx == start_idx;
        const bool with_tail = rhs_arg_static_params_.is_tail
                && vmm_tail_idx.find(vmm_idx) != vmm_tail_idx.cend()
                && IMPLICATION(rhs_broadcasting_strategy
                                == broadcasting_strategy_t::scalar,
                        rhs_arg_static_params_.use_exact_tail_scalar_bcast);
        const Vmm tern_tmp_vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);
        const auto local_vmm_preservation = should_preserve_vmm(
                vmm_idx, vmm_hint, max_vmm_idx, dt_helper_vmm_needed);
        const bool &vmm_preservation_needed = local_vmm_preservation.first;
        const Vmm dst_vmm(vmm_idx);

        // For binary ops with ternary inputs, a temporary vmm will be needed
        // for the additional input.
        if (needs_ternary_input) {
            const auto rhs2_arg_data_type
                    = get_src2_desc(post_op, dst_d).data_type;
            rhs2_arg_addr = prepare_rhs_arg_addr(vmm_idx, rhs_arg_idx + 1,
                    post_op, rhs_arg_params,
                    broadcasting_strategy_t::no_broadcast, true, true);

            const bool ternary_with_tail = rhs_arg_static_params_.is_tail
                    && vmm_tail_idx.find(vmm_idx) != vmm_tail_idx.cend();

            load_rhs(rhs2_arg_data_type, tern_tmp_vmm, rhs2_arg_addr,
                    tail_load_mode, ternary_with_tail);

            if (types::is_integral_dt(rhs2_arg_data_type))
                cvt_to_f32(tern_tmp_vmm);
        }

        // For binary operations with ternary inputs and scalar broadcasting,
        // the src1 address needs to be reloaded as it is overwritten when src2
        // address is loaded.
        const bool load_addr
                = (rhs_broadcasting_strategy == broadcasting_strategy_t::scalar)
                && needs_ternary_input;

        const bool params_differ = rhs_arg_params_differ(vmm_idx, vmm_idx - 1,
                rhs_arg_params, rhs_broadcasting_strategy);
        const bool need_new_addr = is_start_idx || params_differ || load_addr;
        if (need_new_addr) {
            const bool is_baddr_loop_invariant
                    = utils::one_of(rhs_broadcasting_strategy,
                            broadcasting_strategy_t::no_broadcast,
                            broadcasting_strategy_t::per_mb_spatial);
            const bool is_first = is_baddr_loop_invariant
                    ? (is_start_idx || load_addr)
                    : need_new_addr;
            rhs1_arg_addr = prepare_rhs_arg_addr(vmm_idx, rhs_arg_idx, post_op,
                    rhs_arg_params, rhs_broadcasting_strategy, is_first, false);
        }

        if (vmm_preservation_needed) {
            const Vmm vmm_to_preserve(local_vmm_preservation.second);
            push_vmm(host_, vmm_to_preserve);
            if (needs_ternary_input)
                inject_binary_with_ternary_op(post_op, dst_vmm, rhs1_arg_addr,
                        tern_tmp_vmm, with_tail, tail_load_mode);
            else
                inject_binary(post_op, dst_vmm, rhs1_arg_addr, with_tail,
                        tail_load_mode);
            pop_vmm(host_, vmm_to_preserve);
            // in case all Vmm are occupied, Vmm(0) is chosen for tmp by default,
            // so it's content needs to be preserved...

            push_vmm(host_, zero_vmm);
            vmm0_was_preserved = true;
        } else {
            if (needs_ternary_input)
                inject_binary_with_ternary_op(post_op, dst_vmm, rhs1_arg_addr,
                        tern_tmp_vmm, with_tail, tail_load_mode);
            else
                inject_binary(post_op, dst_vmm, rhs1_arg_addr, with_tail,
                        tail_load_mode);
        }
    }
    // ...and restored afterwards
    if (vmm0_was_preserved) pop_vmm(host_, zero_vmm);
    if (post_op.is_prelu() && has_avx512_core_)
        pop_opmask(host_, get_aux_kmask());
}

template <typename Vmm>
Xbyak::Address jit_uni_binary_injector_t<Vmm>::prepare_rhs_arg_addr(int vmm_idx,
        int rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        const broadcasting_strategy_t rhs_broadcasting_strategy, bool is_first,
        bool is_ternary_input) const {

    static constexpr auto rhs_arg_ptr_size = sizeof(const void *);
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    const auto &abi_param_offset = rhs_arg_static_params_.abi_param_offset;
    const auto &rhs_helper_reg = rhs_arg_static_params_.rhs_helper_reg;
    const auto &rhs_md = is_ternary_input
            ? get_src2_desc(post_op, rhs_arg_static_params_.dst_d)
            : get_src1_desc(post_op, rhs_arg_static_params_.dst_d);
    const auto rhs_arg_elem_size = types::data_type_size(rhs_md.data_type);
    const memory_desc_wrapper rhs_d(rhs_md);

    if (is_first || is_ternary_input) {
        host_->mov(rhs_addr_reg, host_->ptr[param1_ + abi_param_offset]);
        host_->mov(rhs_addr_reg,
                host_->ptr[rhs_addr_reg + rhs_arg_idx * rhs_arg_ptr_size]);
    }

    switch (rhs_broadcasting_strategy) {
        case broadcasting_strategy_t::scalar: return host_->ptr_b[rhs_addr_reg];
        case broadcasting_strategy_t::no_broadcast: {
            // The address is not cached for the ternary input to avoid
            // overwriting the address for the binary src input as they are
            // loaded consecutively.
            // TODO: not using caching the addresses for the ternary inputs
            // can slow down the operation. For faster computation,
            // additional cache registers can be introduced to handle
            // the ternary operations.
            // TODO: Currently, there is no broadcasting support for the
            // ternary tensor, hence the address calculation can be carried out
            // within the injector. As the support is extended to include
            // broadcasting strategies like per_oc, it will be beneficial to
            // move the calculation outside to allow iteration over multiple
            // kernels.
            append_no_broadcast_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size, is_first,
                    (!is_ternary_input));

            return host_->ptr[rhs_addr_reg];
        }
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial: {
            append_oc_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size, is_first);

            return rhs_broadcasting_strategy
                            == broadcasting_strategy_t::per_oc_spatial
                    ? host_->ptr_b[rhs_addr_reg]
                    : host_->ptr[rhs_addr_reg];
        }
        case broadcasting_strategy_t::per_oc_d: {
            append_oc_d_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size, is_first);
            return host_->ptr_b[rhs_addr_reg];
        }
        case broadcasting_strategy_t::per_mb_spatial: {
            append_mb_sp_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size, is_first,
                    rhs_d);

            return host_->ptr[rhs_addr_reg];
        }
        case broadcasting_strategy_t::per_mb_w: {
            append_mb_w_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size, is_first);

            return host_->ptr[rhs_addr_reg];
        }
        case broadcasting_strategy_t::per_w: {
            append_w_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size, is_first);

            return host_->ptr[rhs_addr_reg];
        }
        case broadcasting_strategy_t::per_mb: {
            append_mb_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size, is_first);

            return host_->ptr_b[rhs_addr_reg];
        }
        case broadcasting_strategy_t::per_hw: {
            append_hw_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size, is_first);

            return host_->ptr[rhs_addr_reg];
        }
        case broadcasting_strategy_t::batch: {
            append_oc_spatial_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size, is_first);

            return host_->ptr[rhs_addr_reg];
        }
        case broadcasting_strategy_t::spatial: {
            append_mb_oc_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size, is_first);

            return host_->ptr[rhs_addr_reg];
        }
        default: assert(false && "Broadcasting type not supported");
    }

    return host_->ptr[rhs_addr_reg];
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::append_no_broadcast_offset(
        const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
        const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes, bool is_first, bool cache_addr) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();
    if (is_out_addr || is_out_reg) {
        Xbyak::Address out_addr = is_out_addr ? it_out_addr->second
                                              : host_->ptr[it_out_reg->second];
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        const auto &addr_cache_reg = rhs_arg_static_params_.rhs_addr_cache_reg;

        if (is_first || !cache_addr) {
            calculate_no_broadcast_base(out_addr, tmp_reg);
            if (elem_size_bytes > 1) {
                const int shift_val = math::ilog2q(elem_size_bytes);
                host_->sal(tmp_reg, shift_val);
            }
            host_->add(addr_reg, tmp_reg);
            if (cache_addr) host_->mov(addr_cache_reg, addr_reg);
        } else {
            host_->mov(addr_reg, addr_cache_reg);
        }

        if (it_off_val != vmm_idx_to_out_elem_off_val.end()) {
            calculate_no_broadcast_partial(
                    it_off_val->second, tmp_reg, elem_size_bytes);
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_no_broadcast_base(
        Xbyak::Address addr, const Xbyak::Reg64 &out_reg) const {
    // addr may carry a zword[] (512-bit) size annotation because it was built
    // for a Zmm vmm access. LEA only needs the address computation, not the
    // size, so strip the annotation to avoid xbyak STRICT_CHECK_MEM_REG_SIZE
    // flagging the Reg64 vs zword size mismatch.
    host_->lea(out_reg, host_->ptr[addr.getRegExp()]);
    host_->sub(out_reg,
            host_->ptr[param1_ + rhs_arg_static_params_.dst_orig_offset]);
    host_->shr(out_reg,
            math::ilog2q(types::data_type_size(
                    rhs_arg_static_params_.dst_d.data_type())));
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_no_broadcast_partial(
        const dim_t offset, const Xbyak::Reg64 &out_reg,
        dim_t elem_size_bytes) const {
    const auto offset_adj = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    host_->mov(out_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::append_oc_offset(
        const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
        const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes, bool is_first) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        Xbyak::Address out_addr = is_out_addr ? it_out_addr->second
                                              : host_->ptr[it_out_reg->second];
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        const auto &addr_cache_reg = rhs_arg_static_params_.rhs_addr_cache_reg;

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        if (is_first) {
            calculate_no_broadcast_base(out_addr, tmp_reg);

            const auto rax = host_->rax;
            const auto rdx = host_->rdx;
            const auto r8 = host_->r8;

            const injector_utils::conditional_register_preserve_guard_t
                    register_guard {is_out_reg
                                    ? utils::one_of(
                                              it_out_reg->second, rax, rdx, r8)
                                    : false,
                            host_,
                            {is_out_reg ? it_out_reg->second : Xbyak::Reg64()}};

            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_oc_ncsp_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::c_blocked:
                    calculate_oc_blocked_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::nspc:
                    calculate_oc_nspc_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::cspn:
                    calculate_oc_cspn_base(strides, tmp_reg);
                    break;
                default: assert(!"Unknown layout");
            }

            if (elem_size_bytes == 1) {
                host_->add(addr_reg, rax);
            } else {
                const int shift_val = math::ilog2q(elem_size_bytes);
                host_->mov(tmp_reg, rax);
                host_->sal(tmp_reg, shift_val);
                host_->add(addr_reg, tmp_reg);
            }
            host_->mov(addr_cache_reg, addr_reg);
        } else {
            host_->mov(addr_reg, addr_cache_reg);
        }

        if (it_off_val != vmm_idx_to_out_elem_off_val.end()) {
            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_oc_ncsp_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::c_blocked:
                    calculate_oc_blocked_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::nspc:
                    calculate_oc_nspc_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::cspn:
                    calculate_oc_cspn_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                default: assert(!"Unknown layout");
            }
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_ncsp_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // c = (offset % strides[0]) / strides[1]
    // output = rax
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[0]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
    host_->mov(tmp_reg, strides[1]);
    host_->mov(rax, rdx);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_ncsp_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // c = (offset % strides[0]) / strides[1]
    const auto offset_adj
            = ((offset >> math::ilog2q(types::data_type_size(
                        rhs_arg_static_params_.dst_d.data_type())))
                      % strides[0])
            / strides[1];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_blocked_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // c = ((offset % strides[0]) / strides[1]) * strides[ndims - 1] + offset % blk_size
    // output = rax
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const int simd_w = static_cast<int>(
            isa_max_vlen(isa_) / types::data_type_size(dst_d.data_type()));
    const int blk_size = static_cast<int>(dst_d.blocking_desc().inner_blks[0]);
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;

    calculate_oc_ncsp_base(strides, tmp_reg);

    if (blk_size > simd_w) {
        // extract c % blk_size
        host_->mov(r8, rax);
        host_->mov(rax, rdx);
        host_->mov(tmp_reg, blk_size);
        host_->xor_(rdx, rdx);
        host_->div(tmp_reg);
        host_->mov(rax, r8);
        host_->mov(r8, rdx);
    }

    host_->mov(tmp_reg, blk_size);
    host_->mul(tmp_reg);
    if (blk_size > simd_w) host_->add(rax, r8);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_blocked_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // c = ((offset % strides[0]) / strides[1]) * strides[ndims - 1] + offset % blk_size
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const int blk_size = static_cast<int>(dst_d.blocking_desc().inner_blks[0]);
    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto offset_adj = ((offset_shr % strides[0]) / strides[1]) * blk_size
            + offset_shr % blk_size;
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_nspc_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // c = offset % C
    // output = rax
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto C = rhs_arg_static_params_.dst_d.dims()[1];

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, C);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
    host_->mov(rax, rdx);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_nspc_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // c = offset % C
    const auto C = rhs_arg_static_params_.dst_d.dims()[1];
    const auto offset_adj = (offset >> math::ilog2q(types::data_type_size(
                                     rhs_arg_static_params_.dst_d.data_type())))
            % C;
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_cspn_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // c = offset / strides[1]
    // output = rax
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[1]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_cspn_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // c = offset / strides[1]
    const auto offset_adj = (offset >> math::ilog2q(types::data_type_size(
                                     rhs_arg_static_params_.dst_d.data_type())))
            / strides[1];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::append_oc_d_offset(
        const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
        const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes, bool is_first) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        Xbyak::Address out_addr = is_out_addr ? it_out_addr->second
                                              : host_->ptr[it_out_reg->second];
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        const auto &addr_cache_reg = rhs_arg_static_params_.rhs_addr_cache_reg;

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        if (is_first) {
            calculate_no_broadcast_base(out_addr, tmp_reg);

            const auto rax = host_->rax;
            const auto rdx = host_->rdx;

            const injector_utils::conditional_register_preserve_guard_t
                    register_guard {is_out_reg
                                    ? utils::one_of(
                                              it_out_reg->second, rax, rdx)
                                    : false,
                            host_, {it_out_reg->second}};

            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_oc_d_ncsp_base(strides, tmp_reg);
                    break;
                default: assert(!"unimplemented layout");
            }

            if (elem_size_bytes == 1) {
                host_->add(addr_reg, rax);
            } else {
                const int shift_val = math::ilog2q(elem_size_bytes);
                host_->mov(tmp_reg, rax);
                host_->sal(tmp_reg, shift_val);
                host_->add(addr_reg, tmp_reg);
            }
            host_->mov(addr_cache_reg, addr_reg);
        } else {
            host_->mov(addr_reg, addr_cache_reg);
        }

        if (it_off_val != vmm_idx_to_out_elem_off_val.end()) {
            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_oc_d_ncsp_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                default: assert(!"unimplemented layout");
            }
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_d_ncsp_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {

    /* 
    DST: b, g, g/c, sp -> a, b, c, d
    POB: 1, g, g/c, 1 -> 1, b, c, 1
    dstrides = {B*C*D, C*D, D, 1}
    pstrides = {B*C, C, 1, 1}
    dst_offset = a*dstride[0] + b*dstride[1] + c*dstride[2] + d*dstride[3]
               = a*B*C*D + b*C*D + c*D + d
    pob_offset = b*pstride[1] + c
               = b*C + c
    dst_offset = a*dstride[0] + b*dstride[1] + c*dstride[2] + d*dstride[3] = a*B*C*D + b*C*D + c*D + d 
    pob_offset = b*pstride[1] + c = b*C + c

    dst_offset % dstride[0] =  b*dstride[1] + c*dstride[2] + d*dstride[3] = b*C*D + c*D + d
    dst_offset % dstride[0] / dstride[2]  = b*C + c + d/D  ; d/D = 0 -> max(d) = D-1 -> (D-1)/D = 0 floor int div. -> d/D = 0
    dst_offset % dstride[0] / dstride[2] = b*C + c
*/

    // dst_offset % dstride[0] / dstride[2]
    // output = rax
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[0]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
    host_->mov(tmp_reg, strides[2]);
    host_->mov(rax, rdx);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_d_ncsp_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // dst_offset % dstride[0] / dstride[2] = b*C + c
    const auto offset_adj
            = ((offset >> math::ilog2q(types::data_type_size(
                        rhs_arg_static_params_.dst_d.data_type())))
                      % strides[0])
            / strides[2];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::append_mb_sp_offset(
        const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
        const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes, bool is_first,
        const memory_desc_wrapper &rhs_d) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        Xbyak::Address out_addr = is_out_addr ? it_out_addr->second
                                              : host_->ptr[it_out_reg->second];
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        const auto &addr_cache_reg = rhs_arg_static_params_.rhs_addr_cache_reg;

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        if (is_first) {
            calculate_no_broadcast_base(out_addr, tmp_reg);

            const auto rax = host_->rax;
            const auto rdx = host_->rdx;
            const auto r8 = host_->r8;
            const auto r9 = host_->r9;

            const injector_utils::conditional_register_preserve_guard_t
                    register_guard {is_out_reg
                                    ? utils::one_of(it_out_reg->second, rax,
                                              rdx, r8, r9)
                                    : false,
                            host_,
                            {is_out_reg ? it_out_reg->second : Xbyak::Reg64()}};

            if (layout == injector_utils::layout_t::ncsp && rhs_d.is_plain()
                    && !rhs_d.is_dense(true)) {
                calculate_mb_sp_ncsp_base_rhs_strided(dst_d, rhs_d, strides,
                        addr_reg, tmp_reg, elem_size_bytes);
            } else {
                switch (layout) {
                    case injector_utils::layout_t::ncsp:
                        calculate_mb_sp_ncsp_base(strides, tmp_reg);
                        break;
                    case injector_utils::layout_t::c_blocked:
                        calculate_mb_sp_blocked_base(strides, tmp_reg);
                        break;
                    case injector_utils::layout_t::nspc:
                        calculate_mb_sp_nspc_base(strides, tmp_reg);
                        break;
                    case injector_utils::layout_t::cspn:
                        calculate_mb_sp_cspn_base(strides, tmp_reg);
                        break;
                    default: assert(!"Unknown layout");
                }

                if (elem_size_bytes == 1) {
                    host_->add(addr_reg, rax);
                } else {
                    const int shift_val = math::ilog2q(elem_size_bytes);
                    host_->mov(tmp_reg, rax);
                    host_->sal(tmp_reg, shift_val);
                    host_->add(addr_reg, tmp_reg);
                }
            }
            host_->mov(addr_cache_reg, addr_reg);
        } else {
            host_->mov(addr_reg, addr_cache_reg);
        }

        if (it_off_val != vmm_idx_to_out_elem_off_val.end()) {
            if (layout == injector_utils::layout_t::ncsp && rhs_d.is_plain()
                    && !rhs_d.is_dense(true)) {
                calculate_mb_sp_ncsp_partial_rhs_strided(dst_d, rhs_d,
                        it_off_val->second, addr_reg, tmp_reg, elem_size_bytes);
            } else {
                switch (layout) {
                    case injector_utils::layout_t::ncsp:
                        calculate_mb_sp_ncsp_partial(strides,
                                it_off_val->second, tmp_reg, elem_size_bytes);
                        break;
                    case injector_utils::layout_t::c_blocked:
                        calculate_mb_sp_blocked_partial(strides,
                                it_off_val->second, tmp_reg, elem_size_bytes);
                        break;
                    case injector_utils::layout_t::nspc:
                        calculate_mb_sp_nspc_partial(strides,
                                it_off_val->second, tmp_reg, elem_size_bytes);
                        break;
                    case injector_utils::layout_t::cspn:
                        calculate_mb_sp_cspn_partial(strides,
                                it_off_val->second, tmp_reg, elem_size_bytes);
                        break;
                    default: assert(!"Unknown layout");
                }
                host_->add(addr_reg, tmp_reg);
            }
        }
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_sp_ncsp_base_rhs_strided(
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &rhs_d,
        const dim_t *dst_strides, const Xbyak::Reg64 &addr_reg,
        const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const {
    // offset = n*dst_stride_n + c*dst_stride_c + d*dst_stride_d + h*dst_stride_h + w*dst_stride_w
    // rhs_off = n*rhs_stride_n + d*rhs_stride_d + h*rhs_stride_h + w*rhs_stride_w
    // per_mb_spatial: channel c is broadcast, so rem = (offset % dst_stride_n) % dst_stride_c.
    // output = r8
    const auto ndims = dst_d.ndims();
    const auto &rhs_strides = rhs_d.blocking_desc().strides;

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;

    host_->xor_(r8, r8);
    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, dst_strides[0]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
    // rax = n, rdx = rem
    if (rhs_strides[0] != 0) {
        host_->mov(tmp_reg, rhs_strides[0]);
        host_->imul(rax, tmp_reg);
        host_->add(r8, rax);
        // rhs_off += n * rhs_stride_n
    }

    host_->mov(rax, rdx);
    host_->mov(tmp_reg, dst_strides[1]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);

    if (ndims >= 5) {
        host_->mov(rax, rdx);
        host_->mov(tmp_reg, dst_strides[ndims - 3]);
        host_->xor_(rdx, rdx);
        host_->div(tmp_reg);
        if (rhs_strides[ndims - 3] != 0) {
            host_->mov(tmp_reg, rhs_strides[ndims - 3]);
            host_->imul(rax, tmp_reg);
            host_->add(r8, rax);
            // rhs_off += d * rhs_stride_d
        }
    }

    if (ndims >= 4) {
        host_->mov(rax, rdx);
        host_->mov(tmp_reg, dst_strides[ndims - 2]);
        host_->xor_(rdx, rdx);
        host_->div(tmp_reg);
        if (rhs_strides[ndims - 2] != 0) {
            host_->mov(tmp_reg, rhs_strides[ndims - 2]);
            host_->imul(rax, tmp_reg);
            host_->add(r8, rax);
            // rhs_off += h * rhs_stride_h
        }
    }

    if (ndims >= 3) {
        host_->mov(rax, rdx);
        if (dst_strides[ndims - 1] != 1) {
            host_->mov(tmp_reg, dst_strides[ndims - 1]);
            host_->xor_(rdx, rdx);
            host_->div(tmp_reg);
        }
        if (rhs_strides[ndims - 1] != 0) {
            host_->mov(tmp_reg, rhs_strides[ndims - 1]);
            host_->imul(rax, tmp_reg);
            host_->add(r8, rax);
            // rhs_off += w * rhs_stride_w
        }
    }

    if (elem_size_bytes == 1) {
        host_->add(addr_reg, r8);
    } else {
        const int shift_val = math::ilog2q(elem_size_bytes);
        host_->mov(tmp_reg, r8);
        host_->sal(tmp_reg, shift_val);
        host_->add(addr_reg, tmp_reg);
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_sp_ncsp_partial_rhs_strided(
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &rhs_d,
        dim_t dst_offset_bytes, const Xbyak::Reg64 &addr_reg,
        const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const {
    const dim_t rhs_offset
            = rhs_offset_per_mb_spatial(dst_d, rhs_d, dst_offset_bytes);
    if (rhs_offset == 0) return;

    if (elem_size_bytes == 1) {
        host_->add(addr_reg, rhs_offset);
    } else {
        const int shift_val = math::ilog2q(elem_size_bytes);
        const uint64_t rhs_offset_bytes = static_cast<uint64_t>(rhs_offset)
                << shift_val;
        host_->mov(tmp_reg, rhs_offset_bytes);
        host_->add(addr_reg, tmp_reg);
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_sp_ncsp_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_sp_off = (n * (stride_n/C)) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_sp_off = offset - (c * stride_c) - (n * (C - 1)DHW)
    // output = rax
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();
    const auto C_padded = dst_d.padded_dims()[1];
    const auto D = (ndims >= 5) ? dst_d.dims()[ndims - 3] : 1;
    const auto H = (ndims >= 4) ? dst_d.dims()[ndims - 2] : 1;
    const auto W = (ndims >= 3) ? dst_d.dims()[ndims - 1] : 1;

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;
    const auto r9 = host_->r9;

    host_->mov(rax, tmp_reg);
    host_->mov(r9, strides[0]);
    host_->xor_(rdx, rdx);
    host_->div(r9);
    host_->mov(r8, rax);
    // r8 = n
    host_->mov(r9, strides[1]);
    host_->mov(rax, rdx);
    host_->xor_(rdx, rdx);
    host_->div(r9);
    host_->mul(r9);
    // rax = c * stride_c
    host_->sub(tmp_reg, rax);
    // tmp_reg = offset - c * stride_c
    host_->mov(rax, r8);
    // rax = n
    host_->mov(r9, (C_padded - 1) * D * H * W);
    // n(C - 1)DHW = nCDHW - nDHW
    host_->mul(r9);
    // rax = n(C - 1)DHW
    host_->sub(tmp_reg, rax);
    host_->mov(rax, tmp_reg);
    // rax = offset - (c * stride_c) - (n * (C - 1)DHW)
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_sp_ncsp_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_sp_off = (n * (stride_n/C)) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_sp_off = offset - (c * stride_c) - (n * (C - 1)DHW)

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();
    const auto C_padded = dst_d.padded_dims()[1];
    const auto D = (ndims >= 5) ? dst_d.dims()[ndims - 3] : 1;
    const auto H = (ndims >= 4) ? dst_d.dims()[ndims - 2] : 1;
    const auto W = (ndims >= 3) ? dst_d.dims()[ndims - 1] : 1;

    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto c = (offset_shr % strides[0]) / strides[1];
    const auto n = offset_shr / strides[0];
    const auto offset_adj
            = offset_shr - (c * strides[1]) - (n * (C_padded - 1) * D * H * W);
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_sp_blocked_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // mb_sp_off = offset - (c * stride_c) - (n * (C - 1)DHW) - c % blk_size
    // output = rax
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const int simd_w = static_cast<int>(
            isa_max_vlen(isa_) / types::data_type_size(dst_d.data_type()));
    const int blk_size = static_cast<int>(dst_d.blocking_desc().inner_blks[0]);

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;

    if (blk_size > simd_w) {
        // substract c % blk_size
        host_->mov(r8, tmp_reg);
        host_->mov(rax, tmp_reg);
        host_->mov(tmp_reg, blk_size);
        host_->xor_(rdx, rdx);
        host_->div(tmp_reg);
        host_->mov(tmp_reg, r8);
        host_->sub(tmp_reg, rdx);
    }

    calculate_mb_sp_ncsp_base(strides, tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_sp_blocked_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // mb_sp_off = offset - (c * stride_c) - (n * (C - 1)DHW) - c % blk_size

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();
    const auto C_padded = dst_d.padded_dims()[1];
    const auto D = (ndims >= 5) ? dst_d.dims()[ndims - 3] : 1;
    const auto H = (ndims >= 4) ? dst_d.dims()[ndims - 2] : 1;
    const auto W = (ndims >= 3) ? dst_d.dims()[ndims - 1] : 1;
    const int blk_size = static_cast<int>(dst_d.blocking_desc().inner_blks[0]);

    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto c = (offset_shr % strides[0]) / strides[1];
    const auto n = offset_shr / strides[0];
    const auto offset_adj = offset_shr - (c * strides[1])
            - (n * (C_padded - 1) * D * H * W) - c % blk_size;
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_sp_nspc_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // mb_sp_off = nDHW + dHW + hW + w
    // mb_sp_off = offset / C
    // output = rax
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto C = rhs_arg_static_params_.dst_d.padded_dims()[1];

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, C);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_sp_nspc_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // mb_sp_off = nDHW + dHW + hW + w
    // mb_sp_off = offset / C
    const auto C = rhs_arg_static_params_.dst_d.padded_dims()[1];
    const auto offset_adj = (offset >> math::ilog2q(types::data_type_size(
                                     rhs_arg_static_params_.dst_d.data_type())))
            / C;
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_sp_cspn_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // mb_sp_off = dHWN + hWN + wN + n
    // mb_sp_off = offset % stride_c
    // output = rax
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[1]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
    host_->mov(rax, rdx);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_sp_cspn_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // mb_sp_off = dHWN + hWN + wN + n
    // mb_sp_off = offset % stride_c
    const auto offset_adj = (offset >> math::ilog2q(types::data_type_size(
                                     rhs_arg_static_params_.dst_d.data_type())))
            % strides[1];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::append_mb_w_offset(
        const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
        const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes, bool is_first) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        Xbyak::Address out_addr = is_out_addr ? it_out_addr->second
                                              : host_->ptr[it_out_reg->second];
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        const auto &addr_cache_reg = rhs_arg_static_params_.rhs_addr_cache_reg;

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        if (is_first) {
            calculate_no_broadcast_base(out_addr, tmp_reg);

            const auto rax = host_->rax;
            const auto rdx = host_->rdx;
            const auto r8 = host_->r8;
            const auto r9 = host_->r9;

            const injector_utils::conditional_register_preserve_guard_t
                    register_guard {is_out_reg
                                    ? utils::one_of(it_out_reg->second, rax,
                                              rdx, r8, r9)
                                    : false,
                            host_,
                            {is_out_reg ? it_out_reg->second : Xbyak::Reg64()}};

            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_mb_w_ncsp_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::c_blocked:
                    calculate_mb_w_blocked_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::nspc:
                    calculate_mb_w_nspc_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::cspn:
                    calculate_mb_w_cspn_base(strides, tmp_reg);
                    break;
                default: assert(!"Unknown layout");
            }

            if (elem_size_bytes == 1) {
                host_->add(addr_reg, rax);
            } else {
                const int shift_val = math::ilog2q(elem_size_bytes);
                host_->mov(tmp_reg, rax);
                host_->sal(tmp_reg, shift_val);
                host_->add(addr_reg, tmp_reg);
            }
            host_->mov(addr_cache_reg, addr_reg);
        } else {
            host_->mov(addr_reg, addr_cache_reg);
        }

        if (it_off_val != vmm_idx_to_out_elem_off_val.end()) {
            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_mb_w_ncsp_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::c_blocked:
                    calculate_mb_w_blocked_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::nspc:
                    calculate_mb_w_nspc_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::cspn:
                    calculate_mb_w_cspn_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                default: assert(!"Unknown layout");
            }
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_w_ncsp_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_w_off = (n * (stride_n/(C*D*H))) + (w * stride_w)
    // output = rax
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();
    const auto C_padded = dst_d.padded_dims()[1];
    const auto D = (ndims >= 5) ? dst_d.dims()[ndims - 3] : 1;
    const auto H = (ndims >= 4) ? dst_d.dims()[ndims - 2] : 1;

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;
    const auto r9 = host_->r9;

    host_->mov(rax, tmp_reg);
    host_->mov(r9, strides[0]);
    host_->xor_(rdx, rdx);
    host_->div(r9);
    host_->mov(r8, rax);
    // r8 = n

    host_->mov(r9, strides[1]);
    host_->mov(rax, rdx);
    host_->xor_(rdx, rdx);
    host_->div(r9);

    if (ndims >= 5) {
        host_->mov(r9, strides[ndims - 3]);
        host_->mov(rax, rdx);
        host_->xor_(rdx, rdx);
        host_->div(r9);
    }
    if (ndims >= 4) {
        host_->mov(r9, strides[ndims - 2]);
        host_->mov(rax, rdx);
        host_->xor_(rdx, rdx);
        host_->div(r9);
    }
    if (ndims >= 3) {
        host_->mov(r9, strides[ndims - 1]);
        host_->mov(rax, rdx);
        host_->xor_(rdx, rdx);
        host_->div(r9);
        host_->mul(r9);
        host_->mov(tmp_reg, rax);
        // tmp_reg = w * stride_w
    }
    // tmp_reg = w * stride_w
    host_->mov(rax, r8);
    // rax = n
    host_->mov(r9, strides[0] / (C_padded * D * H));
    host_->mul(r9);
    // rax = n * (stride_n/(C*D*H))
    if (ndims >= 3) host_->add(rax, tmp_reg);
    // rax = (n * (stride_n/(C*D*H))) + (w * stride_w)
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_w_ncsp_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_w_off = (n * (stride_n/(C*D*H))) + (w * stride_w)
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();
    const auto C_padded = dst_d.padded_dims()[1];
    const auto D = (ndims >= 5) ? dst_d.dims()[ndims - 3] : 1;
    const auto H = (ndims >= 4) ? dst_d.dims()[ndims - 2] : 1;

    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto n = offset_shr / strides[0];
    const auto w = (offset_shr % strides[ndims - 2]) / strides[ndims - 1];
    const auto offset_adj = (n * (strides[0] / (C_padded * D * H)))
            + (w * strides[ndims - 1]);
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_w_blocked_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // mb_w_off = (n * (stride_n/(C*D*H))) + (w * stride_w)
    // output = rax
    calculate_mb_sp_ncsp_base(strides, tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_w_blocked_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // mb_w_off = (n * (stride_n/(C*D*H))) + (w * stride_w)
    calculate_mb_w_ncsp_partial(strides, offset, tmp_reg, elem_size_bytes);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_w_nspc_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // mb_w_off = nW + w
    // output = rax

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();
    const auto C_padded = dst_d.padded_dims()[1];
    const auto D = (ndims >= 5) ? dst_d.dims()[ndims - 3] : 1;
    const auto H = (ndims >= 4) ? dst_d.dims()[ndims - 2] : 1;

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;
    const auto r9 = host_->r9;

    host_->mov(rax, tmp_reg);
    host_->mov(r9, strides[0]);
    host_->xor_(rdx, rdx);
    host_->div(r9);
    host_->mov(r8, rax);
    // r8 = n
    if (ndims >= 5) {
        host_->mov(r9, strides[ndims - 3]);
        host_->mov(rax, rdx);
        host_->xor_(rdx, rdx);
        host_->div(r9);
    }
    if (ndims >= 4) {
        host_->mov(r9, strides[ndims - 2]);
        host_->mov(rax, rdx);
        host_->xor_(rdx, rdx);
        host_->div(r9);
    }
    if (ndims >= 3) {
        host_->mov(r9, strides[ndims - 1]);
        host_->mov(rax, rdx);
        host_->xor_(rdx, rdx);
        host_->div(r9);
        host_->mov(tmp_reg, rax);
        // tmp_reg = w
    }
    host_->mov(rax, r8);
    // rax = n
    host_->mov(r9, strides[0] / (D * H * C_padded));
    host_->mul(r9);
    // rax = nW
    if (ndims >= 3) host_->add(rax, tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_w_nspc_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // mb_w_off = nW + w
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();
    const auto W = (ndims >= 3) ? dst_d.dims()[ndims - 1] : 1;

    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto n = offset_shr / strides[0];
    const auto w = (offset_shr % strides[ndims >= 4 ? ndims - 2 : 0])
            / strides[ndims - 1];
    const auto offset_adj = n * W + w;
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_w_cspn_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // mb_w_off = wN + n
    // output = rax
    const auto ndims = rhs_arg_static_params_.dst_d.ndims();
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[1]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
    host_->mov(rax, rdx);
    if (ndims >= 5) {
        host_->mov(tmp_reg, strides[ndims - 3]);
        host_->mov(rax, rdx);
        host_->xor_(rdx, rdx);
        host_->div(tmp_reg);
    }
    if (ndims >= 4) {
        host_->mov(tmp_reg, strides[ndims - 2]);
        host_->mov(rax, rdx);
        host_->xor_(rdx, rdx);
        host_->div(tmp_reg);
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_w_cspn_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // mb_w_off = wN + n
    const auto ndims = rhs_arg_static_params_.dst_d.ndims();
    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto offset_adj
            = ndims >= 4 ? offset_shr % strides[ndims - 2] : offset_shr;
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::append_hw_offset(
        const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
        const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes, bool is_first) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        Xbyak::Address out_addr = is_out_addr ? it_out_addr->second
                                              : host_->ptr[it_out_reg->second];
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        const auto &addr_cache_reg = rhs_arg_static_params_.rhs_addr_cache_reg;

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        if (is_first) {
            calculate_no_broadcast_base(out_addr, tmp_reg);

            const auto rax = host_->rax;
            const auto rdx = host_->rdx;
            const auto r8 = host_->r8;

            const injector_utils::conditional_register_preserve_guard_t
                    register_guard {is_out_reg
                                    ? utils::one_of(
                                              it_out_reg->second, rax, rdx, r8)
                                    : false,
                            host_,
                            {is_out_reg ? it_out_reg->second : Xbyak::Reg64()}};

            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_hw_ncsp_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::c_blocked:
                    assert(!"Unimplemented");
                    break;
                case injector_utils::layout_t::nspc:
                    assert(!"Unimplemented");
                    break;
                case injector_utils::layout_t::cspn:
                    assert(!"Unimplemented");
                    break;
                default: assert(!"Unknown layout");
            }

            if (elem_size_bytes == 1) {
                host_->add(addr_reg, rax);
            } else {
                const int shift_val = math::ilog2q(elem_size_bytes);
                host_->mov(tmp_reg, rax);
                host_->sal(tmp_reg, shift_val);
                host_->add(addr_reg, tmp_reg);
            }
            host_->mov(addr_cache_reg, addr_reg);
        } else {
            host_->mov(addr_reg, addr_cache_reg);
        }

        if (it_off_val != vmm_idx_to_out_elem_off_val.end()) {
            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_hw_ncsp_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::c_blocked:
                    assert(!"Unimplemented");
                    break;
                case injector_utils::layout_t::nspc:
                    assert(!"Unimplemented");
                    break;
                case injector_utils::layout_t::cspn:
                    assert(!"Unimplemented");
                    break;
                default: assert(!"Unknown layout");
            }
            host_->add(addr_reg, tmp_reg);
        }
    }
}
template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_hw_ncsp_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = nCHW + cHW + hW + w (for 4D)
    // per_spatial_off = offset % HW

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));

    dim_t divisor = strides[ndims - 3];

    const auto offset_adj = offset_shr % divisor;

    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_hw_ncsp_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = nCHW + cHW + hW + w (for 4D)
    // per_spatial_off = hW + w
    // per_spatial_off = offset % HW

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    host_->mov(rax, tmp_reg);

    dim_t divisor = strides[ndims - 3]; // HW
    host_->mov(tmp_reg, divisor);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg); // rdx = offset % HW

    host_->mov(rax, rdx);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::append_w_offset(
        const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
        const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes, bool is_first) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (!is_out_addr && !is_out_reg) return;

    Xbyak::Address out_addr = is_out_addr ? it_out_addr->second
                                          : host_->ptr[it_out_reg->second];
    const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
    const auto &addr_cache_reg = rhs_arg_static_params_.rhs_addr_cache_reg;

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto strides = dst_d.blocking_desc().strides;
    const auto layout = injector_utils::get_layout_type(dst_d);

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;

    // The width index of a position is that position reduced modulo the row
    // length, so it does not distribute over addition. Deriving it separately
    // for the base and for the per-vmm displacement yields
    // `w(base) + w(off)`, which diverges from the required `w(base + off)` once
    // a displacement reaches a full row. The displacement therefore joins the
    // linear position before the reduction. Only the rhs base pointer stays
    // loop-invariant, so that is what the cache register holds.
    assert(!utils::one_of(addr_cache_reg, rax, rdx, r8)
            && "the rhs address cache register must survive the division");
    if (is_first)
        host_->mov(addr_cache_reg, addr_reg);
    else
        host_->mov(addr_reg, addr_cache_reg);

    calculate_no_broadcast_base(out_addr, tmp_reg);
    if (it_off_val != vmm_idx_to_out_elem_off_val.end()) {
        const auto off_elems = it_off_val->second
                >> math::ilog2q(types::data_type_size(dst_d.data_type()));
        if (off_elems) host_->add(tmp_reg, off_elems);
    }

    const injector_utils::conditional_register_preserve_guard_t register_guard {
            is_out_reg ? utils::one_of(it_out_reg->second, rax, rdx, r8)
                       : false,
            host_, {is_out_reg ? it_out_reg->second : Xbyak::Reg64()}};

    switch (layout) {
        case injector_utils::layout_t::ncsp:
            calculate_w_ncsp_base(strides, tmp_reg);
            break;
        case injector_utils::layout_t::c_blocked:
            calculate_w_blocked_base(strides, tmp_reg);
            break;
        case injector_utils::layout_t::nspc:
            calculate_w_nspc_base(strides, tmp_reg);
            break;
        case injector_utils::layout_t::cspn:
            calculate_w_cspn_base(strides, tmp_reg);
            break;
        default: assert(!"Unknown layout");
    }

    if (elem_size_bytes == 1) {
        host_->add(addr_reg, rax);
    } else {
        const int shift_val = math::ilog2q(elem_size_bytes);
        host_->mov(tmp_reg, rax);
        host_->sal(tmp_reg, shift_val);
        host_->add(addr_reg, tmp_reg);
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_w_ncsp_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // w_off = w * stride_w
    // output = rax
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;

    assert(ndims >= 3);

    host_->mov(rax, tmp_reg);
    host_->mov(r8, strides[ndims - 2]);
    host_->xor_(rdx, rdx);
    host_->div(r8);

    host_->mov(r8, strides[ndims - 1]);
    host_->mov(rax, rdx);
    host_->xor_(rdx, rdx);
    host_->div(r8);
    host_->mul(r8);
    // rax = w * stride_w
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_w_blocked_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    calculate_w_ncsp_base(strides, tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_w_nspc_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // w_off = w
    // output = rax
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;

    assert(ndims >= 3);

    host_->mov(rax, tmp_reg);
    host_->mov(r8, strides[ndims - 2]);
    host_->xor_(rdx, rdx);
    host_->div(r8);

    host_->mov(r8, strides[ndims - 1]);
    host_->mov(rax, rdx);
    host_->xor_(rdx, rdx);
    host_->div(r8);
    // rax = w
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_w_cspn_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // w_off = w
    calculate_w_nspc_base(strides, tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::append_mb_offset(
        const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
        const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes, bool is_first) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        Xbyak::Address out_addr = is_out_addr ? it_out_addr->second
                                              : host_->ptr[it_out_reg->second];
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        const auto &addr_cache_reg = rhs_arg_static_params_.rhs_addr_cache_reg;

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        if (is_first) {
            calculate_no_broadcast_base(out_addr, tmp_reg);

            const auto rax = host_->rax;
            const auto rdx = host_->rdx;
            const auto r8 = host_->r8;

            const injector_utils::conditional_register_preserve_guard_t
                    register_guard {is_out_reg
                                    ? utils::one_of(
                                              it_out_reg->second, rax, rdx, r8)
                                    : false,
                            host_,
                            {is_out_reg ? it_out_reg->second : Xbyak::Reg64()}};

            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_mb_ncsp_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::c_blocked:
                    assert(!"Unimplemented");
                    break;
                case injector_utils::layout_t::nspc:
                    calculate_mb_nspc_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::cspn:
                    calculate_mb_cspn_base(strides, tmp_reg);
                    break;
                default: assert(!"Unknown layout");
            }

            if (elem_size_bytes == 1) {
                host_->add(addr_reg, rax);
            } else {
                const int shift_val = math::ilog2q(elem_size_bytes);
                host_->mov(tmp_reg, rax);
                host_->sal(tmp_reg, shift_val);
                host_->add(addr_reg, tmp_reg);
            }
            host_->mov(addr_cache_reg, addr_reg);
        } else {
            host_->mov(addr_reg, addr_cache_reg);
        }

        if (it_off_val != vmm_idx_to_out_elem_off_val.end()) {
            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_mb_ncsp_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::c_blocked:
                    assert(!"Unimplemented");
                    break;
                case injector_utils::layout_t::nspc:
                    calculate_mb_nspc_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::cspn:
                    calculate_mb_cspn_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                default: assert(!"Unknown layout");
            }
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_ncsp_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_off = offset / stride_n
    // mb_off = n
    // output = rax
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[0]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
    // rax = n
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_ncsp_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_off = offset / stride_n
    // mb_off = n
    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto offset_adj = offset_shr / strides[0];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_nspc_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // mb_off = n;
    calculate_mb_ncsp_base(strides, tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_nspc_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // mb_off = n;
    calculate_mb_ncsp_partial(strides, offset, tmp_reg, elem_size_bytes);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_cspn_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // mb_off = offset % N = offset % strides[ndims-1]
    // mb_off = n
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[ndims - 1]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
    host_->mov(rax, rdx);
    // rax = n
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_cspn_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // mb_off = offset % N = offset % strides[ndims-1]
    // mb_off = n
    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto offset_adj
            = offset_shr % strides[rhs_arg_static_params_.dst_d.ndims() - 1];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::append_oc_spatial_offset(
        const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
        const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes, bool is_first) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        Xbyak::Address out_addr = is_out_addr ? it_out_addr->second
                                              : host_->ptr[it_out_reg->second];
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        const auto &addr_cache_reg = rhs_arg_static_params_.rhs_addr_cache_reg;

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        if (is_first) {
            calculate_no_broadcast_base(out_addr, tmp_reg);

            const auto rax = host_->rax;
            const auto rdx = host_->rdx;
            const auto r8 = host_->r8;

            const injector_utils::conditional_register_preserve_guard_t
                    register_guard {is_out_reg
                                    ? utils::one_of(
                                              it_out_reg->second, rax, rdx, r8)
                                    : false,
                            host_,
                            {is_out_reg ? it_out_reg->second : Xbyak::Reg64()}};

            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_oc_spatial_ncsp_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::c_blocked:
                    assert(!"Unimplemented");
                    break;
                case injector_utils::layout_t::nspc:
                    calculate_oc_spatial_nspc_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::cspn:
                    calculate_oc_spatial_cspn_base(strides, tmp_reg);
                    break;
                default: assert(!"Unknown layout");
            }

            if (elem_size_bytes == 1) {
                host_->add(addr_reg, rax);
            } else {
                const int shift_val = math::ilog2q(elem_size_bytes);
                host_->mov(tmp_reg, rax);
                host_->sal(tmp_reg, shift_val);
                host_->add(addr_reg, tmp_reg);
            }
            host_->mov(addr_cache_reg, addr_reg);
        } else {
            host_->mov(addr_reg, addr_cache_reg);
        }

        if (it_off_val != vmm_idx_to_out_elem_off_val.end()) {
            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_oc_spatial_ncsp_partial(strides,
                            it_off_val->second, tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::c_blocked:
                    assert(!"Unimplemented");
                    break;
                case injector_utils::layout_t::nspc:
                    calculate_oc_spatial_nspc_partial(strides,
                            it_off_val->second, tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::cspn:
                    calculate_oc_spatial_cspn_partial(strides,
                            it_off_val->second, tmp_reg, elem_size_bytes);
                    break;
                default: assert(!"Unknown layout");
            }
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_spatial_ncsp_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // oc_spatial_off = offset % stride_n
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[0]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
    host_->mov(rax, rdx);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_spatial_ncsp_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // oc_spatial_off = offset % stride_n
    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto offset_adj = offset_shr % strides[0];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_spatial_nspc_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // oc_spatial_off = offset % stride_n
    calculate_oc_spatial_ncsp_base(strides, tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_spatial_nspc_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // oc_spatial_off = offset % stride_n
    calculate_oc_spatial_ncsp_partial(
            strides, offset, tmp_reg, elem_size_bytes);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_spatial_cspn_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // oc_spatial_off = offset / strides[ndims - 1]
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    const auto rax = host_->rax;
    const auto rdx = host_->rdx;

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[ndims - 1]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_oc_spatial_cspn_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // oc_spatial_off = offset / strides[ndims - 1]
    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto offset_adj
            = offset_shr / strides[rhs_arg_static_params_.dst_d.ndims() - 1];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::append_mb_oc_offset(
        const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
        const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes, bool is_first) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        Xbyak::Address out_addr = is_out_addr ? it_out_addr->second
                                              : host_->ptr[it_out_reg->second];
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        const auto &addr_cache_reg = rhs_arg_static_params_.rhs_addr_cache_reg;

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        if (is_first) {
            calculate_no_broadcast_base(out_addr, tmp_reg);

            const auto rax = host_->rax;
            const auto rdx = host_->rdx;
            const auto r8 = host_->r8;

            const injector_utils::conditional_register_preserve_guard_t
                    register_guard {is_out_reg
                                    ? utils::one_of(
                                              it_out_reg->second, rax, rdx, r8)
                                    : false,
                            host_,
                            {is_out_reg ? it_out_reg->second : Xbyak::Reg64()}};

            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_mb_oc_ncsp_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::c_blocked:
                    assert(!"Unimplemented");
                    break;
                case injector_utils::layout_t::nspc:
                    calculate_mb_oc_nspc_base(strides, tmp_reg);
                    break;
                case injector_utils::layout_t::cspn:
                    calculate_mb_oc_cspn_base(strides, tmp_reg);
                    break;
                default: assert(!"Unknown layout");
            }

            if (elem_size_bytes == 1) {
                host_->add(addr_reg, rax);
            } else {
                const int shift_val = math::ilog2q(elem_size_bytes);
                host_->mov(tmp_reg, rax);
                host_->sal(tmp_reg, shift_val);
                host_->add(addr_reg, tmp_reg);
            }
            host_->mov(addr_cache_reg, addr_reg);
        } else {
            host_->mov(addr_reg, addr_cache_reg);
        }

        if (it_off_val != vmm_idx_to_out_elem_off_val.end()) {
            switch (layout) {
                case injector_utils::layout_t::ncsp:
                    calculate_mb_oc_ncsp_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::c_blocked:
                    assert(!"Unimplemented");
                    break;
                case injector_utils::layout_t::nspc:
                    calculate_mb_oc_nspc_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                case injector_utils::layout_t::cspn:
                    calculate_mb_oc_cspn_partial(strides, it_off_val->second,
                            tmp_reg, elem_size_bytes);
                    break;
                default: assert(!"Unknown layout");
            }
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_oc_ncsp_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = nCDHW + cDHW + dHW + hW + w
    // mb_oc_off = nC + c
    // mb_oc_off = offset / DHW
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;

    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[1]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_oc_ncsp_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = nCDHW + cDHW + dHW + hW + w
    // mb_oc_off = offset / DHW
    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto offset_adj = offset_shr / strides[1];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_oc_nspc_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // mb_oc_off = nC + c
    // mb_oc_off = (offset / DHWC) * C + offset % C
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    host_->mov(r8, tmp_reg);
    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[0]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg); // rax = nDHWC / DHWC
    host_->mov(tmp_reg, strides[ndims - 1]);
    host_->mul(tmp_reg); // rax = n * C
    host_->mov(tmp_reg, rax);
    host_->mov(rax, r8);
    host_->mov(r8, strides[ndims - 1]);
    host_->xor_(rdx, rdx);
    host_->div(r8); // rdx = offset % C
    host_->mov(rax, rdx);
    host_->add(rax, tmp_reg); // rax = nC + c
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_oc_nspc_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = nDHWC + dHWC + hWC + wC + c
    // mb_oc_off = nC + c
    // mb_oc_off = (offset / DHWC) * C + offset % C

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto offset_adj = (offset_shr / strides[0]) * strides[ndims - 1]
            + offset_shr % strides[ndims - 1];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_oc_cspn_base(
        const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // mb_oc_off = cN + n
    // mb_oc_off = (offset / DHWN) * N + offset % N
    const auto rax = host_->rax;
    const auto rdx = host_->rdx;
    const auto r8 = host_->r8;

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    host_->mov(r8, tmp_reg);
    host_->mov(rax, tmp_reg);
    host_->mov(tmp_reg, strides[2]);
    host_->xor_(rdx, rdx);
    host_->div(tmp_reg); // rax = cDHWN / DHWN
    host_->mov(tmp_reg, strides[ndims - 1]);
    host_->mul(tmp_reg); // rax = c * N
    host_->mov(tmp_reg, rax);
    host_->mov(rax, r8);
    host_->mov(r8, strides[ndims - 1]);
    host_->xor_(rdx, rdx);
    host_->div(r8); // rdx = offset % N
    host_->mov(rax, rdx);
    host_->add(rax, tmp_reg); // rax = cN + n
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::calculate_mb_oc_cspn_partial(
        const dim_t *strides, const dim_t offset, const Xbyak::Reg64 &tmp_reg,
        dim_t elem_size_bytes) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // mb_oc_off = cN + n
    // mb_oc_off = (offset / DHWN) * N + offset % N

    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    const auto offset_shr = offset >> math::ilog2q(types::data_type_size(
                                    rhs_arg_static_params_.dst_d.data_type()));
    const auto offset_adj = (offset_shr / strides[2]) * strides[ndims - 1]
            + offset_shr % strides[ndims - 1];
    host_->mov(tmp_reg,
            elem_size_bytes > 1 ? offset_adj << math::ilog2q(elem_size_bytes)
                                : offset_adj);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::inject_binary(
        const dnnl_post_ops::entry_t &post_op, Vmm dst,
        const Xbyak::Address &rhs_addr, bool with_tail,
        const tail_lode_mode_t tail_load_mode) const {

    const bool is_prelu = post_op.is_prelu();
    const auto alg = is_prelu ? alg_kind::undef : post_op.binary.alg;
    const bool cmp_op = utils::one_of(alg, alg_kind::binary_ge,
            alg_kind::binary_gt, alg_kind::binary_le, alg_kind::binary_lt,
            alg_kind::binary_eq, alg_kind::binary_ne);
    const auto rhs_arg_data_type
            = get_src1_desc(post_op, rhs_arg_static_params_.dst_d).data_type;
    const bool scalar_f32
            = rhs_addr.isBroadcast() && rhs_arg_data_type == data_type::f32;
    const bool with_tail_not_fusable_to_binary_op
            = with_tail && !isa_has_masks(isa_);
    const bool cmp_op_with_tail_vector_mem_operand
            = cmp_op && with_tail && !rhs_addr.isBroadcast();
    const bool process_rhs_arg_using_tmp_vmm
            = rhs_arg_data_type != data_type::f32
            || (scalar_f32 && !has_avx512_core_)
            || with_tail_not_fusable_to_binary_op
            || !binary_op_with_unaligned_mem_operand_allowed_
            || (cmp_op && !has_avx512_core_)
            || cmp_op_with_tail_vector_mem_operand;

    if (process_rhs_arg_using_tmp_vmm) {

        const Vmm tmp_vmm = Vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);

        if (rhs_addr.isBroadcast())
            execute_broadcast(rhs_arg_data_type, tmp_vmm,
                    remove_bcast_bit(rhs_addr), tail_load_mode, with_tail);
        else
            load_rhs(rhs_arg_data_type, tmp_vmm, rhs_addr, tail_load_mode,
                    with_tail);

        if (types::is_integral_dt(rhs_arg_data_type)) cvt_to_f32(tmp_vmm);

        if (is_prelu)
            execute_prelu(dst, tmp_vmm);
        else
            execute_binary(alg, dst, dst, tmp_vmm);
    } else {
        const auto lhs = dst;
        if (with_tail) {
            assert(isa_has_masks(isa_));
            assert(rhs_arg_static_params_.is_opmask_set()
                    && "Opmask is not set for tail loading avx512");
            const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;
            dst = dst | tail_opmask | host_->T_z;
        }

        if (is_prelu)
            execute_prelu(dst, rhs_addr);
        else
            execute_binary(alg, dst, lhs, rhs_addr);
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::inject_binary_with_ternary_op(
        const dnnl_post_ops::entry_t &post_op, Vmm dst,
        const Xbyak::Address &rhs_addr, Vmm tmp_vmm, bool with_tail,
        const tail_lode_mode_t tail_load_mode) const {
    const auto alg = post_op.binary.alg;
    const auto rhs_arg_data_type
            = get_src1_desc(post_op, rhs_arg_static_params_.dst_d).data_type;

    if (alg == alg_kind::binary_select) {
        // The select op is implemented by adding the results of
        // the src0 tensor ANDed with the select input and the
        // src1 tensor ANDed with the complemented select input.
        // While a more direct implementation can be obtained by
        // blending the tensors, the current approach reserves
        // fewer registers for operation.
        if (is_superset(isa_, avx512_core)) {
            const auto &cmp_mask = rhs_arg_static_params_.tail_opmask;
            push_opmask(host_, cmp_mask);
            push_vmm(host_, dst);
            host_->vxorps(dst, dst, dst);
            host_->vpcmpgtd(cmp_mask | host_->T_z, tmp_vmm, dst);
            pop_vmm(host_, dst);
            host_->vxorps(tmp_vmm, tmp_vmm, tmp_vmm);
            host_->vblendmps(dst | cmp_mask, tmp_vmm, dst);
            host_->knotw(cmp_mask, cmp_mask);
            host_->vpmovm2b(tmp_vmm, cmp_mask);
            pop_opmask(host_, cmp_mask);
            push_vmm(host_, dst);

            if (rhs_addr.isBroadcast())
                execute_broadcast(rhs_arg_data_type, dst,
                        remove_bcast_bit(rhs_addr), tail_load_mode, with_tail);
            else
                load_rhs(rhs_arg_data_type, dst, rhs_addr, tail_load_mode,
                        with_tail);
            if (types::is_integral_dt(rhs_arg_data_type)) cvt_to_f32(dst);

            push_opmask(host_, cmp_mask);
            host_->vpmovb2m(cmp_mask, tmp_vmm);

            host_->vxorps(tmp_vmm, tmp_vmm, tmp_vmm);
            host_->vblendmps(tmp_vmm | cmp_mask, tmp_vmm, dst);
            pop_opmask(host_, cmp_mask);
            pop_vmm(host_, dst);
            host_->vpaddd(dst, dst, tmp_vmm);
        } else {
            push_vmm(host_, dst);
            host_->vxorps(dst, dst, dst);

            host_->vcmpps(tmp_vmm, tmp_vmm, dst, jit_generator_t::_cmp_nle_us);
            pop_vmm(host_, dst);
            host_->vpand(dst, dst, tmp_vmm);
            push_vmm(host_, dst);
            host_->vpcmpeqd(dst, dst, dst);
            host_->vpandn(tmp_vmm, tmp_vmm, dst);

            if (rhs_addr.isBroadcast())
                execute_broadcast(rhs_arg_data_type, dst,
                        remove_bcast_bit(rhs_addr), tail_load_mode, with_tail);
            else
                load_rhs(rhs_arg_data_type, dst, rhs_addr, tail_load_mode,
                        with_tail);
            if (types::is_integral_dt(rhs_arg_data_type)) cvt_to_f32(dst);

            host_->vpand(tmp_vmm, dst, tmp_vmm);
            pop_vmm(host_, dst);
            host_->vpaddd(dst, dst, tmp_vmm);
        }
    } else {
        assert(!"unsupported algorithm");
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::execute_broadcast(
        const data_type_t &data_type, const Vmm &tmp_reg,
        const Xbyak::Address &rhs_addr, const tail_lode_mode_t tail_load_mode,
        bool with_tail) const {
    if (with_tail) {
        if (tail_load_mode == tail_lode_mode_t::DYNAMIC
                || (tail_load_mode == tail_lode_mode_t::DEFAULT
                        && has_avx512_core_)) {
            if (has_avx512_core_)
                execute_broadcast_tail_with_opmask(
                        data_type, tmp_reg, rhs_addr);
            else
                execute_broadcast_tail_with_gpr(data_type, tmp_reg, rhs_addr);
        } else
            execute_broadcast_tail_statically(data_type, tmp_reg, rhs_addr,
                    rhs_arg_static_params_.tail_size);
    } else
        execute_broadcast_no_tail(data_type, tmp_reg, rhs_addr);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::load_rhs(const data_type_t &data_type,
        const Vmm &tmp_reg, const Xbyak::Address &rhs_addr,
        const tail_lode_mode_t tail_load_mode, bool with_tail) const {
    if (with_tail) {
        if (tail_load_mode == tail_lode_mode_t::DYNAMIC
                || (tail_load_mode == tail_lode_mode_t::DEFAULT
                        && has_avx512_core_)) {
            if (has_avx512_core_)
                load_rhs_tail_dynamically_with_opmask(
                        data_type, tmp_reg, rhs_addr);
            else
                load_rhs_tail_dynamically_with_gpr(data_type, tmp_reg);
        } else
            load_rhs_tail_statically(data_type, tmp_reg, rhs_addr);
    } else
        load_rhs_no_tail(data_type, tmp_reg, rhs_addr);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::load_acc_as_f32(const Vmm &dst,
        const Xbyak::Reg64 &base, dim_t byte_off, data_type_t data_type,
        bool with_tail, const tail_lode_mode_t tail_load_mode) const {

    // The RHS address register, reused here to point at accumulator memory
    // (see the declaration on reusing the RHS load path).
    const auto addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    const bool byte_off_fits = byte_off <= std::numeric_limits<int>::max();

    // The address needs its own register (`addr_reg`) when it cannot be
    // represented as a `[base + disp]` operand. Two cases force that:
    //   1. AVX2 tail loads: the helper reads the pointer from `addr_reg` and
    //      ignores the operand.
    //   2. A byte offset that doesn't fit int range.
    // No-tail and AVX-512 opmask tail loads take the operand directly.
    const bool need_scratch_reg
            = (with_tail && !has_avx512_core_) || !byte_off_fits;

    // This is a direct entry point so the register-preserve guard is not in
    // effect here. Since we need to follow the defined ABI we preserve the
    // helper GPRs when the caller requested it.
    const bool preserve_gpr = rhs_arg_static_params_.preserve_gpr_helpers;
    if (need_scratch_reg) {
        if (preserve_gpr) host_->push(addr_reg);

        if (byte_off_fits) {
            host_->lea(addr_reg, host_->ptr[base + (int)byte_off]);
        } else {
            // Use a scratch register to handle large offsets.
            const auto tmp_reg = rhs_arg_static_params_.rhs_helper_reg;
            if (preserve_gpr) host_->push(tmp_reg);

            host_->mov(tmp_reg, byte_off);
            host_->lea(addr_reg, host_->ptr[base + tmp_reg]);

            if (preserve_gpr) host_->pop(tmp_reg);
        }
    }
    const auto addr = need_scratch_reg ? host_->ptr[addr_reg]
                                       : host_->ptr[base + (int)byte_off];
    load_rhs(data_type, dst, addr, tail_load_mode, with_tail);

    if (types::is_integral_dt(data_type)) cvt_to_f32(dst);
    if (need_scratch_reg && preserve_gpr) host_->pop(addr_reg);
}

template <typename Vmm>
Xbyak::Address jit_uni_binary_injector_t<Vmm>::remove_bcast_bit(
        const Xbyak::Address &rhs_addr) const {
    return Xbyak::Address(rhs_addr.getBit(), false, rhs_addr.getRegExp());
}

template <typename Vmm>
Xbyak::Opmask jit_uni_binary_injector_t<Vmm>::get_aux_kmask() const {
    auto tail_mask_idx = rhs_arg_static_params_.tail_opmask.getIdx();
    return Xbyak::Opmask(tail_mask_idx < 7 ? tail_mask_idx + 1 : 1);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::cvt_to_f32(const Vmm &tmp_vmm) const {
    host_->uni_vcvtdq2ps(tmp_vmm, tmp_vmm);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::execute_broadcast_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    assert(is_data_supported(isa_, data_type) && "unsupported data type");
    switch (data_type) {
        case data_type::f32: host_->uni_vbroadcastss(tmp_vmm, rhs_addr); break;
        case data_type::s32: host_->uni_vpbroadcastd(tmp_vmm, rhs_addr); break;
        case data_type::s8:
        case data_type::u8:
            execute_broadcast_s8u8_no_tail(data_type, tmp_vmm, rhs_addr);
            break;
        case data_type::f16:
            if (has_avx512_core_fp16_)
                host_->vcvtph2psx(tmp_vmm, host_->ptr_b[rhs_addr.getRegExp()]);
            else if (is_superset(isa_, avx2_vnni_2))
                host_->vbcstnesh2ps(tmp_vmm, rhs_addr);
            else
                assert(!"unsupported ISA for given data type");
            break;
        case data_type::f8_e5m2:
            assert(f8_e5m2_cvt_);
            f8_e5m2_cvt_->bcst_f8_to_f32(tmp_vmm, rhs_addr);
            break;
        case data_type::f8_e4m3:
            assert(f8_e4m3_cvt_);
            f8_e4m3_cvt_->bcst_f8_to_f32(tmp_vmm, rhs_addr);
            break;
        case data_type::bf16:
            if (has_avx512_core_) {
                host_->vpbroadcastw(tmp_vmm, rhs_addr);
                host_->vpslld(tmp_vmm, tmp_vmm, 0x10);
            } else if (is_superset(isa_, avx2_vnni_2)) {
                host_->vbcstnebf162ps(tmp_vmm, rhs_addr);
            } else
                assert(!"unsupported ISA for given data type");
            break;
        default: assert(!"unsupported data type");
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    assert(utils::one_of(data_type, data_type::s8, data_type::u8)
            && "unsupported data type");

    const Xbyak::Xmm tmp_xmm = Xbyak::Xmm(tmp_vmm.getIdx());

    host_->uni_vpinsrb(tmp_xmm, tmp_xmm, rhs_addr, 0);
    if (data_type == data_type::s8)
        host_->uni_vpmovsxbd(tmp_xmm, tmp_xmm);
    else
        host_->uni_vpmovzxbd(tmp_vmm, tmp_xmm);
    host_->uni_vpbroadcastd(tmp_vmm, tmp_xmm);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::execute_broadcast_tail_with_opmask(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {

    assert(is_data_supported(isa_, data_type) && "unsupported data type");
    assert(rhs_arg_static_params_.is_opmask_set()
            && "Opmask is not set for tail loading avx512");
    const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;

    switch (data_type) {
        case data_type::f32:
            host_->vbroadcastss(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::s32:
            host_->vpbroadcastd(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::s8:
        case data_type::u8: {
            const Xbyak::Xmm xmm(tmp_vmm.getIdx());

            host_->uni_vpinsrb(xmm, xmm, rhs_addr, 0);
            if (data_type == data_type::s8)
                host_->uni_vpmovsxbd(xmm, xmm);
            else if (data_type == data_type::u8)
                host_->uni_vpmovzxbd(xmm, xmm);
            host_->uni_vpbroadcastd(tmp_vmm | tail_opmask | host_->T_z, xmm);
            break;
        }
        case data_type::f16:
            if (has_avx512_core_fp16_)
                host_->vcvtph2psx(tmp_vmm | tail_opmask | host_->T_z,
                        host_->ptr_b[rhs_addr.getRegExp()]);
            else
                assert(!"unsupported masked tail processing");
            break;
        case data_type::f8_e5m2:
            assert(f8_e5m2_cvt_);
            f8_e5m2_cvt_->bcst_f8_to_f32(
                    tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::f8_e4m3:
            assert(f8_e4m3_cvt_);
            f8_e4m3_cvt_->bcst_f8_to_f32(
                    tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::bf16:
            host_->vpbroadcastw(tmp_vmm, rhs_addr);
            host_->vpslld(tmp_vmm | tail_opmask | host_->T_z, tmp_vmm, 0x10);
            break;
        default: return;
    }
}

static constexpr int xmm_size_elem = 4;

static void load_tail_avx(jit_generator_t *host, int ymm_idx, int tail_size,
        const std::function<void()> &init_op,
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

static Xbyak::uint8 MM_SHUFFLE(
        Xbyak::uint8 z, Xbyak::uint8 y, Xbyak::uint8 x, Xbyak::uint8 w) {
    return static_cast<Xbyak::uint8>(
            ((z) << 6) | ((y) << 4) | ((x) << 2) | (w));
}

static void execute_broadcast_f32_tail_avx(jit_generator_t *host,
        const Xbyak::Ymm &vmm, const Xbyak::Address &rhs_addr, int tail_size) {

    const auto vmm_idx = vmm.getIdx();
    const auto tmp_xmm = Xbyak::Xmm(vmm_idx);
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
            host, vmm_idx, tail_size, init_op, upper_half_op, lower_half_op);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::execute_broadcast_tail_statically(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr, const int tail_size) const {
    // An opmask-capable ISA has no static tail path. `execute_broadcast` sends
    // its tails to `execute_broadcast_tail_with_opmask`, so arriving here means
    // the caller asked for a tail mode this ISA does not implement.
    if (has_avx512_core_) {
        assert(!"unsupported tail load mode");
        return;
    }

    if (is_superset(isa_, avx2_vnni_2)
            && utils::one_of(data_type, data_type::bf16, data_type::f16)) {
        const auto tmp_xmm = Xbyak::Xmm(tmp_vmm.getIdx());
        host_->uni_vxorps(tmp_vmm, tmp_vmm, tmp_vmm);
        // The RHS is a scalar broadcast; repeat the one source element.
        for (int i = 0; i < tail_size; ++i)
            host_->vpinsrw(tmp_xmm, tmp_xmm, rhs_addr, i);
        if (data_type == data_type::bf16) {
            host_->vpmovzxwd(tmp_vmm, tmp_xmm);
            host_->vpslld(tmp_vmm, tmp_vmm, 16);
        } else {
            host_->vcvtph2ps(tmp_vmm, tmp_xmm);
        }
        return;
    }

    host_->uni_vxorps(tmp_vmm, tmp_vmm, tmp_vmm);

    if (data_type == data_type::f32 || data_type == data_type::s32) {
        execute_broadcast_f32_tail_avx(host_, tmp_vmm, rhs_addr, tail_size);
    } else if (data_type == data_type::u8 || data_type == data_type::s8) {
        const auto tmp_xmm = Xbyak::Xmm(tmp_vmm.getIdx());
        for (int i = 0; i < tail_size; i++)
            host_->uni_vpinsrb(tmp_xmm, tmp_xmm, rhs_addr, i);

        if (data_type == data_type::s8)
            host_->uni_vpmovsxbd(tmp_vmm, tmp_xmm);
        else
            host_->uni_vpmovzxbd(tmp_vmm, tmp_xmm);
    } else
        assert(!"unsupported data type");
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::execute_broadcast_tail_with_gpr(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {

    const Xbyak::Reg64 &reg_tmp = rhs_arg_static_params_.rhs_helper_reg;
    const Xbyak::Reg64 &reg_tail_size = rhs_arg_static_params_.reg_tail_size;

    auto runtime_tail_load = [&](int load_size) {
        execute_broadcast_tail_statically(
                data_type, tmp_vmm, rhs_addr, load_size);
    };
    host_->runtime_tail_process<Vmm>(reg_tail_size, reg_tmp, runtime_tail_load);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::load_rhs_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    assert(is_data_supported(isa_, data_type) && "unsupported data type");
    switch (data_type) {
        case data_type::f32:
        case data_type::s32: host_->uni_vmovups(tmp_vmm, rhs_addr); break;
        case data_type::s8:
        case data_type::u8:
            load_rhs_i8_no_tail(data_type, tmp_vmm, rhs_addr);
            break;
        case data_type::f16:
            if (has_avx512_core_fp16_)
                host_->vcvtph2psx(tmp_vmm, rhs_addr);
            else if (is_superset(isa_, avx2_vnni_2))
                host_->vcvtph2ps(tmp_vmm, rhs_addr);
            else
                assert(!"unsupported ISA for given data type");
            break;
        case data_type::f8_e5m2:
            assert(f8_e5m2_cvt_);
            f8_e5m2_cvt_->vcvt_f8_to_f32(tmp_vmm, rhs_addr);
            break;
        case data_type::f8_e4m3:
            assert(f8_e4m3_cvt_);
            f8_e4m3_cvt_->vcvt_f8_to_f32(tmp_vmm, rhs_addr);
            break;
        case data_type::bf16:
            if (has_avx512_core_ || is_superset(isa_, avx2_vnni_2)) {
                host_->vpmovzxwd(tmp_vmm, rhs_addr);
                host_->vpslld(tmp_vmm, tmp_vmm, 0x10);
                break;
            }
        default: assert(!"unsupported data type");
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::load_rhs_i8_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    if (data_type == data_type::s8)
        host_->uni_vpmovsxbd(tmp_vmm, rhs_addr);
    else if (data_type == data_type::u8)
        host_->uni_vpmovzxbd(tmp_vmm, rhs_addr);
    else
        assert(!"unsupported data type");
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::load_rhs_tail_dynamically_with_opmask(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    assert(is_data_supported(isa_, data_type) && "unsupported data type");
    assert(rhs_arg_static_params_.is_opmask_set()
            && "Opmask is not set for tail loading avx512");

    const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;

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
        case data_type::f16:
            if (has_avx512_core_fp16_)
                host_->vcvtph2psx(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            else
                assert(!"unsupported masked tail processing");
            break;
        case data_type::f8_e5m2:
            assert(f8_e5m2_cvt_);
            f8_e5m2_cvt_->vcvt_f8_to_f32(
                    tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::f8_e4m3:
            assert(f8_e4m3_cvt_);
            f8_e4m3_cvt_->vcvt_f8_to_f32(
                    tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::bf16:
            host_->vpmovzxwd(tmp_vmm | tail_opmask | host_->T_z, rhs_addr);
            host_->vpslld(tmp_vmm | tail_opmask | host_->T_z, tmp_vmm, 0x10);
            break;
        default: return;
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::load_rhs_tail_dynamically_with_gpr(
        const data_type_t &data_type, const Vmm &tmp_vmm) const {

    const bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    const Xbyak::Reg64 &reg_addr = rhs_arg_static_params_.rhs_addr_reg;
    const Xbyak::Reg64 &reg_tmp = rhs_arg_static_params_.rhs_helper_reg;
    const Xbyak::Reg64 &reg_tail_size = rhs_arg_static_params_.reg_tail_size;
    const Xbyak::Xmm x = Xbyak::Xmm(tmp_vmm.getIdx());
    const Xbyak::Ymm y = Xbyak::Ymm(tmp_vmm.getIdx());

    auto runtime_tail_load = [&](int load_size) {
        if (is_ymm)
            host_->load_data(data_type, y, reg_addr, 0, load_size);
        else
            host_->load_data(data_type, x, reg_addr, 0, load_size);
    };

    host_->runtime_tail_process<Vmm>(reg_tail_size, reg_tmp, runtime_tail_load);
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::load_rhs_tail_statically(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    // An opmask-capable ISA has no static tail path. `load_rhs` sends its tails
    // to `load_rhs_tail_dynamically_with_opmask`, so arriving here means the
    // caller asked for a tail mode this ISA does not implement.
    if (has_avx512_core_) {
        assert(!"unsupported tail load mode");
        return;
    }

    const auto &tail_size = rhs_arg_static_params_.tail_size;
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;

    // Sub-dword floating point is widened in place from a packed tail, so it
    // shares neither the zeroing nor the element assembly the dword paths
    // below need.
    if (is_superset(isa_, avx2_vnni_2)
            && utils::one_of(data_type, data_type::bf16, data_type::f16)) {
        const auto tmp_lower_vmm =
                typename vreg_traits_t<Vmm>::Vmm_lower_t(tmp_vmm.getIdx());
        host_->load_bytes(tmp_lower_vmm, rhs_addr_reg, 0,
                static_cast<int>(tail_size * sizeof(bfloat16_t)));
        if (data_type == data_type::bf16) {
            host_->vpmovzxwd(tmp_vmm, tmp_lower_vmm);
            host_->vpslld(tmp_vmm, tmp_vmm, 16);
        } else // f16
            host_->vcvtph2ps(tmp_vmm, tmp_lower_vmm);
        return;
    }

    if (!utils::one_of(data_type, data_type::f32, data_type::s32, data_type::s8,
                data_type::u8))
        assert(!"unsupported data type");

    host_->load_data(data_type, tmp_vmm, rhs_addr_reg, 0, tail_size);
}

template <typename Vmm>
template <typename T>
void jit_uni_binary_injector_t<Vmm>::execute_cmp_binary(const Vmm &dst,
        const Vmm &lhs, const T &rhs, const unsigned int cmp_predicate) const {
    // The rhs operand shape selects the sequence, and it also pins the ISA. A
    // memory operand needs the opmask form because it only arrives from the
    // `inject_binary` path that fuses the tail mask into the binary op, which
    // exists on AVX-512 alone. A `Zmm` operand needs it because VEX `vcmpps`
    // has no 512-bit encoding. Every other case is a `Ymm` or `Xmm` register,
    // which keeps the VEX sequence even on an AVX-512 host. A host without
    // opmasks never reaches the first branch: `inject_binary` routes each of
    // its compares through a temporary vector register.
    const bool rhs_requires_opmask = std::is_same<T, Xbyak::Zmm>::value
            || std::is_same<T, Xbyak::Address>::value;
    if (rhs_requires_opmask) {
        assert(has_avx512_core_ && "compare with an opmask requires AVX-512");
        // For GreaterEqual op, replace 0xFFFFFFFF by 1
        // which was returned by vcmpps.
        const auto &cmp_mask = rhs_arg_static_params_.tail_opmask;
        const Xbyak::Xmm xreg_one
                = Xbyak::Xmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);
        const Xbyak::Reg64 reg_tmp = rhs_arg_static_params_.rhs_helper_reg;

        push_opmask(host_, cmp_mask);
        host_->vcmpps(cmp_mask, lhs, rhs, static_cast<uint8_t>(cmp_predicate));
        host_->mov(reg_tmp, float2int(1));
        host_->uni_vmovq(xreg_one, reg_tmp);
        // broadcast 1.0f with mask
        host_->vbroadcastss(dst | cmp_mask | host_->T_z, xreg_one);
        // pop tail mask from stack
        pop_opmask(host_, cmp_mask);
    } else {
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
}

template <typename Vmm>
template <typename T>
void jit_uni_binary_injector_t<Vmm>::execute_binary(alg_kind_t binary_alg,
        const Vmm &dst, const Vmm &lhs, const T &rhs) const {
    switch (binary_alg) {
        case alg_kind::binary_add: host_->uni_vaddps(dst, lhs, rhs); break;
        case alg_kind::binary_mul: host_->uni_vmulps(dst, lhs, rhs); break;
        case alg_kind::binary_max: host_->uni_vmaxps(dst, lhs, rhs); break;
        case alg_kind::binary_min: host_->uni_vminps(dst, lhs, rhs); break;
        case alg_kind::binary_div: host_->uni_vdivps(dst, lhs, rhs); break;
        case alg_kind::binary_sub: host_->uni_vsubps(dst, lhs, rhs); break;
        case alg_kind::binary_ge:
            execute_cmp_binary(dst, lhs, rhs, jit_generator_t::_cmp_nlt_us);
            break;
        case alg_kind::binary_gt:
            execute_cmp_binary(dst, lhs, rhs, jit_generator_t::_cmp_nle_us);
            break;
        case alg_kind::binary_le:
            execute_cmp_binary(dst, lhs, rhs, jit_generator_t::_cmp_le_os);
            break;
        case alg_kind::binary_lt:
            execute_cmp_binary(dst, lhs, rhs, jit_generator_t::_cmp_lt_os);
            break;
        case alg_kind::binary_eq:
            execute_cmp_binary(dst, lhs, rhs, jit_generator_t::_cmp_eq_oq);
            break;
        case alg_kind::binary_ne:
            execute_cmp_binary(dst, lhs, rhs, jit_generator_t::_cmp_neq_uq);
            break;
        default: assert(!"unsupported algorithm");
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::execute_prelu(
        const Vmm &dst, const Xbyak::Operand &rhs) const {
    Vmm tmp_vmm = Vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);
    if (is_superset(isa_, avx512_core)) {
        assert(rhs.isMEM());
        Vmm dst_vmm = Vmm(dst.getIdx());
        Xbyak::Opmask maybe_tail_kmask = Xbyak::Opmask(dst.getOpmaskIdx());
        Xbyak::Opmask aux_kmask = get_aux_kmask();
        host_->vxorps(tmp_vmm, tmp_vmm, tmp_vmm);
        host_->vcmpps(aux_kmask | maybe_tail_kmask, dst_vmm, tmp_vmm,
                jit_generator_t::_cmp_le_os);
        host_->vmulps(dst_vmm | aux_kmask, dst_vmm, rhs);
    } else {
        // Three operand version
        host_->uni_vmulps(tmp_vmm, dst, rhs);
        host_->uni_vblendvps(dst, dst, tmp_vmm, dst);
    }
}

template <typename Vmm>
void jit_uni_binary_injector_t<Vmm>::compute_vector(int idx, int rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    compute_vector_range({idx}, rhs_arg_idx, post_op, rhs_arg_params);
}

template class jit_uni_binary_injector_t<Xbyak::Zmm>;
template class jit_uni_binary_injector_t<Xbyak::Ymm>;

#undef VCHECK_BIN_INJ_BOOL

} // namespace binary_injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
