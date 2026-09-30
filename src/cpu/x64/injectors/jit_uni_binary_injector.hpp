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

#ifndef CPU_X64_INJECTORS_JIT_UNI_BINARY_INJECTOR_HPP
#define CPU_X64_INJECTORS_JIT_UNI_BINARY_INJECTOR_HPP

#include <array>
#include <cassert>
#include <functional>
#include <map>
#include <utility>
#include <vector>
#include <unordered_set>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "cpu/binary_injector_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace binary_injector {
using dnnl::impl::cpu::binary_injector_utils::get_src1_desc;
using dnnl::impl::cpu::binary_injector_utils::get_src2_desc;
using dnnl::impl::cpu::binary_injector_utils::prepare_binary_args;

bcast_set_t get_all_strategies_supported_by_injector();

bool binary_args_broadcast_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

bool any_binary_postop_rhs_non_scalar_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d);
bool any_binary_postop_rhs_with_ternary_scalar_bcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d);
bool is_ternary_bcast_supported(
        const memory_desc_t &src2_md, const memory_desc_wrapper &dst_d);
bool any_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set,
        const std::function<bool(const memory_desc_wrapper &)> &predicate);

bool any_binary_postop_rhs_per_w_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

void extend_binary_args_per_w(const post_ops_t &post_ops,
        const std::vector<const void *> &orig_post_ops_binary_rhs_arg_vec,
        std::vector<const void *> &post_ops_binary_rhs_arg_vec,
        uint8_t *expanded_rhs, const std::vector<dim_t> &expanded_elems_len);

/*
 * Represents params related to all binary post-ops right-hand side arguments
 * (arg1) that don't change during jit_uni_binary_injector_t object lifetime
 * and between compute_vector_range calls.
 *
 * @param rhs_dt_helper_vmm_idx - index of vmm helper used when loading data for
 * calculations. Treated as hint from user. If inside compute_vector_range hint
 * turns out to be invalid, it will be overwriten by register preserving logic inside
 * binary injector.
 * @param rhs_addr_reg - gpr register, used as the currently processed address of
 * rhs tensor slice. Data of rhs(arg1) for the binary operation is loaded from address
 * stored inside rhs_addr_reg.
 * @param rhs_helper_reg - gpr register used as helper for calculations during data
 * loading phase.
 * @param rhs_addr_cache_reg - gpr register used for caching part of calculated
 * offset, this register is always preserved.
 * @param preserve_gpr_helpers - determines whether gpr registers specified above
 * should be preserved (pushed to stack and poped back afterwords) between
 * compute_vector_range calls.
 * @param preserve_vmm_helper - determines whether vmm helper register specified
 * above should be preserved between compute_vector_range calls.
 * @param abi_param_offset - offset to rhs tensor from first binary post-op operation
 * specified by user from runtime structure passed to kernel as abi param 1.
 * @param dst_orig_offset - offset 0 to destination tensor
 * @param dst_d - descriptor of destination tensor (result after applying all post-ops
 * operations).
 * @param tail_opmask - register with loaded by user mask, used in avx512 for load with
 * tail handling.
 * @param tail_size - size of processed tail in elements.
 * @param use_exact_tail_scalar_bcast - in case of scalar broadcast user can disable
 * loading data with tail, usually bcast through entire vector is faster (usually 1 instruction)
 * vs. broadcasting limited by tail size (potentially several instructions). In case
 * when user during storing ignores values from vmm above tail size, setting this option to
 * false can result in better performance.
 * @param reg_tail_size - register with loaded size of tail, used in AVX2 and AVX-512
 * for load with tail in runtime.
 */
struct rhs_arg_static_params_t {
    rhs_arg_static_params_t(int rhs_dt_helper_vmm_idx,
            const Xbyak::Reg64 &rhs_addr_reg,
            const Xbyak::Reg64 &rhs_helper_reg,
            const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
            bool preserve_vmm_helper, int abi_param_offset,
            dim_t dst_orig_offset, const memory_desc_wrapper &dst_d,
            int tail_size = 0, bool use_exact_tail_scalar_bcast = false);
    rhs_arg_static_params_t(int rhs_dt_helper_vmm_idx,
            const Xbyak::Reg64 &rhs_addr_reg,
            const Xbyak::Reg64 &rhs_helper_reg,
            const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
            bool preserve_vmm_helper, int abi_param_offset,
            dim_t dst_orig_offset, const memory_desc_wrapper &dst_d,
            int tail_size, const Xbyak::Opmask &tail_opmask,
            bool use_exact_tail_scalar_bcast);
    rhs_arg_static_params_t(int rhs_dt_helper_vmm_idx,
            const Xbyak::Reg64 &rhs_addr_reg,
            const Xbyak::Reg64 &rhs_helper_reg,
            const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
            bool preserve_vmm_helper, int abi_param_offset,
            dim_t dst_orig_offset, const memory_desc_wrapper &dst_d,
            int tail_size, const Xbyak::Opmask &tail_opmask,
            const Xbyak::Reg64 &reg_tail_size,
            bool use_exact_tail_scalar_bcast);

    bool is_opmask_set() const noexcept { return is_opmask_set_; }

    mutable int rhs_dt_helper_vmm_idx;
    Xbyak::Reg64 rhs_addr_reg;
    Xbyak::Reg64 rhs_helper_reg;
    Xbyak::Reg64 rhs_addr_cache_reg;
    bool preserve_gpr_helpers;
    bool preserve_vmm_helper;
    int abi_param_offset;
    dim_t dst_orig_offset;
    memory_desc_wrapper dst_d;
    int tail_size;
    Xbyak::Opmask tail_opmask;
    bool use_exact_tail_scalar_bcast;
    Xbyak::Reg64 reg_tail_size;
    bool is_tail;

private:
    rhs_arg_static_params_t(int rhs_dt_helper_vmm_idx,
            const Xbyak::Reg64 &rhs_addr_reg,
            const Xbyak::Reg64 &rhs_helper_reg,
            const Xbyak::Reg64 &rhs_addr_cache_reg, bool preserve_gpr_helpers,
            bool preserve_vmm_helper, int abi_param_offset,
            dim_t dst_orig_offset, const memory_desc_wrapper &dst_d,
            int tail_size, const Xbyak::Opmask &tail_opmask,
            bool use_exact_tail_scalar_bcast, const Xbyak::Reg64 &reg_tail_size,
            bool is_opmask_set);

    bool is_opmask_set_;
};

/*
 * Represents params required by jit_uni_binary_injector_t that don't change
 * during it's entire lifetime.
 *
 * @param param1 - register storing abi param1. At the moment of calling
 * compute_vector_range method can be different than the default one defined
 * inside jit_generator_t.
 * @param bcast_set_t supported_strategy_set - set allowing disabling particular
 * bcast strategies
 * @param rhs_arg_static_params - params related to all binary post-ops right-hand side
 * arguments that don't change during entire lifetime of jit_uni_binary_injector_t
 * object.
 */
struct static_params_t {
    static_params_t(const Xbyak::Reg64 &param1,
            const bcast_set_t &supported_strategy_set,
            const rhs_arg_static_params_t &rhs_arg_static_params,
            fp8_conversion_e5m2_t *f8_e5m2_cvt,
            fp8_conversion_e4m3_t *f8_e4m3_cvt);
    static_params_t(const Xbyak::Reg64 &param1,
            const bcast_set_t &supported_strategy_set,
            const rhs_arg_static_params_t &rhs_arg_static_params);
    static_params_t(const Xbyak::Reg64 &param1,
            const rhs_arg_static_params_t &rhs_arg_static_params);

    Xbyak::Reg64 param1;
    const bcast_set_t supported_strategy_set;
    rhs_arg_static_params_t rhs_arg_static_params;
    // Both fp8 (e5m2 and e4m3) binary post-ops data types are possible.
    // Therefore, we need both fp8 converters.
    fp8_conversion_e5m2_t *f8_e5m2_cvt_ {nullptr};
    fp8_conversion_e4m3_t *f8_e4m3_cvt_ {nullptr};
};

/*
 * Mode of data load with tail for rhs:
 * STATIC - load based on given integer.
 * DYNAMIC - load based on opmask or 64-bit register.
 * DEFAULT - DYNAMIC for avx512, STATIC for others.
 */

enum class tail_lode_mode_t { STATIC, DYNAMIC, DEFAULT };

/*
 * Maps rhs_arg_idx of a binary post-op to the index of the vmm that holds its
 * scalar RHS as f32 in all lanes.
 */
using preloaded_rhs_t = std::map<int, int>;

/*
 * Represents params passed to compute_vector_range method of
 * jit_uni_binary_injector_t that can be different for each call.
 * Contains configurable std::maps where key is vmm index and value is
 * offset in elements. The offset value identifies tensor slice in particular
 * vmm. This is utilized by broadcasting mechanism. Offset, depending on the
 * implementation particular kernels, can be passed as value (usually during
 * unrolling), inside operand, under memory address.
 *
 * @param vmm_idx_to_out_addr - vmm mapped to address of destination tensor with offset,
 * used to calculate offset in no_broadcast strategy, but also in other strategies whose
 * calculations are based on no_broadcast strategy.
 * @param vmm_idx_to_out_reg - vmm mapped to register containing address of destination
 * with offset, used to calculate offset in no_broadcast strategy, but also in other
 * strategies whose calculations are based on no_broadcast strategy.
 * @param vmm_idx_to_out_elem_off_val - vmm mapped to offset in elements passed as raw
 * value intended to use in no_broadcast strategy, but also in other
 * strategies whose calculations are based on no_broadcast strategy.
 * @param vmm_tail_idx - vmm indices that contains data don't fill the whole vector (tail).
 * @param is_dynamic_tail_load - determines whether to load with tail in
 * runtime (based on the value from reg_tail_size or opmask) or based on given
 * integer.
 * @param preloaded_rhs - scalar RHS values already held in vmms, as returned by
 * jit_uni_postops_injector_t::preload_scalar_vector_range. The injector uses
 * these registers instead of reading the RHS from memory.
 */

struct rhs_arg_dynamic_params_t {
    std::map<int, Xbyak::Address> vmm_idx_to_out_addr;
    std::map<int, Xbyak::Reg64> vmm_idx_to_out_reg;
    std::map<int, dim_t> vmm_idx_to_out_elem_off_val;

    std::unordered_set<int> vmm_tail_idx_;
    tail_lode_mode_t tail_load_mode = tail_lode_mode_t::DEFAULT;
    preloaded_rhs_t preloaded_rhs;
};

/*
 * Checks if src1 data type is supported by binary injector.
 */
bool is_data_supported(cpu_isa_t isa, data_type_t data_type);

/*
 * Checks if broadcast of src1 is supported by binary injector.
 */
bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Checks if binary injection for given args is supported.
 */
bool is_supported(cpu_isa_t isa, const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Main mechanism responsible for injecting binary postops supporting various
 * isa: avx2, avx512 with core, bf16 extensions as well as data
 * types: f32, bf16, s32, u8, s8.
 *
 * The ISA to generate for is taken from the host generator, which already
 * carries it as its `max_cpu_isa()` ceiling. `Vmm` stays a template parameter
 * to keep the vector width statically typed.
 */
template <typename Vmm>
class jit_uni_binary_injector_t {
public:
    jit_uni_binary_injector_t(
            jit_generator_t *host, const static_params_t &static_params);

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * ordered set of vector registers' indexes. Function loads appropriate
     * slice of rhs tensor for computations based on internally determined
     * broadcast strategy and information about stored data in particular vmm
     * described inside rhs_arg_params.
     */
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            int rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * range <start_idx, end_idx) of vector registers' indexes. Function loads
     * appropriate slice of rhs tensor for computations based on internally
     * determined broadcast strategy and information about stored data in particular
     * vmm described inside rhs_arg_params.
     */
    void compute_vector_range(int start_idx, int end_idx, int rhs_arg_idx,
            const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * a single vector register index. Function loads appropriate slice of rhs tensor
     * for computations based on internally determined broadcast strategy and information
     * about stored data in particular vmm described inside rhs_arg_params.
     */
    void compute_vector(int idx, int rhs_arg_idx,
            const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Loads the scalar RHS of `post_op` into Vmm(vmm_idx) as f32 in all lanes.
     * Returns false and emits nothing if the post-op is not eligible. On
     * success, the host may add {rhs_arg_idx, vmm_idx} to
     * rhs_arg_dynamic_params_t::preloaded_rhs if Vmm(vmm_idx) is free.
     */
    bool preload_scalar_rhs(int vmm_idx, int rhs_arg_idx,
            const dnnl_post_ops::entry_t &post_op) const;

    /*
     * Reads a value from `base + byte_off` into vmm `dst` as f32, reusing the
     * RHS load path (`load_rhs` and its helpers). The sum post-op uses it to
     * read the accumulator's previous value, so the RHS infrastructure
     * (`rhs_addr_reg`, tail mask, integer-to-f32 conversion) operates on
     * accumulator memory. `byte_off` values past `int` range go through a
     * scratch register.
     */
    void load_acc_as_f32(const Vmm &dst, const Xbyak::Reg64 &base,
            dim_t byte_off, data_type_t data_type, bool with_tail,
            tail_lode_mode_t tail_load_mode = tail_lode_mode_t::DEFAULT) const;

private:
    /*
     * Checks if the RHS of `post_op` is a single value that one vmm can hold
     * for all vectors, see preload_scalar_rhs.
     */
    bool is_scalar_rhs_preloadable(const dnnl_post_ops::entry_t &post_op) const;
    /*
     * Determines if hint passed by user is valid (is inside range
     * <start_idx, end_idx>). If not it returns new vmm idx value that will be
     * used as temporary vmm in future computations.
     */
    int adjust_temp_vmm_hint(
            int user_hint, int start_idx, int end_idx, int max_vmm_idx) const;
    /*
     * Taking into account rhs_broadcasting_strategy and information from user
     * about tensor slice (rhs_arg_params) stored in Vmm(vmm_idx) calculates
     * address of rhs tensor slice needed for binary operation and returns
     * ptr to it.
     */
    Xbyak::Address prepare_rhs_arg_addr(int vmm_idx, int rhs_arg_idx,
            const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params,
            const broadcasting_strategy_t rhs_broadcasting_strategy,
            bool is_first, bool is_ternary_input) const;
    /*
     * Loads data and applies particular binary operation.
     */
    void inject_binary(const dnnl_post_ops::entry_t &post_op, Vmm dst,
            const Xbyak::Address &rhs_addr, bool with_tail,
            const tail_lode_mode_t tail_load_mode) const;

    /*
     * Loads data and applies binary operation that require ternary inputs.
     */
    void inject_binary_with_ternary_op(const dnnl_post_ops::entry_t &post_op,
            Vmm dst, const Xbyak::Address &rhs_addr, Vmm tmp_vmm,
            bool with_tail, const tail_lode_mode_t tail_load_mode) const;
    /*
     * Helper functions responsible for preparing rhs tensor slice address.
     */
    void append_no_broadcast_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes, bool is_first,
            bool cache_addr) const;
    void calculate_no_broadcast_base(
            Xbyak::Address addr, const Xbyak::Reg64 &out_reg) const;
    void calculate_no_broadcast_partial(const dim_t offset,
            const Xbyak::Reg64 &out_reg, dim_t elem_size_bytes) const;

    void append_oc_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes,
            bool is_first) const;
    void calculate_oc_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_ncsp_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_oc_blocked_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_blocked_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_oc_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_nspc_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_oc_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_cspn_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;

    void append_oc_d_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes,
            bool is_first) const;
    void calculate_oc_d_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_d_ncsp_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;

    void append_mb_sp_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes, bool is_first,
            const memory_desc_wrapper &rhs_d) const;
    void calculate_mb_sp_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_sp_ncsp_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_mb_sp_ncsp_base_rhs_strided(const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &rhs_d, const dim_t *dst_strides,
            const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
            dim_t elem_size_bytes) const;
    void calculate_mb_sp_ncsp_partial_rhs_strided(
            const memory_desc_wrapper &dst_d, const memory_desc_wrapper &rhs_d,
            dim_t dst_offset_bytes, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_mb_sp_blocked_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_sp_blocked_partial(const dim_t *strides,
            const dim_t offset, const Xbyak::Reg64 &tmp_reg,
            dim_t elem_size_bytes) const;
    void calculate_mb_sp_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_sp_nspc_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_mb_sp_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_sp_cspn_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;

    void append_mb_w_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes,
            bool is_first) const;
    void calculate_mb_w_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_w_ncsp_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_mb_w_blocked_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_w_blocked_partial(const dim_t *strides,
            const dim_t offset, const Xbyak::Reg64 &tmp_reg,
            dim_t elem_size_bytes) const;
    void calculate_mb_w_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_w_nspc_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_mb_w_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_w_cspn_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;

    void append_w_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes,
            bool is_first) const;
    void calculate_w_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_w_blocked_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_w_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_w_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;

    void append_mb_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes,
            bool is_first) const;
    void calculate_mb_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_ncsp_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_mb_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_nspc_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_mb_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_cspn_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;

    void append_oc_spatial_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes,
            bool is_first) const;
    void calculate_oc_spatial_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_spatial_ncsp_partial(const dim_t *strides,
            const dim_t offset, const Xbyak::Reg64 &tmp_reg,
            dim_t elem_size_bytes) const;
    void calculate_oc_spatial_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_spatial_nspc_partial(const dim_t *strides,
            const dim_t offset, const Xbyak::Reg64 &tmp_reg,
            dim_t elem_size_bytes) const;
    void calculate_oc_spatial_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_oc_spatial_cspn_partial(const dim_t *strides,
            const dim_t offset, const Xbyak::Reg64 &tmp_reg,
            dim_t elem_size_bytes) const;

    void append_hw_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes,
            bool is_first) const;
    void calculate_hw_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_hw_ncsp_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;

    void append_mb_oc_offset(
            const std::map<int, Xbyak::Address> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak::Reg64> &vmm_idx_to_out_reg,
            const std::map<int, dim_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes,
            bool is_first) const;
    void calculate_mb_oc_ncsp_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_oc_ncsp_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_mb_oc_nspc_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_oc_nspc_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;
    void calculate_mb_oc_cspn_base(
            const dim_t *strides, const Xbyak::Reg64 &tmp_reg) const;
    void calculate_mb_oc_cspn_partial(const dim_t *strides, const dim_t offset,
            const Xbyak::Reg64 &tmp_reg, dim_t elem_size_bytes) const;

    template <typename T>
    void execute_cmp_binary(const Vmm &dst, const Vmm &lhs, const T &rhs,
            const unsigned int cmp_predicate) const;
    template <typename T>
    void execute_binary(alg_kind_t binary_alg, const Vmm &dst, const Vmm &lhs,
            const T &rhs) const;
    void execute_prelu(const Vmm &dst, const Xbyak::Operand &rhs) const;

    /*
     * Used in scalar broadcast strategy, broadcasting single value of given
     * data type over entire vector Vmm register.
     */
    void execute_broadcast(const data_type_t &data_type, const Vmm &tmp_reg,
            const Xbyak::Address &rhs_addr,
            const tail_lode_mode_t tail_load_mode,
            bool with_tail = false) const;
    void load_rhs(const data_type_t &data_type, const Vmm &tmp_reg,
            const Xbyak::Address &rhs_addr,
            const tail_lode_mode_t tail_load_mode,
            bool with_tail = false) const;
    void execute_broadcast_tail_with_opmask(const data_type_t &data_type,
            const Vmm &tmp_reg, const Xbyak::Address &rhs_addr) const;
    void execute_broadcast_tail_statically(const data_type_t &data_type,
            const Vmm &tmp_reg, const Xbyak::Address &rhs_addr,
            const int tail_size) const;
    void execute_broadcast_tail_with_gpr(const data_type_t &data_type,
            const Vmm &tmp_reg, const Xbyak::Address &rhs_addr) const;
    void load_rhs_tail_dynamically_with_opmask(const data_type_t &data_type,
            const Vmm &tmp_vmm, const Xbyak::Address &rhs_addr) const;
    void load_rhs_tail_dynamically_with_gpr(
            const data_type_t &data_type, const Vmm &tmp_vmm) const;
    void load_rhs_tail_statically(const data_type_t &data_type,
            const Vmm &tmp_vmm, const Xbyak::Address &rhs_addr) const;
    void execute_broadcast_no_tail(const data_type_t &data_type,
            const Vmm &tmp_vmm, const Xbyak::Address &rhs_addr) const;
    void execute_broadcast_s8u8_no_tail(const data_type_t &data_type,
            const Vmm &tmp_vmm, const Xbyak::Address &rhs_addr) const;
    void load_rhs_no_tail(const data_type_t &data_type, const Vmm &tmp_reg,
            const Xbyak::Address &rhs_addr) const;
    void load_rhs_i8_no_tail(const data_type_t &data_type, const Vmm &tmp_reg,
            const Xbyak::Address &rhs_addr) const;
    void cvt_to_f32(const Vmm &tmp_reg) const;
    /*
     * Returns pair consisting of flag indication preservation is needed for vmm
     * index in second member that should be used as temporary vmm inside inject
     * binary.
     */
    std::pair<bool, int> should_preserve_vmm(int curr_idx, int vmm_hint,
            int max_vmm_idx, bool dt_helper_vmm_needed) const;
    /*
     * Used in isa != avx512 where m32bcst is not supported, replaces ptr_b
     * with ptr.
     */
    Xbyak::Address remove_bcast_bit(const Xbyak::Address &rhs_addr) const;

    /*
    * Returns an Opmask register, that can be used from temporary use.
    * The returned register will be different from tail_opmask.
    * Note: The Opmask register returned may need to be preserved.
    */
    Xbyak::Opmask get_aux_kmask() const;

    jit_generator_t *host_;
    fp8_conversion_e5m2_t *f8_e5m2_cvt_ {nullptr};
    fp8_conversion_e4m3_t *f8_e4m3_cvt_ {nullptr};

    const rhs_arg_static_params_t rhs_arg_static_params_;
    const Xbyak::Reg64 param1_;
    const bcast_set_t supported_strategy_set_;
    const cpu_isa_t isa_ = host_->max_cpu_isa();
    const bool has_avx512_core_ = is_superset(isa_, avx512_core);
    const bool has_avx512_core_fp16_ = is_superset(isa_, avx512_core_fp16);
    const bool has_avx2_ = is_superset(isa_, avx2);

    static constexpr int sizeof_reg64 = 8;
    /*
     * Instructions from SSE/AVX used to compute binary result like vaddps where
     * second operand is memory, require mem operand to be 16/32 byte explicitly
     * aligned. (Intel Manual chapter 2.4).
     * Rule is relaxed from AVX2 (Intel Manual chapter 14.9).
     * When using benchdnn zmalloc_protect doesn't guarantee that tensor memory
     * address is 64 byte aligned, which can cause segmentation fault.
     */
    const bool binary_op_with_unaligned_mem_operand_allowed_
            = is_superset(isa_, avx2);
};

} // namespace binary_injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
