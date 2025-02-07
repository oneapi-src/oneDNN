/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_CODEGEN_KERNEL_HPP
#define GPU_INTEL_JIT_CODEGEN_KERNEL_HPP

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif

#include "common/cpp_compat.hpp"

#include "common/impl_registration.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/jit/codegen/operand.hpp"
#include "gpu/intel/jit/codegen/register_allocator.hpp"
#include "gpu/intel/jit/codegen/register_scope.hpp"
#include "gpu/intel/jit/codegen/reorder.hpp"
#include "gpu/intel/jit/emulation.hpp"
#include "gpu/intel/jit/generator.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/ir_builder.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/ir/walk_order.hpp"
#include "ngen/ngen.hpp"
#include "ngen/ngen_register_allocator.hpp"
#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <template <ngen::HW> class KernelT>
struct ir_generator_t : public generator_base_t {
    ir_generator_t(const kernel_desc_base_t &kernel_desc)
        : kernel_name_(kernel_desc.kernel_name()), kernel_desc_(kernel_desc) {}

    const char *kernel_name() const override { return kernel_name_.c_str(); }

    xpu::binary_t get_binary(const ocl::engine_t *engine) override {
        try {
#define CASE(hw) \
    case ngen::HW::hw: { \
        KernelT<ngen::HW::hw> kernel(kernel_desc_, engine); \
        return kernel.getBinary(engine->context(), engine->device()); \
    }
            auto *device_info = engine->device_info();
            auto hw = convert_dnnl_arch_to_ngen(device_info->gpu_arch());
            switch (hw) {
                REG_GEN9_ISA(CASE(Gen9));
                REG_GEN11_ISA(CASE(Gen11));
                REG_XELP_ISA(CASE(XeLP));
                REG_XEHP_ISA(CASE(XeHP));
                REG_XEHPG_ISA(CASE(XeHPG));
                REG_XEHPC_ISA(CASE(XeHPC));
                REG_XE2_ISA(CASE(Xe2));
                REG_XE3_ISA(CASE(Xe3));
                default: gpu_assert(false) << "Unexpected GPU architecture";
            }
#undef CASE
        } catch (ngen::out_of_registers_exception &) { return xpu::binary_t(); }
        return xpu::binary_t();
    }

private:
    std::string kernel_name_;
    const kernel_desc_base_t &kernel_desc_;
};

class expr_binding_t {
public:
    expr_binding_t(ngen::HW hw) : hw_(hw) {}

    ~expr_binding_t() {
        if (!cpp_compat::uncaught_exceptions()) {
            gpu_assert(expr2dst_.empty()) << "Detected missing unbind_dst().";
        }
    }

    bool is_dst_bound(const expr_t &expr) const {
        return expr2dst_.count(expr) == 1;
    }

    ngen_operand_t get_dst(const expr_t &expr) const {
        gpu_assert(is_dst_bound(expr)) << "Destination is not bound: " << expr;
        return expr2dst_.at(expr);
    }

    void bind_dst(const expr_t &expr, const ngen_operand_t &operand) {
        gpu_assert(!expr.is_empty());
        auto ret = expr2dst_.insert({expr, operand});
        gpu_assert(ret.second) << "Already bound: " << expr;
    }

    void unbind_dst(const expr_t &expr) {
        gpu_assert(!expr.is_empty());
        auto it = expr2dst_.find(expr);
        gpu_assert(it != expr2dst_.end());
        expr2dst_.erase(it);
    }

    bool is_bound(const expr_t &expr) const {
        return expr2operand_.count(expr) == 1;
    }

    ngen_operand_t get(const expr_t &expr, bool allow_empty = false) const {
        if (expr.is_empty()) return ngen_operand_t();
        if (!is_bound(expr)) {
            if (!allow_empty)
                gpu_assert(false) << "Operand is not bound: " << expr;
            return ngen_operand_t();
        }
        return expr2operand_.at(expr);
    }

    void bind(const expr_t &expr, const ngen::Subregister &sub) {
        bind(expr, ngen_operand_t(reg_buf_data_t(hw_, sub)));
    }

    void bind(const expr_t &expr, const ngen_operand_t &operand) {
        if (is_dst_bound(expr)) unbind_dst(expr);

        auto op_to_bind = operand;

        // Operand is with predicate - can't bind.
        if (operand.mod().getPredCtrl() != ngen::PredCtrl::None) return;

        int esize = operand.mod().getExecSize();
        if (esize == 0) esize = 1;
        if (esize != expr.type().elems() && !expr.type().is_bool()) {
            gpu_assert(expr.type().is_scalar() || esize == 1)
                    << "Expected broadcast.";
            if (operand.is_reg_buf_data() && esize != 1) {
                // Bind scalar expression to the first vector element.
                op_to_bind = operand.reg_buf_data().format(
                        0, ngen::DataType::invalid, 1);
            }
        }

        auto ret = expr2operand_.insert({expr, op_to_bind});
        gpu_assert(ret.second) << "Already bound: " << expr;
    }

    void unbind(const expr_t &expr) {
        gpu_assert(!expr.is_empty());

        auto it = expr2operand_.find(expr);
        gpu_assert(it != expr2operand_.end());
        expr2operand_.erase(it);
    }

private:
    ngen::HW hw_;
    object_map_t<expr_t, ngen_operand_t> expr2dst_;
    object_map_t<expr_t, ngen_operand_t> expr2operand_;
};

template <ngen::HW hw>
class expr_evaluator_t;

template <ngen::HW hw>
class ir_to_ngen_t;

template <ngen::HW hw>
class ir_kernel_t : public generator_t<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    friend class expr_evaluator_t<hw>;
    friend class ir_to_ngen_t<hw>;
    friend class send_impl_t;

    ir_kernel_t(const kernel_desc_base_t &desc, const impl::engine_t *engine,
            const debug_config_t &debug_config)
        : generator_t<hw>(debug_config)
        , kernel_name_(desc.kernel_name())
        , exec_cfg_(desc.exec_cfg(engine))
        , local_range_(desc.local_range())
        , require_dpas_(desc.with_dpas())
        , regs_(exec_cfg_.regs())
        , ra_(hw, desc.kernel_name())
        , emu_strategy(hw, exec_cfg_.hw().stepping_id()) {
        desc.init_kernel_iface(kernel_iface_);
        setStepping(exec_cfg_.hw().stepping_id());
        ra_.setRegisterCount(regs_);
    }

    ir_kernel_t(const std::string &kernel_name, const exec_config_t &exec_cfg,
            const compute::range_t &local_range, bool require_dpas,
            const debug_config_t &debug_config)
        : generator_t<hw>(debug_config)
        , kernel_name_(kernel_name)
        , exec_cfg_(exec_cfg)
        , local_range_(local_range)
        , require_dpas_(require_dpas)
        , regs_(exec_cfg.regs())
        , ra_(hw, kernel_name)
        , emu_strategy(hw, exec_cfg.hw().stepping_id()) {
        setStepping(exec_cfg.hw().stepping_id());
        ra_.setRegisterCount(regs_);
    }

    const exec_config_t &exec_cfg() { return exec_cfg_; }
    const kernel_iface_t &kernel_iface() const { return kernel_iface_; }
    void set_kernel_iface(const kernel_iface_t &kernel_iface) {
        kernel_iface_ = kernel_iface;
    }

    void setup_interface(const stmt_t &kernel_body = stmt_t()) {
        externalName(kernel_name_);
        requireLocalID(3);
        requireLocalSize();
        requireGRF(regs_);
        requireSIMD(exec_cfg_.simd());
        requireBarrier();
        if (require_dpas_) requireDPAS();
        if (has_send_atomics(kernel_body)) requireGlobalAtomics();

        for (int i = 0; i < kernel_iface_.nargs(); i++) {
            auto &name = kernel_iface_.arg_name(i);
            auto &type = kernel_iface_.arg_type(i);
            if (type.is_ptr()) {
                newArgument(name, ngen::ExternalArgumentType::GlobalPtr,
                        ngen::GlobalAccessType::Stateless);
            } else {
                newArgument(name, to_ngen(type));
            }
        }

        if (!kernel_body.is_empty() && local_range_) {
            int slm_size = alloc_manager_t(kernel_body)
                                   .total_size(alloc_kind_t::slm);
            int max_slm_size = compute::device_info_t::max_slm_size_per_tg(
                    convert_ngen_arch_to_dnnl(hw), thread_group_size(),
                    regs_ > 128);
            if (slm_size > max_slm_size) {
                // TODO: Use status code for this check.
                gpu_except_not_implemented("SLM size limit is exceeded.");
            }
            requireSLM(slm_size);
        }

        finalizeInterface();
    }

    void generate_prologue() {
        setDefaultNoMask();
        setDefaultAutoSWSB(true);

        prologue();

        // Claim registers.
        ra_.claim(r0);
        for (int i = 0; i < 3; i++)
            ra_.claim(getLocalID(i));

        for (int i = 0; i < kernel_iface_.nargs(); i++) {
            ra_.claim(getArgument(kernel_iface_.arg_name(i)));
        }

        if (emu_strategy.emulate64) {
            emu_state.temp[0] = ra_.alloc();
            emu_state.temp[1] = ra_.alloc();
        }
        // Enable IEEE f32 -> s32 rounding and f64/f32/f16 denormals.
        or_(1, cr0, cr0, uint16_t(0x14C0));

        // Allocate and initialize signal header for future use.
        if (require_signal_header_) {
            signal_header_ = ra_.alloc();
            barrierheader(signal_header_);
        }
    }

    void bind_external_vars(const stmt_t &kernel_body,
            const walk_order_t &kernel_grid_walk_order,
            expr_binding_t &expr_binding) {
        bind_external_vars(kernel_body, expr_binding);
        bind_kernel_grid_walk_order(kernel_grid_walk_order, expr_binding);
    }

    void bind_external_vars(
            const stmt_t &kernel_body, expr_binding_t &expr_binding) {
        alloc_manager_t alloc_mgr(kernel_body);

        // Bind grid indices.
        int r0_sub_idxs[] = {1, 6, 7};
        for (int i = 0; i < 3; i++) {
            auto tmp = ra_.template alloc_sub<int32_t>();
            mov(1, tmp, r0.ud(r0_sub_idxs[i]));
            expr_binding.bind(ir_builder_t::tg_idxs()[i], tmp);
        }

        // Bind local IDs.
        for (int i = 0; i < 3; i++) {
            expr_binding.bind(
                    ir_builder_t::local_ids()[i], getLocalID(i).uw(0));
        }

        // Bind arguments.
        for (int i = 0; i < kernel_iface_.nargs(); i++) {
            auto &arg_var = kernel_iface_.arg_var(i);
            auto &name = kernel_iface_.arg_name(i);
            if (arg_var.type().is_ptr()) {
                auto alloc_buf
                        = alloc_mgr.find_buffer(name, /*allow_empty=*/true);
                if (alloc_buf.is_empty()) {
                    gpu_warning() << "Unused argument: " << arg_var;
                    continue;
                }
                gpu_assert(alloc_buf.is_same(arg_var));
            }
            expr_binding.bind(arg_var, getArgument(name));
        }

        // Bind SLM buffer (SLM loads/stores use 0-based offsets).
        auto slm_buf = alloc_mgr.find_buffer("slm", /*allow_empty=*/true);
        if (!slm_buf.is_empty()) expr_binding.bind(slm_buf, to_ngen(expr_t(0)));
    }

    void bind_kernel_grid_walk_order_blocked(const ngen::Subregister &id,
            const std::vector<std::pair<int, int>> &blocks,
            const std::vector<int> &dims, const std::vector<expr_t> &grid_vars,
            expr_binding_t &expr_binding) {
        int ndims = (int)dims.size();
        int nblocks = (int)blocks.size();
        std::vector<ngen::Subregister> rem_dims(ndims);
        std::vector<ngen::Subregister> dim_idxs(ndims);
        for (int i = 0; i < ndims; i++) {
            rem_dims[i] = ra_.alloc_sub<int32_t>();
            dim_idxs[i] = ra_.alloc_sub<int32_t>();
            emov(1, rem_dims[i], dims[i]);
            emov(1, dim_idxs[i], 0);
        }

        auto mul_add = [&](const ngen::Subregister &dst,
                               const ngen::Subregister &src0,
                               const ngen::Subregister &src1, uint32_t src2) {
            bool is_src2_16_bit
                    = (src2 <= std::numeric_limits<uint16_t>::max());
            if (hw >= ngen::HW::XeLP && is_src2_16_bit && false) {
                mad(1, dst, src0, src1, src2);
            } else {
                auto tmp = ra_.alloc_sub<uint64_t>();
                mul(1, tmp.d(0), src1, src2 & 0xFFFF);
                mul(1, tmp.d(1), src1, src2 >> 16);
                shl<uint32_t>(1, tmp.ud(1), tmp.ud(1), 16);
                add(1, tmp.d(0), tmp.d(1), tmp.d(0));
                add(1, dst, src0, tmp.d(0));
                ra_.safeRelease(tmp);
            }
        };

        auto _id = ra_.alloc_sub<int32_t>();
        auto qot = ra_.alloc_sub<int32_t>();
        auto rem = ra_.alloc_sub<int32_t>();
        auto rem_size = ra_.alloc_sub<uint32_t>();
        auto rounded = ra_.alloc_sub<int32_t>();
        emov(1, _id, id);
        for (int i = nblocks - 1; i >= 0; i--) {
            int dim_idx = blocks[i].first;
            int inner_block_size = 1;
            for (int j = 0; j < i; j++) {
                if (blocks[j].first == dim_idx)
                    inner_block_size *= blocks[j].second;
            }
            emov(1, rem_size, inner_block_size);
            for (int j = 0; j < ndims; j++) {
                if (j == dim_idx) continue;
                emul(1, rem_size, rem_size, rem_dims[j]);
            }
            eidiv(1, qot, rem, _id, rem_size);
            emov(1, _id, rem);
            mul_add(dim_idxs[dim_idx], qot, dim_idxs[dim_idx],
                    blocks[i].second);
            emul(1, rounded, qot, inner_block_size);
            eadd(1, rounded, rem_dims[dim_idx], -rounded);
            min_(1, rem_dims[dim_idx], rounded, inner_block_size);
        }
        ra_.safeRelease(_id);
        ra_.safeRelease(qot);
        ra_.safeRelease(rem);
        ra_.safeRelease(rem_size);
        ra_.safeRelease(rounded);

        for (int i = 0; i < ndims; i++)
            ra_.safeRelease(rem_dims[i]);

        for (int i = 0; i < ndims; i++) {
            expr_binding.bind(grid_vars[i], dim_idxs[i]);
        }
    }

    void bind_kernel_grid_walk_order_non_blocked(const ngen::Subregister &id,
            const std::vector<std::pair<int, int>> &blocks,
            const std::vector<expr_t> &grid_vars,
            expr_binding_t &expr_binding) {
        int nblocks = (int)blocks.size();
        gpu_assert((int)grid_vars.size() == nblocks);
        if (nblocks == 1) {
            expr_binding.bind(grid_vars[0], id);
            return;
        }
        auto _id = ra_.alloc_sub<int32_t>();
        emov(1, _id, id);
        for (int i = 0; i < nblocks; i++) {
            int dim_idx = blocks[i].first;
            auto idx = ra_.alloc_sub<int32_t>();
            eidiv(1, _id, idx, _id, (uint32_t)blocks[i].second);
            expr_binding.bind(grid_vars[dim_idx], idx);
        }
        ra_.safeRelease(_id);
    }

    void bind_kernel_grid_walk_order(
            const walk_order_t &walk_order, expr_binding_t &expr_binding) {
        const int grid_ndims = 3;
        ngen::Subregister grid_ids[grid_ndims] = {r0.ud(1), r0.ud(6), r0.ud(7)};
        for (int i = 0; i < grid_ndims; i++) {
            std::vector<std::pair<int, int>> blocks;
            std::unordered_map<pvar_t, int> dim_map;
            auto to_dim_idx = [&](const pvar_t &dim) {
                if (dim_map.count(dim) != 0) return dim_map.at(dim);
                int idx = (int)dim_map.size();
                dim_map.emplace(dim, idx);
                return idx;
            };
            for (auto &b : walk_order.blocks()) {
                if (b.grid_id != i) continue;
                blocks.emplace_back(to_dim_idx(b.dim), b.size);
            }
            if (dim_map.empty()) continue;
            std::vector<int> dims;
            std::vector<expr_t> grid_vars;
            dims.resize(dim_map.size());
            grid_vars.resize(dim_map.size());
            for (auto &kv : dim_map) {
                dims[kv.second] = walk_order.dim_size(kv.first);
                grid_vars[kv.second] = walk_order.grid_var(kv.first);
            }
            if (walk_order.is_blocked(i) || gpu_utils::dev_getenv("B", false)) {
                bind_kernel_grid_walk_order_blocked(
                        grid_ids[i], blocks, dims, grid_vars, expr_binding);
            } else {
                bind_kernel_grid_walk_order_non_blocked(
                        grid_ids[i], blocks, grid_vars, expr_binding);
            }
        }
    }

    void generate_epilogue() {
        epilogue();
        pad_kernel();
    }

    // Kernel padding for instruction prefetch.
    void pad_kernel() {
        for (int rep = 0; rep < 8; rep++)
            nop();
    }

    void emov(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0) {
        if (dst.is_reg_data()) {
            if (src0.is_reg_data()) {
                emov(mod, dst.reg_data(), src0.reg_data());
            } else if (src0.is_reg_buf_data()) {
                emov(mod, dst.reg_data(), src0.reg_buf_data().reg_data());
            } else if (src0.is_immediate()) {
                emov(mod, dst.reg_data(), src0.immediate());
            } else if (dst.type() == ngen::DataType::uw
                    || dst.type() == ngen::DataType::ud) {
                emov(mod, dst.reg_data(), src0.flag_register());
                if (src0.is_negated()) {
                    not_(mod, dst.reg_data(), dst.reg_data());
                }
            } else {
                emov(mod | src0.flag_register_mod(), dst.reg_data(), 1);
                emov(mod | ~src0.flag_register_mod(), dst.reg_data(), 0);
            }
        } else {
            // dst is a flag register.
            gpu_assert(!dst.is_negated());
            auto _mod = mod;
            _mod.setExecSize(1);
            if (src0.is_reg_data()) {
                emov(_mod, dst.flag_register(), src0.reg_data());
            } else {
                emov(_mod, dst.flag_register(), src0.immediate());
            }
        }
    }

    void eadd(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src0.is_immediate()) {
            gpu_assert(src1.is_reg_data());
            eadd(mod, dst, src1, src0);
            return;
        }
        if (src1.is_reg_data()) {
            eadd(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            if (ngen_is_qw(src1.type())) {
                auto tmp = ra_.alloc_sub(src1.type());
                emov(1, tmp, src1.immediate());
                eadd(mod, dst.reg_data(), src0.reg_data(), tmp);
                ra_.safeRelease(tmp);
            } else {
                eadd(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
            }
        }
    }

    void emul(const ngen::InstructionModifier &mod_, const ngen_operand_t &dst_,
            const ngen_operand_t &src0_, const ngen_operand_t &src1_) {
        int width = mod_.getExecSize();
        int esize = ngen_is_dw(src0_.type()) && ngen_is_dw(src1_.type())
                ? 8
                : width;
        int step = esize;
        auto src0 = src0_;
        auto src1 = src1_;
        auto dst = dst_;
        auto mod = mod_;
        if (src0.is_immediate()) {
            gpu_assert(src1.is_reg_data());
            emul(mod, dst, src1, src0);
            return;
        }
        if (src1.is_reg_data()) {
            for (int i = 0; i < width; i += step) {
                step = std::min(step, width - i);
                step = utils::rnd_down_pow2(step);
                esize = step;
                mod.setExecSize(esize);
                auto subreg = [&](const ngen_operand_t &src) {
                    auto hs = src.reg_buf_data().hs();
                    int stride = hs == 0 ? 1 : esize;
                    return src.sub_reg_data(i, stride);
                };
                src0 = subreg(src0_);
                src1 = subreg(src1_);
                dst = dst_.sub_reg_data(i, 1);
                if (ngen_is_dw(src1.type()) && ngen_is_w(src0.type())) {
                    emul(mod, dst.reg_data(), src1.reg_data(), src0.reg_data());
                } else {
                    emul(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
                }
            }
        } else {
            auto &src1_imm = src1.immediate();
            if (ngen_is_qw(dst.type()) || ngen_is_w(src1_imm.getType())) {
                emul(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
                return;
            }
            if (ngen_is_dw(src1_imm.getType())) {
                gpu_assert(mod.getExecSize() == 1);
                auto tmp = ra_.alloc_sub<int64_t>();
                if (ngen_is_w(src0.type())) {
                    auto tmp_src1 = ra_.alloc_sub<int32_t>();
                    emov(mod, tmp_src1.d(0), src0.reg_data());
                    emul(mod, tmp.q(0), tmp_src1.d(0), src1_imm);
                    ra_.safeRelease(tmp_src1);
                } else {
                    emul(mod, tmp.q(0), src0.reg_data(), src1_imm);
                }
                emov(mod, dst.reg_data(), tmp.reinterpret(0, dst.type()));
                ra_.safeRelease(tmp);
                return;
            }
            emul(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void eadd3(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &_src0, const ngen_operand_t &_src1,
            const ngen_operand_t &_src2) {
        auto src0 = _src0;
        auto src1 = _src1;
        auto src2 = _src2;
        auto scope = ngen_register_scope_t(ra_);
        align_src_dst_offset(this, scope, mod, dst, src0);
        align_src_dst_offset(this, scope, mod, dst, src1);
        if (hw >= ngen::HW::XeHP) {
            if (src2.is_reg_data()) {
                align_src_dst_offset(this, scope, mod, dst, src2);
                add3(mod, dst.reg_data(), fixup_ternary_rgn(src0.reg_data()),
                        fixup_ternary_rgn(src1.reg_data()), src2.reg_data());
            } else {
                add3(mod, dst.reg_data(), fixup_ternary_rgn(src0.reg_data()),
                        fixup_ternary_rgn(src1.reg_data()), src2.immediate());
            }
            return;
        }
        add(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        if (src2.is_reg_data()) {
            align_src_dst_offset(this, scope, mod, dst, src2);
            add(mod, dst.reg_data(), dst.reg_data(), src2.reg_data());
        } else {
            add(mod, dst.reg_data(), dst.reg_data(), src2.immediate());
        }
    }

    void emad(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &_src0, const ngen_operand_t &_src1,
            const ngen_operand_t &_src2) {
        auto src0 = _src0;
        auto src1 = _src1;
        auto src2 = _src2;
        auto scope = ngen_register_scope_t(ra_);
        align_src_dst_offset(this, scope, mod, dst, src1);
        if (src2.is_reg_data()) {
            align_src_dst_offset(this, scope, mod, dst, src0);
            align_src_dst_offset(this, scope, mod, dst, src2);
            mad(mod, dst.reg_data(), fixup_ternary_rgn(src0.reg_data()),
                    fixup_ternary_rgn(src1.reg_data()), src2.reg_data());
        } else if (hw < ngen::HW::XeLP) {
            align_src_dst_offset(this, scope, mod, dst, src0);
            mul(mod, dst.reg_data(), src1.reg_data(), src2.immediate());
            add(mod, dst.reg_data(), dst.reg_data(), src0.reg_data());
        } else if (src0.is_immediate()
                && (ngen_is_dw(src0.type())
                        || src0.type() == ngen::DataType::uw)) {
            // dword immediate src0 is not supported, move to a register.
            auto tmp_src0 = scope.alloc_sub(src0.type());
            mov(1, tmp_src0, src0.immediate());
            mad(mod, dst.reg_data(), tmp_src0,
                    fixup_ternary_rgn(src1.reg_data()), src2.immediate());
        } else {
            align_src_dst_offset(this, scope, mod, dst, src0);
            mad(mod, dst.reg_data(), fixup_ternary_rgn(src0.reg_data()),
                    fixup_ternary_rgn(src1.reg_data()), src2.immediate());
        }
    }

    void ediv(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (!src1.is_immediate()) {
            // Immediate src0 is not supported with fdiv_ieee.
            if (src0.is_immediate() && hw >= ngen::HW::XeHPC) {
                auto tmp_src0 = ra_.alloc_sub(src0.type());
                mov(mod, tmp_src0, src0.immediate());
                efdiv(mod, dst, ngen_operand_t(reg_buf_data_t(hw, tmp_src0)),
                        src1);
                ra_.safeRelease(tmp_src0);
            } else {
                efdiv(mod, dst, src0, src1);
            }
        } else {
            auto &src1_imm = src1.immediate();
            if (to_ir(src0.type()).is_fp()) {
                ngen::Immediate src1_inv_value(1.f / to_cpp<float>(src1_imm));
                emul(mod, dst, src0, src1_inv_value);
            } else {
                int32_t src1_value = to_cpp<int32_t>(src1_imm);
                gpu_assert(0 < src1_value && src1_value <= INT32_MAX)
                        << src1_value;
                eidiv(mod, dst.reg_data(), ngen::Subregister(), src0.reg_data(),
                        src1_value);
            }
        }
    }

    void efdiv(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        int esize = mod.getExecSize();
        int grf_size = ngen::GRF::bytes(hw);
        int div_esize = std::min(esize, grf_size / int(sizeof(float)));

        gpu_assert(dst.type() == ngen::DataType::f);
        gpu_assert(src0.type() == ngen::DataType::f);
        gpu_assert(src1.type() == ngen::DataType::f);

        if (src1.reg_data().getHS() != 0) {
            int nregs = std::max(1, (mod.getExecSize() * 4) / grf_size);
            auto s1 = src1.reg_data();
            auto tmp_range_ = ra_.alloc_range(nregs);
            auto tmp = tmp_range_[0].retype(s1.getType());
            auto t1 = tmp.f(s1.getOffset())
                              .setRegion(s1.getVS(), s1.getWidth(), s1.getHS());
            inv(mod, t1, s1);
            emul(mod, dst.reg_data(), src0.reg_data(), t1);
            ra_.safeRelease(tmp);
            return;
        }

        // fdiv_ieee() is not supported in XeHPG so we use a less precise, inv-based sequence.
        if (hw < ngen::HW::XeHPC) {
            auto tmp = ra_.alloc_sub<float>();
            inv(1, tmp, src1.reg_data());
            emul(mod, dst, src0, ngen_operand_t(reg_buf_data_t(hw, tmp)));
            ra_.safeRelease(tmp);
            return;
        }

        auto one = ra_.alloc().f();
        auto zero = ra_.alloc().f();
        auto tmp = ra_.alloc_range(4);

        auto div_mod = ngen::InstructionModifier(mod);
        div_mod.setExecSize(div_esize);

        mov(div_mod, one, ngen::Immediate(1));
        mov(div_mod, zero, ngen::Immediate(0));

        for (int i = 0; i < mod.getExecSize(); i += div_esize) {
            // Copy to temporary registers to ensure dst, num and denom are
            // distinct as required for fdiv_ieee.
            auto d = dst.sub_reg_data(i, div_esize).reg_data();
            auto s0 = src0.sub_reg_data(i, div_esize).reg_data();
            auto s1 = src1.sub_reg_data(i, 1).reg_data();
            bool force_spill = overlaps(div_esize, d, s0)
                    || overlaps(div_esize, d, s1)
                    || overlaps(div_esize, s0, s1);
            bool d_spill = force_spill || (d.getHS() != 1);
            bool s0_spill = force_spill || (s0.getHS() != 1);
            bool s1_spill = force_spill || (s1.getHS() != 1);
            auto dst_rd = w_spill(d, div_esize, d_spill);
            auto src0_rd = r_spill(s0, div_esize, s0_spill);
            auto src1_rd = r_spill(s1, div_esize, s1_spill);
            // Enable mask as fdiv_ieee relies on masked if/endif flow.
            setDefaultNoMask(false);
            fdiv_ieee(div_mod, f0[0], dst_rd(), src0_rd(), src1_rd(), zero, one,
                    tmp);
            setDefaultNoMask(true);
        }

        ra_.safeRelease(one);
        ra_.safeRelease(zero);
        ra_.safeRelease(tmp);
    }

    void emod(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        gpu_assert(src1.is_immediate());
        auto &src1_imm = src1.immediate();
        int32_t src1_value = to_cpp<int32_t>(src1_imm);
        gpu_assert(0 < src1_value && src1_value <= INT32_MAX) << src1_value;
        eidiv(mod, ngen::Subregister(), dst.reg_data(), src0.reg_data(),
                src1_value);
    }

    void eshl(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            shl(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            shl(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void eshr(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            shr(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            shr(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emin(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            min_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            min_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emax(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            max_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            max_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void ecmp(const ngen::InstructionModifier &mod, const ngen_operand_t &src0,
            const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            cmp(mod, src0.reg_data(), src1.reg_data());
        } else if (utils::one_of(src1.immediate().getType(), ngen::DataType::q,
                           ngen::DataType::uq)) {
            auto tmp = src1.immediate().getType() == ngen::DataType::uq
                    ? ra_.alloc().uq()
                    : ra_.alloc().q();
            mov(1, tmp, src1.immediate());
            cmp(mod, src0.reg_data(), tmp);
            ra_.safeRelease(tmp);
        } else {
            cmp(mod, src0.reg_data(), src1.immediate());
        }
    }

    void ecmp(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            cmp(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else if (utils::one_of(src1.immediate().getType(), ngen::DataType::q,
                           ngen::DataType::uq)) {
            auto tmp = src1.immediate().getType() == ngen::DataType::uq
                    ? ra_.alloc().uq()
                    : ra_.alloc().q();
            mov(1, tmp, src1.immediate());
            cmp(mod, dst.reg_data(), src0.reg_data(), tmp);
            ra_.safeRelease(tmp);
        } else {
            cmp(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void eand(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src0.is_reg_data() && src1.is_reg_data()) {
            and_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            if (src0.is_reg_data())
                and_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
            else
                and_(mod, dst.reg_data(), src1.reg_data(), src0.immediate());
        }
    }

    // Emulates integer division by a non-constant (rounding towards negative
    // infinity).
    // Requirements:
    //     INT32_MIN <= x <= UINT32_MAX
    //     0         <  y <= INT32_MAX
    // Computes:
    //     qot = x / y
    //     rem = x % y
    // See ir_utils::idiv_magicgu_packed() for information
    // about magic calculation.
    void eidiv(const ngen::InstructionModifier &mod, const ngen::RegData &qot,
            const ngen::RegData &rem, const ngen::RegData &x,
            const ngen::RegData &_y, const ngen::RegData &_magic) {
        gpu_assert(x.getHS() == 0);
        gpu_assert(_y.getType() == ngen::DataType::ud);
        gpu_assert(_magic.getHS() == 0);
        gpu_assert(_magic.getType() == ngen::DataType::uq);

        bool x_signed = utils::one_of(x.getType(), ngen::DataType::b,
                ngen::DataType::w, ngen::DataType::d);
        auto div_type = (x_signed ? ngen::DataType::d : ngen::DataType::ud);
        auto magic = ngen::Subregister(
                _magic, _magic.getOffset(), _magic.getType());
        auto y = ngen::Subregister(_y, _y.getOffset(), _y.getType());
        auto m = magic.ud(0);
        auto p = magic.ud(1);

        auto x_tmp = ra_.alloc().retype(div_type);
        auto qot_tmp = ra_.alloc().retype(div_type);
        auto p_tmp = ra_.alloc_sub<uint32_t>();
        auto _x = x_tmp[0];
        auto _qot = qot_tmp[0];
        mov(1, _x, x);

        auto acc = acc0.retype(div_type);
        mul(1, acc[0], _x, m.uw(0));
        mach(1, _qot, _x, m);
        add(1, p_tmp, p, -32);
        cmp(1 | ge | f0[0], p, 32);
        shr<uint32_t>(1 | f0[0], _qot, _qot, p_tmp);
        shr<uint32_t>(1 | ~f0[0], _qot, _x, p);
        if (!qot.isInvalid()) mov(mod, qot, _qot);

        if (!rem.isInvalid()) {
            // rem = x - qot * y
            auto tmp = ra_.alloc_sub<uint64_t>();
            mul(1, tmp.ud(0), _qot, y.uw(0));
            mul(1, tmp.ud(1), _qot, y.uw(1));
            shl<uint32_t>(1, tmp.ud(1), tmp.ud(1), 16);
            add(1, tmp.ud(0), tmp.ud(1), tmp.ud(0));
            add(mod, rem, x, -tmp.ud(0));
            ra_.safeRelease(tmp);
        }

        ra_.safeRelease(x_tmp);
        ra_.safeRelease(qot_tmp);
        ra_.safeRelease(p_tmp);
    }

    // Emulates integer division by a non-constant (rounding towards negative
    // infinity). This version is based on FP inverse and does not require a
    // pre-computed "magic" value. Note, that cr0 register is updated/restored
    // to use RTZ mode when converting float -> int.
    // Requirements (validated range):
    //    -2^20 <= x <= 2^20
    //     0    <  y <= 2^20
    // Computes:
    //     qot = x / y
    //     rem = x % y
    void eidiv(const ngen::InstructionModifier &mod, const ngen::RegData &_qot,
            const ngen::RegData &rem, const ngen::RegData &x,
            const ngen::RegData &_y, bool update_cr0_fp_to_int_rtz = true) {
        gpu_assert(mod.getExecSize() == 1);
        gpu_assert(_y.getType() == ngen::DataType::ud);
        auto cr0_save = ra_.alloc_sub<uint32_t>();
        auto f_tmp = ra_.alloc_sub<float>();
        auto x_tmp = ra_.alloc_sub<float>();
        auto qot_tmp = ra_.alloc_sub<int32_t>();
        auto y = ngen::Subregister(_y, _y.getOffset(), _y.getType());
        mov(1, cr0_save, cr0);
        // Set RTZ rounding mode when converting float to int.
        and_(1, cr0, cr0, ~0x1000);
        mov(1, f_tmp, y);
        mov(1, x_tmp, x);
        inv(1, f_tmp, f_tmp);
        add(1, f_tmp.ud(0), f_tmp.ud(0), 1);
        mul(1, f_tmp, x_tmp, f_tmp);
        mov(mod, qot_tmp, f_tmp);
        if (!rem.isInvalid()) {
            auto tmp = ra_.alloc_sub<int64_t>();
            mul(1, tmp.d(0), qot_tmp, y.uw(0));
            mul(1, tmp.d(1), qot_tmp, y.uw(1));
            shl<uint32_t>(1, tmp.ud(1), tmp.ud(1), 16);
            add(1, tmp.d(0), tmp.d(1), tmp.d(0));
            add(mod, rem, x, -tmp.d(0));
            ra_.safeRelease(tmp);
        }
        if (!_qot.isInvalid()) mov(mod, _qot, qot_tmp);
        mov(1, cr0, cr0_save);
        ra_.safeRelease(cr0_save);
        ra_.safeRelease(f_tmp);
        ra_.safeRelease(x_tmp);
        ra_.safeRelease(qot_tmp);
    }

    // Emulates integer division by a constant (rounding towards negative
    // infinity)
    // Requirements:
    //     INT32_MIN <= x <= UINT32_MAX
    //     0         <  y <= INT32_MAX
    // Computes:
    //     qot = x / y
    //     rem = x % y
    void eidiv(const ngen::InstructionModifier &mod, const ngen::RegData &qot,
            const ngen::RegData &rem, const ngen::RegData &x, uint32_t y) {
        bool x_signed = utils::one_of(x.getType(), ngen::DataType::b,
                ngen::DataType::w, ngen::DataType::d);
        auto div_type = (x_signed ? ngen::DataType::d : ngen::DataType::ud);
        gpu_assert(x.getHS() == 0);
        if (ngen::utils::is_zero_or_pow2(y)) {
            auto _x = get_subregister(x);
            if (x.getNeg() || (x == qot) || (x == rem)) {
                // Negation modifier has bitwise semantics with shr/and so x
                // needs to be arithmetically negated first.
                _x = ra_.alloc_sub(div_type);
                mov(1, _x, x);
            }
            if (!qot.isInvalid()) shr(mod, qot, _x, ngen::utils::log2(y));
            if (!rem.isInvalid()) and_(mod, rem, _x, y - 1);
            if (_x != x) ra_.safeRelease(_x);
            return;
        }

        uint32_t m = 0, p = 0;
        ir_utils::idiv_magicgu(y, m, p);

        auto x_tmp = ra_.alloc().retype(div_type);
        auto qot_tmp = ra_.alloc().retype(div_type);
        auto _x = x_tmp[0];
        auto _qot = qot_tmp[0];
        mov(1, _x, x);

        // qot = (x * m) >> p
        bool use_mach = true;
        if (use_mach) {
            auto acc = acc0.retype(div_type);
            mul(1, acc[0], _x, m & 0xFFFF);
            mach(1, _qot, _x, m);
            shr<uint32_t>(1, _qot, _qot, p - 32);
        } else {
            auto q_tmp = qot_tmp.retype(ngen::DataType::q);
            emul(1, q_tmp[0], _x, m);
            eshr(1, q_tmp.uq(0), q_tmp.uq(0), p);
        }

        if (!rem.isInvalid()) {
            // rem = x - qot * y
            bool y_is_16_bit = (y <= static_cast<uint32_t>(
                                        std::numeric_limits<int16_t>::max()));
            if (hw >= ngen::HW::XeLP && y_is_16_bit) {
                mad(mod, rem, x, _qot, -int16_t(y));
            } else {
                auto tmp = ra_.alloc_sub<uint64_t>();
                mul(1, tmp.ud(0), _qot, y & 0xFFFF);
                mul(1, tmp.ud(1), _qot, y >> 16);
                shl<uint32_t>(1, tmp.ud(1), tmp.ud(1), 16);
                add(1, tmp.ud(0), tmp.ud(1), tmp.ud(0));
                add(mod, rem, x, -tmp.ud(0));
                ra_.safeRelease(tmp);
            }
        }
        if (!qot.isInvalid()) mov(mod, qot, _qot);

        ra_.safeRelease(x_tmp);
        ra_.safeRelease(qot_tmp);
    }

    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0) {
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, emu_strategy);
    }
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::Immediate src0) {
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, emu_strategy);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        if (ngen_is_xf(dst.getType())) {
            mul(mod, dst, src0, src1);
            return;
        }
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1) {
        if (ngen_is_xf(dst.getType())) {
            mul(mod, dst, src0, src1);
            return;
        }
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void eshl(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1) {
        EmulationImplementation::eshl<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void eshr(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1) {
        EmulationImplementation::eshr<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }

protected:
    // Helper RAII class allocating a temporary GRF buffer aligned at a
    // register boundary for instructions that require aligned operands.
    class spiller_t {
    public:
        // rd - register region to align
        // esize - execution size used with the register region
        // read - whether operand is to be used as input (needs pre-copy)
        // write - whether operand is to be used as output (needs post-copy)
        // force_copy - always copy the region (even if it's aligned)
        spiller_t(ir_kernel_t<hw> *host, const ngen::RegData &rd, int esize,
                bool read, bool write, bool force_copy)
            : host_(host), rd_(rd), esize_(esize), read_(read), write_(write) {
            if (rd.getOffset() == 0 && !force_copy) return;

            int w = rd.getWidth();
            int hs = rd.getHS();
            int vs = rd.getVS();
            int grf_size = ngen::GRF::bytes(hw);
            int regs = utils::div_up(
                    std::max(esize * hs, 1) * rd.getBytes(), grf_size);
            tmp_range_ = host_->ra_.alloc_range(regs);
            auto tmp = tmp_range_[0].retype(rd_.getType());
            tmp_ = ngen::RegisterRegion(tmp, vs, w, hs);
            if (read_) host_->mov(esize_, to_xd(tmp_), to_xd(rd_));
        }

        spiller_t(spiller_t &&other) : spiller_t(other) {
            other.tmp_range_ = ngen::GRFRange();
        }

        ngen::RegData operator()() const {
            return tmp_.isInvalid() ? rd_ : tmp_;
        }

        ~spiller_t() {
            if (tmp_range_.isInvalid()) return;
            if (write_) host_->mov(esize_, to_xd(rd_), to_xd(tmp_));
            host_->ra_.safeRelease(tmp_range_);
        }

    private:
        spiller_t(const spiller_t &) = default;

        static ngen::RegData to_xd(const ngen::RegData &rd) {
            auto ret = rd;
            switch (rd.getBytes()) {
                case 1: ret.setType(ngen::DataType::ub); break;
                case 2: ret.setType(ngen::DataType::uw); break;
                case 4: ret.setType(ngen::DataType::ud); break;
                default: gpu_error_not_expected();
            }
            return ret;
        }

        ir_kernel_t<hw> *host_ = nullptr;
        ngen::RegData rd_;
        int esize_;
        bool read_ = false;
        bool write_ = false;
        ngen::GRFRange tmp_range_;
        ngen::RegData tmp_;
    };

    spiller_t spill(const ngen::RegData &rd, int esize, bool read, bool write,
            bool force_copy) {
        return spiller_t(this, rd, esize, read, write, force_copy);
    }

    spiller_t r_spill(
            const ngen::RegData &rd, int esize, bool force_copy = false) {
        return spill(rd, esize, true, false, force_copy);
    }

    spiller_t w_spill(
            const ngen::RegData &rd, int esize, bool force_copy = false) {
        return spill(rd, esize, false, true, force_copy);
    }

    static bool overlaps(
            int esize, const ngen::RegData &a, const ngen::RegData &b) {
        int grf_size = ngen::GRF::bytes(hw);
        int a_beg = a.getBase() * grf_size + a.getByteOffset();
        int b_beg = b.getBase() * grf_size + b.getByteOffset();
        int a_end = a_beg + std::max(esize * a.getHS(), 1) * a.getBytes() - 1;
        int b_end = b_beg + std::max(esize * b.getHS(), 1) * b.getBytes() - 1;
        a_beg /= grf_size;
        b_beg /= grf_size;
        a_end /= grf_size;
        b_end /= grf_size;
        if (a_beg <= b_beg && b_beg <= a_end) return true;
        if (a_beg <= b_end && b_end <= a_end) return true;
        return false;
    }

    int thread_group_size() const {
        gpu_assert(local_range_);
        int local_size = 1;
        for (int i = 0; i < (int)local_range_.ndims(); i++) {
            local_size *= (int)local_range_[i];
        }
        return ir_utils::safe_divide(local_size, exec_cfg_.simd());
    }

    static ngen::RegData fixup_ternary_rgn(const ngen::RegData &r) {
        ngen::RegData retn = r;
        return ((retn.getHS() == 1) && (retn.getVS() == retn.getWidth()))
                ? retn.setRegion(1, 1, 0)
                : retn;
    }

    kernel_iface_t kernel_iface_;
    std::string kernel_name_;
    exec_config_t exec_cfg_;
    compute::range_t local_range_;
    bool require_dpas_;
    bool require_signal_header_ = false;
    int regs_;
    reg_allocator_t ra_;
    ngen::GRF signal_header_;

    EmulationStrategy emu_strategy;
    EmulationState emu_state;
};

#define IR_KERNEL_EMULATION_FORWARD(hw) \
    using ir_kernel_t<hw>::emov; \
    using ir_kernel_t<hw>::eadd; \
    using ir_kernel_t<hw>::emul; \
    using ir_kernel_t<hw>::eshl; \
    using ir_kernel_t<hw>::eshr;

#define IR_KERNEL_FORWARD(hw) \
    NGEN_FORWARD_OPENCL(hw) \
    IR_KERNEL_EMULATION_FORWARD(hw) \
    using ir_kernel_t<hw>::exec_cfg; \
    using ir_kernel_t<hw>::kernel_iface; \
    using ir_kernel_t<hw>::set_kernel_iface; \
    using ir_kernel_t<hw>::setup_interface; \
    using ir_kernel_t<hw>::bind_external_vars; \
    using ir_kernel_t<hw>::generate_prologue; \
    using ir_kernel_t<hw>::generate_epilogue; \
    using ir_kernel_t<hw>::emu_strategy; \
    using ir_kernel_t<hw>::ra_;

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic pop
#endif

#endif
