/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_JIT_CODEGEN_KERNEL_HPP
#define GPU_JIT_CODEGEN_KERNEL_HPP

#include "common/cpp_compat.hpp"

#include "gpu/jit/codegen/operand.hpp"
#include "gpu/jit/codegen/register_allocator.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/ngen/ngen.hpp"
#include "gpu/jit/ngen/ngen_register_allocator.hpp"

#include "gpu/jit/gemm/emulation.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

inline size_t icache_size(ngen::HW arch) {
    switch (arch) {
        case gpu_gen9: return 48 * 1024;
        case gpu_gen11: return 48 * 1024;
        case gpu_xe_lp: return 48 * 1024;
        case gpu_xe_hp: return 48 * 1024;
        case gpu_xe_hpg: return 96 * 1024;
        case gpu_xe_hpc: return 80 * 1024;
        default: return 0;
    }
}

template <template <ngen::HW> class KernelT, ngen::HW arch, typename... ArgsT>
std::unique_ptr<jit::jit_generator_base> make_generator(ArgsT &&...args) {

    auto raw_kernel = new KernelT<arch>(std::forward<ArgsT>(args)...);
    if (raw_kernel->getRootStreamLength() > icache_size(arch)) {
        ir_warning() << raw_kernel->kernel_name()
                     << " larger than icache, kernel: "
                     << raw_kernel->getRootStreamLength()
                     << " bytes, icache: " << icache_size(arch) << " bytes\n";
    }
    return std::unique_ptr<jit::jit_generator_base>(raw_kernel);
}

template <template <ngen::HW> class KernelT, typename... ArgsT>
compute::kernel_t make_kernel(
        gpu_primitive_t *primitive, engine_t *engine, ArgsT &&...args) {
    using namespace compute;
    kernel_t kernel;

    if (primitive->cache_blob()) {
        status_t status = primitive->create_kernel(engine, &kernel, nullptr);
        if (status != status::success) return kernel_t();
        return kernel;
    }

    auto *compute_engine = utils::downcast<compute_engine_t *>(engine);
    auto *device_info = compute_engine->device_info();
    auto arch = convert_dnnl_arch_to_ngen(device_info->gpu_arch());

    std::unique_ptr<jit::jit_generator_base> jit_kernel;
#define CASE(gpu_arch) \
    case gpu_arch: \
        jit_kernel = make_generator<KernelT, gpu_arch>( \
                std::forward<ArgsT>(args)...); \
        break;
    switch (arch) {
        REG_GEN9_ISA(CASE(gpu_gen9));
        REG_GEN11_ISA(CASE(gpu_gen11));
        REG_XELP_ISA(CASE(gpu_xe_lp));
        REG_XEHP_ISA(CASE(gpu_xe_hp));
        REG_XEHPG_ISA(CASE(gpu_xe_hpg));
        REG_XEHPC_ISA(CASE(gpu_xe_hpc));
        default: break;
    }
#undef CASE

#ifdef DNNL_DEV_MODE
    gpu_gen_t actual_arch = ngen::HW::Unknown;
    switch (device_info->gpu_arch()) {
        case gpu_arch_t::gen9: actual_arch = gpu_gen9; break;
        case gpu_arch_t::gen11: actual_arch = gpu_gen11; break;
        case gpu_arch_t::xe_lp: actual_arch = gpu_xe_lp; break;
        case gpu_arch_t::xe_hp: actual_arch = gpu_xe_hp; break;
        case gpu_arch_t::xe_hpg: actual_arch = gpu_xe_hpg; break;
        case gpu_arch_t::xe_hpc: actual_arch = gpu_xe_hpc; break;
        case gpu_arch_t::unknown: actual_arch = ngen::HW::Unknown; break;
    }
    ir_assert(actual_arch == arch)
            << "Cannot emulate executing gpu_arch environment";
#endif

    if (!jit_kernel) return kernel_t();

    status_t status
            = primitive->create_kernel(engine, &kernel, jit_kernel.get());
    if (status != status::success) return kernel_t();
    return kernel;
}

class expr_binding_t {
public:
    expr_binding_t(ngen::HW hw) : hw_(hw) {}

    ~expr_binding_t() {
        if (!cpp_compat::uncaught_exceptions()) {
            ir_assert(expr2dst_.empty()) << "Detected missing unbind_dst().";
        }
    }

    bool is_dst_bound(const expr_t &expr) const {
        return expr2dst_.count(expr) == 1;
    }

    ngen_operand_t get_dst(const expr_t &expr) const {
        ir_assert(is_dst_bound(expr)) << "Destination is not bound: " << expr;
        return expr2dst_.at(expr);
    }

    void bind_dst(const expr_t &expr, const ngen_operand_t &operand) {
        ir_assert(!expr.is_empty());
        auto ret = expr2dst_.insert({expr, operand});
        ir_assert(ret.second) << "Already bound: " << expr;
    }

    void unbind_dst(const expr_t &expr) {
        ir_assert(!expr.is_empty());
        auto it = expr2dst_.find(expr);
        ir_assert(it != expr2dst_.end());
        expr2dst_.erase(it);
    }

    bool is_bound(const expr_t &expr) const {
        return expr2operand_.count(expr) == 1;
    }

    ngen_operand_t get(const expr_t &expr, bool allow_empty = false) const {
        if (expr.is_empty()) return ngen_operand_t();
        if (!is_bound(expr)) {
            if (!allow_empty)
                ir_assert(false) << "Operand is not bound: " << expr;
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
            ir_assert(expr.type().is_scalar() || esize == 1)
                    << "Expected broadcast.";
            if (operand.is_reg_buf_data() && esize != 1) {
                // Bind scalar expression to the first vector element.
                op_to_bind = operand.reg_buf_data().format(
                        0, ngen::DataType::invalid, 1);
            }
        }

        auto ret = expr2operand_.insert({expr, op_to_bind});
        ir_assert(ret.second) << "Already bound: " << expr;
    }

    void unbind(const expr_t &expr) {
        ir_assert(!expr.is_empty());

        auto it = expr2operand_.find(expr);
        ir_assert(it != expr2operand_.end());
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

enum class grf_mode_t {
    any, // Kernel sets optimal grf mode
    matches, // Propogate grf mode to avoid context switch
    small, // Force small grf_mode
    large, // Force large grf_mode
};

template <ngen::HW hw>
class ir_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    friend class expr_evaluator_t<hw>;
    friend class ir_to_ngen_t<hw>;
    friend class send_impl_t;

    ir_kernel_t(const std::string &kernel_name, const exec_config_t &exec_cfg,
            const kernel_info_t &kernel_info,
            const compute::nd_range_t &nd_range, bool require_dpas,
            grf_mode_t grf_mode = grf_mode_t::any)
        : kernel_name_(kernel_name)
        , exec_cfg_(exec_cfg)
        , kernel_info_(kernel_info)
        , nd_range_(nd_range)
        , require_dpas_(require_dpas)
        , regs_((grf_mode == grf_mode_t::large)             ? 256
                          : (grf_mode == grf_mode_t::small) ? 128
                                                            : exec_cfg.regs())
        , ra_(hw, kernel_name,
                  grf_mode == grf_mode_t::any ? reg_allocator_t::warn_all
                                              : reg_allocator_t::warn_default)
        , emu_strategy(hw, exec_cfg.hw_cfg().stepping_id()) {
        setStepping(exec_cfg.hw_cfg().stepping_id());
        ra_.setRegisterCount(regs_);
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

        for (int i = 0; i < kernel_info_.nargs(); i++) {
            auto &name = kernel_info_.arg_name(i);
            auto &type = kernel_info_.arg_type(i);
            if (type.is_ptr()) {
                newArgument(name, ngen::ExternalArgumentType::GlobalPtr);
            } else {
                newArgument(name, to_ngen(type));
            }
        }

        if (!kernel_body.is_empty()) {
            int slm_size = alloc_manager_t(kernel_body)
                                   .total_size(alloc_kind_t::slm);
            int max_slm_size = compute::device_info_t::max_slm_size_per_tg(
                    convert_ngen_arch_to_dnnl(hw), thread_group_size(),
                    regs_ > 128);
            if (slm_size > max_slm_size) {
                // TODO: Use status code for this check.
                ir_except_not_implemented("SLM size limit is exceeded.");
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

        for (int i = 0; i < kernel_info_.nargs(); i++) {
            ra_.claim(getArgument(kernel_info_.arg_name(i)));
        }

        if (emu_strategy.emulate64) {
            emu_state.temp[0] = ra_.alloc();
            emu_state.temp[1] = ra_.alloc();
        }
        // Enable IEEE f32 -> s32 rounding and f32/f16 denormals.
        or_(1, cr0, cr0, uint16_t(0x1480));

        // Allocate and initialize signal header for future use.
        if (require_signal_header_) {
            signal_header_ = ra_.alloc();
            barrierheader(signal_header_);
        }
    }

    void bind_external_vars(const stmt_t &kernel_body,
            const grid_info_t &kernel_grid,
            const std::array<expr_t, 3> &local_id,
            expr_binding_t &expr_binding) {
        alloc_manager_t alloc_mgr(kernel_body);

        // Bind grid indices.
        int r0_sub_idxs[] = {1, 6, 7};
        for (int i = 0; i < 3; i++) {
            auto tmp = ra_.template alloc_sub<int32_t>();
            mov(1, tmp, r0.ud(r0_sub_idxs[i]));
            expr_binding.bind(kernel_grid.idx(i), tmp);
        }

        // Bind local IDs.
        for (int i = 0; i < 3; i++) {
            expr_binding.bind(local_id[i], getLocalID(i).uw(0));
        }

        // Bind arguments.
        for (int i = 0; i < kernel_info_.nargs(); i++) {
            auto &arg_var = kernel_info_.arg_var(i);
            auto &name = kernel_info_.arg_name(i);
            if (arg_var.type().is_ptr()) {
                auto alloc_buf = alloc_mgr.find_buffer(name);
                ir_assert(alloc_buf.is_same(arg_var));
            }
            expr_binding.bind(arg_var, getArgument(name));
        }

        // Bind SLM buffer (SLM loads/stores use 0-based offsets).
        auto slm_buf = alloc_mgr.find_buffer("slm", /*allow_empty=*/true);
        if (!slm_buf.is_empty()) expr_binding.bind(slm_buf, to_ngen(expr_t(0)));
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
            ir_assert(!dst.is_negated());
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
            ir_assert(src1.is_reg_data());
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
            ir_assert(src1.is_reg_data());
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
                ir_assert(mod.getExecSize() == 1);
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
            const ngen_operand_t &src0, const ngen_operand_t &src1,
            const ngen_operand_t &src2) {
        if (hw >= ngen::HW::XeHP) {
            if (src2.is_reg_data()) {
                add3(mod, dst.reg_data(), src0.reg_data(), src1.reg_data(),
                        src2.reg_data());
            } else {
                add3(mod, dst.reg_data(), src0.reg_data(), src1.reg_data(),
                        src2.immediate());
            }
            return;
        }
        add(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        if (src2.is_reg_data()) {
            add(mod, dst.reg_data(), dst.reg_data(), src2.reg_data());
        } else {
            add(mod, dst.reg_data(), dst.reg_data(), src2.immediate());
        }
    }

    void emad(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1,
            const ngen_operand_t &src2) {
        if (src2.is_reg_data()) {
            mad(mod, dst.reg_data(), src0.reg_data(), src1.reg_data(),
                    src2.reg_data());
        } else if (hw < ngen::HW::XeLP) {
            mul(mod, dst.reg_data(), src1.reg_data(), src2.immediate());
            add(mod, dst.reg_data(), dst.reg_data(), src0.reg_data());
        } else if (src0.is_immediate()
                && (ngen_is_dw(src0.type())
                        || src0.type() == ngen::DataType::uw)) {
            // dword immediate src0 is not supported, move to a register.
            auto tmp_src0 = ra_.alloc_sub(src0.type());
            mov(1, tmp_src0, src0.immediate());
            mad(mod, dst.reg_data(), tmp_src0, src1.reg_data(),
                    src2.immediate());
            ra_.safeRelease(tmp_src0);
        } else {
            mad(mod, dst.reg_data(), src0.reg_data(), src1.reg_data(),
                    src2.immediate());
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
            int32_t src1_value = to_cpp<int32_t>(src1_imm);
            ir_assert(0 < src1_value && src1_value <= INT32_MAX) << src1_value;
            eidiv(mod, dst.reg_data(), ngen::Subregister(), src0.reg_data(),
                    src1_value);
        }
    }

    void efdiv(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        int esize = mod.getExecSize();
        int grf_size = ngen::GRF::bytes(hw);
        int div_esize = std::min(esize, grf_size / int(sizeof(float)));

        ir_assert(dst.type() == ngen::DataType::f);
        ir_assert(src0.type() == ngen::DataType::f);
        ir_assert(src1.type() == ngen::DataType::f);
        ir_assert(src1.reg_data().getHS() == 0);

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
            auto dst_rd = w_spill(d, div_esize, force_spill);
            auto src0_rd = r_spill(s0, div_esize, force_spill);
            auto src1_rd = r_spill(s1, div_esize, force_spill);
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
        ir_assert(src1.is_immediate());
        auto &src1_imm = src1.immediate();
        int32_t src1_value = to_cpp<int32_t>(src1_imm);
        ir_assert(0 < src1_value && src1_value <= INT32_MAX) << src1_value;
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
        } else {
            cmp(mod, src0.reg_data(), src1.immediate());
        }
    }

    void ecmp(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            cmp(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
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

    // Adapted version of magicgu function from Hacker's Delight 10-15.
    static void eidiv_magicgu(uint32_t d, uint32_t &m, uint32_t &p) {
        uint32_t s32_max = std::numeric_limits<int32_t>::max();
        ir_assert(d != 0 && d <= s32_max);
        uint64_t nc = (s32_max / d) * d - 1;
        for (p = 32; p < 64; p++) {
            uint64_t _2p = 1LL << p;
            if (_2p > nc * (d - 1 - (_2p - 1) % d)) {
                m = (_2p + d - 1 - (_2p - 1) % d) / d;
                return;
            }
        }
        ir_error_not_expected();
    }

    // Emulates integer division by a constant.
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
        ir_assert(x.getHS() == 0);
        if (ngen::utils::is_zero_or_pow2(y)) {
            auto _x = get_subregister(x);
            if (x.getNeg()) {
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
        eidiv_magicgu(y, m, p);

        auto x_tmp = ra_.alloc().retype(div_type);
        auto qot_tmp = ra_.alloc().retype(div_type);
        auto _x = x_tmp[0];
        auto _qot = qot_tmp[0];
        mov(1, _x, x);

        // qot = (x * m) >> p
        auto acc = acc0.retype(div_type);
        mul(1, acc[0], _x, m & 0xFFFF);
        mach(1, _qot, _x, m);
        shr<uint32_t>(1, _qot, _qot, p - 32);
        if (!qot.isInvalid()) mov(mod, qot, _qot);

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
                default: ir_error_not_expected();
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
        int local_size = 1;
        ir_assert(nd_range_.local_range());
        for (int i = 0; i < (int)nd_range_.ndims(); i++) {
            local_size *= (int)nd_range_.local_range()[i];
        }
        return ir_utils::safe_divide(local_size, exec_cfg_.simd());
    }

    std::string kernel_name_;
    exec_config_t exec_cfg_;
    kernel_info_t kernel_info_;
    compute::nd_range_t nd_range_;
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
    using ir_kernel_t<hw>::setup_interface; \
    using ir_kernel_t<hw>::bind_external_vars; \
    using ir_kernel_t<hw>::generate_prologue; \
    using ir_kernel_t<hw>::generate_epilogue; \
    using ir_kernel_t<hw>::emu_strategy; \
    using ir_kernel_t<hw>::ra_;

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
