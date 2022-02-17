/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef GPU_JIT_CONV_CONV_KERNEL_HPP
#define GPU_JIT_CONV_CONV_KERNEL_HPP

#include "common/cpp_compat.hpp"

#include "gpu/jit/conv/bank_conflict_allocation.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/fma_support.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/kernel_builder.hpp"
#include "gpu/jit/conv/kernel_info.hpp"
#include "gpu/jit/conv/message_support.hpp"
#include "gpu/jit/conv/ngen_proxy.hpp"
#include "gpu/jit/conv/post_op_support.hpp"
#include "gpu/jit/conv/reduce_support.hpp"
#include "gpu/jit/conv/reg_buf.hpp"
#include "gpu/jit/conv/reorder_support.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/ngen/ngen.hpp"
#include "gpu/jit/ngen/ngen_register_allocator.hpp"

#include "gpu/jit/gemm/emulation.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

template <typename T>
T to_cpp(const ngen::Immediate &imm) {
    auto u64 = uint64_t(imm);
    switch (imm.getType()) {
        case ngen::DataType::w:
            return (T)utils::bit_cast<std::array<int16_t, 4>>(u64)[0];
        case ngen::DataType::uw:
            return (T)utils::bit_cast<std::array<uint16_t, 4>>(u64)[0];
        case ngen::DataType::d:
            return (T)utils::bit_cast<std::array<int32_t, 2>>(u64)[0];
        case ngen::DataType::ud:
            return (T)utils::bit_cast<std::array<uint32_t, 2>>(u64)[0];
        case ngen::DataType::q: return (T)utils::bit_cast<int64_t>(u64);
        case ngen::DataType::uq: return (T)utils::bit_cast<uint64_t>(u64);
        default: ir_error_not_expected();
    }
    return 0;
}

// type_t to ngen::DataType convertor.
ngen::DataType to_ngen(const type_t &type) {
    ir_assert(type.is_scalar()) << "Expected scalar type.";

#define CASE(_kind, ngen_enum) \
    if (type.kind() == type_kind_t::_kind) return ngen::DataType::ngen_enum

    CASE(bf16, bf);
    CASE(f16, hf);
    CASE(f32, f);
    CASE(s16, w);
    CASE(s32, d);
    CASE(s64, q);
    CASE(s8, b);
    CASE(u16, uw);
    CASE(u32, ud);
    CASE(u64, uq);
    CASE(u8, ub);

    if (type == type_t::byte_ptr()) return ngen::DataType::uq;

#undef CASE
    ir_error_not_expected();
    return ngen::DataType::invalid;
}

ngen::Immediate to_ngen(
        const expr_t &expr, const type_t &type = type_t::undef()) {
    ir_assert(expr.type().is_scalar()) << "Vector types are not supported.";
    if (expr.is<int_imm_t>()) {
        auto &imm = expr.as<int_imm_t>();
        // No conversion.
        if (utils::one_of(type, type_t::undef(), expr.type()))
            return ngen::Immediate(imm.value);
            // Do conversion.
#define CASE(cpp_type) \
    if (type.is_cpp<cpp_type>()) return ngen::Immediate(cpp_type(imm.value))

        CASE(int16_t);
        CASE(int32_t);
        CASE(int64_t);
        CASE(uint16_t);
        CASE(uint32_t);
        CASE(uint64_t);

#undef CASE
        ir_error_not_expected() << "Can't convert expression: " << expr;
    } else if (expr.is<float_imm_t>()) {
        ir_assert(utils::one_of(type, type_t::undef(), type_t::f32()))
                << "Conversion is not supported.";
        auto &imm = expr.as<float_imm_t>();
        return ngen::Immediate(imm.value);
    }
    ir_error_not_expected() << "Can't convert expression: " << expr;
    return ngen::Immediate();
}

ngen::Bundle to_ngen(const ngen_proxy::Bundle &bundle) {
    return ngen::Bundle(bundle.bank_id, bundle.bundle_id);
}

ngen::InstructionModifier to_ngen(
        const ngen_proxy::InstructionModifier &mod_proxy) {
    ngen::InstructionModifier mod;
    if (mod_proxy.is_atomic) mod |= ngen::ThreadCtrl::Atomic;
    if (!mod_proxy.sbid.is_empty()) mod |= ngen::SBID(mod_proxy.sbid.token).set;
    return mod;
}

ngen::AtomicOp to_ngen(ngen_proxy::AtomicOp atomic_op) {
    switch (atomic_op) {
        case ngen_proxy::AtomicOp::fadd: return ngen::AtomicOp::fadd;
        default: ir_error_not_expected();
    }
    return ngen::AtomicOp(std::numeric_limits<uint16_t>::max());
}

ngen::ConditionModifier cmp_op_to_ngen(op_kind_t op_kind) {
    ir_assert(is_cmp_op(op_kind));
    switch (op_kind) {
        case op_kind_t::_eq: return ngen::ConditionModifier::eq;
        case op_kind_t::_ne: return ngen::ConditionModifier::ne;
        case op_kind_t::_ge: return ngen::ConditionModifier::ge;
        case op_kind_t::_gt: return ngen::ConditionModifier::gt;
        case op_kind_t::_le: return ngen::ConditionModifier::le;
        case op_kind_t::_lt: return ngen::ConditionModifier::lt;
        default: ir_error_not_expected();
    }
    return ngen::ConditionModifier::none;
}

ngen::Immediate ngen_negate(const ngen::Immediate &imm) {
    switch (imm.getType()) {
        case ngen::DataType::w: return ngen::Immediate(-to_cpp<int16_t>(imm));
        case ngen::DataType::d: return ngen::Immediate(-to_cpp<int32_t>(imm));
        case ngen::DataType::f: return ngen::Immediate(-to_cpp<float>(imm));
        default: ir_error_not_expected();
    }
    return ngen::Immediate();
}

bool ngen_is_qw(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::q, ngen::DataType::uq);
}

bool ngen_is_dw(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::d, ngen::DataType::ud);
}

bool ngen_is_w(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::w, ngen::DataType::uw);
}

bool ngen_is_b(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::b, ngen::DataType::ub);
}

bool ngen_is_xf(ngen::DataType type) {
    return utils::one_of(
            type, ngen::DataType::bf, ngen::DataType::hf, ngen::DataType::f);
}

inline ngen::Subregister get_subregister(
        ngen::HW hw, ngen::DataType type, const ngen::GRFRange &r, int idx) {
    int grf_size = ngen::GRF::bytes(hw);
    int type_size = ngen::getBytes(type);
    int off = idx * type_size;
    return r[off / grf_size].sub((off % grf_size) / type_size, type);
}

inline ngen::Subregister get_subregister(const ngen::RegData &rd) {
    return ngen::Subregister(rd, rd.getOffset(), rd.getType());
}

enum class ngen_operand_kind_t {
    invalid,
    immediate,
    reg_buf_data,
    flag_register
};

// Wrapper to generalize ngen::FlagRegister, ngen::RegData, reg_buf_data_t and
// ngen::Immediate operands.
class ngen_operand_t {
public:
    ngen_operand_t() : kind_(ngen_operand_kind_t::invalid) {}

    ngen_operand_t(const ngen::FlagRegister &flag)
        : kind_(ngen_operand_kind_t::flag_register)
        , ptr_(new ngen::FlagRegister(flag),
                  destroy<ngen_operand_kind_t::flag_register>) {}

    ngen_operand_t(const reg_buf_data_t &reg_buf_data)
        : kind_(ngen_operand_kind_t::reg_buf_data)
        , ptr_(new reg_buf_data_t(reg_buf_data),
                  destroy<ngen_operand_kind_t::reg_buf_data>) {}

    ngen_operand_t(const ngen::Immediate &imm)
        : kind_(ngen_operand_kind_t::immediate)
        , ptr_(new ngen::Immediate(imm),
                  destroy<ngen_operand_kind_t::immediate>) {}

    template <typename T>
    ngen_operand_t(const T &other, const ngen::InstructionModifier &mod)
        : ngen_operand_t(other) {
        mod_ = mod;
    }

    const ngen::Immediate &immediate() const {
        ir_assert(is_immediate());
        return *(const ngen::Immediate *)ptr_.get();
    }

    const reg_buf_data_t &reg_buf_data() const {
        ir_assert(is_reg_buf_data());
        return *(const reg_buf_data_t *)ptr_.get();
    }

    ngen::RegData reg_data() const {
        auto &rd = reg_buf_data().reg_data();
        return is_negated_ ? -rd : rd;
    }

    const ngen::FlagRegister &flag_register() const {
        ir_assert(is_flag_register());
        return *(const ngen::FlagRegister *)ptr_.get();
    }

    ngen::InstructionModifier flag_register_mod() const {
        ngen::InstructionModifier mod;
        mod |= flag_register();
        return !is_negated() ? mod : ~mod;
    }

    const ngen::InstructionModifier &mod() const { return mod_; }

    bool is_invalid() const { return kind_ == ngen_operand_kind_t::invalid; }

    bool is_immediate() const {
        return kind_ == ngen_operand_kind_t::immediate;
    }

    bool is_reg_buf_data() const {
        return kind_ == ngen_operand_kind_t::reg_buf_data;
    }

    bool is_reg_data() const { return is_reg_buf_data(); }

    bool is_flag_register() const {
        return kind_ == ngen_operand_kind_t::flag_register;
    }

    bool is_negated() const { return is_negated_; }

    ngen::DataType type() const {
        if (is_immediate()) return immediate().getType();
        if (is_reg_buf_data()) return reg_buf_data().type();
        ir_error_not_expected();
        return ngen::DataType::invalid;
    }

    ngen_operand_t operator-() const {
        if (is_immediate()) return ngen_operand_t(ngen_negate(immediate()));
        if (is_reg_buf_data() || is_flag_register()) {
            auto ret = *this;
            ret.is_negated_ = !ret.is_negated_;
            return ret;
        }
        ir_error_not_expected();
        return ngen_operand_t();
    }

    ngen_operand_t reinterpret(const type_t &new_type) const {
        ir_assert(new_type.is_scalar());
        return ngen_operand_t(
                reg_buf_data().reinterpret(to_ngen(new_type)), mod_);
    }

    // Creates an operand with the requested register region based on the
    // existing region. off - offset in elements of the region data type.
    ngen_operand_t sub_reg_data(int off, int exec_size) const {
        int off_bytes = off * ngen::getBytes(reg_buf_data().type())
                * reg_buf_data().hs();
        auto rd = reg_buf_data().format(off_bytes, ngen::DataType::invalid,
                exec_size, reg_buf_data().hs());
        return ngen_operand_t(rd, exec_size);
    }

    bool operator==(const ngen_operand_t &other) const {
        if (kind_ != other.kind_) return false;
        if (mod_.getAll() != other.mod_.getAll()) return false;
        switch (kind_) {
            case ngen_operand_kind_t::immediate: {
                auto &this_imm = immediate();
                auto &other_imm = other.immediate();
                return (this_imm.getType() == other_imm.getType())
                        && (uint64_t(this_imm) == uint64_t(other_imm));
            }
            case ngen_operand_kind_t::flag_register:
                return flag_register() == other.flag_register();
            case ngen_operand_kind_t::reg_buf_data:
                return reg_buf_data() == other.reg_buf_data();
            default: ir_error_not_expected();
        }
        return false;
    }

private:
    template <ngen_operand_kind_t kind>
    static void destroy(void *ptr) {
        if (!ptr) return;

        switch (kind) {
            case ngen_operand_kind_t::immediate:
                delete (ngen::Immediate *)ptr;
                break;
            case ngen_operand_kind_t::reg_buf_data:
                delete (reg_buf_data_t *)ptr;
                break;
            case ngen_operand_kind_t::flag_register:
                delete (ngen::FlagRegister *)ptr;
                break;
            default: ir_error_not_expected();
        }
    }

    ngen_operand_kind_t kind_;
    std::shared_ptr<void> ptr_;
    ngen::InstructionModifier mod_;

    // Whether the operand is negated. Applicable to flag registers and
    // register data operands only. Negation of immediate operands is directly
    // supported through nGEN API.
    bool is_negated_ = false;
};

template <typename T>
T to_cpp(ngen::HW hw, const ngen_operand_t &op) {
    ir_assert(op.is_immediate());
    return to_cpp<T>(op.immediate());
}

// Maintains scoped allocations which are automatically released when the scope
// is destructed.
class ngen_register_scope_t {
public:
    ngen_register_scope_t(reg_allocator_t &ra) : ra_(ra) {}

    ngen_register_scope_t(const ngen_register_scope_t &) = delete;

    ngen_register_scope_t(ngen_register_scope_t &&other)
        : ra_(other.ra_)
        , grf_ranges_(std::move(other.grf_ranges_))
        , subregisters_(std::move(other.subregisters_)) {}

    reg_allocator_t &register_allocator() { return ra_; }

    ngen::HW hw() const { return ra_.hardware(); }

    ~ngen_register_scope_t() { clear(); }

    void clear() {
        for (auto &r : grf_ranges_)
            ra_.safeRelease(r);
        for (auto &s : subregisters_)
            ra_.safeRelease(s);
        for (auto &f : flags_)
            ra_.safeRelease(f);
        grf_ranges_.clear();
        subregisters_.clear();
        flags_.clear();
    }

    ngen::GRFRange find_grf_range(int base, int byte_offset) const {
        if (byte_offset != 0) return ngen::GRFRange();
        for (auto &r : grf_ranges_)
            if (r.getBase() == base) return r;
        return ngen::GRFRange();
    }

    ngen::Subregister find_sub(int base, int byte_offset) const {
        for (auto &s : subregisters_)
            if (s.getBase() == base && s.getByteOffset() == byte_offset)
                return s;
        return ngen::Subregister();
    }

    ngen::GRFRange try_alloc_range(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        auto ret = ra_.try_alloc_range(regs, base_bundle);
        if (!ret.isInvalid()) grf_ranges_.push_back(ret);
        return ret;
    }

    ngen::GRFRange alloc_range(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        auto ret = ra_.alloc_range(regs, base_bundle);
        grf_ranges_.push_back(ret);
        return ret;
    }

    reg_buf_t alloc_reg_buf(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        auto range = ra_.alloc_range(regs, base_bundle);
        grf_ranges_.push_back(range);
        return reg_buf_t(ra_.hardware(), range);
    }

    reg_buf_data_t alloc_reg_buf_data(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        return alloc_reg_buf(regs, base_bundle);
    }

    reg_buf_data_t alloc_reg_data(const type_t &type, int stride_bytes = -1,
            ngen::Bundle bundle = ngen::Bundle()) {
        if (type.is_scalar()) {
            auto sub = alloc_sub(to_ngen(type), bundle);
            return reg_buf_data_t(hw(), sub);
        }

        int type_size = type.scalar().size();
        if (stride_bytes == -1) stride_bytes = type_size;
        int grf_size = ngen::GRF::bytes(hw());
        int regs = utils::div_up(type.elems() * stride_bytes, grf_size);
        auto buf = alloc_reg_buf(regs, bundle);
        reg_buf_data_t rbd(buf);
        return rbd.format(0, to_ngen(type.scalar()), type.elems(),
                stride_bytes / type_size);
    }

    ngen::GRF alloc(ngen::Bundle bundle = ngen::Bundle()) {
        auto range = ra_.alloc_range(1, bundle);
        grf_ranges_.push_back(range);
        return range[0];
    }

    ngen::Subregister alloc_sub(
            ngen::DataType type, ngen::Bundle bundle = ngen::Bundle()) {
        auto ret = ra_.alloc_sub(type, bundle);
        subregisters_.push_back(ret);
        return ret;
    }

    ngen::FlagRegister alloc_flag() {
        auto ret = ra_.alloc_flag();
        flags_.push_back(ret);
        return ret;
    }

    void claim(const ngen::GRFRange &range) {
        ra_.claim(range);
        grf_ranges_.push_back(range);
    }

    void claim(const ngen::Subregister &sub) {
        ra_.claim(sub);
        subregisters_.push_back(sub);
    }

    template <typename T>
    void safeRelease(T &t) {
        ra_.safeRelease(t);
    }

private:
    reg_allocator_t &ra_;

    std::vector<ngen::GRFRange> grf_ranges_;
    std::vector<ngen::Subregister> subregisters_;
    std::vector<ngen::FlagRegister> flags_;
};

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
        if (esize != expr.type().elems()) {
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

template <ngen::HW hw>
class conv_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    friend class expr_evaluator_t<hw>;
    friend class ir_to_ngen_t<hw>;
    friend class send_impl_t;

    conv_kernel_t(const conv_config_t &cfg, const convolution_pd_t *pd,
            const kernel_info_t &kernel_info, bool force_large_grf = false);

    void setup_interface(
            const stmt_t &kernel_body, const kernel_info_t &kernel_info) {
        externalName("gen_conv");
        requireLocalID(3);
        requireLocalSize();
        requireGRF(regs_);
        requireSIMD(cfg_.simd_size);
        requireBarrier();
        if (utils::one_of(cfg_.fma_kind, fma_kind_t::dpas, fma_kind_t::dpasw))
            requireDPAS();
        if (cfg_.do_atomic_update) requireGlobalAtomics();

        for (int i = 0; i < kernel_info.nargs(); i++) {
            auto &name = kernel_info.arg_name(i);
            auto &type = kernel_info.arg_type(i);
            if (type.is_ptr()) {
                newArgument(name, ngen::ExternalArgumentType::GlobalPtr);
            } else {
                newArgument(name, to_ngen(type));
            }
        }

        int slm_size
                = alloc_manager_t(kernel_body).total_size(alloc_kind_t::slm);
        requireSLM(slm_size);

        finalizeInterface();
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
            } else if (dst.type() == ngen::DataType::uw) {
                emov(mod, dst.reg_data(), src0.flag_register());
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
        if (src1.is_reg_data()) {
            eadd(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            eadd(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emul(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src0.is_immediate()) {
            ir_assert(src1.is_reg_data());
            emul(mod, dst, src1, src0);
            return;
        }
        if (src1.is_reg_data()) {
            emul(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
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
        } else {
            mad(mod, dst.reg_data(), src0.reg_data(), src1.reg_data(),
                    src2.immediate());
        }
    }

    void ediv(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (!src1.is_immediate()) {
            efdiv(mod, dst, src0, src1);
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
        ir_assert(!src1.is_immediate());
        auto one = ra_.alloc().f();
        auto zero = ra_.alloc().f();

        auto tmp = ra_.alloc_range(4);

        int esize = mod.getExecSize();
        int grf_size = ngen::GRF::bytes(hw);
        int div_esize = std::min(esize, grf_size / int(sizeof(float)));

        int tmp_regs = utils::div_up(esize * int(sizeof(float)), grf_size);
        auto src0_tmp = ra_.alloc_range(tmp_regs);
        auto src1_tmp = ra_.alloc_range(tmp_regs);

        // Copy to temporary registers to ensure dst, num and denom are
        // distinct as required for fdiv_ieee.
        mov(mod, src0_tmp[0].f(), src0.reg_data());
        mov(mod, src1_tmp[0].f(), src1.reg_data());

        auto div_mod = ngen::InstructionModifier(mod);
        div_mod.setExecSize(div_esize);

        mov(div_mod, one, ngen::Immediate(1));
        mov(div_mod, zero, ngen::Immediate(0));

        // Enable mask as fdiv_ieee relies on masked if/endif flow.
        setDefaultNoMask(false);

        for (int i = 0; i < mod.getExecSize(); i += div_esize) {
            fdiv_ieee(div_mod, f0[0], dst.sub_reg_data(i, div_esize).reg_data(),
                    src0_tmp[i / div_esize].f(), src1_tmp[i / div_esize].f(),
                    zero, one, tmp);
        }

        ra_.safeRelease(one);
        ra_.safeRelease(zero);
        ra_.safeRelease(src0_tmp);
        ra_.safeRelease(src1_tmp);
        ra_.safeRelease(tmp);

        setDefaultNoMask(true);
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

    void eand(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            and_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            and_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
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
    //     0 <= x <= UINT32_MAX
    //     0 <  y <= INT32_MAX
    // Computes:
    //     qot = x / y
    //     rem = x % y
    void eidiv(const ngen::InstructionModifier &mod, const ngen::RegData &qot,
            const ngen::RegData &rem, const ngen::RegData &x, uint32_t y) {
        ir_assert(x.getHS() == 0);
        if (ngen::utils::is_zero_or_pow2(y)) {
            auto _x = get_subregister(x);
            if (x.getNeg()) {
                // Negation modifier has bitwise semantics with shr/and so x
                // needs to be arithmetically negated first.
                _x = ra_.alloc_sub(x.getType());
                mov(1, _x, x);
            }
            if (!qot.isInvalid()) shr(mod, qot, _x, ngen::utils::log2(y));
            if (!rem.isInvalid()) and_(mod, rem, _x, y - 1);
            if (_x != x) ra_.safeRelease(_x);
            return;
        }

        uint32_t m = 0, p = 0;
        eidiv_magicgu(y, m, p);

        auto _x = ra_.alloc().ud();
        auto _qot = ra_.alloc().ud();
        mov(1, _x, x);

        // qot = (x * m) >> p
        mul(1, acc0.ud(0), _x, m & 0xFFFF);
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

        ra_.safeRelease(_x);
        ra_.safeRelease(_qot);
    }

    friend struct dnnl::impl::gpu::jit::EmulationImplementation;
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

private:
    const conv_config_t &cfg_;
    const int regs_;
    reg_allocator_t ra_;
    ngen::GRF signal_header_;

    EmulationStrategy emu_strategy = EmulationStrategy(hw);
    EmulationState emu_state;
};

template <ngen::HW hw = ngen::HW::Unknown>
class zero_out_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    zero_out_kernel_t(const conv_config_t &cfg, const convolution_pd_t *pd,
            const kernel_info_t &kernel_info)
        : ra_(hw, "zero_out_kernel_t") {
        externalName("zero_out");
        requireLocalID(1);
        requireLocalSize();
        requireGRF(cfg.regs);
        requireSIMD(cfg.simd_size);
        if (cfg.is_dpas_or_dpasw_fma()) requireDPAS();

        for (int i = 0; i < kernel_info.nargs(); i++) {
            auto &name = kernel_info.arg_name(i);
            auto &type = kernel_info.arg_type(i);
            if (type.is_ptr()) {
                newArgument(name, ngen::ExternalArgumentType::GlobalPtr);
            } else {
                newArgument(name, to_ngen(type));
            }
        }

        finalizeInterface();

        // Claim registers.
        ra_.claim(r0);
        ra_.claim(getLocalID(0));
        ra_.claim(getLocalSize(0));

        std::vector<std::string> arg_names(kernel_info.nargs());
        for (int i = 0; i < kernel_info.nargs(); i++) {
            arg_names[i] = kernel_info.arg_name(i);
            ra_.claim(getArgument(arg_names[i]));
        }

        setDefaultNoMask();
        setDefaultAutoSWSB(true);

        bool use_a64 = false;
        // XXX: Stateful messages don't work on XeHPC.
        use_a64 = (hw == ngen::HW::XeHPC);

        prologue();

        if (emu_strategy.emulate64) {
            emu_state.temp[0] = ra_.alloc();
            emu_state.temp[1] = ra_.alloc();
        }

        auto ptr = getArgument(arg_names[0]);
        auto surf = Surface(getArgumentSurface(arg_names[0]));
        auto size = getArgument(arg_names[1]);
        auto global_id = ra_.alloc_sub<uint32_t>();
        auto off0 = ra_.alloc_sub<uint32_t>();

        mul(1, global_id, r0.ud(1), getLocalSize(0).uw());
        add(1, global_id, global_id, getLocalID(0));
        shl(1, off0, global_id, math::ilog2q(bytes_per_thr / cfg.simd_size));

        int grf_size = ngen::GRF::bytes(hw);
        int bytes_per_store = 16;
        int ud_size = sizeof(uint32_t);
        int uq_size = sizeof(uint64_t);

        auto zero = ra_.alloc_range(bytes_per_store * ud_size / grf_size);
        auto off_vec = ra_.alloc_range(bytes_per_thr * ud_size / grf_size);
        auto ptr_vec = ra_.alloc_range(bytes_per_thr * uq_size / grf_size);

        for (int i = 0; i < bytes_per_store * ud_size; i += 64) {
            auto z = get_subregister(hw, ngen::DataType::ud, zero, i);
            mov(16, z, 0);
        }

        auto idx_vec = ra_.alloc().uw();
        mov(8, idx_vec, ngen::Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));

        for (int i = 0; i < bytes_per_thr; i += 8) {
            auto off_sub_vec
                    = get_subregister(hw, ngen::DataType::ud, off_vec, i)(1);
            add3(8, off_sub_vec, off0, idx_vec, i);
            if (use_a64) {
                auto ptr_sub_vec = get_subregister(
                        hw, ngen::DataType::uq, ptr_vec, i)(1);
                eadd(8, ptr_sub_vec, ptr, off_sub_vec);
            }
        }

        for (int i = 0; i < bytes_per_thr; i += bytes_per_store) {
            auto off_sub_vec
                    = get_subregister(hw, ngen::DataType::ud, off_vec, i)(1);
            cmp(16 | lt | f0[0], off_sub_vec, size);
            if (use_a64) {
                auto h_a64
                        = get_subregister(hw, ngen::DataType::uq, ptr_vec, i);
                store(16 | f0[0], ngen::scattered_byte(), A64, h_a64, zero[0]);
            } else {
                auto h_bts = off_sub_vec;
                store(16 | f0[0], ngen::scattered_byte(), surf, h_bts, zero[0]);
            }
        }

        epilogue();
    }

    friend struct dnnl::impl::gpu::jit::EmulationImplementation;

    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0) {
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, emu_strategy);
    }

    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }

    static const int bytes_per_thr;

private:
    reg_allocator_t ra_;
    EmulationStrategy emu_strategy = EmulationStrategy(hw);
    EmulationState emu_state;
};

template <ngen::HW hw>
const int zero_out_kernel_t<hw>::bytes_per_thr = 128;

template <ngen::HW hw = ngen::HW::Unknown>
class compensation_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    ngen::Subregister src_base_ptr, wei_base_ptr, dst_base_ptr;

    int simd, grf_size, grf_simd;
    int a_size, b_size, c_size;
    int num_a_regs, num_b_regs, num_c_regs;
    int num_a_headers;
    int num_regs_in_chunk, num_chunks;
    int kdhw;

    ngen::GRFRange b, c;
    ngen::GRFRange a_headers, b_headers;
    std::vector<ngen::GRF> a;
    std::vector<ngen::GRF> b_temp;
    std::vector<ngen::GRF> c_temp;

    static const int ow_block;

    compensation_kernel_t(const conv_config_t &cfg, const convolution_pd_t *pd,
            const kernel_info_t &kernel_info, bool is_edge)
        : cfg(cfg), ra_(hw, "compensation_kernel_t") {
        externalName("compensation");
        requireLocalID(1);
        requireLocalSize();
        requireGRF(cfg.regs);
        requireSIMD(cfg.simd_size);
        if (cfg.is_dpas_or_dpasw_fma()) requireDPAS();

        for (int i = 0; i < kernel_info.nargs(); i++) {
            auto &name = kernel_info.arg_name(i);
            auto &type = kernel_info.arg_type(i);
            if (type.is_ptr()) {
                newArgument(name, ngen::ExternalArgumentType::GlobalPtr);
            } else {
                newArgument(name, to_ngen(type));
            }
        }

        finalizeInterface();

        simd = cfg.simd_size;
        grf_size = ngen::GRF::bytes(hw);
        grf_simd = grf_size / sizeof(int);
        kdhw = cfg.kw * cfg.kh * cfg.kd;

        const auto &conf = cfg.zp_cfg;
        a_size = conf.ic_block * sizeof(int);
        b_size = conf.ic_block * conf.oc_block * sizeof(char);
        c_size = conf.oc_block * sizeof(int);

        // Claim registers.
        ra_.claim(r0);
        ra_.claim(getLocalID(0));
        ra_.claim(getLocalSize(0));

        std::vector<std::string> arg_names(kernel_info.nargs());
        for (int i = 0; i < kernel_info.nargs(); i++) {
            arg_names[i] = kernel_info.arg_name(i);
            ra_.claim(getArgument(arg_names[i]));
        }

        setDefaultNoMask();
        setDefaultAutoSWSB(true);

        prologue();

        if (emu_strategy.emulate64) {
            emu_state.temp[0] = ra_.alloc();
            emu_state.temp[1] = ra_.alloc();
        }

        src_base_ptr = getArgument("src");
        wei_base_ptr = getArgument("wei");
        dst_base_ptr = getArgument("dst");

        if (is_edge)
            body_edge();
        else
            body_common();

        epilogue();
    }

    void body_edge() {
        const auto &conf = cfg.zp_cfg;
        auto sp_idx = r0.ud(1);
        auto oc_block_idx = r0.ud(6);
        auto group_idx = r0.ud(7);

        init_abc_regs();

        // zero out c
        for (int i = 0; i < num_c_regs; ++i)
            mov(grf_simd, c[i].f(), float(0));

        auto reg_osp = ra_.alloc();
        auto reg_isp = ra_.alloc();
        auto reg_isp_init = ra_.alloc();
        auto reg_k = ra_.alloc();
        auto ow = reg_osp.d(0);
        auto oh = reg_osp.d(1);
        auto od = reg_osp.d(2);
        auto iw = reg_isp.d(0);
        auto ih = reg_isp.d(1);
        auto id = reg_isp.d(2);
        auto iw_init = reg_isp_init.d(0);
        auto ih_init = reg_isp_init.d(1);
        auto id_init = reg_isp_init.d(2);
        auto kw = reg_k.d(0);
        auto kh = reg_k.d(1);
        auto kd = reg_k.d(2);
        auto do_kw = reg_k.uw(8);
        auto do_kh = reg_k.uw(10);
        auto do_kd = reg_k.uw(12);

        auto ohd = ra_.alloc().d(0);
        eidiv(ohd, ow, sp_idx.d(), int(cfg.ow));
        eidiv(od, oh, ohd, int(cfg.oh));
        ra_.release(ngen::GRF(ohd.getBase()));

        mul(1, iw_init.d(), ow.d(), int(cfg.sw));
        mul(1, ih_init.d(), oh.d(), int(cfg.sh));
        mul(1, id_init.d(), od.d(), int(cfg.sd));
        add(1, iw_init.d(), iw_init.d(), int(-cfg.pw));
        add(1, ih_init.d(), ih_init.d(), int(-cfg.ph));
        add(1, id_init.d(), id_init.d(), int(-cfg.pd));

        const int iw_max = cfg.iw - cfg.kw * (1 + cfg.dw);
        const int ih_max = cfg.ih - cfg.kh * (1 + cfg.dh);
        const int id_max = cfg.id - cfg.kd * (1 + cfg.dd);

        mov(1, f1[1].uw(), int(0));
        cmp(1 | ge | f1[1], null.d(), iw_init, int(0));
        cmp(1 | f1[1] | le, null.d(), iw_init, int(iw_max));
        if (cfg.ndims > 3) {
            cmp(1 | f1[1] | ge, null.d(), ih_init, int(0));
            cmp(1 | f1[1] | le, null.d(), ih_init, int(ih_max));
        }
        if (cfg.ndims > 4) {
            cmp(1 | f1[1] | ge, null.d(), id_init, int(0));
            cmp(1 | f1[1] | le, null.d(), id_init, int(id_max));
        }

        if (cfg.kd == 1) {
            mov(1, id, id_init);
            mov(1, f0[0], int(0));
            cmp(1 | ge | f0[0], null.d(), id, int(0));
            cmp(1 | f0[0] | lt, null.d(), id, int(cfg.id));
            mov(1, do_kd, f0[0].uw());
        }
        if (cfg.kh == 1) {
            mov(1, ih, ih_init);
            mov(1, f0[1], int(0));
            cmp(1 | ge | f0[1], null.d(), ih, int(0));
            cmp(1 | f0[1] | lt, null.d(), ih, int(cfg.ih));
            mov(1, do_kh, f0[1].uw());
        }
        if (cfg.kw == 1) {
            mov(1, iw, iw_init);
            mov(1, f1[0], int(0));
            cmp(1 | ge | f1[0], null.d(), iw, int(0));
            cmp(1 | f1[0] | lt, null.d(), iw, int(cfg.iw));
            mov(1, do_kw, f1[0].uw());
        }

        const bool icb_padded = cfg.ic % conf.ic_block != 0;
        const int32_t last_icb_size = cfg.ic - (conf.icb - 1) * conf.ic_block;
        const int32_t last_icb_mask = ~(~0u << last_icb_size);

        ngen::Label icb_loop, skip_loop;
        fixup_control_flow();
        if_(16 | ~f1[1] | any16h, skip_loop, skip_loop);

        init_a_headers(group_idx);
        init_b_headers(oc_block_idx, group_idx);

        zero_out_c_temp();

        auto reg_count = ra_.alloc();
        auto icb_count = reg_count.d(0);
        auto ic_mask = reg_count.d(4);
        mov(1, icb_count, int(0));
        mov(1, ic_mask, uint32_t(~0u));

        mark(icb_loop);

        if (icb_padded) {
            mov(1, f0[1], int(0));
            cmp(1 | lt | f0[1], null.d(), icb_count, int(conf.icb - 1));
            sel(1 | f0[1] | any16h, ic_mask, ic_mask, int(last_icb_mask));
            mov(1, f1.d(), ic_mask);
            load_a_masked();
        } else {
            load_a_no_mask();
        }

        ngen::Label kw_loop, kh_loop, kd_loop;

        if (cfg.kd != 1) {
            mov(1, id, id_init);
            mov(1, kd, int(0));
            mark(kd_loop);
            mov(1, f0[0], int(0));
            cmp(1 | ge | f0[0], null.d(), id, int(0));
            cmp(1 | f0[0] | lt, null.d(), id, int(cfg.id));
            mov(1, do_kd, f0[0]);
            add(1, id, id, int(1 + cfg.dd));
        }
        if (cfg.kh != 1) {
            mov(1, ih, ih_init);
            mov(1, kh, int(0));
            mark(kh_loop);
            mov(1, f0[1], int(0));
            cmp(1 | ge | f0[1], null.d(), ih, int(0));
            cmp(1 | f0[1] | lt, null.d(), ih, int(cfg.ih));
            mov(1, do_kh, f0[1]);
            add(1, ih, ih, int(1 + cfg.dh));
        }
        if (cfg.kw != 1) {
            mov(1, iw, iw_init);
            mov(1, kw, int(0));
            mark(kw_loop);
            mov(1, f1[0], int(0));
            cmp(1 | ge | f1[0], null.d(), iw, int(0));
            cmp(1 | f1[0] | lt, null.d(), iw, int(cfg.iw));
            mov(1, do_kw, f1[0]);
            add(1, iw, iw, int(1 + cfg.dw));
        }

        ngen::Label do_mul, skip_mul;

        auto wei_ptr = b_headers[0].uq(0);
        const int wei_block = conf.ic_block * conf.oc_block * sizeof(char);

        and_(1, f1[1], do_kd, do_kh);
        and_(1, f1[1], f1[1], do_kw);

        fixup_control_flow();
        if_(16 | f1[1] | any16h, do_mul, skip_mul);

        eadd(1, wei_ptr, wei_ptr, int(wei_block));

        else_(16, skip_mul, skip_mul);
        mark(do_mul);

        load_b();
        multiply();

        mark(skip_mul);
        endif(16);

        if (cfg.kw != 1) {
            add(1, kw, kw, int(1));
            mov(1, f0[1], int(0));
            cmp(1 | lt | f0[1], null.d(), kw, int(cfg.kw));
            fixup_control_flow();
            while_(16 | f0[1] | any16h, kw_loop);
        }
        if (cfg.kh != 1) {
            add(1, kh, kh, int(1));
            mov(1, f1[0], int(0));
            cmp(1 | lt | f1[0], null.d(), kh, int(cfg.kh));
            fixup_control_flow();
            while_(16 | f1[0] | any16h, kh_loop);
        }
        if (cfg.kd != 1) {
            add(1, kd, kd, int(1));
            mov(1, f1[1], int(0));
            cmp(1 | lt | f1[1], null.d(), kd, int(cfg.kd));
            fixup_control_flow();
            while_(16 | f1[1] | any16h, kd_loop);
        }

        add(1, icb_count, icb_count, int(1));
        mov(1, f1[1], int(0));
        cmp(1 | lt | f1[1], null.d(), icb_count, int(conf.icb));
        fixup_control_flow();
        while_(16 | f1[1] | any16h, icb_loop);

        ra_.release(reg_count);

        finalize();

        mark(skip_loop);
        endif(16);

        // store
        auto dst_header = ra_.alloc();
        auto dst_ptr = dst_header.uq(0);
        mov(grf_simd, dst_header.f(), float(0.f));

        const int dst_ow_stride = conf.oc_block * sizeof(int);
        const int dst_oh_stride = cfg.ow * dst_ow_stride;
        const int dst_od_stride = cfg.oh * dst_oh_stride;
        const int dst_ocb_stride = cfg.od * dst_od_stride;
        const int dst_g_stride
                = utils::div_up(cfg.oc, conf.oc_block) * dst_ocb_stride;

        auto dst_ow_offset = ra_.alloc().uq(0);
        auto dst_oh_offset = ra_.alloc().uq(0);
        auto dst_od_offset = ra_.alloc().uq(0);
        auto dst_ocb_offset = ra_.alloc().uq(0);
        auto dst_g_offset = ra_.alloc().uq(0);

        emul(1, dst_ow_offset, ow, int(dst_ow_stride));
        emul(1, dst_oh_offset, oh, int(dst_oh_stride));
        emul(1, dst_od_offset, od, int(dst_od_stride));
        emul(1, dst_ocb_offset, oc_block_idx, int(dst_ocb_stride));
        emul(1, dst_g_offset, group_idx, int(dst_g_stride));
        eadd(1, dst_ptr, dst_base_ptr, dst_ow_offset);
        eadd(1, dst_ptr, dst_ptr, dst_oh_offset);
        eadd(1, dst_ptr, dst_ptr, dst_od_offset);
        eadd(1, dst_ptr, dst_ptr, dst_ocb_offset);
        eadd(1, dst_ptr, dst_ptr, dst_g_offset);

        sync(ngen::SyncFunction::allwr, 0xffff);
        if (conf.oc_block == 32) {
            store(16, ngen::block_oword(8), A64, dst_header, c[0]);
        } else if (conf.oc_block == 8) {
            store(16, ngen::block_oword(2), A64, dst_header, c[0]);
        } else {
            ir_error_not_expected();
        }

        ra_.release(ngen::GRF(dst_ow_offset.getBase()));
        ra_.release(ngen::GRF(dst_oh_offset.getBase()));
        ra_.release(ngen::GRF(dst_od_offset.getBase()));
        ra_.release(ngen::GRF(dst_ocb_offset.getBase()));
        ra_.release(ngen::GRF(dst_g_offset.getBase()));
        ra_.release(reg_osp);
        ra_.release(reg_isp);
        ra_.release(reg_isp_init);
        ra_.release(reg_k);
    }

    void body_common() {
        const auto &conf = cfg.zp_cfg;
        auto oc_block_idx = r0.ud(1);
        auto group_idx = r0.ud(6);

        init_a_headers(group_idx);
        init_b_headers(oc_block_idx, group_idx);

        init_abc_regs();

        // zero out c
        for (int i = 0; i < num_c_regs; ++i)
            mov(grf_simd, c[i].f(), float(0));

        loop_common();

        finalize();

        // store
        auto dst_header = ra_.alloc();
        auto dst_ptr = dst_header.uq(0);
        mov(grf_simd, dst_header.f(), float(0.f));

        const int dst_ocb_stride = conf.oc_block * sizeof(int);
        const int dst_g_stride
                = utils::div_up(cfg.oc, conf.oc_block) * dst_ocb_stride;

        auto dst_ocb_offset = ra_.alloc().uq(0);
        auto dst_g_offset = ra_.alloc().uq(0);

        emul(1, dst_ocb_offset, oc_block_idx, int(dst_ocb_stride));
        emul(1, dst_g_offset, group_idx, int(dst_g_stride));
        eadd(1, dst_ptr, dst_base_ptr, dst_ocb_offset);
        eadd(1, dst_ptr, dst_ptr, dst_g_offset);

        sync(ngen::SyncFunction::allwr, 0xffff);
        if (conf.oc_block == 32) {
            store(16, ngen::block_oword(8), A64, dst_header, c[0]);
        } else if (conf.oc_block == 8) {
            store(16, ngen::block_oword(2), A64, dst_header, c[0]);
        } else {
            ir_error_not_expected();
        }

        ra_.release(ngen::GRF(dst_ocb_offset.getBase()));
        ra_.release(ngen::GRF(dst_g_offset.getBase()));
    }

    void loop_common() {
        const auto &conf = cfg.zp_cfg;
        const bool unroll_ic = (kdhw == 1) && cfg.ic <= 512;
        const bool unroll_k = kdhw <= 49;

        const bool icb_padded = cfg.ic % conf.ic_block != 0;
        const int32_t last_icb_size = cfg.ic - (conf.icb - 1) * conf.ic_block;
        const int32_t last_icb_mask = ~(~0u << last_icb_size);

        zero_out_c_temp();

        if (unroll_ic) {
            for (int i = 0; i < conf.icb; ++i) {
                const bool is_last_block = (i == conf.icb - 1);
                if (is_last_block && icb_padded) {
                    mov(1, f1.ud(), uint32_t(last_icb_mask));
                    load_a_masked();
                } else {
                    load_a_no_mask();
                }
                load_b();
                multiply();
            }
        } else {
            auto reg_count = ra_.alloc();
            auto icb_count = reg_count.d(0);
            auto ic_mask = reg_count.d(4);
            mov(1, icb_count, int(0));
            mov(1, ic_mask, uint32_t(~0u));

            ngen::Label icb_loop;
            mark(icb_loop);

            if (icb_padded) {
                mov(1, f0[1], int(0));
                cmp(1 | lt | f0[1], null.d(), icb_count, int(conf.icb - 1));
                sel(1 | f0[1] | any16h, ic_mask, ic_mask, int(last_icb_mask));
                mov(1, f1.d(), ic_mask);
                load_a_masked();
            } else {
                load_a_no_mask();
            }

            if (unroll_k) {
                for (int i = 0; i < kdhw; ++i) {
                    load_b();
                    multiply();
                }
            } else {
                auto k_count = ra_.alloc().d(0);
                mov(1, k_count, int(0));

                ngen::Label k_loop;
                mark(k_loop);

                load_b();
                multiply();

                mov(1, f1[0], int(0));
                add(1, k_count, k_count, int(1));
                cmp(1 | lt | f1[0], null.d(), k_count, int(kdhw));
                fixup_control_flow();
                while_(16 | f1[0] | any16h, k_loop);

                ra_.release(ngen::GRF(k_count.getBase()));
            }

            mov(1, f1[1], int(0));
            add(1, icb_count, icb_count, int(1));
            cmp(1 | lt | f1[1], null.d(), icb_count, int(conf.icb));
            fixup_control_flow();
            while_(16 | f1[1] | any16h, icb_loop);

            ra_.release(reg_count);
        }
    }

    void init_a_headers(const ngen::Subregister &group_idx) {
        const auto &conf = cfg.zp_cfg;
        if (conf.is_common_src_zero_point || !conf.is_runtime_src_zero_points)
            return;

        const bool use_a64 = (hw >= ngen::HW::XeHPC);
        const int addr_size = use_a64 ? sizeof(int64_t) : sizeof(int32_t);
        num_a_headers = utils::div_up(conf.ic_block * addr_size, grf_size);

        // prepare src headers
        a_headers = ra_.alloc_range(num_a_headers);
        auto idx_vec = ra_.alloc().uw();
        auto g_offset = ra_.alloc().ud();
        emul(1, g_offset.ud(0), group_idx, int(cfg.ic));
        mov(8, idx_vec.uw(0), ngen::Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        mov(8, idx_vec.uw(8),
                ngen::Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
        if (use_a64) shl(16, idx_vec, idx_vec, int(2)); // sizeof(int32_t)

        for (int off = 0, exec_size = grf_size / addr_size; off < conf.ic_block;
                off += exec_size) {
            const int r = off / exec_size;
            if (use_a64) {
                if (off == 0) {
                    eadd(exec_size, a_headers[r].uq(), src_base_ptr,
                            g_offset.ud(0));
                    eadd(exec_size, a_headers[r].uq(), a_headers[r].uq(),
                            idx_vec);
                } else
                    eadd(exec_size, a_headers[r].uq(), a_headers[0].uq(),
                            int(off));
            } else {
                if (off == 0)
                    add(exec_size, a_headers[r].ud(), g_offset.ud(0), idx_vec);
                else
                    add(exec_size, a_headers[r].ud(), a_headers[0].ud(),
                            int(off));
            }
        }

        ra_.release(idx_vec);
        ra_.release(g_offset);
    }

    void init_b_headers(const ngen::Subregister &oc_block_idx,
            const ngen::Subregister &group_idx) {
        const auto &conf = cfg.zp_cfg;
        const int wei_block = conf.ic_block * conf.oc_block;
        const int wei_g_stride
                = conf.ocb * conf.icb * kdhw * wei_block * sizeof(char);
        const int wei_ocb_stride = conf.icb * kdhw * wei_block * sizeof(char);

        b_headers = ra_.alloc_range(1);
        auto wei_ptr = b_headers[0].uq(0);
        auto ocb_offset = ra_.alloc().ud();
        auto g_offset = ra_.alloc().ud();
        emul(1, ocb_offset.ud(0), oc_block_idx, int(wei_ocb_stride));
        emul(1, g_offset.ud(0), group_idx, int(wei_g_stride));
        eadd(1, wei_ptr, wei_base_ptr, ocb_offset.ud(0));
        eadd(1, wei_ptr, wei_ptr, g_offset.ud(0));
        ra_.release(ocb_offset);
        ra_.release(g_offset);
    }

    void init_abc_regs() {
        num_a_regs = utils::div_up(a_size, grf_size);
        num_b_regs = utils::div_up(b_size, grf_size);
        num_c_regs = utils::div_up(c_size, grf_size);

        c = ra_.alloc_range(num_c_regs, ngen::Bundle(0, ngen::Bundle::any));
        b = ra_.alloc_range(num_b_regs, ngen::Bundle(1, ngen::Bundle::any));

        const auto &conf = cfg.zp_cfg;
        ir_assert(simd == conf.oc_inner);
        ir_assert(conf.oc_inner % grf_simd == 0);
        num_regs_in_chunk = conf.oc_inner / grf_simd;
        num_chunks = conf.oc_outer * conf.ic_inner * num_regs_in_chunk;

        b_temp.resize(num_chunks);
        c_temp.resize(num_chunks);
        for (int r = 0; r < num_chunks; ++r) {
            const int bundle_id_c = (r + 7) % ngen::Bundle::bundle_count(hw);
            const int bundle_id_b = (r + 8) % ngen::Bundle::bundle_count(hw);
            const auto bundle_c = ngen::Bundle(0, bundle_id_c);
            const auto bundle_b = ngen::Bundle(1, bundle_id_b);
            c_temp[r] = ra_.alloc_range(num_regs_in_chunk, bundle_c)[0];
            b_temp[r] = ra_.alloc_range(num_regs_in_chunk, bundle_b)[0];
        }

        a.resize(num_a_regs);
        for (auto &r : a)
            r = ra_.alloc(ngen::Bundle(0, ngen::Bundle::any));
    }

    void zero_out_c_temp() {
        for (auto &reg : c_temp)
            mov(simd, reg.f(), float(0));
    }

    void load_a_no_mask() {
        const auto &conf = cfg.zp_cfg;
        if (conf.is_common_src_zero_point) return;

        auto a_surf = Surface(getArgumentSurface("src"));
        const bool use_a64 = (hw >= ngen::HW::XeHPC);
        const int elem_size = use_a64 ? sizeof(int32_t) : 1;
        ngen::AddressBase address_base = use_a64 ? A64 : a_surf;

        if (conf.ic_block == 32) {
            const int addr_size = use_a64 ? sizeof(int64_t) : sizeof(int32_t);
            const int data_size = sizeof(int32_t);
            for (int off = 0; off < conf.ic_block; off += grf_simd) {
                const auto rd = off * data_size / grf_size;
                const auto rh = off * addr_size / grf_size;
                load(grf_simd, a[rd], ngen::scattered_dword(), address_base,
                        a_headers[rh]);
                if (use_a64)
                    eadd(grf_simd, a_headers[rh].uq(), a_headers[rh].uq(),
                            int(32 * elem_size));
                else
                    add(grf_simd, a_headers[rh].ud(), a_headers[rh].ud(),
                            int(32 * elem_size));
            }
        } else if (conf.ic_block == 4) {
            load(4, a[0], ngen::scattered_dword(), address_base, a_headers[0]);
            if (use_a64)
                eadd(grf_simd, a_headers[0].uq(), a_headers[0].uq(),
                        int(4 * elem_size));
            else
                add(grf_simd, a_headers[0].ud(), a_headers[0].ud(),
                        int(4 * elem_size));
        } else {
            ir_error_not_expected();
        }
    }

    void load_a_masked() {
        const auto &conf = cfg.zp_cfg;
        if (conf.is_common_src_zero_point) return;

        auto a_surf = Surface(getArgumentSurface("src"));
        const bool use_a64 = (hw >= ngen::HW::XeHPC);
        const int elem_size = use_a64 ? sizeof(int32_t) : 1;
        ngen::AddressBase address_base = use_a64 ? A64 : a_surf;

        const int addr_size = use_a64 ? sizeof(int64_t) : sizeof(int32_t);
        const int data_size = sizeof(int32_t);

        const uint32_t sbid_mask = ~(~0u << (conf.ic_block / grf_simd));
        sync(ngen::SyncFunction::allwr, sbid_mask);

        if (conf.ic_block == 32) {
            for (int off = 0, sbid = 0; off < conf.ic_block;
                    off += grf_simd, sbid++) {
                const auto rd = off * data_size / grf_size;
                const auto rh = off * addr_size / grf_size;
                setDefaultAutoSWSB(false);
                fixup_control_flow();
                load(grf_simd | f1[0] | ngen::SBID(sbid), a[rd],
                        ngen::scattered_dword(), address_base, a_headers[rh]);
                setDefaultAutoSWSB(true);
                if (use_a64)
                    eadd(grf_simd, a_headers[rh].uq(), a_headers[rh].uq(),
                            int(32 * elem_size));
                else
                    add(grf_simd, a_headers[rh].ud(), a_headers[rh].ud(),
                            int(32 * elem_size));
                shr(1, f1[0].ud(), f1[0].ud(), 8);
            }
        } else if (conf.ic_block == 4) {
            setDefaultAutoSWSB(false);
            fixup_control_flow();
            load(4 | f1[0] | ngen::SBID(0), a[0], ngen::scattered_dword(),
                    address_base, a_headers[0]);
            setDefaultAutoSWSB(true);
            if (use_a64)
                eadd(grf_simd, a_headers[0].uq(), a_headers[0].uq(),
                        int(4 * elem_size));
            else
                add(grf_simd, a_headers[0].ud(), a_headers[0].ud(),
                        int(4 * elem_size));
        } else {
            ir_error_not_expected();
        }
    }

    void load_b() {
        auto b_ptr = b_headers[0].uq();
        if (b_size % 128 == 0) {
            for (int byte_off = 0; byte_off < b_size; byte_off += 128) {
                load(16, b[byte_off / grf_size], ngen::block_oword(8), A64,
                        b_headers[0]);
                eadd(1, b_ptr, b_ptr, int(128));
            }
        } else if (b_size == 32) {
            load(16, b[0], ngen::block_oword(2), A64, b_headers[0]);
            eadd(1, b_ptr, b_ptr, int(32));
        } else {
            ir_error_not_expected();
        }
    }

    void reduce_block() {
        const auto &conf = cfg.zp_cfg;
        auto tmp = ra_.alloc_range(num_regs_in_chunk);
        mov(simd, tmp[0].d(), int(0x01010101));
        for (int ic_outer = 0; ic_outer < conf.ic_outer; ++ic_outer) {
            for (int oc_outer = 0; oc_outer < conf.oc_outer; ++oc_outer) {
                const int sx = num_regs_in_chunk
                        * (ic_outer + oc_outer * conf.ic_outer);
                const int dx = num_regs_in_chunk * oc_outer;
                dp4a(simd, c[dx].d(), c[dx].d(), tmp[0].d(), b[sx].d());
            }
        }
        ra_.release(tmp);
    }

    void multiply_block() {
        const auto &conf = cfg.zp_cfg;
        ir_assert(conf.ic_inner == 4);
        for (int ic_outer = 0; ic_outer < conf.ic_outer; ++ic_outer) {
            for (int oc_outer = 0; oc_outer < conf.oc_outer; ++oc_outer) {
                for (int ic_inner = 0; ic_inner < conf.ic_inner; ++ic_inner) {
                    const int sx = ic_inner;
                    const int sy = num_regs_in_chunk
                            * (ic_outer + oc_outer * conf.ic_outer);
                    const int dx = ic_inner + conf.ic_inner * oc_outer;
                    const int stride = conf.ic_inner;
                    mov(simd, b_temp[dx].w(0)(2), b[sy].b(sx)(stride));
                }
            }
            for (int oc_outer = 0; oc_outer < conf.oc_outer; ++oc_outer) {
                for (int ic_inner = 0; ic_inner < conf.ic_inner; ++ic_inner) {
                    const int ic = ic_inner + ic_outer * conf.ic_inner;
                    const int sx = ic % grf_simd;
                    const int sy = ic / grf_simd;
                    const int dx = ic_inner + conf.ic_inner * oc_outer;
                    mad(simd, c_temp[dx].d(), c_temp[dx].d(), a[sy].d(sx),
                            b_temp[dx].w(0)(2));
                }
            }
        }
    }

    void multiply() {
        const auto &conf = cfg.zp_cfg;
        if (conf.is_common_src_zero_point)
            reduce_block();
        else
            multiply_block();
    }

    void finalize() {
        const auto &conf = cfg.zp_cfg;
        if (conf.is_common_src_zero_point) {
            // multiply after reduce
            auto zp_common = a[0].d();
            if (conf.is_runtime_src_zero_points) {
                auto header = ra_.alloc();
                mov(grf_simd, header.f(), float(0.f));
                emov(1, header.uq(), src_base_ptr);
                load(1, zp_common, ngen::scattered_dword(), A64, header);
                ra_.release(header);
            } else {
                mov(1, zp_common.d(), int(conf.common_src_zero_point));
            }
            for (int r = 0; r < num_c_regs; ++r) {
                mul(grf_simd, acc0.d(), c[r].d(), zp_common.uw(0));
                macl(grf_simd, c[r].d(), c[r].d(), zp_common.ud(0));
            }
        } else {
            // sum temp accs
            ir_assert(conf.ic_inner == 4);
            std::vector<ngen::GRFRange> tmp(conf.oc_outer);
            for (auto &r : tmp)
                r = ra_.alloc_range(num_regs_in_chunk);
            for (int oc_outer = 0; oc_outer < conf.oc_outer; ++oc_outer) {
                const int sx = oc_outer * conf.ic_inner;
                const int dx = num_regs_in_chunk * oc_outer;
                add(simd, c[dx].d(), c_temp[sx + 0].d(), c_temp[sx + 1].d());
                add(simd, tmp[oc_outer][0].d(), c_temp[sx + 2].d(),
                        c_temp[sx + 3].d());
                add(simd, c[dx].d(), c[dx].d(), tmp[oc_outer][0].d());
            }
            for (auto &r : tmp)
                ra_.release(r);
        }
    }

    void fixup_control_flow() {
        if (hw >= ngen::HW::XeHPC) return;

        csel(4 | eq | f0[0] | ngen::SWSB<float>(1), r0.w(0)(1), r0.w(0)(1),
                r0.w(0)(1), r0.w(0)(1));
        csel(4 | eq | f0[0] | ngen::SWSB<int>(1), r0.f(0)(1), r0.f(0)(1),
                r0.f(0)(1), r0.f(0)(1));
    };

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
    //     0 <= x <= UINT32_MAX
    //     0 <  y <= INT32_MAX
    // Computes:
    //     qot = x / y
    //     rem = x % y
    void eidiv(const ngen::RegData &qot, const ngen::RegData &rem,
            const ngen::RegData &x, uint32_t y) {
        if (ngen::utils::is_zero_or_pow2(y)) {
            if (!qot.isInvalid()) shr(1, qot, x, ngen::utils::log2(y));
            if (!rem.isInvalid()) and_(1, rem, x, y - 1);
            return;
        }

        uint32_t m = 0, p = 0;
        eidiv_magicgu(y, m, p);

        auto _x = ra_.alloc().ud();
        auto _qot = ra_.alloc().ud();
        mov(1, _x, x);

        // qot = (x * m) >> p
        mul(1, acc0.ud(0), _x, m & 0xFFFF);
        mach(1, _qot, _x, m);
        shr<uint32_t>(1, _qot, _qot, p - 32);
        if (!qot.isInvalid()) mov(1, qot, _qot);

        if (!rem.isInvalid()) {
            // rem = x - qot * y
            bool y_is_16_bit = (y <= static_cast<uint32_t>(
                                        std::numeric_limits<int16_t>::max()));
            if (hw >= ngen::HW::XeLP && y_is_16_bit) {
                mad(1, rem, x, _qot, -int16_t(y));
            } else {
                auto tmp = ra_.alloc_sub<uint64_t>();
                mul(1, tmp, _qot, y);
                add(1, rem, x, -tmp.ud(0));
                ra_.safeRelease(tmp);
            }
        }

        ra_.safeRelease(_x);
        ra_.safeRelease(_qot);
    }

    friend struct dnnl::impl::gpu::jit::EmulationImplementation;

    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0) {
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
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1) {
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }

private:
    conv_config_t cfg;

    reg_allocator_t ra_;
    EmulationStrategy emu_strategy = EmulationStrategy(hw);
    EmulationState emu_state;
};

template <ngen::HW hw>
const int compensation_kernel_t<hw>::ow_block = 8;

// Evaluates expression by emitting instructions with nGEN.
template <ngen::HW hw>
class expr_evaluator_t : public ir_visitor_t {
public:
    expr_evaluator_t(conv_kernel_t<hw> *host,
            const expr_binding_t &expr_binding, ngen_register_scope_t &scope)
        : host_(host), expr_binding_(expr_binding), scope_(scope) {}

    bool is_int_up_convert(const expr_t &e, type_t &type) const {
        auto it = int_up_converts_.find(e);
        if (it == int_up_converts_.end()) return false;
        type = it->second;
        return true;
    }

    // If `dst_operand` is not empty, use its pre-allocated location for the
    // result.
    ngen_operand_t eval(const expr_t &e,
            const ngen_operand_t &dst_operand = ngen_operand_t()) {
        if (!dst_operand.is_invalid()) {
            ir_assert(dst_operand.mod().getExecSize() != 0);
        }
        if (expr_binding_.is_bound(e)) {
            if (!dst_operand.is_invalid()) {
                host_->emov(
                        dst_operand.mod(), dst_operand, expr_binding_.get(e));
                return dst_operand;
            }
        } else {
            if (!dst_operand.is_invalid())
                expr_binding_.bind_dst(e, dst_operand);
            visit(e);
        }

        return expr_binding_.get(e, /*allow_empty=*/true);
    }

    std::vector<ngen_operand_t> eval(const std::vector<expr_t> &exprs) {
        std::vector<ngen_operand_t> ret;
        for (auto &e : exprs) {
            if (!expr_binding_.is_bound(e)) visit(e);
            ret.push_back(expr_binding_.get(e));
        }
        return ret;
    }

    void _visit(const binary_op_t &obj) override {
        auto dst_op = alloc_dst_op(obj);
        auto mod = dst_op.mod();

        switch (obj.op_kind) {
            case op_kind_t::_and: {
                if (obj.type.is_bool()) {
                    eval(obj.a, dst_op);
                    eval(obj.b,
                            ngen_operand_t(
                                    dst_op, mod | dst_op.flag_register_mod()));
                    break;
                }
                // else fall through to the default label.
            }
            default: {
                // Some cases require pre-allocated register regions with
                // special strides for a/b.
                auto a_out_op = maybe_alloc_strided_op(obj.type, obj.a);
                auto b_out_op = maybe_alloc_strided_op(obj.type, obj.b);
                auto src0_op = eval(obj.a, a_out_op);
                auto src1_op = eval(obj.b, b_out_op);

                // XXX: (q x d) case is not supported. Try to downgrade it to
                // (d x d) based on the previous assignments.
                if (obj.op_kind == op_kind_t::_mul && obj.a.type().is_x64()) {
                    type_t orig_type;
                    if (is_int_up_convert(obj.a, orig_type)) {
                        src0_op = src0_op.reinterpret(orig_type);
                        // XXX: sync workaround is to fix an issue with
                        // mul(q, d, d) instruction on XeHP. For some reason
                        // the result is incorrect when dst and src0 are
                        // accessed from the same register.
                        host_->sync(ngen::SyncFunction::nop,
                                ngen::SWSB<uint64_t>(1));
                    } else {
                        ir_error_not_expected();
                    }
                }
                ebinary(obj, mod, dst_op, src0_op, src1_op);
                break;
            }
        }

        bind(obj, dst_op);
    }

    void _visit(const bool_imm_t &obj) override {
        // Scalar booleans must never be directly lowered:
        // - Booleans are mapped to flag registers
        // - Flag register stores vector of boolean vectors
        // - All boolean values in IR must be expressed by shuffle_t objects
        // - _visit(shuffle_t *) must properly handle vector of booleans -> flag
        //   register lowering
        ir_error_not_expected();
    }

    void _visit(const cast_t &obj) override {
        auto &from_type = obj.expr.type();
        auto &to_type = obj.type;

        ir_assert(from_type != to_type) << "Equal types are not expected.";

        if (is_const(obj.expr) && !to_type.is_bool()) {
            bind(obj, to_ngen(obj.expr, to_type));
            return;
        }

        auto dst_op = alloc_dst_op(obj);

        // Handle ptr -> u64 and u64 -> ptr casts.
        if (utils::one_of(obj.type, type_t::u64(), type_t::byte_ptr())
                && utils::one_of(
                        obj.expr.type(), type_t::u64(), type_t::byte_ptr())) {
            eval(obj.expr, dst_op);
            bind(obj, dst_op);
            return;
        }

        // Handle integer (down-)conversion, assume bitwise equality in this
        // case. Examples: d <-> ud, d -> w, q -> d.
        bool is_int_convert = from_type.is_scalar() && to_type.is_scalar()
                && from_type.is_int() && to_type.is_int();
        bool is_int_down_convert
                = is_int_convert && from_type.size() >= to_type.size();
        bool is_int_up_convert
                = is_int_convert && from_type.size() < to_type.size();
        if (is_int_down_convert) {
            eval(obj.expr, dst_op.reinterpret(from_type));
            bind(obj, dst_op);
            return;
        }

        auto expr_op = eval(obj.expr);
        auto mod = dst_op.mod();
        if (obj.saturate) mod |= host_->sat;
        host_->emov(mod, dst_op, expr_op);
        if (is_int_up_convert) int_up_converts_.emplace(obj, from_type);
        bind(obj, dst_op);
    }

    void _visit(const float_imm_t &obj) override { bind(obj, to_ngen(obj)); }

    void _visit(const int_imm_t &obj) override { bind(obj, to_ngen(obj)); }

    void _visit(const load_t &obj) override {
        auto &type = obj.type;
        auto buf_op = eval(obj.buf);
        auto off_op = eval(obj.off);
        int stride;
        if (obj.has_default_stride()) {
            stride = 1;
        } else {
            ir_assert(obj.stride % type.scalar().size() == 0);
            stride = obj.stride / type.scalar().size();
        }
        auto load_rbd
                = buf_op.reg_buf_data().format(to_cpp<int>(off_op.immediate()),
                        to_ngen(type.scalar()), type.elems(), stride);
        bind(obj, load_rbd);
    }

    void _visit(const ptr_t &obj) override {
        auto base_op = eval(obj.base);

        if (is_zero(obj.off)) {
            bind(obj, base_op);
            return;
        }

        ir_assert(base_op.is_reg_buf_data());

        int off = to_cpp<int>(obj.off);
        bind(obj, base_op.reg_buf_data().format(off, ngen::DataType::ub));
    }

    void _visit(const shuffle_t &obj) override {
        int elems = obj.elems();
        if (obj.type.is_bool() && is_shuffle_const(obj)) {
            auto dst_op = alloc_dst_op(obj);
            auto e_shuffle = expr_t(obj);
            ir_assert(dst_op.is_flag_register()) << e_shuffle;
            ir_assert(!dst_op.is_negated()) << e_shuffle;
            uint16_t flag_mask = 0;
            for (int i = elems - 1; i >= 0; i--) {
                flag_mask <<= 1;
                flag_mask |= (to_cpp<bool>(e_shuffle[i]) ? 1 : 0);
            }
            if (dst_op.mod().getPredCtrl() == ngen::PredCtrl::None) {
                host_->emov(1, dst_op, ngen::Immediate(flag_mask));
            } else {
                ir_assert(dst_op.mod().getFlagReg() == dst_op.flag_register());
                host_->and_(1, dst_op.flag_register(), dst_op.flag_register(),
                        ngen::Immediate(flag_mask));
            }
            bind(obj, dst_op);
            return;
        }

        if (obj.is_broadcast()) {
            if (obj.type.is_bool()) {
                auto dst_op = alloc_dst_op(obj);
                eval(obj.vec[0], dst_op);
                bind(obj, dst_op);
            } else {
                auto scalar_op = eval(obj.vec[0]);
                bind(obj, scalar_op);
            }
            return;
        }

        // tuples: <offset, length, idx>
        std::vector<std::tuple<int, int, int>> chunks;
        for (int i = 0; i < elems; i++) {
            int idx = obj.idx[i];
            if (chunks.empty() || std::get<2>(chunks.back()) != idx) {
                chunks.emplace_back(i, 1, idx);
            } else {
                std::get<1>(chunks.back())++;
            }
        }

        auto dst_op = alloc_dst_op(obj);
        for (auto &chunk : chunks) {
            int off = std::get<0>(chunk);
            int length = std::get<1>(chunk);
            int idx = std::get<2>(chunk);
            // Split length into powers of two.
            while (length > 0) {
                int exec_size = (1 << math::ilog2q(length));
                auto chunk_op = dst_op.sub_reg_data(off, exec_size);
                eval(obj.vec[idx], ngen_operand_t(chunk_op, exec_size));
                length -= exec_size;
                off += exec_size;
            }
        }
        bind(obj, dst_op);
    }

    void _visit(const ternary_op_t &obj) override {
        switch (obj.op_kind) {
            case op_kind_t::_add3:
            case op_kind_t::_mad: {
                auto dst_op = alloc_dst_op(obj);
                auto mod = dst_op.mod();
                auto src0_op = eval(obj.a);
                auto src1_op = eval(obj.b);
                auto src2_op = eval(obj.c);
                if (obj.op_kind == op_kind_t::_add3) {
                    host_->eadd3(mod, dst_op, src0_op, src1_op, src2_op);
                } else {
                    host_->emad(mod, dst_op, src0_op, src1_op, src2_op);
                }
                bind(obj, dst_op);
                break;
            }
            default: ir_error_not_expected();
        }
    }

    void _visit(const unary_op_t &obj) override {
        ir_assert(obj.op_kind == op_kind_t::_minus);
        auto a_op = eval(obj.a);
        bind(obj, -a_op);
    }

    void _visit(const var_t &obj) override {
        ir_assert(expr_binding_.is_bound(obj))
                << "Variable is not defined: " << expr_t(obj);
    }

private:
    ngen_operand_t alloc_dst_op(const expr_t &e) {
        ir_assert(!expr_binding_.is_bound(e)) << "Already evaluated: " << e;
        if (expr_binding_.is_dst_bound(e)) return expr_binding_.get_dst(e);

        // Expression is not bound yet, allocate new storage and bind.
        ngen_operand_t op;
        if (e.type().is_bool()) {
            op = ngen_operand_t(scope_.alloc_flag(), e.type().elems());
        } else {
            op = ngen_operand_t(
                    scope_.alloc_reg_data(e.type()), e.type().elems());
        }
        expr_binding_.bind_dst(e, op);
        return op;
    }

    ngen_operand_t alloc_tmp(const expr_t &e) {
        return ngen_operand_t(
                scope_.alloc_reg_data(e.type()), e.type().elems());
    }

    // Pre-allocates a strided register region for expression `e` if needed.
    ngen_operand_t maybe_alloc_strided_op(
            const type_t &res_type, const expr_t &e) {
        // Need q-strided region for `e` if res_type is q/uq and `e` is of a
        // sub-q data type and not a scalar.
        if (e.type().is_scalar()) return ngen_operand_t();
        if (!utils::one_of(res_type.scalar(), type_t::s64(), type_t::u64()))
            return ngen_operand_t();
        if (utils::one_of(e.type().scalar(), type_t::s64(), type_t::u64()))
            return ngen_operand_t();

        auto *shuffle = e.as_ptr<shuffle_t>();
        if (shuffle && shuffle->is_broadcast()) return ngen_operand_t();

        return ngen_operand_t(
                scope_.alloc_reg_data(e.type(), res_type.scalar().size()),
                e.type().elems());
    }

    void bind(const expr_t &e, const ngen_operand_t &op) {
        if (!expr_binding_.is_dst_bound(e)) {
            expr_binding_.bind(e, op);
            return;
        }
        auto dst_op = expr_binding_.get_dst(e);
        if (dst_op == op) {
            expr_binding_.bind(e, op);
            return;
        }
        // Expression is already bound, move to the location it was bound to.
        // This is required for immediate values - they are bound as is but
        // sometimes we need them to be moved to registers.
        host_->emov(dst_op.mod(), dst_op, op);
        expr_binding_.bind(e, dst_op);
    }

    void ebinary(const binary_op_t &obj, const ngen::InstructionModifier &mod,
            const ngen_operand_t &_dst, const ngen_operand_t &_src0,
            const ngen_operand_t &_src1) {
        auto dst = _dst;
        auto src0 = _src0;
        auto src1 = _src1;
        align_src_dst_offset(host_, scope_, mod, dst, src0, src1);
        switch (obj.op_kind) {
            case op_kind_t::_add: host_->eadd(mod, dst, src0, src1); break;
            case op_kind_t::_sub: host_->eadd(mod, dst, src0, -src1); break;
            case op_kind_t::_mul: host_->emul(mod, dst, src0, src1); break;
            case op_kind_t::_div: host_->ediv(mod, dst, src0, src1); break;
            case op_kind_t::_mod: host_->emod(mod, dst, src0, src1); break;
            case op_kind_t::_shl: host_->eshl(mod, dst, src0, src1); break;
            case op_kind_t::_shr: host_->eshr(mod, dst, src0, src1); break;
            case op_kind_t::_min: host_->emin(mod, dst, src0, src1); break;
            case op_kind_t::_max: host_->emax(mod, dst, src0, src1); break;
            case op_kind_t::_ge:
            case op_kind_t::_gt:
            case op_kind_t::_le:
            case op_kind_t::_lt:
            case op_kind_t::_eq:
            case op_kind_t::_ne: {
                ir_assert(!dst.is_negated()) << "Destination can't be negated.";
                ngen::InstructionModifier cmp_mod = mod;
                cmp_mod |= cmp_op_to_ngen(obj.op_kind);
                cmp_mod |= dst.flag_register();
                host_->ecmp(cmp_mod, src0, src1);
                break;
            }
            case op_kind_t::_and: host_->eand(mod, dst, src0, src1); break;
            case op_kind_t::_prelu: {
                int grf_size = ngen::GRF::bytes(hw);
                int esize = mod.getExecSize();
                int regs = utils::div_up(esize * int(sizeof(float)), grf_size);
                auto temp = scope_.alloc_reg_buf_data(regs).format(
                        0, ngen::DataType::f, esize);
                host_->emul(mod, temp, dst, src1);
                host_->csel(mod | host_->le, dst.reg_data(), temp,
                        dst.reg_data(), dst.reg_data());
                break;
            }
            default:
                ir_error_not_expected()
                        << "Unknown kind: " << to_string(obj.op_kind);
        }
    }

    conv_kernel_t<hw> *host_;
    expr_binding_t expr_binding_;
    ngen_register_scope_t &scope_;

    object_eq_map_t<expr_t, type_t> int_up_converts_;
};

template <typename DataSpecT, typename = void>
struct atomic_helper_t {
    template <typename GeneratorT>
    static void call(GeneratorT *, ngen::AtomicOp,
            const ngen::InstructionModifier &, const DataSpecT &,
            ngen::AddressBase, const ngen::RegData &, const ngen::RegData &) {
        ir_error_not_expected()
                << "Unknown DataSpec: atomics are not supported.";
    }
};

template <typename DataSpecT>
struct atomic_helper_t<DataSpecT,
        typename std::enable_if<
                std::is_same<DataSpecT, ngen::scattered_dword>::value>::type> {
    template <typename GeneratorT>
    static void call(GeneratorT *host, ngen::AtomicOp atomic_op,
            const ngen::InstructionModifier &mod, const DataSpecT &spec,
            ngen::AddressBase base, const ngen::RegData &addr,
            const ngen::RegData &data) {
        host->atomic(atomic_op, mod, spec, base, addr, data);
    }
};

// Helper to emit send instructions.
class send_impl_t {
public:
    send_impl_t(ngen::HW hw, const send_t &send) : hw_(hw), send_(send) {}

    template <typename GeneratorT, typename T>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const ngen::InstructionModifier &mod,
            const ngen::RegData &surf_base_addr, int surf_bti,
            const ngen::RegData &header, const T &data) {

        auto access_type = send_.access_type;
        auto data_type = send_.data_type;
        auto data_elems = send_.data_elems;
        auto address_model = send_.address_model;
        auto atomic_op = send_.atomic_op;

        bool is_read = (access_type == ngen_proxy::Access::Read);
        ngen::AddressBase address_base;
        if (address_model == ngen_proxy::AddressModel::ModelBTS) {
            address_base = ngen::AddressBase::createBTS(surf_bti);
        } else if (address_model == ngen_proxy::AddressModel::ModelA64) {
            address_base = ngen::AddressBase::createA64(true);
        } else if (address_model == ngen_proxy::AddressModel::ModelSLM) {
            address_base = ngen::AddressBase::createSLM();
        } else {
            ir_error_not_expected();
        }

        if (data_type == type_t::byte()) {
            emit_load_or_store(is_read, atomic_op, host, mod,
                    ngen::scattered_byte(data_elems), address_base, header,
                    data);
        } else if (data_type == type_t::dword()) {
            emit_load_or_store(is_read, atomic_op, host, mod,
                    ngen::scattered_dword(data_elems), address_base, header,
                    data);
        } else if (data_type == type_t::qword()) {
            emit_load_or_store(is_read, atomic_op, host, mod,
                    ngen::scattered_qword(data_elems), address_base, header,
                    data);
        } else if (data_type == type_t::oword()) {
            emit_load_or_store(is_read, atomic_op, host, mod,
                    ngen::block_oword(data_elems), address_base, header, data);
        } else if (data_type == type_t::hword()) {
            emit_load_or_store(is_read, atomic_op, host, mod,
                    ngen::block_hword(data_elems), address_base, header, data);
        } else {
            ir_error_not_expected();
        }
    }

private:
    template <typename GeneratorT, typename DataSpecT>
    void emit_load_or_store(bool is_read, ngen_proxy::AtomicOp atomic_op,
            GeneratorT *host, const ngen::InstructionModifier &mod,
            const DataSpecT &spec, ngen::AddressBase base,
            const ngen::RegData &addr, const ngen::RegData &data) {
        bool is_atomic = (atomic_op != ngen_proxy::AtomicOp::undef);

        if (hw_ == ngen::HW::XeHPC) {
            if (maybe_promote_to_lsc(host, mod, data, spec, base, addr)) {
                return;
            }
        }
        ir_assert(!send_.is_prefetch) << "Prefetches are not supported.";

        if (is_read) {
            ir_assert(!is_atomic) << "Unexpected atomic loads.";
            host->load(mod, data, spec, base, addr);
        } else {
            if (is_atomic) {
                atomic_helper_t<DataSpecT>::call(
                        host, to_ngen(atomic_op), mod, spec, base, addr, data);
            } else {
                host->store(mod, spec, base, addr, data);
            }
        }
    }

    template <typename GeneratorT, typename DataSpecT>
    bool maybe_promote_to_lsc(GeneratorT *host,
            const ngen::InstructionModifier &mod, const ngen::RegData &data,
            const DataSpecT &spec, const ngen::AddressBase &base,
            const ngen::RegData &addr) {
        if (send_.is_atomic()) return false;
        if (!send_.is_a64()) return false;
        if (send_.type != message_type_t::block) return false;
        if (send_.slots != 1) return false;

        int size = send_.size();

        int lsc_data_size = 4; // Use D32.
        int lsc_vector_size = size / lsc_data_size;
        if (lsc_data_size * lsc_vector_size != size) return false;

        if (send_.is_read()) {
            host->load.ugm(1 | mod, data,
                    ngen::block(ngen::DataSizeLSC::D32, lsc_vector_size)
                            | ngen::CacheSettingsLSC::L1C_L3C,
                    host->A64, addr);
        } else {
            host->store.ugm(1 | mod,
                    ngen::block(ngen::DataSizeLSC::D32, lsc_vector_size)
                            | ngen::CacheSettingsLSC::L1WB_L3WB,
                    host->A64, addr, data);
        }

        return true;
    }

    ngen::HW hw_;
    const send_t &send_;
};

// Reinterprets layouts to wider data type (up to 4 bytes).
// Example: 16a16b (s8 type) -> 16a4b (s32 type)
static bool try_reinterpret_to_wider_type(layout_t &src, layout_t &dst,
        const tensor_t &tile = {}, bool do_update = true,
        int *new_size_out = nullptr) {
    if (src.blocks().empty() || dst.blocks().empty()) return false;
    if (src.type() != dst.type()) return false;

    auto &s0 = src.blocks()[0];
    auto &d0 = dst.blocks()[0];
    if (s0.dim_idx != d0.dim_idx) return false;
    if (int(s0.stride) != 1) return false;
    if (int(d0.stride) != 1) return false;

    int old_size = src.type().size();
    int s0_old_size = int(s0.block) * old_size;
    int d0_old_size = int(d0.block) * old_size;

    int new_size = math::gcd(s0_old_size, d0_old_size);
    new_size = math::gcd(new_size, 4); // Try types up to 4 bytes.
    if (new_size <= old_size) return false;

    auto tile_ok = [&](const layout_t &l) {
        if (tile.is_empty()) return true;
        int factor = new_size / old_size;
        if (tile(l.blocks()[0].dim_idx) % factor != 0) return false;
        return true;
    };

    auto strides_ok = [&](const layout_t &l) {
        for (int i = 1; i < int(l.blocks().size()); i++) {
            auto &b = l.blocks()[i];
            if (int(b.stride) * old_size % new_size != 0) return false;
        }
        return true;
    };

    while (new_size > old_size) {
        bool ok = true;
        ok &= (tile_ok(src) && tile_ok(dst));
        ok &= (strides_ok(src) && strides_ok(dst));
        if (ok) {
            if (do_update) {
                src = src.reinterpret(type_t::s(new_size * 8));
                dst = dst.reinterpret(type_t::s(new_size * 8));
            }
            if (new_size_out) *new_size_out = new_size;
            return true;
        }
        new_size /= 2;
    }
    return false;
}

// Implementation of GRF reorder between 2D dense layouts.
// Requirements for A -> B reorder:
// - A and B must have the same data type
// - Layouts must be 2D and dense
// Reorder may require several steps, in this case a temporary buffer T is
// allocated. For example: A -> T -> B or A -> B -> T -> B
class reorder_2d_impl_t {
public:
    reorder_2d_impl_t(
            ngen::HW hw, const layout_t &src_layout, const layout_t &dst_layout)
        : hw_(hw), src_(src_layout), dst_(dst_layout) {
        ir_assert(src_.type() == dst_.type());
        tile_ = find_2d_tile(src_, dst_);
    }

    const tensor_t &tile() const { return tile_; }

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src_rd, const reg_buf_data_t &dst_rd) {
        int a_idx, b_idx;
        int tile_a, tile_b;
        tile_to_2d_dims(tile_, a_idx, b_idx, tile_a, tile_b);

        // Convert src/dst to 2D layouts.
        dim_assignment_t to_ab(src_.ndims(), 2);
        to_ab.assign(a_idx, 0);
        to_ab.assign(b_idx, 1);
        auto src_ab = to_ab.map(src_);
        auto dst_ab = to_ab.map(dst_);

        // Find minimal cost reorder path between layouts.
        auto path = find_min_cost_path(hw_, src_ab, dst_ab, tile_a, tile_b);

        // Allocate a temporary GRF buffer if needed.
        reg_buf_data_t tmp;
        if (path.size() > 1) {
            const int grf_size = ngen::GRF::bytes(hw_);
            tmp = scope.alloc_reg_buf_data(
                    utils::div_up(dst_ab.size(), grf_size));
        }

        // Iterate through found reorders.
        auto *prev_layout = &src_ab;
        auto prev_rd = src_rd;
        int path_len = int(path.size());
        auto &orig_type = src_ab.type();
        for (int i = 0; i < path_len; i++) {
            auto &step = path[i];
            auto &tile = step.tile;
            auto &type = step.type;
            auto *next_layout = &step.layout;

            // x -> y reorder.
            auto x = prev_layout->map(tile).reinterpret(type);
            auto y = next_layout->map(tile).reinterpret(type);

            bool use_dst = ((path_len - i) % 2 == 1);
            auto next_rd = (use_dst ? dst_rd : tmp);
            auto &x_blocks = x.blocks();
            auto &y_blocks = y.blocks();
            ir_assert(x_blocks.size() <= 1);
            ir_assert(y_blocks.size() <= 1);
            int x_stride = (x_blocks.empty() ? 1 : int(x_blocks[0].stride));
            int y_stride = (y_blocks.empty() ? 1 : int(y_blocks[0].stride));
            int width = int(tile.elems()) * orig_type.size() / type.size();
            next_layout->for_each_tile(
                    tile, [&](const std::vector<dim_t> &start) {
                        int prev_off = int(prev_layout->offset_in_bytes(start));
                        int next_off = int(next_layout->offset_in_bytes(start));
                        auto x_sub = prev_rd.format(prev_off, to_ngen(type), 1);
                        auto y_sub = next_rd.format(next_off, to_ngen(type), 1);
                        emit_reorder_1d_tile(hw_, host, scope, width, x_sub,
                                x_stride, y_sub, y_stride);
                    });
            prev_layout = next_layout;
            prev_rd = next_rd;
        }
    }

    // Returns the biggest common 2D tile that is innermost for both layouts.
    // The returned tile contains at most max_elems elements. If match_outer is
    // true, then outer parts of both layouts are required to be equal.
    // Returns an empty tensor if the requested tile is not found.
    static tensor_t find_2d_tile(const layout_t &a, const layout_t &b,
            int max_elems = std::numeric_limits<int>::max(),
            bool match_outer = false) {
        std::vector<dim_t> tile_dims(a.ndims(), 1);
        if (a.blocks().empty() || b.blocks().empty())
            return tensor_t(tile_dims);

        auto non_one_ndims = [](const tensor_t &t) {
            int ret = 0;
            for (dim_t d : t.dims())
                ret += (d != 1 ? 1 : 0);
            return ret;
        };

        layout_iterator_t a_it(a);
        layout_iterator_t b_it(b);

        tensor_t max_tile;
        for (;;) {
            auto a_tile = a_it.tile();
            auto b_tile = b_it.tile();
            if (non_one_ndims(a_tile) > 2 || non_one_ndims(b_tile) > 2) break;
            dim_t a_elems = a_tile.elems();
            dim_t b_elems = b_tile.elems();

            bool tile_ok = true;
            if (!a_tile.is_equal(b_tile)) tile_ok = false;
            if (match_outer) {
                auto a_outer = a_it.outer_layout();
                auto b_outer = b_it.outer_layout();
                if (!a_outer.is_equal(b_outer)) tile_ok = false;
            }
            if (tile_ok) {
                if (a_it.nblocks() > max_tile_blocks) break;
                if (b_it.nblocks() > max_tile_blocks) break;
                if (a_tile.elems() > max_elems) break;
                max_tile = a_tile;
                if (!a_it.has_next() || !b_it.has_next()) break;
                ++a_it;
                ++b_it;
            } else if (a_elems <= b_elems) {
                if (!a_it.has_next()) break;
                ++a_it;
            } else {
                if (!b_it.has_next()) break;
                ++b_it;
            }
        }
        return max_tile;
    }

    static const int max_tile_blocks = 4;

private:
    // Helper class to incrementally increase a sub-layout of the given layout.
    // One step - adding the minimal factor of the next remaining block. Used
    // to find the minimal tile between two layouts that is innermost for both
    // layouts.
    struct layout_iterator_t {
        layout_iterator_t(const layout_t &l) : l(l), block_idx(-1), block(1) {}

        bool has_next() const {
            dim_t b = block;
            int b_idx = block_idx;
            while (b == 1) {
                b_idx++;
                if (b_idx >= int(l.blocks().size())) return false;
                b = int(l.blocks()[b_idx].block);
            }
            return true;
        }

        layout_iterator_t &operator++() {
            ir_assert(has_next());
            while (block == 1) {
                block_idx++;
                block = int(l.blocks()[block_idx].block);
            }
            // Find smallest factor.
            for (int factor = 2; factor <= int(block); factor++) {
                if (block % factor == 0) {
                    block /= factor;
                    return *this;
                }
            }

            ir_error_not_expected();
            return *this;
        }

        tensor_t tile() const {
            std::vector<dim_t> dims(l.ndims(), 1);
            for (int i = 0; i <= block_idx; i++) {
                auto &b = l.blocks()[i];
                int b_block = b.block;
                if (i == block_idx) b_block /= block;
                dims[b.dim_idx] *= b_block;
            }
            return tensor_t(dims);
        }

        int nblocks() const { return block_idx + 1; }

        layout_t outer_layout() const {
            auto &blocks = l.blocks();
            std::vector<block_t> outer_blocks;
            if (block > 1) {
                auto &b = blocks[block_idx];
                outer_blocks.push_back(b);
                outer_blocks[0].block = block;
                outer_blocks[0].stride = b.stride * (b.block / block);
            }
            outer_blocks.insert(outer_blocks.end(),
                    blocks.begin() + block_idx + 1, blocks.end());
            return layout_t(l.type(), l.ndims(), l.offset(), outer_blocks);
        }

        const layout_t &l;

        int block_idx;
        dim_t block;
    };

    // Represents 2D reorder corresponding to (a x b) tile.
    struct edge_t {
        edge_t() = default;
        edge_t(int idx, int a, int b) : idx(idx), a(a), b(b) {}

        tensor_t tile() const { return tensor_t({a, b}); }

        std::string str() const {
            std::ostringstream oss;
            oss << "edge(idx = " << idx << ", a = " << a << ", b = " << b
                << ")";
            return oss.str();
        }

        int idx; // Identifier of the edge.
        int a = 0, b = 0; // Specify tile (a x b).
    };

    // Represents GRF layout between edges-reorders.
    struct vertex_t {
        vertex_t(ngen::HW hw, int idx, const layout_t &layout)
            : hw(hw), idx(idx), layout(layout) {}

        std::string str() const {
            std::ostringstream oss;
            oss << "vertex(idx = " << idx << ", layout = " << layout << ")";
            return oss.str();
        }

        void set_edges(const std::vector<edge_t> &edges) {
            adj_edge_type_masks.resize(edges.size());
            int type_size = layout.type().size();
            for (int i = 0; i < int(edges.size()); i++) {
                auto &e = edges[i];
                auto tile = e.tile();
                int max_type_size;
                bool ok = try_reinterpret_to_wider_type(
                        layout, layout, tile, false, &max_type_size);
                if (!ok) max_type_size = type_size;
                int from = math::ilog2q(type_size);
                int to = math::ilog2q(max_type_size);
                for (int j = from; j <= to; j++) {
                    type_t type = type_t::u(8 << j);
                    if (can_reorder(tile, type))
                        adj_edge_type_masks[i] |= (1 << j);
                }
            }
        }

        void add_neighbor(const vertex_t *v) { adj_vertices.push_back(v); }

        bool is_neighbor(const vertex_t &v) const {
            for (auto *n : adj_vertices)
                if (n == &v) return true;
            return false;
        }

        // Check the following limitations:
        // - Assume at most one block (maybe with non-dense stride)
        // - Horizontal stride must be <= 4 for GRF region
        // - GRF region can't span more than 2 registers
        bool can_reorder(const tensor_t &tile, const type_t &type) const {
            auto ab_layout = layout.map(tile).reinterpret(type);
            int nblocks = int(ab_layout.blocks().size());
            if (nblocks == 0) return true;
            if (nblocks > 1) return false;
            auto &last = ab_layout.blocks().back();
            int max_stride = int(last.stride * last.block);
            if (last.stride > 4) return false;
            int max_stride_bytes = max_stride * type.size();
            int grf_size = ngen::GRF::bytes(hw);
            if (max_stride_bytes > 2 * grf_size) return false;
            return true;
        }

        // Finds the minimal cost of reordering from this vertex to vertex v.
        int cost(const vertex_t &v, const std::vector<edge_t> &edges,
                edge_t &min_edge, type_t &min_type) const {
            int min_cost = std::numeric_limits<int>::max();
            for (int i = 0; i < int(edges.size()); i++) {
                type_t i_min_type;
                int new_cost = cost(edges[i], v, i_min_type);
                if (new_cost < min_cost) {
                    min_cost = new_cost;
                    min_edge = edges[i];
                    min_type = i_min_type;
                }
            }
            return min_cost;
        }

        // Finds the minimal cost of reordering from this vertex to vertex `v`
        // through edge `e`. If the reorder is possible, `type` contains the
        // reorder type with the minimal cost.
        int cost(const edge_t &e, const vertex_t &v, type_t &type) const {
            uint32_t mask = (adj_edge_type_masks[e.idx]
                    & v.adj_edge_type_masks[e.idx]);
            if (mask == 0) return std::numeric_limits<int>::max();
            int cur_size = layout.type().size();
            int cur_cost = layout.elems() / (e.a * e.b);
            int min_log_bytes = math::ilog2q(cur_size);
            int max_log_bytes = 3;
            int min_cost = std::numeric_limits<int>::max();
            for (int i = min_log_bytes; i <= max_log_bytes; i++) {
                if ((mask & (1 << i)) == 0) continue;
                if (i > min_log_bytes) {
                    ir_assert(!layout.blocks().empty());
                    ir_assert(!v.layout.blocks().empty());
                    int dim_idx0 = layout.blocks()[0].dim_idx;
                    int dim_idx1 = v.layout.blocks()[0].dim_idx;
                    if (dim_idx0 != dim_idx1) continue;
                }
                min_cost = cur_cost;
                type = type_t::u(8 << i);
                break;
            }
            return min_cost;
        }

        ngen::HW hw;
        int idx; // Identifier of the vertex.
        layout_t layout; // Layout of the vertex.
        // Specifies a bitmask for every edge: if adj_edge_type_masks[E_idx]
        // has b-th bit set then this vertex can be reordered through E edge
        // using the data type with size 2^b bytes.
        std::vector<uint32_t> adj_edge_type_masks;
        std::vector<const vertex_t *> adj_vertices; // Adjacent vertices.
    };

    // Represents a reorder step.
    struct reorder_step_t {
        reorder_step_t() = default;
        reorder_step_t(const layout_t &layout, const tensor_t &tile,
                const type_t &type)
            : layout(layout), tile(tile), type(type) {}

        layout_t layout; // Destination layout.
        tensor_t tile; // Tile corresponding to one instruction.
        type_t type; // Registers should be reinterpreted to `type` for reorder.
    };

    // Extracts dimension sizes and their indices from a multidimensional
    // tensor.
    static void tile_to_2d_dims(
            const tensor_t &tile, int &a_idx, int &b_idx, int &a, int &b) {
        a_idx = -1;
        b_idx = -1;
        for (int i = 0; i < tile.ndims(); i++) {
            if (tile.dims()[i] == 1) continue;
            if (a_idx == -1) {
                a_idx = i;
                continue;
            }
            if (b_idx == -1) {
                b_idx = i;
                continue;
            }
            ir_error_not_expected();
        }

        for (int i = 0; i < tile.ndims(); i++) {
            if (utils::one_of(i, a_idx, b_idx)) continue;
            if (a_idx == -1) {
                a_idx = i;
                continue;
            }
            if (b_idx == -1) {
                b_idx = i;
                continue;
            }
        }

        if (a_idx > b_idx) std::swap(a_idx, b_idx);

        a = tile.dims()[a_idx];
        b = tile.dims()[b_idx];
    }

    // Finds the optimal sequence of reorders between src and dst layouts.
    static std::vector<reorder_step_t> find_min_cost_path(ngen::HW hw,
            const layout_t &src, const layout_t &dst, int tile_a, int tile_b) {
        // Create all possible edges - 2D reorders.
        std::vector<edge_t> edges;
        for (int a = 1; a <= tile_a; a *= 2) {
            for (int b = 1; b <= tile_b; b *= 2) {
                int idx = int(edges.size());
                edges.emplace_back(idx, a, b);
            }
        }

        int nedges = int(edges.size());

        // Create all possible layouts for tile_a x tile_b tensor.
        std::vector<vertex_t> vertices;
        std::vector<std::vector<std::pair<int, uint32_t>>> edge_vertices(
                nedges);
        auto all_layouts = generate_all_layouts(src.type(), tile_a, tile_b);
        for (auto &l : all_layouts) {
            // Skip if too many blocks.
            if (int(l.blocks().size()) > max_tile_blocks) continue;
            int v_idx = int(vertices.size());
            vertices.emplace_back(hw, v_idx, l);
            auto &v = vertices.back();
            // Pass all known reorders, the vertex/layout will filter out
            // incompatible reorders.
            v.set_edges(edges);
            // Store all vertices adjacent to a specific edge.
            for (int i = 0; i < nedges; i++) {
                uint32_t mask = v.adj_edge_type_masks[i];
                if (mask != 0) edge_vertices[i].emplace_back(v_idx, mask);
            }
        }

        // Find neighbors between all vertices.
        int nvertices = int(vertices.size());
        for (int i = 0; i < nvertices; i++) {
            auto &v = vertices[i];
            for (int j = 0; j < nedges; j++) {
                uint32_t mask = v.adj_edge_type_masks[j];
                if (mask != 0) {
                    for (auto &idx_mask : edge_vertices[j]) {
                        int v_idx = idx_mask.first;
                        if (v_idx == i) continue;
                        uint32_t common_mask = (mask
                                & vertices[v_idx].adj_edge_type_masks[j]);
                        if (common_mask != 0) v.add_neighbor(&vertices[v_idx]);
                    }
                }
            }
        }

        // Identify source and destination vertices.
        int src_idx = -1;
        int dst_idx = -1;
        for (int i = 0; i < nvertices; i++) {
            auto &v = vertices[i];
            if (src_idx == -1
                    && v.layout.is_strictly_equal(
                            src, /*compare_offset=*/false))
                src_idx = i;
            if (dst_idx == -1
                    && v.layout.is_strictly_equal(
                            dst, /*compare_offset=*/false))
                dst_idx = i;
        }

        ir_assert(src_idx != -1);
        ir_assert(dst_idx != -1);

        // Layouts are the same, just copy.
        if (src_idx == dst_idx) {
            auto &v = vertices[src_idx];
            edge_t min_edge;
            type_t min_type;
            v.cost(v, edges, min_edge, min_type);
            reorder_step_t step(v.layout, min_edge.tile(), min_type);
            return {step};
        }

        // Dijkstra's algorithm, find the minimal cost path between src and
        // dst. Use the number of instructions to estimate the cost.
        int inf_cost = std::numeric_limits<int>::max();
        std::vector<int> cost(nvertices, inf_cost);
        std::vector<int> prev(nvertices);
        std::vector<reorder_step_t> reorder_steps(nvertices);
        std::vector<bool> seen(nvertices, false);
        cost[src_idx] = 0;
        for (int i = 0; i < nvertices; i++) {
            int min_idx = -1;
            int min_cost = inf_cost;
            for (int j = 0; j < nvertices; j++) {
                if (seen[j]) continue;
                if (cost[j] < min_cost) {
                    min_idx = j;
                    min_cost = cost[j];
                }
            }
            seen[min_idx] = true;
            auto &v_min = vertices[min_idx];
            for (auto *v : v_min.adj_vertices) {
                edge_t min_edge;
                type_t min_type;
                int new_cost = cost[min_idx]
                        + v_min.cost(*v, edges, min_edge, min_type);
                if (new_cost < cost[v->idx]) {
                    cost[v->idx] = new_cost;
                    prev[v->idx] = min_idx;
                    reorder_steps[v->idx] = reorder_step_t(
                            v->layout, min_edge.tile(), min_type);
                }
            }
        }

        // Sanity check, ensure the reorder sequence is not too long.
        int max_cost = 256;
        ir_assert(cost[dst_idx] <= max_cost);
        MAYBE_UNUSED(max_cost);

        // Restore the shortest reorder path.
        std::vector<reorder_step_t> ret;
        int idx = dst_idx;
        while (idx != src_idx) {
            ret.push_back(reorder_steps[idx]);
            idx = prev[idx];
        }
        std::reverse(ret.begin(), ret.end());
        return ret;
    }

    // Returns all possible layouts for (a x b) tensor.
    static std::vector<layout_t> generate_all_layouts(
            const type_t &type, int a, int b) {
        std::vector<layout_t> ret;
        std::vector<block_t> blocks;
        generate_all_layouts_impl(ret, blocks, type, a, b, 1);
        return ret;
    }

    static void generate_all_layouts_impl(std::vector<layout_t> &layouts,
            std::vector<block_t> &blocks, const type_t &type, int a, int b,
            int stride) {
        if (a == 1 && b == 1) {
            layouts.emplace_back(type, 2, 0, blocks);
            return;
        }
        bool iterate_a = true;
        bool iterate_b = true;

        // Avoid repeating indices to keep only unique layouts.
        if (!blocks.empty()) {
            auto &last = blocks.back();
            iterate_a &= (last.dim_idx != 0);
            iterate_b &= (last.dim_idx != 1);
        }

        if (iterate_a) {
            for (int a_blk = 2; a_blk <= a; a_blk++) {
                if (a % a_blk != 0) continue;
                blocks.emplace_back(0, a_blk, stride);
                generate_all_layouts_impl(
                        layouts, blocks, type, a / a_blk, b, stride * a_blk);
                blocks.pop_back();
            }
        }
        if (iterate_b) {
            for (int b_blk = 2; b_blk <= b; b_blk++) {
                if (b % b_blk != 0) continue;
                blocks.emplace_back(1, b_blk, stride);
                generate_all_layouts_impl(
                        layouts, blocks, type, a, b / b_blk, stride * b_blk);
                blocks.pop_back();
            }
        }
    }

    ngen::HW hw_;

    layout_t src_;
    layout_t dst_;

    tensor_t tile_;
};

template <ngen::HW hw = ngen::HW::Unknown>
class reorder_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    reorder_kernel_t(const conv_config_t &cfg, const convolution_pd_t *pd,
            const kernel_info_t &kernel_info, const layout_t &src_layout,
            const layout_t &dst_layout)
        : simd_size_(cfg.simd_size), ra_(hw, "reorder_kernel_t") {
        externalName("reorder");
        requireLocalID(1);
        requireLocalSize();
        requireGRF(cfg.regs);
        requireSIMD(simd_size_);
        if (cfg.is_dpas_or_dpasw_fma()) requireDPAS();

        for (int i = 0; i < kernel_info.nargs(); i++) {
            auto &name = kernel_info.arg_name(i);
            auto &type = kernel_info.arg_type(i);
            if (type.is_ptr()) {
                newArgument(name, ngen::ExternalArgumentType::GlobalPtr);
            } else {
                newArgument(name, to_ngen(type));
            }
        }

        finalizeInterface();

        // Claim registers.
        ra_.claim(r0);
        ra_.claim(getLocalID(0));
        ra_.claim(getLocalSize(0));

        std::vector<std::string> arg_names(kernel_info.nargs());

        for (int i = 0; i < kernel_info.nargs(); i++) {
            arg_names[i] = kernel_info.arg_name(i);
            ra_.claim(getArgument(kernel_info.arg_name(i)));
        }

        setDefaultNoMask();
        setDefaultAutoSWSB(true);

        prologue();

        if (emu_strategy.emulate64) {
            emu_state.temp[0] = ra_.alloc();
            emu_state.temp[1] = ra_.alloc();
        }

        src_ptr_ = getArgument(arg_names[0]);
        dst_ptr_ = getArgument(arg_names[1]);
        src_surf_ = Surface(getArgumentSurface(arg_names[0]));
        dst_surf_ = Surface(getArgumentSurface(arg_names[1]));
        elems_ = getArgument(arg_names[2]);

        global_id_ = ra_.alloc_sub<uint32_t>();

        mul(1, global_id_, r0.ud(1), getLocalSize(0).uw());
        add(1, global_id_, global_id_, getLocalID(0));

        int elems_per_thr;
        if (is_f32_to_bf16(src_layout, dst_layout)) {
            emit_f32_to_bf16();
        } else if (is_2d_reorder(src_layout, dst_layout, elems_per_thr)) {
            emit_2d_reorder(src_layout, dst_layout, elems_per_thr);
        } else {
            ir_error_not_expected();
        }

        epilogue();
    }

    void emit_f32_to_bf16() {
        int elems_per_thr = f32_to_bf16_elems_per_thr();

        bool use_a64 = false;
        // XXX: Stateful messages don't work on XeHPC.
        use_a64 = (hw == ngen::HW::XeHPC);

        int grf_size = ngen::GRF::bytes(hw);
        int ud_size = sizeof(uint32_t);
        int uq_size = sizeof(uint64_t);
        int f_size = sizeof(float);
        int bf_size = sizeof(uint16_t);

        auto elem_vec = ra_.alloc_range(elems_per_thr * ud_size / grf_size);
        auto elem_vec_q_strided
                = ra_.alloc_range(elems_per_thr * uq_size / grf_size);
        auto src_ptr_vec = ra_.alloc_range(elems_per_thr * uq_size / grf_size);

        auto get_elem = [&](int i) {
            return get_subregister(hw, ngen::DataType::ud, elem_vec, i);
        };

        auto get_elem_q_strided = [&](int i) {
            return get_subregister(
                    hw, ngen::DataType::ud, elem_vec_q_strided, i * 2);
        };

        auto S = ra_.alloc_range(elems_per_thr * f_size / grf_size);
        // D is for bf16 but allocated as dword-strided to use with
        // scattered_byte(2) messages.
        auto D = ra_.alloc_range(elems_per_thr * f_size / grf_size);

        auto get_src_reg = [&](int i) {
            return get_subregister(hw, ngen::DataType::f, S, i);
        };

        auto get_dst_reg = [&](int i) {
            return get_subregister(hw, ngen::DataType::bf, D, i);
        };

        auto idx_vec = ra_.alloc().uw();
        mov(8, idx_vec, ngen::Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        for (int i = 0; i < elems_per_thr; i += 8)
            shl(8, get_elem(i), global_id_,
                    math::ilog2q(elems_per_thr / simd_size_));
        for (int i = 0; i < elems_per_thr; i += 8) {
            add3(8, get_elem(i), get_elem(i), idx_vec, i);
            if (use_a64) {
                auto src_ptr_sub_vec = get_subregister(
                        hw, ngen::DataType::uq, src_ptr_vec, i)(1);
                emov(8, get_elem_q_strided(i)(2), get_elem(i)(1));
                eshl(8, src_ptr_sub_vec, get_elem_q_strided(i)(2),
                        math::ilog2q(f_size));
                eadd(8, src_ptr_sub_vec, src_ptr_sub_vec, src_ptr_);
            }
        }

        int elems_per_load = 16;
        for (int i = 0; i < elems_per_thr; i += elems_per_load) {
            cmp(16 | lt | f0[0], get_elem(i)(1), elems_);
            if (use_a64) {
                auto h_a64 = get_subregister(
                        hw, ngen::DataType::uq, src_ptr_vec, i);
                load(16 | f0[0], get_src_reg(i), ngen::scattered_dword(), A64,
                        h_a64);
            } else {
                auto h_bts = get_elem(i);
                load(16 | f0[0], get_src_reg(i), ngen::scattered_dword(),
                        src_surf_, h_bts);
            }
        }

        int mov_step = (grf_size == 32 ? 8 : 16);
        for (int i = 0; i < elems_per_thr; i += mov_step) {
            // dst is dword-strided.
            mov(mov_step, get_dst_reg(i * 2)(2), get_src_reg(i)(1));
        }

        auto dst_header = ra_.alloc_range(
                elems_per_load * (use_a64 ? uq_size : ud_size) / grf_size);
        for (int i = 0; i < elems_per_thr; i += elems_per_load) {
            for (int j = 0; j < elems_per_load; j += 8) {
                ngen::RegData h;
                if (use_a64) {
                    int off = j * uq_size;
                    h = dst_header[off / grf_size].uq(
                            (off % grf_size) / uq_size)(1);
                } else {
                    int off = j * ud_size;
                    h = dst_header[off / grf_size].ud(
                            (off % grf_size) / ud_size)(1);
                }
                emov(8, get_elem_q_strided(i + j)(2), get_elem(i + j)(1));
                eshl(8, h, get_elem_q_strided(i + j)(2), math::ilog2q(bf_size));
                if (use_a64) eadd(8, h, h, dst_ptr_);
            }

            cmp(16 | lt | f0[0], get_elem(i)(1), elems_);
            if (use_a64) {
                store(16 | f0[0], ngen::scattered_byte(2), A64, dst_header[0],
                        get_dst_reg(i * 2));
            } else {
                store(16 | f0[0], ngen::scattered_byte(2), dst_surf_,
                        dst_header[0], get_dst_reg(i * 2));
            }
        }
    }

    void emit_2d_reorder(
            const layout_t &_src, const layout_t &_dst, int elems_per_thr) {
        auto tile = _src.split_into_max_tile(elems_per_thr, /*is_dense=*/true);
        ir_assert(!tile.is_empty()) << "Can't split " << _src;

        auto src = _src.map(tile);
        auto dst = _dst.map(tile);

        int src_size = src.type().size();
        int dst_size = dst.type().size();
        int src_tile_bytes = src_size * elems_per_thr;
        int dst_tile_bytes = dst_size * elems_per_thr;

        int grf_size = ngen::GRF::bytes(hw);

        auto S = ra_.alloc_range(utils::div_up(src_tile_bytes, grf_size));
        auto D = ra_.alloc_range(utils::div_up(dst_tile_bytes, grf_size));

        auto src_header = ra_.alloc();
        auto dst_header = ra_.alloc();

        // Prepare headers for loads and stores.
        eshl(1, src_header.uq(0), global_id_,
                math::ilog2q(elems_per_thr / simd_size_ * src_size));
        eshl(1, dst_header.uq(0), global_id_,
                math::ilog2q(elems_per_thr / simd_size_ * dst_size));
        eadd(1, src_header.uq(0), src_header.uq(0), src_ptr_);
        eadd(1, dst_header.uq(0), dst_header.uq(0), dst_ptr_);

        int oword_bytes = 16;

        // Load source tile.
        int src_off = 0;
        while (src_tile_bytes > 0) {
            for (int i = 3; i >= 0; i--) {
                int size = (1 << i) * oword_bytes;
                if (src_tile_bytes >= size) {
                    load(16, S[src_off / grf_size],
                            ngen::block_oword(size / oword_bytes), A64,
                            src_header);
                    eadd(1, src_header.uq(0), src_header.uq(0), size);
                    src_tile_bytes -= size;
                    src_off += size;
                    break;
                }
            }
        }

        // Reorder source tile to destination tile.
        ngen_register_scope_t scope(ra_);
        reorder_2d_impl_t r(hw, src, dst);
        reg_buf_t S_buf(hw, S);
        reg_buf_t D_buf(hw, D);
        r.emit(this, scope, S_buf, D_buf);

        // Store destination tile.
        int dst_off = 0;
        while (dst_tile_bytes > 0) {
            for (int i = 3; i >= 0; i--) {
                int size = (1 << i) * oword_bytes;
                if (dst_tile_bytes >= size) {
                    store(16, ngen::block_oword(size / oword_bytes), A64,
                            dst_header, D[dst_off / grf_size]);
                    eadd(1, dst_header.uq(0), dst_header.uq(0), size);
                    dst_tile_bytes -= size;
                    dst_off += size;
                    break;
                }
            }
        }
    }

    friend struct dnnl::impl::gpu::jit::EmulationImplementation;

    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0) {
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
            const ngen::RegData &src0, const ngen::Immediate &src1) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }

    template <typename DT = void>
    void eshl(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, uint16_t src1) {
        EmulationImplementation::eshl<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }

    static compute::nd_range_t nd_range(
            int simd, const layout_t &src, const layout_t &dst) {
        ir_assert(src.elems() == dst.elems());

        if (is_f32_to_bf16(src, dst)) {
            int elems_per_thr = f32_to_bf16_elems_per_thr();
            return compute::nd_range_t(
                    {utils::div_up(src.elems(), elems_per_thr) * simd});
        }

        int elems_per_thr;
        if (is_2d_reorder(src, dst, elems_per_thr)) {
            return compute::nd_range_t(
                    {utils::div_up(src.elems(), elems_per_thr) * simd});
        }

        ir_error_not_expected();
        return compute::nd_range_t();
    }

private:
    static bool is_f32_to_bf16(const layout_t &src, const layout_t &dst) {
        if (src.type() != type_t::f32()) return false;
        if (dst.type() != type_t::bf16()) return false;
        if (src.retype(type_t::u8()) != dst.retype(type_t::u8())) return false;
        return true;
    }

    static int f32_to_bf16_elems_per_thr() { return 32; }

    static bool is_2d_reorder(
            const layout_t &src, const layout_t &dst, int &elems_per_thr) {
        if (src.type() != dst.type()) return false;

        const int hword_bytes = 32;
        const int min_bytes_per_thr = hword_bytes;
        const int max_bytes_per_thr = 32 * hword_bytes;

        int type_size = src.type().size();
        int max_elems_per_thr = max_bytes_per_thr / type_size;

        auto tile = reorder_2d_impl_t::find_2d_tile(
                src, dst, max_elems_per_thr, /*match_outer=*/true);

        if (tile.is_empty()) return false;

        elems_per_thr = tile.elems();
        int bytes_per_thr = elems_per_thr * type_size;
        if (bytes_per_thr % hword_bytes != 0) return false;
        if (bytes_per_thr < min_bytes_per_thr) return false;
        if (bytes_per_thr > max_bytes_per_thr) return false;

        return true;
    }

    int simd_size_;
    reg_allocator_t ra_;
    EmulationStrategy emu_strategy = EmulationStrategy(hw);
    EmulationState emu_state;

    ngen::Subregister src_ptr_;
    ngen::Subregister dst_ptr_;
    ngen::AddressBase src_surf_;
    ngen::AddressBase dst_surf_;
    ngen::Subregister elems_;
    ngen::Subregister global_id_;
};

// Aligns src offset with dst offset when src is not broadcasted.
template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const reg_buf_data_t &dst,
        reg_buf_data_t &src);

// Performs 1D reorder, possibly with strides and type conversion.
template <typename GeneratorT>
void emit_reorder_1d_tile(ngen::HW hw, GeneratorT *host,
        ngen_register_scope_t &scope, int width, const reg_buf_data_t &_src,
        int src_stride, const reg_buf_data_t &_dst, int dst_stride) {
    auto src = _src;
    auto dst = _dst;
    ngen::DataType src_type = src.type();
    ngen::DataType dst_type = dst.type();
    // Replace (float -> float) by (int -> int) as word/dword moves have less
    // restrictions.
    if (src_type == dst_type && ngen_is_xf(src_type)) {
        src_type = to_ngen(type_t::u(ngen::getBytes(src_type) * 8));
        dst_type = src_type;
        src = src.reinterpret(src_type);
        dst = dst.reinterpret(dst_type);
    }

    int grf_size = ngen::GRF::bytes(hw);
    int src_type_size = ngen::getBytes(src_type);
    int dst_type_size = ngen::getBytes(dst_type);
    int src_stride_bytes = src_stride * src_type_size;
    int dst_stride_bytes = dst_stride * dst_type_size;
    bool dst_b = ngen_is_b(dst_type);
    bool dst_bf = (dst_type == ngen::DataType::bf);
    bool dst_d = ngen_is_dw(dst_type);
    bool dst_f = (dst_type == ngen::DataType::f);
    bool dst_hf = (dst_type == ngen::DataType::hf);
    bool dst_xf = dst_bf || dst_f || dst_hf;
    bool src_b = ngen_is_b(src_type);
    bool src_hf = (src_type == ngen::DataType::hf);
    bool src_bf = (src_type == ngen::DataType::bf);
    bool src_d = ngen_is_dw(src_type);
    bool src_f = (src_type == ngen::DataType::f);
    bool src_xf = src_bf || src_f || src_hf;
    bool f_to_xf = (src_f && (dst_bf || dst_hf));

    auto get_step = [&]() {
        int step = (width < 16 ? 8 : 16);

        // f32 -> bf16 or f32 -> f16: SIMD16 does not support mixed mode move.
        if (hw < ngen::HW::XeHPC)
            if (f_to_xf) step = std::min(step, 8);

        // Max supported stride is 4.
        if (src_stride > 4 || dst_stride > 4) step = 1;

        return step;
    };

    // bf16 -> f32:
    // - bf16 must be packed: use left shift instead.
    if (src_bf && dst_f) {
        int step = get_step();
        for (int i = 0; i < width; i += step) {
            int esize = std::min(step, width - i);
            ir_assert(math::is_pow2(esize));
            auto s = src.subregister(
                    i, esize, src_stride_bytes, ngen::DataType::uw);
            auto d = dst.subregister(
                    i, esize, dst_stride_bytes, ngen::DataType::ud);
            host->eshl(esize, d(dst_stride), s(src_stride), 16);
        }
        return;
    }

    // d -> bf/hf:
    // - Use d -> f -> bf/hf conversion with temporary
    if (src_d && (dst_bf || dst_hf)) {
        auto tmp = scope.alloc_reg_buf_data(
                                utils::div_up(
                                        int(width * sizeof(float)), grf_size))
                           .format(0, ngen::DataType::f);
        emit_reorder_1d_tile(hw, host, scope, width, src, src_stride, tmp, 1);
        emit_reorder_1d_tile(hw, host, scope, width, tmp, 1, dst, dst_stride);
        return;
    }

    // f32/f16/s32 -> s8/u8 and s8/u8 -> f32/s32
    // - Use saturation
    // - s8/u8 must be DW-strided: use temporary
    bool d_or_f_to_b = (src_d || src_f) && dst_b;
    bool b_to_d_or_f = (dst_d || dst_f) && src_b;
    bool hf_to_b = src_hf && dst_b;
    if (d_or_f_to_b || b_to_d_or_f || hf_to_b) {
        if (dst_d || dst_f) ir_assert(dst_stride_bytes == 4);
        if (src_d || src_f) ir_assert(src_stride_bytes == 4);
        if (src_hf) ir_assert(src_stride_bytes == 2);
        if (dst_b) ir_assert(utils::one_of(dst_stride_bytes, 1, 4));
        if (src_b) ir_assert(utils::one_of(src_stride_bytes, 1, 4));
        int step = get_step();
        const int grf_size = ngen::GRF::bytes(hw);
        auto tmp = scope.alloc_reg_buf_data(
                utils::div_up(int(step * sizeof(uint32_t)), grf_size));
        for (int i = 0; i < width; i += step) {
            int esize = std::min(step, width - i);
            ir_assert(math::is_pow2(esize));

            auto s = src.subregister(i, esize, src_stride_bytes);
            auto d = dst.subregister(i, esize, dst_stride_bytes);
            if (src_d || src_f || src_hf) {
                // d -> b.
                if (dst_stride_bytes == 1) {
                    auto t = tmp.subregister(0, dst_type)(4);
                    host->emov(esize | host->sat, t, s(1));
                    host->emov(esize, d(1), t);
                } else {
                    host->emov(esize | host->sat, d(4), s(1));
                }
            } else {
                // b -> d.
                // hf -> d.
                if (esize == 1) {
                    // Direct x8 -> x32 scalar cast is not always
                    // supported. Use intermediate cast to s16.
                    auto t = tmp.subregister(0, ngen::DataType::w)(1);
                    host->emov(esize, t, s);
                    host->emov(esize, d, t);
                } else if (src_stride_bytes == 1) {
                    auto t = tmp.subregister(0, src_type)(4);
                    host->emov(esize, t, s(1));
                    host->emov(esize, d(1), t);
                } else {
                    host->emov(esize, d(1), s(4));
                }
            }
        }
        return;
    }

    // Perform regular move.
    int step = get_step();
    for (int i = 0; i < width; i += step) {
        int esize = std::min(step, width - i);
        ir_assert(math::is_pow2(esize));
        auto s = src.format(i * src_stride_bytes, ngen::DataType::invalid,
                esize, src_stride);
        auto d = dst.format(i * dst_stride_bytes, ngen::DataType::invalid,
                esize, dst_stride);
        // Float pipe has some register regioning limitations. If mov is not
        // allowed then fix regioning by switching to integer pipe which has
        // less limitations.
        if (esize > 1 && s.hs() != 0 && (src_xf || dst_xf)
                && s.offset() != d.offset()) {
            bool ok = false;
            bool s_half_grf_aligned = (src_hf || src_bf)
                    && utils::one_of(s.byte_offset(), 0, grf_size / 2);
            bool d_half_grf_aligned = (dst_hf || dst_bf)
                    && utils::one_of(d.byte_offset(), 0, grf_size / 2);
            if (dst_f && d.offset() == 0 && s_half_grf_aligned) ok = true;
            if (src_f && s.offset() == 0 && d_half_grf_aligned) ok = true;
            if (!ok) {
                auto i_type = to_ngen(type_t::u(ngen::getBytes(src_type) * 8));
                s = s.reinterpret(i_type);
                align_src_dst_offset(host, scope, esize, d, s);
                s = s.reinterpret(src_type);
            }
        }
        host->emov(esize, d, s);
    }
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const reg_buf_data_t &dst,
        reg_buf_data_t &src) {
    int src_stride = src.hs();
    // src is broadcasted, no need to align, return.
    if (src_stride == 0) return;

    int src_type_size = ngen::getBytes(src.type());
    int src_off = src.offset();
    int dst_off = dst.offset();
    // src is aligned with dst, return.
    if (src_off == dst_off) return;

    int esize = mod.getExecSize();
    int grf_size = ngen::GRF::bytes(scope.hw());
    int src_size = std::max(src_type_size * esize * src_stride, src_type_size);

    auto new_src = scope.alloc_reg_buf_data(
            utils::div_up(src_size + dst_off * src_type_size, grf_size));
    new_src = new_src.format(
            dst_off * src_type_size, src.type(), esize, src_stride);
    emit_reorder_1d_tile(scope.hw(), host, scope, esize, src, src_stride,
            new_src, src_stride);
    src = new_src;
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const reg_buf_data_t &dst,
        reg_buf_data_t &src0, reg_buf_data_t &src1) {
    align_src_dst_offset(host, scope, mod, dst, src0);
    align_src_dst_offset(host, scope, mod, dst, src1);
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
        ngen_operand_t &src) {
    if (!dst.is_reg_data()) return;
    if (!src.is_reg_data()) return;

    auto rd = src.reg_buf_data();
    align_src_dst_offset(host, scope, mod, dst.reg_buf_data(), rd);
    if (rd == src.reg_buf_data()) return;

    bool is_negated = src.is_negated();
    src = ngen_operand_t(rd, src.mod());
    if (is_negated) src = -src;
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
        ngen_operand_t &src0, ngen_operand_t &src1) {
    align_src_dst_offset(host, scope, mod, dst, src0);
    align_src_dst_offset(host, scope, mod, dst, src1);
}

class reorder_impl_t {
public:
    reorder_impl_t(ngen::HW hw, const reorder_t &reorder)
        : hw_(hw)
        , src_layout_(reorder.src_layout)
        , dst_layout_(reorder.dst_layout) {
        try_reinterpret_to_wider_type(src_layout_, dst_layout_);

        // Pure bf moves are not supported.
        if (utils::everyone_is(
                    type_t::bf16(), src_layout_.type(), dst_layout_.type())) {
            src_layout_ = src_layout_.retype(type_t::u16());
            dst_layout_ = dst_layout_.retype(type_t::u16());
        }
    }

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src, const reg_buf_data_t &dst) {
        if (try_emit_2d(host, scope, src, dst)) return;
        emit_1d(host, scope, src, dst);
    }

private:
    template <typename GeneratorT>
    void emit_1d(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src_rd, const reg_buf_data_t &dst_rd) {
        int src_stride;
        int dst_stride;
        auto tile = find_max_tile_with_fixed_stride(
                src_layout_, dst_layout_, src_stride, dst_stride);

        int tile_elems = int(tile.elems());
        auto &src_type = src_layout_.type();
        auto &dst_type = dst_layout_.type();
        dst_layout_.for_each_tile(tile, [&](const std::vector<dim_t> &start) {
            int src_off = int(src_layout_(start) * src_type.size());
            int dst_off = int(dst_layout_(start) * dst_type.size());
            auto sub_src = src_rd.format(src_off, to_ngen(src_type), 1);
            auto sub_dst = dst_rd.format(dst_off, to_ngen(dst_type), 1);

            ngen_register_scope_t tile_scope(scope.register_allocator());
            emit_reorder_1d_tile(hw_, host, tile_scope, tile_elems, sub_src,
                    src_stride, sub_dst, dst_stride);
        });
    }

    static tensor_t find_max_2d_dense_tile(const layout_t &a_layout,
            const layout_t &b_layout, dim_t _max_elems) {
        dim_t max_elems = _max_elems;
        for (auto &l : {&a_layout, &b_layout}) {
            dim_t stride = 1;
            dim_t elems = 1;
            int non_one_ndims = 0;
            std::vector<bool> seen(l->ndims());
            for (auto &b : l->blocks()) {
                // Tile is not dense anymore, break.
                if (dim_t(b.stride) != stride) break;
                stride = dim_t(b.stride) * b.block;

                if (b.block == 1) continue;
                if (!seen[b.dim_idx]) {
                    seen[b.dim_idx] = true;
                    non_one_ndims++;
                }
                // Tile is not 2D anymore, break.
                if (non_one_ndims > 2) break;
                elems *= b.block;
            }
            max_elems = std::min(max_elems, elems);
        }
        return a_layout.split_into_max_tile(max_elems, /*is_dense=*/true);
    }

    template <typename GeneratorT>
    bool try_emit_2d(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src_rd, const reg_buf_data_t &dst_rd) {
        if (src_layout_.type() != dst_layout_.type()) return false;
        if (!src_layout_.is_dense()) return false;
        if (!dst_layout_.is_dense()) return false;

        int max_tile_size = 512;
        int max_tile_elems = max_tile_size / src_layout_.type().size();
        auto tile = find_max_2d_dense_tile(
                src_layout_, dst_layout_, max_tile_elems);

        // Couldn't find tile, 2D reorder is not supported.
        if (tile.is_empty()) return false;

        auto src_tile_layout = src_layout_.map(tile);
        auto dst_tile_layout = dst_layout_.map(tile);
        if (!dst_tile_layout.is_dense()) return false;

        // Set layout offset to 0 since the offset is handled by fixing up the
        // register input to try_emit_2d_impl
        src_tile_layout.set_offset(0);
        dst_tile_layout.set_offset(0);

        bool ok = true;
        auto type = to_ngen(src_layout_.type());
        src_layout_.for_each_tile(tile, [&](const std::vector<dim_t> &start) {
            auto src_off = src_layout_.offset_in_bytes<dim_t>(start);
            auto dst_off = dst_layout_.offset_in_bytes<dim_t>(start);
            auto src_tile_rd = src_rd.format(int(src_off), type);
            auto dst_tile_rd = dst_rd.format(int(dst_off), type);

            ngen_register_scope_t tile_scope(scope.register_allocator());
            ok &= try_emit_2d_impl(host, tile_scope, src_tile_layout,
                    dst_tile_layout, src_tile_rd, dst_tile_rd);
        });
        return ok;
    }

    template <typename GeneratorT>
    bool try_emit_2d_impl(GeneratorT *host, ngen_register_scope_t &scope,
            const layout_t &src_layout, const layout_t &dst_layout,
            const reg_buf_data_t &src_rd, const reg_buf_data_t &dst_rd) {
        // Try to allocate/release a temporary buffer to avoid out_of_registers
        // exception.
        const int grf_size = ngen::GRF::bytes(hw_);
        auto dummy = scope.try_alloc_range(
                utils::div_up(dst_layout.size(), grf_size));
        if (dummy.isInvalid()) {
            ir_warning() << "Can't allocate buffer for 2D reorder. Reorder "
                            "performance may be suboptimal.\n";
            return false;
        }

        // Allocation succeeded, can proceed further.
        scope.safeRelease(dummy);

        reorder_2d_impl_t r(hw_, src_layout, dst_layout);
        int tile_elems = int(r.tile().elems());
        if (tile_elems < 16 || tile_elems > 512) return false;

        r.emit(host, scope, src_rd, dst_rd);
        return true;
    }

    static tensor_t find_max_tile_with_fixed_stride(const layout_t &src,
            const layout_t &dst, int &src_stride, int &dst_stride) {
        // 1. Split layouts to have aligned blocks.
        auto a = src;
        auto b = dst;
        layout_t::align_layouts(a, b);

        // 2. Find the max innermost tile.
        auto a_blocks = a.blocks();
        auto b_blocks = b.blocks();

        std::vector<dim_t> tile_dims(a.ndims(), 1);
        src_stride = (a_blocks.empty() ? 1 : int(a_blocks[0].stride));
        dst_stride = (b_blocks.empty() ? 1 : int(b_blocks[0].stride));
        int src_cur_stride = src_stride;
        int dst_cur_stride = dst_stride;

        int min_blocks = int(std::min(a_blocks.size(), b_blocks.size()));
        for (int i = 0; i < min_blocks; i++) {
            auto &ab = a_blocks[i];
            auto &bb = b_blocks[i];
            if (ab.dim_idx != bb.dim_idx || ab.block != bb.block) break;

            // Strides are supported for the innermost block only.
            if (src_cur_stride != int(ab.stride)) break;
            if (dst_cur_stride != int(bb.stride)) break;

            src_cur_stride = int(ab.block * ab.stride);
            dst_cur_stride = int(bb.block * bb.stride);
            tile_dims[ab.dim_idx] *= ab.block;
        }
        return tensor_t(tile_dims);
    }

    ngen::HW hw_;
    layout_t src_layout_;
    layout_t dst_layout_;
};

class reduce_impl_t {
public:
    reduce_impl_t(ngen::HW hw, const reduce_t &reduce)
        : hw_(hw)
        , src_layout_(reduce.src_layout)
        , dst_layout_(reduce.dst_layout) {}

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src_rd, const reg_buf_data_t &dst_rd) {
        auto &src_type = src_layout_.type();
        auto &dst_type = dst_layout_.type();

        bool is_inplace = (src_rd.base() == dst_rd.base()
                && src_rd.byte_offset() == dst_rd.byte_offset());
        if (is_inplace) {
            ir_assert(src_type == dst_type)
                    << "Inplace operation is supported for the same type only.";
        }

        std::vector<bool> seen(src_layout_.size() * src_type.size());

        tensor_t tile = find_1d_tile();
        src_layout_.for_each_tile(
                tile, [&](const std::vector<dim_t> &src_start) {
                    auto dst_start = src_start;
                    for (int i = 0; i < dst_layout_.ndims(); i++) {
                        if (dst_layout_.dims()[i] == 1) dst_start[i] = 0;
                    }
                    int src_off = int(src_layout_(src_start) * src_type.size());
                    int dst_off = int(dst_layout_(dst_start) * dst_type.size());

                    if (is_inplace) {
                        bool same_src_dst = (dst_off == src_off);
                        if (!seen[dst_off] && !same_src_dst) {
                            ir_error_not_expected()
                                    << "Invalid inplace reduction.";
                        }
                        seen[dst_off] = true;
                        if (same_src_dst) return;
                    }

                    auto sub_src
                            = src_rd.subregister(src_off, to_ngen(src_type));
                    auto sub_dst
                            = dst_rd.subregister(dst_off, to_ngen(dst_type));
                    host->add(int(tile.elems()), sub_dst(1), sub_dst(1),
                            sub_src(1));
                });
    }

private:
    tensor_t find_1d_tile() const {
        auto a = src_layout_;
        auto b = dst_layout_;
        layout_t::align_layouts(a, b);

        ir_assert(!a.blocks().empty());
        ir_assert(!b.blocks().empty());

        auto &a0 = a.blocks()[0];
        auto &b0 = b.blocks()[0];

        ir_assert(a0.is_equal(b0)) << "Incompatible layouts for reduction.";
        ir_assert(dim_t(a0.stride) == 1) << "Reduction is not supported.";

        int grf_size = ngen::GRF::bytes(hw_);
        int a_grf_elems = grf_size / a.type().size();
        int b_grf_elems = grf_size / b.type().size();

        int min_step = std::min(a_grf_elems, b_grf_elems);
        int max_step = 2 * min_step;

        min_step = std::min(8, min_step);

        ir_assert(a0.block % min_step == 0) << "Reduction is not supported.";

        std::vector<dim_t> tile_dims(src_layout_.ndims(), 1);
        tile_dims[a0.dim_idx]
                = ir_utils::max_divisor(int(a0.block), {min_step, max_step});

        return tensor_t(tile_dims);
    }

    ngen::HW hw_;
    layout_t src_layout_;
    layout_t dst_layout_;
};

// Lowers IR to nGEN.
template <ngen::HW hw>
class ir_to_ngen_t : public ir_visitor_t {
public:
    ir_to_ngen_t(conv_kernel_t<hw> *host, const expr_binding_t &expr_binding)
        : host_(host)
        , expr_binding_(expr_binding)
        , simd_size_(host->cfg_.simd_size) {}

    ~ir_to_ngen_t() {
#ifdef GEN_CONV_DEBUG
        if (bank_conflicts_ > 0)
            ir_warning() << "Found bank conflicts: " << bank_conflicts_
                         << std::endl;
        if (bundle_conflicts_ > 0)
            ir_warning() << "Found bundle conflicts: " << bundle_conflicts_
                         << std::endl;
#endif
    }

    void _visit(const alloc_t &obj) override {
        auto scope = register_scope();
        bool do_alloc = (obj.kind == alloc_kind_t::grf);
        bool use_bc_alloc = false;
        if (do_alloc) {
            reg_buf_t rb;
            if (obj.has_attr<bank_conflict_attr_t>()) {
                rb = create_bank_conflict_allocation(obj);
                use_bc_alloc = true;
            } else {
                int grf_size = ngen::GRF::bytes(hw);
                int regs = utils::div_up(obj.size, grf_size);
                rb = scope.alloc_reg_buf(regs);
            }
            if (obj.has_attr<grf_permute_attr_t>()) {
                auto &attr = obj.get_attr<grf_permute_attr_t>();
                rb.set_grf_permutation(*attr.grf_perm);
            }
            expr_binding_.bind(obj.buf, reg_buf_data_t(rb));
        }
        visit(obj.body);
        if (do_alloc) expr_binding_.unbind(obj.buf);
        if (use_bc_alloc) release_bank_conflict_allocation(obj);
    }

    void _visit(const for_t &obj) override {
        auto scope = register_scope();
        auto var_op = scope.alloc_reg_data(obj.var.type());
        auto init_op = eval(obj.init, scope);
        auto bound_op = eval(obj.bound, scope);

        ngen::Label loop_label;
        loop_end_labels_.emplace_back();

        host_->emov(1, var_op, init_op);
        expr_binding_.bind(obj.var, var_op);
        host_->mark(loop_label);
        visit(obj.body);

        host_->mark(loop_end_labels_.back());
        loop_end_labels_.pop_back();

        host_->eadd(1, var_op, var_op, ngen::Immediate(1));
        host_->ecmp(1 | host_->lt | host_->f0[0], var_op, bound_op);
        host_->jmpi(1 | host_->f0[0], loop_label);
        expr_binding_.unbind(obj.var);
    }

    void _visit(const func_call_t &obj) override {
        auto scope = register_scope();
        auto &func = obj.func;
        if (func.is<dpas_t>()) {
            auto arg_ops = eval(obj.args, scope);
            dpas(func.as<dpas_t>(), arg_ops, obj.attr);
        } else if (func.is<mad_t>()) {
            auto arg_ops = eval(obj.args, scope);
            mad(scope, func.as<mad_t>(), arg_ops, obj.attr);
        } else if (func.is<reduce_t>()) {
            auto arg_ops = eval(obj.args, scope);
            ir_assert(obj.attr.is_empty()) << "Unexpected attribute.";
            reduce(scope, func.as<reduce_t>(), arg_ops);
        } else if (func.is<reorder_t>()) {
            auto arg_ops = eval(obj.args, scope);
            ir_assert(obj.attr.is_empty()) << "Unexpected attribute.";
            reorder(scope, func.as<reorder_t>(), reorder_t::arg_src_buf(obj),
                    arg_ops);
        } else if (func.is<send_t>()) {
            auto &send_func = func.as<send_t>();
            auto args = obj.args;
            auto &mem_buf = send_t::arg_mem_buf(args);
            auto &mask = send_t::arg_mask(args);
            // If all channels are disabled for writing, quick return.
            if (all_of(mask, expr_t(false))) {
                if (send_func.is_read()) {
                    auto reg_buf_op = eval(send_t::arg_reg_buf(args), scope);
                    zero_out_data_payload(send_func, send_func.eff_mask_count,
                            reg_buf_op.reg_buf_data());
                }
                return;
            }
            // If all channels are enabled, do not use mask.
            if (all_of(mask, expr_t(true))) mask = expr_t();
            auto arg_ops = eval(args, scope);
            send(scope, func.as<send_t>(), mem_buf, arg_ops, obj.attr);
        } else if (func.is<eltwise_t>()) {
            auto &eltwise_func = func.as<eltwise_t>();
            auto arg_ops = eval(obj.args, scope);
            eltwise(scope, eltwise_func, arg_ops);
        } else if (func.is_equal(funcs::barrier_func())) {
            barrier(obj.attr);
        } else if (func.is_equal(funcs::barrier_wait_func())) {
            barrier_wait();
        } else if (func.is_equal(funcs::signal_func())) {
            signal(obj.attr);
        } else if (func.is_equal(funcs::slm_fence_func())) {
            slm_fence(obj.attr);
        } else {
            ir_error_not_expected() << object_t(obj);
        }
    }

    void _visit(const if_t &obj) override {
        ir_assert(obj.cond.is<shuffle_t>());
        ir_assert(obj.cond.as<shuffle_t>().elems() == simd_size_);

        bool has_else = !obj.else_body.is_empty();
        auto scope = register_scope();
        auto cond_op = eval(obj.cond, scope);

        if (try_emit_if_continue(obj, cond_op)) return;

        ngen::Label l_else;
        ngen::Label l_endif;
        host_->if_(simd_size_ | cond_op.flag_register(),
                has_else ? l_else : l_endif, l_endif);
        visit(obj.body);
        if (has_else) {
            host_->else_(simd_size_, l_endif, l_endif);
            host_->mark(l_else);
            visit(obj.else_body);
        }
        host_->mark(l_endif);
        host_->endif(simd_size_);
    }

    void _visit(const let_t &obj) override {
        if (obj.value.is_empty()) {
            // External variable, must be already bound.
            ir_assert(expr_binding_.is_bound(obj.var))
                    << "Variable is not defined: " << obj.var;
            visit(obj.body);
            return;
        }

        auto scope = register_scope();
        if (is_const(obj.value) || is_shuffle_const(obj.value)
                || obj.var.type() != obj.value.type()) {
            auto &var_type = obj.var.type();
            auto var_op = scope.alloc_reg_data(var_type);
            eval(obj.value, scope, ngen_operand_t(var_op, var_type.elems()));
            expr_binding_.bind(obj.var, var_op);
        } else {
            auto value_op = eval(obj.value, scope);
            expr_binding_.bind(obj.var, value_op);
        }

        auto var_op = expr_binding_.get(obj.var);

        // At this point the scope contains allocations for temporary
        // expressions. We need to 1) query and later re-claim the allocation
        // for the let variable in a new scope and 2) release the current scope
        // allocations to reduce GRF consumption.
        ngen::GRFRange var_grf_range;
        ngen::Subregister var_sub;

        if (var_op.is_reg_data()) {
            auto var_rd = var_op.reg_data();
            var_grf_range = scope.find_grf_range(
                    var_rd.getBase(), var_rd.getByteOffset());
            var_sub = scope.find_sub(var_rd.getBase(), var_rd.getByteOffset());
        }

        // Release the current scope allocations.
        scope.clear();

        // Claim the let variable allocation.
        auto var_scope = register_scope();
        if (!var_grf_range.isInvalid()) {
            var_scope.claim(var_grf_range);
        } else if (!var_sub.isInvalid()) {
            var_scope.claim(var_sub);
        }

        visit(obj.body);
        expr_binding_.unbind(obj.var);
    }

    void _visit(const store_t &obj) override {
        auto scope = register_scope();
        auto buf_op = eval(obj.buf, scope);
        auto off = to_cpp<int>(obj.off);
        auto mask_op = eval(obj.mask, scope);

        auto &type = obj.value.type();

        int stride;
        if (obj.has_default_stride()) {
            stride = 1;
        } else {
            ir_assert(obj.stride % type.scalar().size() == 0);
            stride = obj.stride / type.scalar().size();
        }

        ngen::InstructionModifier mod = type.elems();
        if (!mask_op.is_invalid()) mod |= mask_op.flag_register_mod();
        auto dst_rbd = buf_op.reg_buf_data().format(
                off, to_ngen(type.scalar()), type.elems(), stride);
        ngen_operand_t dst(dst_rbd, mod);
        eval(obj.value, scope, dst);
    }

private:
    ngen_register_scope_t register_scope() {
        return ngen_register_scope_t(host_->ra_);
    }

#ifdef GEN_CONV_DEBUG
    void check_bank_conflicts(const ngen::InstructionModifier &mod,
            const ngen::RegData &_src0, const ngen::RegData &_src1,
            const ngen::RegData &_src2, bool is_dpas = false) {
        int esize = mod.getExecSize();
        int hw_simd = (hw >= ngen::HW::XeHPC ? 16 : 8);
        auto shift = [](const ngen::RegData &rd, int exec_off) {
            if (exec_off == 0 || rd.isNull()) return rd;
            int type_size = ngen::getBytes(rd.getType());
            int w = (exec_off % rd.getWidth());
            int h = (exec_off / rd.getWidth());
            int off = rd.getByteOffset()
                    + (w * rd.getHS() + h * rd.getVS()) * type_size;
            int grf_size = ngen::GRF::bytes(hw);
            int shifted_base = rd.getBase() + off / grf_size;
            int shifted_off = off % grf_size;
            auto ret = rd;
            ret.setBase(shifted_base);
            ret.setOffset(ir_utils::safe_divide(shifted_off, type_size));
            return ret;
        };
        for (int i = 0; i < esize; i += hw_simd) {
            auto src0 = shift(_src0, i);
            auto src1 = shift(_src1, i);
            auto src2 = shift(_src2, i);
            bool same_bank01 = ngen::Bundle::same_bank(hw, src0, src1);
            bool same_bank02 = ngen::Bundle::same_bank(hw, src0, src2);
            if (is_dpas) {
                if (same_bank02) bank_conflicts_++;
            } else {
                if (same_bank01 && same_bank02) bank_conflicts_++;
                if (ngen::Bundle::conflicts(hw, src0, src1)
                        || ngen::Bundle::conflicts(hw, src0, src2)
                        || ngen::Bundle::conflicts(hw, src1, src2)) {
                    bundle_conflicts_++;
                }
            }
        }
    }
#else
    template <typename... ArgsT>
    void check_bank_conflicts(const ArgsT &...) {}
#endif

    reg_buf_t create_bank_conflict_allocation(const alloc_t &alloc) {
        auto &bc_attr = alloc.get_attr<bank_conflict_attr_t>();
        auto it = bc_allocations_.find(bc_attr);
        if (it != bc_allocations_.end()) {
            it->second.retain();
            return it->second.get_reg_buf(alloc.buf);
        }
        auto bca = bank_conflict_allocation_t::create(
                host_->ra_, host_->regs_, bc_attr);
        if (bca.is_empty()) return {};

        auto ret = bc_allocations_.emplace(bc_attr, std::move(bca));
        return ret.first->second.get_reg_buf(alloc.buf);
    }

    void release_bank_conflict_allocation(const alloc_t &alloc) {
        auto &bc_attr = alloc.get_attr<bank_conflict_attr_t>();
        auto it = bc_allocations_.find(bc_attr);
        ir_assert(it != bc_allocations_.end());
        it->second.release(alloc.buf);
        if (it->second.refs() == 0) bc_allocations_.erase(bc_attr);
    }

    void signal(const func_call_attr_t &attr) {
        ngen::InstructionModifier mod;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        host_->barriermsg(mod, host_->signal_header_);
    }

    void barrier_wait() { host_->barrierwait(); }

    void slm_fence(const func_call_attr_t &attr) {
        auto scope = register_scope();
        auto tmp = scope.alloc();
        ngen::InstructionModifier mod;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);

        const int dwords = ngen::GRF::bytes(hw) / sizeof(int32_t);
        host_->slmfence(mod, tmp, host_->r0);
        host_->template mov<int32_t>(dwords, host_->null, tmp);
    }

    void barrier(const func_call_attr_t &attr) {
        auto scope = register_scope();
        auto tmp = scope.alloc();
        ngen::InstructionModifier mod;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);

        const int dwords = ngen::GRF::bytes(hw) / sizeof(int32_t);
        host_->slmfence(mod, tmp, host_->r0);
        host_->template mov<int32_t>(dwords, host_->null, tmp);
        host_->barriermsg(mod, host_->signal_header_);
        host_->barrierwait();
    }

    void dpas(const dpas_t &dpas_func, const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        auto dst = dpas_t::arg_dst(args).reg_buf_data();
        auto src1 = dpas_t::arg_src1(args).reg_buf_data();
        auto src2 = dpas_t::arg_src2(args).reg_buf_data();

        if (dpas_func.is_dpasw) dst = dst.unpermute();

        int simd = dpas_func.simd_size;

        ngen::RegData src0;
        auto &src0_op = dpas_t::arg_src0(args);
        if (!src0_op.is_immediate()) {
            auto src0_rbd = src0_op.reg_buf_data().format(
                    0, to_ngen(dpas_func.dst_type), simd, 1);
            if (dpas_func.is_dpasw) src0_rbd = src0_rbd.unpermute();
            src0 = src0_rbd;
        } else {
            ir_assert(src0_op.is_immediate());
            ir_assert(to_cpp<int32_t>(src0_op.immediate()) == 0);
            src0 = host_->null.retype(to_ngen(dpas_func.dst_type));
        }

        dst = dst.format(0, to_ngen(dpas_func.dst_type), simd, 1);
        src1 = src1.format(0, to_ngen(dpas_func.src1_type), simd, 1);
        int src2_width = (dpas_func.is_dp4a() ? 1 : simd);
        int src2_stride = (dpas_func.is_dp4a() ? 0 : 1);
        src2 = src2.format(
                0, to_ngen(dpas_func.src2_type), src2_width, src2_stride);

        ngen::InstructionModifier mod = simd_size_;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        check_bank_conflicts(mod, src0, src1, src2, /*is_dpas=*/true);
        if (dpas_func.is_dpasw) {
            host_->dpasw(mod, dpas_func.sdepth, dpas_func.rcount, dst, src0,
                    src1, src2);
        } else if (dpas_func.is_dp4a()) {
            if (src0.isNull()) {
                host_->mov(mod, dst, 0);
                host_->dp4a(mod, dst, dst, src1, src2);
            } else {
                host_->dp4a(mod, dst, src0, src1, src2);
            }
        } else {
            host_->dpas(mod, dpas_func.sdepth, dpas_func.rcount, dst, src0,
                    src1, src2);
        }
    }

    void mad(ngen_register_scope_t &scope, const mad_t &mad_func,
            const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        auto dst = mad_t::arg_dst(args).reg_buf_data();
        auto src1 = mad_t::arg_src1(args).reg_buf_data();
        auto src2 = mad_t::arg_src2(args).reg_buf_data();

        ngen::RegData src0;
        auto &src0_op = mad_t::arg_src0(args);
        if (!src0_op.is_immediate()) {
            src0 = src0_op.reg_buf_data()
                           .format(0, to_ngen(mad_func.dst_type),
                                   mad_func.simd_size)
                           .reg_data();
        } else {
            ir_assert(src0_op.is_immediate());
            ir_assert(to_cpp<int32_t>(src0_op.immediate()) == 0);
            src0 = host_->null;
            src0.setType(to_ngen(mad_func.dst_type));
        }

        dst = dst.format(0, to_ngen(mad_func.dst_type), mad_func.simd_size);

        int src1_width = (mad_func.src1_stride == 0 ? 1 : mad_func.simd_size);
        int src2_width = (mad_func.src2_stride == 0 ? 1 : mad_func.simd_size);
        src1 = src1.format(0, to_ngen(mad_func.src1_type), src1_width,
                mad_func.src1_stride);
        src2 = src2.format(0, to_ngen(mad_func.src2_type), src2_width,
                mad_func.src2_stride);

        ngen::InstructionModifier mod = simd_size_;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);

        check_bank_conflicts(mod, src0, src1, src2, /*is_dpas=*/false);
        if (src0.isNull()) {
            host_->mul(mod, dst, src1, src2);
        } else {
            ir_assert(dst.byte_offset() == src0.getByteOffset())
                    << "dst/src0 must be aligned to the same GRF offset.";
            auto _src1 = src1;
            auto _src2 = src2;
            align_src_dst_offset(host_, scope, mod, dst, _src1, _src2);
            host_->mad(mod, dst, src0, _src1, _src2);
        }
    }

    void reduce(ngen_register_scope_t &scope, const reduce_t &reduce_func,
            const std::vector<ngen_operand_t> &args) {
        auto &src_op = reduce_t::arg_src_buf(args);
        auto &dst_op = reduce_t::arg_dst_buf(args);

        reduce_impl_t reduce_impl(hw, reduce_func);
        reduce_impl.emit(
                host_, scope, src_op.reg_buf_data(), dst_op.reg_buf_data());
    }

    void reorder(ngen_register_scope_t &scope, const reorder_t &reorder_func,
            const expr_t &src_buf, const std::vector<ngen_operand_t> &args) {
        auto &src_op = reorder_t::arg_src_buf(args);
        auto &dst_op = reorder_t::arg_dst_buf(args);

        reorder_impl_t reorder_impl(hw, reorder_func);
        reorder_impl.emit(
                host_, scope, src_op.reg_buf_data(), dst_op.reg_buf_data());
    }

    void zero_out_data_payload(const send_t &send_func,
            const ngen::InstructionModifier &_mod, const reg_buf_data_t &rd) {
        bool is_per_slot = (send_func.mask_count() > 1);

        auto get_modifier = [&](int exec_size) {
            if (is_per_slot) {
                ir_assert(_mod.getExecSize() == exec_size);
                auto mod = _mod;
                mod = ~mod;
                mod.setSWSB({});
                return mod;
            }
            return ngen::InstructionModifier(exec_size);
        };

        int ud_size = sizeof(uint32_t);
        int send_size = send_func.register_size();
        int grf_size = ngen::GRF::bytes(hw);
        int step = (is_per_slot ? send_func.mask_count() * ud_size
                                : 2 * grf_size);
        for (int i = 0; i < send_size; i += step) {
            int exec_size;
            if (is_per_slot) {
                exec_size = send_func.eff_mask_count;
            } else {
                exec_size = std::min(step, send_size - i) / ud_size;
            }
            auto sub_rd_mov
                    = rd.format(i, ngen::DataType::f, exec_size).reg_data();
            host_->emov(
                    get_modifier(exec_size), sub_rd_mov, ngen::Immediate(0.0f));
        }
    }

    void send(ngen_register_scope_t &scope, const send_t &send_func,
            const expr_t &mem_buf, const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        send_impl_t spec_impl(hw, send_func);
        auto &mem_off_op = send_t::arg_mem_off(args);
        auto &reg_buf_op = send_t::arg_reg_buf(args);
        auto &mask_op = send_t::arg_mask(args);

        ngen::RegData mem_buf_rd;
        int surf_bti = -1;
        switch (send_func.address_model) {
            case ngen_proxy::AddressModel::ModelSLM: break;
            case ngen_proxy::AddressModel::ModelBTS: {
                auto &buf_name = mem_buf.as<var_t>().name;
                surf_bti = host_->getArgumentSurface(buf_name);
                break;
            }
            case ngen_proxy::AddressModel::ModelA64: {
                auto &mem_buf_op = send_t::arg_mem_buf(args);
                mem_buf_rd = mem_buf_op.reg_data();
                break;
            }
            default: ir_error_not_expected();
        }
        ngen::InstructionModifier mod = send_func.eff_mask_count;
        ir_assert(math::is_pow2(mod.getExecSize()));
        if (!attr.is_empty())
            mod |= to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        if (!mask_op.is_invalid()) mod |= mask_op.flag_register_mod();

        // Zero-out inactive channels.
        if (send_func.is_read() && !send_func.is_prefetch
                && mod.getPredCtrl() != ngen::PredCtrl::None) {
            zero_out_data_payload(send_func, mod, reg_buf_op.reg_buf_data());
        }

        // Emit send instruction.
        auto rd = send_maybe_make_dense_payload(scope, send_func, reg_buf_op);
        spec_impl.emit(host_, scope, mod, mem_buf_rd, surf_bti,
                mem_off_op.reg_data(), rd);
    }

    ngen::RegData send_maybe_make_dense_payload(ngen_register_scope_t &scope,
            const send_t &send_func, const ngen_operand_t &op_buf) const {
        if (send_func.is_prefetch) return ngen::RegData(host_->null);

        auto &buf = op_buf.reg_buf_data();
        int size = send_func.register_size();
        bool is_dense = buf.is_dense(size);
        if (is_dense) return buf.reg_data();

        if (send_func.is_read()) {
            ir_error_not_expected()
                    << "Expected dense GRF region for load message.";
            return ngen::RegData();
        }

        ir_assert(send_func.is_write());

        // Reorder buffer to a dense buffer for store.
        int grf_size = ngen::GRF::bytes(hw);
        int regs = utils::div_up(size, grf_size);

        auto tmp = scope.alloc_range(regs);

        int dwords = ngen::GRF::bytes(hw) / sizeof(int32_t);
        int max_step = 2;
        for (int i = 0; i < regs; i += max_step) {
            int step = std::min(max_step, regs - i);
            int esize = step * dwords;
            auto src = buf.subregister(i * grf_size, ngen::DataType::ud)(1);
            auto dst = tmp[i].ud(0)(1);
            host_->emov(esize, dst, src);
        }
        return tmp[0];
    }

    void eltwise(ngen_register_scope_t &scope, const eltwise_t &func,
            const std::vector<ngen_operand_t> &args) {
        int elems = to_cpp<int>(hw, eltwise_t::arg_elems(args));
        auto &data_op = eltwise_t::arg_data(args);
        auto data_rd = data_op.reg_buf_data();

        int grf_size = ngen::GRF::bytes(hw);
        ir_assert(elems * sizeof(float) % grf_size == 0)
                << "Partial GRF updates are not supported.";
        ir_assert(data_rd.byte_offset() == 0)
                << "Data must be aligned to GRF boundary.";

        jit_eltwise_injector_f32<hw> inj(
                host_, func.alg_kind, func.alpha, func.beta, func.scale);
        auto scratch = scope.alloc_range(inj.preferred_scratch_regs());
        inj.set_scratch(scratch);
        inj.prepare();

        int regs = elems * sizeof(float) / grf_size;
        int step = 2;
        for (int i = 0; i < regs; i += step) {
            int cur_regs = std::min(step, regs - i);
            auto cur_rd = data_rd.format(i * grf_size);
            inj.compute(ngen::GRFRange(cur_rd.base(), cur_regs));
        }
    }

    bool try_emit_if_continue(const if_t &obj, const ngen_operand_t &cond_op) {
        if (!obj.else_body.is_empty()) return false;
        auto *call = obj.body.as_ptr<func_call_t>();
        if (!call) return false;
        if (!call->func.is_equal(funcs::continue_func())) return false;

        ir_assert(!loop_end_labels_.empty())
                << "Can't emit continue: no label found.";
        host_->jmpi(1 | cond_op.flag_register(), loop_end_labels_.back());
        return true;
    }

    ngen_operand_t eval(const expr_t &e, ngen_register_scope_t &scope,
            const ngen_operand_t &dst_operand = ngen_operand_t()) {
        expr_evaluator_t<hw> expr_evaluator(host_, expr_binding_, scope);
        return expr_evaluator.eval(e, dst_operand);
    }

    std::vector<ngen_operand_t> eval(
            const std::vector<expr_t> &exprs, ngen_register_scope_t &scope) {
        expr_evaluator_t<hw> expr_evaluator(host_, expr_binding_, scope);
        return expr_evaluator.eval(exprs);
    }

    conv_kernel_t<hw> *host_;
    expr_binding_t expr_binding_;
    int simd_size_;

    std::vector<ngen::Label> loop_end_labels_;

#ifdef GEN_CONV_DEBUG
    int bank_conflicts_ = 0;
    int bundle_conflicts_ = 0;
#endif

    object_map_t<alloc_attr_t, bank_conflict_allocation_t> bc_allocations_;
};

template <ngen::HW hw>
conv_kernel_t<hw>::conv_kernel_t(const conv_config_t &cfg,
        const convolution_pd_t *pd, const kernel_info_t &kernel_info,
        bool force_large_grf)
    : cfg_(cfg)
    , regs_(!force_large_grf ? cfg.regs : 256)
    , ra_(hw, "conv_kernel_t", reg_allocator_t::warn_all) {

    ra_.setRegisterCount(regs_);

    // XXX: BWD_W does 32x32 multiplication in the inner loop which may cause
    // hangs when using with split barrier. Switch to emulation to work around
    // the issue.
    if (cfg_.is_bwd_w) emu_strategy.emulate64 = true;

    // Build IR for the kernel.
    kernel_builder_t builder(cfg, pd, kernel_info);
    stmt_t body = builder.stmt();

    alloc_manager_t alloc_mgr(body);

    setup_interface(body, kernel_info);

    setDefaultNoMask();
    setDefaultAutoSWSB(true);

    prologue();

    // Claim registers.
    ra_.claim(r0);
    for (int i = 0; i < 3; i++)
        ra_.claim(getLocalID(i));

    for (int i = 0; i < kernel_info.nargs(); i++) {
        ra_.claim(getArgument(kernel_info.arg_name(i)));
    }

    if (emu_strategy.emulate64) {
        emu_state.temp[0] = ra_.alloc();
        emu_state.temp[1] = ra_.alloc();
    }
    // Enable IEEE f32 -> s32 rounding and f32/f16 denormals.
    or_(1, cr0, cr0, uint16_t(0x1480));

    // Allocate and initialize signal header for future use.
    signal_header_ = ra_.alloc();
    barrierheader(signal_header_);

    // Bind "external" variables.
    expr_binding_t expr_binding(hw);

    // Bind grid indices.
    int r0_sub_idxs[] = {1, 6, 7};
    for (int i = 0; i < 3; i++) {
        auto tmp = ra_.alloc_sub<int32_t>();
        mov(1, tmp, r0.ud(r0_sub_idxs[i]));
        expr_binding.bind(builder.kernel_grid_idx(i), tmp);
    }

    // Bind local IDs.
    for (int i = 0; i < 3; i++) {
        expr_binding.bind(builder.local_id(i), getLocalID(i).uw(0));
    }

    // Bind arguments.
    for (int i = 0; i < kernel_info.nargs(); i++) {
        auto &arg_var = kernel_info.arg_var(i);
        auto &name = kernel_info.arg_name(i);
        if (arg_var.type().is_ptr()) {
            auto alloc_buf = alloc_mgr.find_buffer(name);
            ir_assert(alloc_buf.is_same(arg_var));
        }
        expr_binding.bind(arg_var, getArgument(name));
    }

    // Bind SLM buffer (SLM loads/stores use 0-based offsets).
    auto slm_buf = alloc_mgr.find_buffer("slm", /*allow_empty=*/true);
    if (!slm_buf.is_empty()) { expr_binding.bind(slm_buf, to_ngen(expr_t(0))); }

    // Generate assembly from IR.
    ir_to_ngen_t<hw> visitor(this, expr_binding);
    visitor.visit(body);

    epilogue();
    pad_kernel();

#ifdef GEN_CONV_DEBUG
    if (ra_.get_peak_grf_usage() > cfg_.estimated_peak_grf_usage) {
        ir_warning()
                << "conv_kernel_t register usage underestimated: estimate = "
                << cfg_.estimated_peak_grf_usage
                << ", actual = " << ra_.get_peak_grf_usage() << "\n";
    }
#endif
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
