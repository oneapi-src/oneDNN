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

#ifndef GPU_JIT_CONV_MESSAGE_SUPPORT_HPP
#define GPU_JIT_CONV_MESSAGE_SUPPORT_HPP

#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/conv/utils.hpp"
#include "gpu/jit/ngen/ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Send operation kind.
enum class send_op_t {
    atomic_fadd,
    load,
    prefetch,
    store,
};

// Send address model.
enum class send_address_t {
    a64,
    bts,
    slm,
};

// Function representing send messages.
class send_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(send_t, func_impl_t)

    static func_t make(ngen::HW hw, send_op_t op, send_address_t address,
            const type_t &type, int slots) {
        return func_t(new send_t(hw, op, address, type, slots));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return (hw == other.hw) && (op == other.op)
                && (address == other.address) && (type == other.type)
                && (slots == other.slots);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(hw, op, address, type, slots);
    }

    std::string str() const override {
        std::ostringstream oss;
        switch (op) {
            case send_op_t::atomic_fadd: oss << "atomic_fadd"; break;
            case send_op_t::load: oss << "load"; break;
            case send_op_t::prefetch: oss << "prefetch"; break;
            case send_op_t::store: oss << "store"; break;
            default: ir_error_not_expected();
        }
        oss << ".";
        if (is_scattered()) oss << slots << "x";
        oss << type.str();
        return oss.str();
    }

    IR_DEFINE_ARG_GET(mem_buf, 0)
    IR_DEFINE_ARG_GET(mem_off, 1)
    IR_DEFINE_ARG_GET(reg_buf, 2)
    IR_DEFINE_ARG_GET(mask, 3)

    stmt_t operator()(const expr_t &mem_buf, const expr_t &mem_off,
            const expr_t &reg_buf, const expr_t &mask) const {
        return call({mem_buf, mem_off, reg_buf, mask});
    }

    bool is_atomic() const { return op == send_op_t::atomic_fadd; }
    bool is_load() const { return op == send_op_t::load; }
    bool is_prefetch() const { return op == send_op_t::prefetch; }
    bool is_store() const { return op == send_op_t::store; }
    bool is_a64() const { return address == send_address_t::a64; }
    bool is_bts() const { return address == send_address_t::bts; }
    bool is_slm() const { return address == send_address_t::slm; }

    bool is_block() const {
        return utils::one_of(
                type.kind(), type_kind_t::oword, type_kind_t::hword);
    }

    bool is_scattered() const { return !is_block(); }

    // Size of memory (global memory or SLM) to access.
    int access_size() const { return type.size() * slots; }

    int payload_type_stride() const {
        if (type.kind() == type_kind_t::byte) return 4;
        return type.size();
    }

    // Full size of payload GRF buffer for this message. Buffer may be strided
    // and/or require GRF boundary round-up.
    int payload_size() const {
        int sz = payload_type_stride() * slots;
        return utils::rnd_up(sz, grf_size());
    }

    int alignment() const {
        if (is_block()) return type.scalar().size();
        return 1;
    }

    int mask_size() const {
        if (is_block()) {
            // Block messages use SIMT1 execution mask (one mask per message)
            // on XeHPC+.
            if (is_xe_hpc_plus()) return type.size();
            return 4;
        }

        if (is_scattered()) return type.size();

        ir_error_not_expected();
        return 0;
    }

    int nmasks() const {
        int masks = ir_utils::safe_divide(type.size() * slots, mask_size());
        if (masks > 16) {
            ir_assert(is_block())
                    << "Round-robin masking applies to block messages only.";
            ir_assert(masks % 16 == 0);
            masks = 16;
        }
        return masks;
    }

    int address_size() const { return is_a64() ? 8 : 4; }

    type_t address_type(bool is_signed = false, int elems = 1) const {
        int bits = address_size() * 8;
        return is_signed ? type_t::s(bits, elems) : type_t::u(bits, elems);
    }

    // Size of header in bytes.
    int header_size() const {
        return utils::rnd_up(address_size() * slots, grf_size());
    }

    // Generates a statement to store (and maybe convert) the offset to the
    // message header according to the message description.
    stmt_t create_offset_store(const expr_t &header_buf, const expr_t &mem_buf,
            const expr_t &mem_off, bool is_signed_offset = false) const;

    bool is_supported() const;

    static std::vector<func_t> get_all(ngen::HW hw, send_op_t op,
            send_address_t address, const type_t &mem_type);

    ngen::HW hw;
    send_op_t op;
    send_address_t address;
    type_t type;
    int slots;

private:
    int grf_size() const { return ngen::GRF::bytes(hw); }

    bool is_xe_hpc_plus() const { return hw >= ngen::HW::XeHPC; }

    send_t(ngen::HW hw, send_op_t op, send_address_t address,
            const type_t &type, int slots)
        : hw(hw), op(op), address(address), type(type), slots(slots) {}
};

class memory_walker_t;
class layout_walker_t;

// Generates loads or stores to move data between memory (global or SLM) and
// GRF. Memory view is a parameter. GRF payload layout is deduced
// automatically, according to the decomposition into messages.
class access_builder_t {
public:
    access_builder_t(ngen::HW hw, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const view_t &mem_view,
            const expr_t &mem_buf, const expr_t &reg_buf, send_op_t send_op,
            send_address_t send_address);
    access_builder_t(access_builder_t &&);
    ~access_builder_t();

    const layout_t &reg_layout() const { return reg_layout_; }
    int reg_buf_size() const {
        return utils::rnd_up(reg_layout_.size(), grf_size());
    }
    const stmt_t &stmt() const { return stmt_; }

    std::string str() const {
        std::ostringstream oss;
        oss << "Memory view:          " << mem_view_ << std::endl;
        oss << "Register layout:      " << reg_layout_ << std::endl;
        oss << "Register buffer:      " << reg_buf_ << std::endl;
        oss << "Register buffer size: " << reg_buf_size() << " ("
            << reg_buf_size() / grf_size() << " regs)" << std::endl;
        oss << "Statement:            " << std::endl << stmt_;
        return oss.str();
    }

private:
    void build();
    bool try_build(const layout_t &try_layout);
    std::vector<layout_t> candidate_payload_layouts() const;
    stmt_t create_send_stmt(const send_t &send);
    int grf_size() const { return ngen::GRF::bytes(hw_); }

    ngen::HW hw_;
    view_t mem_view_;
    expr_t mem_buf_;
    expr_t reg_buf_;
    send_op_t send_op_;
    send_address_t send_address_;

    type_t mem_type_;

    std::unique_ptr<memory_walker_t> mem_walker_;
    std::unique_ptr<layout_walker_t> reg_layout_walker_;

    layout_t reg_layout_;
    stmt_t stmt_;
};

inline access_builder_t make_access_builder(ngen::HW hw, ir_context_t &ir_ctx,
        const constraint_set_t &cset, const view_t &mem_view,
        const expr_t &mem_buf, const expr_t &reg_buf, send_op_t send_op,
        send_address_t send_address) {
    return access_builder_t(hw, ir_ctx, cset, mem_view, mem_buf, reg_buf,
            send_op, send_address);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
