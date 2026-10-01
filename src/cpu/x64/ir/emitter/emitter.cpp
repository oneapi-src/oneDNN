/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <vector>
#include <unordered_map>

#include "cpu/x64/ir/emitter/backend_avx2.hpp"
#include "cpu/x64/ir/emitter/backend_avx512.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/postops_injector.hpp"
#include "cpu/x64/utils/jit_regops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

template <typename backend_t>
void emit(backend_t &be, const ir_t &ir, const reg_alloc_result_t &alloc,
        const reg_config_t &rc, data_section_t &data,
        postops_injector_t *postops) {

    // The backend holds the generator.
    jit_generator_t &gen = be.gen();

    // One label per loop, keyed by the `loop_begin` instruction index. A
    // `loop_end` jumps back to `labels[op.match]`. A map, so only loops get an
    // entry.
    std::unordered_map<int, Xbyak::Label> labels;
    // One label per `label` id, for `label`/`jmp`/`jz`.
    std::vector<Xbyak::Label> label_id_to_label(ir.n_labels());

    // if vreg needs to be spilled
    auto spilled
            = [&](vreg_t vr) { return alloc.assignments[(int)vr].spilled; };
    // get a physical register from a virtual one
    auto phys = [&](vreg_t vr) { return alloc.assignments[(int)vr].phys; };
    // get a stack slot for a virtual register
    auto slot = [&](vreg_t vr) {
        return gen.ptr[gen.rsp + (int)alloc.assignments[(int)vr].slot];
    };
    // stack slot byte offset for a vec spill
    auto slot_off
            = [&](vreg_t vr) { return (int)alloc.assignments[(int)vr].slot; };
    // Data type of a vec vreg.
    auto dt_of = [&](vreg_t vr) { return ir.vreg_info()[(int)vr].dt; };

    // Temps of the operation being lowered (see `temp_reg_t`).
    const std::vector<temp_reg_t> *op_temps = nullptr;

    // Temp that holds the spilled `vr` during the current operation. There is
    // none when the operation exceeds `max_temps_per_op` or needs more
    // registers than the file holds. The kernel then cannot be emitted.
    auto temp_of = [&](vreg_t vr) -> int {
        for (const temp_reg_t &t : *op_temps)
            if (t.vreg == vr) return t.phys;
        JIT_ASSERT_RET(!"emit: spilled operand has no temp register", 0);
        return 0;
    };

    // Physical register that holds `vr` during the current operation, without
    // a reload. It is the temp when `vr` is spilled.
    auto reg_of
            = [&](vreg_t vr) { return spilled(vr) ? temp_of(vr) : phys(vr); };

    // Move a spilled vec value between its stack slot and a register, as a
    // vector load/store against the stack frame (`rsp`).
    const int rsp_idx = gen.rsp.getIdx();
    auto spill_reload
            = [&](vreg_t vr, int p) { be.vload_raw(p, rsp_idx, slot_off(vr)); };
    auto spill_store = [&](vreg_t vr, int p) {
        be.vstore_raw(rsp_idx, slot_off(vr), p);
    };

    // Resolve a virtual register that an instruction READS (use) to a
    // concrete physical register, hiding whether the allocator spilled it:
    //   - not spilled: the value is already in a physical register, so just
    //     return that register (no extra instruction).
    //   - spilled: the value lives on the stack slot, so emit a reload into its
    //     temp and return the temp.
    // Each spilled operand of an operation has its own temp, so an instruction
    // with several spilled operands reloads each into a different register.
    // These helpers handle only reads. Writing a spilled result back is done
    // by the defining instruction (compute into the temp, then store to the
    // slot).
    //
    // gpr reloads are ISA-neutral (a plain `mov`), so `gpr_use` emits them
    // directly. A spilled vec source is reloaded through the backend, since the
    // reload instruction is ISA-specific. The `vec_use` returns a physical
    // index rather than a typed register.
    auto gpr_use = [&](vreg_t vr) -> Xbyak::Reg64 {
        const Xbyak::Reg64 r(reg_of(vr));
        // reload the spilled gpr from its stack slot
        if (spilled(vr)) gen.mov(r, slot(vr));
        return r;
    };

    auto vec_use = [&](vreg_t vr) -> int {
        const int r = reg_of(vr);
        // reload the spilled vector register from its stack slot
        if (spilled(vr)) spill_reload(vr, r);
        return r;
    };

    // Lower each IR instruction. Spilled operands are handled as follows:
    //
    // - Inputs that an instruction reads are accessed through gpr_use/vec_use.
    //   These return the register directly, or reload the value from its spill
    //   slot into its temp if needed.
    //
    // - The output that an instruction writes is handled separately inside each
    //   case. If the destination is spilled, we reload it first (for
    //   read-modify-write operations), perform the operation in its temp, and
    //   then store the result back to its spill slot.
    //
    // The temps of an operation differ from each other and from every register
    // that holds a value live across the operation, so a reload never clobbers
    // another operand.

    // Fixed registers the injector reads are set up once, ahead of the IR,
    // rather than at every `inject_postops` operation. The pattern does not
    // change between operations, and an operation inside a loop would otherwise
    // rebuild it on every iteration.
    if (postops) postops->init(gen);

    for (int i = 0; i < ir.n_ops(); i++) {
        const op_t &op = ir.ops()[i];
        op_temps = &alloc.temps[i];
        switch (op.kind) {
            // General-purpose register ops. ISA-neutral, emitted directly.
            case op_kind_t::mov_imm: {
                const Xbyak::Reg64 d(reg_of(op.dst));
                gen.mov(d, op.imm);
                if (spilled(op.dst)) gen.mov(slot(op.dst), d);
                break;
            }
            case op_kind_t::mov_reg: {
                Xbyak::Reg64 s = gpr_use(op.s0);
                if (!spilled(op.dst))
                    gen.mov(Xbyak::Reg64(phys(op.dst)), s);
                else
                    gen.mov(slot(op.dst), s);
                break;
            }
            case op_kind_t::add_imm: {
                const Xbyak::Reg64 d = gpr_use(op.dst);
                gen.add(d, op.imm);
                if (spilled(op.dst)) gen.mov(slot(op.dst), d);
                break;
            }
            case op_kind_t::add_reg: {
                const Xbyak::Reg64 s = gpr_use(op.s0);
                const Xbyak::Reg64 d = gpr_use(op.dst);
                gen.add(d, s);
                if (spilled(op.dst)) gen.mov(slot(op.dst), d);
                break;
            }
            case op_kind_t::load: {
                Xbyak::Reg64 base = op.mem.is_param ? Xbyak::Reg64(rc.param_reg)
                                                    : gpr_use(op.mem.base);
                const Xbyak::Reg64 d(reg_of(op.dst));
                gen.mov(d, gen.ptr[base + (int)op.mem.disp]);
                if (spilled(op.dst)) gen.mov(slot(op.dst), d);
                break;
            }

            // Vector ops. Emitting the instruction is the backend's job. A
            // spilled dst is always stored back to its slot after the op.
            // A read-modify-write (`rmw`) op also reloads a spilled dst before
            // the op. An op that overwrites dst does not.
            case op_kind_t::vzero: { // overwrites dst
                int d = reg_of(op.dst);
                be.vzero(d);
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vload: { // overwrites dst
                int base = gpr_use(op.mem.base).getIdx();
                int d = reg_of(op.dst);
                be.vload(d, base, op.mem.disp, op.mem_dt, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vstore: {
                int base = gpr_use(op.mem.base).getIdx();
                int s = vec_use(op.s0);
                be.vstore(base, op.mem.disp, s, op.mem_dt, dt_of(op.s0));
                break;
            }
            case op_kind_t::vload_scalar: { // overwrites dst
                int base = gpr_use(op.mem.base).getIdx();
                int d = reg_of(op.dst);
                be.vload_scalar(d, base, op.mem.disp, op.mem_dt, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vstore_scalar: {
                int base = gpr_use(op.mem.base).getIdx();
                int s = vec_use(op.s0);
                be.vstore_scalar(base, op.mem.disp, s, op.mem_dt, dt_of(op.s0));
                break;
            }
            case op_kind_t::vload_bcast: { // overwrites dst
                int base = gpr_use(op.mem.base).getIdx();
                int d = reg_of(op.dst);
                be.vload_bcast(d, base, op.mem.disp, op.mem_dt, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vdot: { // rmw: reads and writes dst
                int d = vec_use(op.dst);
                int a = vec_use(op.s0);
                int b = vec_use(op.s1);
                be.vdot(d, a, b, dt_of(op.s0));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vadd: { // rmw: reads and writes dst
                int d = vec_use(op.dst);
                int s = vec_use(op.s0);
                be.vadd(d, s, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vmul: { // rmw: reads and writes dst
                int d = vec_use(op.dst);
                int s = vec_use(op.s0);
                be.vmul(d, s, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vhreduce: { // reads and writes dst, overwrites ws
                int d = vec_use(op.dst);
                int ws = reg_of(op.s0);
                be.vhreduce(d, ws, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                if (spilled(op.s0)) spill_store(op.s0, ws);
                break;
            }

            // Lowers to the external JIT injector (see `postops_injector_t`).
            // Spilled operands go through their temps like those of any other
            // operation. The injector saves and restores every other register
            // it borrows.
            case op_kind_t::inject_postops: {
                const auto &args = ir.inject_postops_args()[(int)op.imm];
                std::vector<int> acc_phys;
                acc_phys.reserve(args.acc.size());
                for (vreg_t v : args.acc)
                    acc_phys.push_back(vec_use(v));
                // An eltwise-only chain has no base pointer.
                const int base_phys = args.base_ptr == vreg_t::none
                        ? -1
                        : gpr_use(args.base_ptr).getIdx();
                JIT_ASSERT(postops && "inject_postops: missing injector");
                postops->inject(acc_phys, base_phys, args.out_byte_off);
                for (size_t a = 0; a < args.acc.size(); a++)
                    if (spilled(args.acc[a]))
                        spill_store(args.acc[a], acc_phys[a]);
                break;
            }

            // Mask ops. Emitting the instruction is the backend's job. The
            // allocator does not spill masks to make room (see
            // `max_temps_per_op`). A mask ends up spilled only when no register
            // is left for it, which the asserts reject.
            case op_kind_t::set_mask_imm: {
                JIT_ASSERT(!spilled(op.dst) && "set_mask_imm: mask spilled");
                be.set_mask_imm(phys(op.dst), (int)op.imm, data);
                break;
            }
            case op_kind_t::vload_masked: { // overwrites dst
                int base = gpr_use(op.mem.base).getIdx();
                int d = reg_of(op.dst);
                JIT_ASSERT(!spilled(op.s1) && "vload_masked: mask spilled");
                be.vload_masked(d, base, op.mem.disp, phys(op.s1), op.mem_dt,
                        dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vstore_masked: {
                int base = gpr_use(op.mem.base).getIdx();
                int s = vec_use(op.s0);
                JIT_ASSERT(!spilled(op.s1) && "vstore_masked: mask spilled");
                be.vstore_masked(base, op.mem.disp, s, phys(op.s1), op.mem_dt,
                        dt_of(op.s0));
                break;
            }

            // prefetcht0 is base x86-64, so emit it directly.
            case op_kind_t::prefetch: {
                Xbyak::Reg64 base = gpr_use(op.mem.base);
                gen.prefetcht0(gen.ptr[base + (int)op.mem.disp]);
                break;
            }

            // Control flow. ISA-neutral, emitted directly.
            case op_kind_t::loop_begin: {
                const Xbyak::Reg64 c(reg_of(op.dst));
                if (op.init_is_reg) {
                    Xbyak::Reg64 iv = gpr_use(op.s0);
                    gen.mov(c, iv);
                } else {
                    gen.mov(c, op.imm);
                }
                if (spilled(op.dst)) gen.mov(slot(op.dst), c);
                gen.L(labels[i]); // body start
                break;
            }
            case op_kind_t::loop_end: {
                // dec sets ZF, so the back-edge is a plain jnz with no cmp. The
                // counter starts >= 1 and lands on exactly 0, so jnz matches jg.
                const Xbyak::Reg64 c = gpr_use(op.dst);
                gen.dec(c);
                if (spilled(op.dst)) gen.mov(slot(op.dst), c);
                gen.jnz(labels[op.match]); // back-edge to the matching loop_begin
                break;
            }
            case op_kind_t::label: {
                gen.L(label_id_to_label[(int)op.label_id]);
                break;
            }
            case op_kind_t::jmp: {
                gen.jmp(label_id_to_label[(int)op.label_id],
                        Xbyak::CodeGenerator::T_NEAR);
                break;
            }
            case op_kind_t::jz: {
                Xbyak::Reg64 c = gpr_use(op.s0);
                gen.cmp(c, 0);
                gen.jz(label_id_to_label[(int)op.label_id],
                        Xbyak::CodeGenerator::T_NEAR);
                break;
            }
        }
    }
}

void emit(jit_generator_t &gen, const ir_t &ir, const reg_alloc_result_t &alloc,
        const reg_config_t &reg_cfg, data_section_t &data,
        postops_injector_t *postops) {
    const cpu_isa_t isa = gen.max_cpu_isa();
    if (is_superset(isa, avx512_core)) {
        avx512_backend_t be(gen, isa);
        emit(be, ir, alloc, reg_cfg, data, postops);
    } else {
        avx2_backend_t be(gen, isa);
        emit(be, ir, alloc, reg_cfg, data, postops);
    }
}

void emit_data_section(jit_generator_t &gen, data_section_t &data) {
    data.begin_offset = gen.getSize();
    for (auto &c : data.constants) {
        gen.align(data_section_t::alignment);
        gen.L(c.second);
        for (unsigned char byte : c.first)
            gen.db(byte);
    }
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
