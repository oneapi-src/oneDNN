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

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iomanip>
#include <vector>

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/utils.hpp"

#include "cpu/x64/ir/dump.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

namespace {

const char *kind_name(op_kind_t kind) {
    switch (kind) {
        case op_kind_t::mov_imm: return "mov_imm";
        case op_kind_t::mov_reg: return "mov_reg";
        case op_kind_t::add_imm: return "add_imm";
        case op_kind_t::add_reg: return "add_reg";
        case op_kind_t::load: return "load";
        case op_kind_t::vzero: return "vzero";
        case op_kind_t::vload: return "vload";
        case op_kind_t::vstore: return "vstore";
        case op_kind_t::vload_scalar: return "vload_scalar";
        case op_kind_t::vstore_scalar: return "vstore_scalar";
        case op_kind_t::vload_bcast: return "vload_bcast";
        case op_kind_t::vdot: return "vdot";
        case op_kind_t::vadd: return "vadd";
        case op_kind_t::vmul: return "vmul";
        case op_kind_t::vhreduce: return "vhreduce";
        case op_kind_t::inject_postops: return "inject_postops";
        case op_kind_t::set_mask_imm: return "set_mask_imm";
        case op_kind_t::vload_masked: return "vload_masked";
        case op_kind_t::vstore_masked: return "vstore_masked";
        case op_kind_t::prefetch: return "prefetch";
        case op_kind_t::loop_begin: return "loop_begin";
        case op_kind_t::loop_end: return "loop_end";
        case op_kind_t::label: return "label";
        case op_kind_t::jmp: return "jmp";
        case op_kind_t::jz: return "jz";
    }
    assert(!"unknown op kind");

    return "?";
}

bool is_valid(const ir_t &ir, vreg_t v) {
    return (int)v >= 0 && (int)v < ir.n_vregs();
}

// Returns `g<id>`, `<dt>:v<id>`, or `m<id>` by the kind of `v`.
std::string vreg_str(const ir_t &ir, vreg_t v) {
    if (v == vreg_t::none) return "none";

    const std::string id = std::to_string((int)v);
    if (!is_valid(ir, v)) return "?" + id;

    switch (ir.vreg_info()[(int)v].kind) {
        case reg_kind_t::gpr: return "g" + id;
        case reg_kind_t::vec:
            return std::string(dnnl_dt2str(ir.vreg_info()[(int)v].dt)) + ":v"
                    + id;
        case reg_kind_t::mask: return "m" + id;
    }
    assert(!"unknown reg kind");

    return "?" + id;
}

std::string label_str(label_t l) {
    return "L" + std::to_string((int)l);
}

// Returns the memory operand of `op`. `data` is the vec vreg the access moves,
// or `none` for an access that moves no vec value (e.g. `load`, `prefetch`).
std::string mem_str(const ir_t &ir, const op_t &op, vreg_t data) {
    std::string s;
    if (data != vreg_t::none && op.mem_dt != data_type::undef) {
        s += dnnl_dt2str(op.mem_dt);
        s += ":";
    }
    s += "[";
    s += op.mem.is_param ? "param" : vreg_str(ir, op.mem.base);
    if (op.mem.disp >= 0) s += "+";
    s += std::to_string(op.mem.disp);
    s += "]";

    return s;
}

// Returns the operands of an `inject_postops` op from its side-table entry:
// the accumulators, then `base=` unless the base pointer is `none`, then
// `off=` unless the offsets are empty.
std::string postops_str(const ir_t &ir, const op_t &op) {
    const auto &table = ir.inject_postops_args();
    if (op.imm < 0 || op.imm >= (dim_t)table.size())
        return "args=?" + std::to_string(op.imm);

    const inject_postops_args_t &args = table[(int)op.imm];

    std::string s;
    for (size_t i = 0; i < args.acc.size(); i++) {
        if (i > 0) s += ", ";
        s += vreg_str(ir, args.acc[i]);
    }

    if (args.base_ptr != vreg_t::none)
        s += ", base=" + vreg_str(ir, args.base_ptr);

    if (!args.out_byte_off.empty()) {
        s += ", off=[";
        for (size_t i = 0; i < args.out_byte_off.size(); i++) {
            if (i > 0) s += ", ";
            s += std::to_string(args.out_byte_off[i]);
        }
        s += "]";
    }

    return s;
}

// Returns the text of one operation, without index or indentation. Every kind
// prints as its name followed by its operands, except the control-flow
// markers. A loop prints as an opening and a closing brace, and a label prints
// as `L<id>:`.
std::string op_str(const ir_t &ir, const op_t &op) {
    const auto r = [&](vreg_t v) { return vreg_str(ir, v); };
    const auto mem = [&](vreg_t data) { return mem_str(ir, op, data); };
    const std::string k = std::string(kind_name(op.kind)) + " ";

    switch (op.kind) {
        case op_kind_t::mov_imm:
        case op_kind_t::add_imm:
        case op_kind_t::set_mask_imm:
            return k + r(op.dst) + ", " + std::to_string(op.imm);
        case op_kind_t::mov_reg:
        case op_kind_t::add_reg:
        case op_kind_t::vadd:
        case op_kind_t::vmul:
        case op_kind_t::vhreduce: return k + r(op.dst) + ", " + r(op.s0);
        case op_kind_t::load: return k + r(op.dst) + ", " + mem(vreg_t::none);
        case op_kind_t::vzero: return k + r(op.dst);
        case op_kind_t::vload:
        case op_kind_t::vload_scalar:
        case op_kind_t::vload_bcast: return k + r(op.dst) + ", " + mem(op.dst);
        case op_kind_t::vstore:
        case op_kind_t::vstore_scalar: return k + mem(op.s0) + ", " + r(op.s0);
        case op_kind_t::vdot:
            return k + r(op.dst) + ", " + r(op.s0) + ", " + r(op.s1);
        case op_kind_t::inject_postops: return k + postops_str(ir, op);
        case op_kind_t::vload_masked:
            return k + r(op.dst) + ", " + mem(op.dst) + ", " + r(op.s1);
        case op_kind_t::vstore_masked:
            return k + mem(op.s0) + ", " + r(op.s0) + ", " + r(op.s1);
        case op_kind_t::prefetch: return k + mem(vreg_t::none);
        case op_kind_t::loop_begin:
            return "loop " + r(op.dst) + " = "
                    + (op.init_is_reg ? r(op.s0) : std::to_string(op.imm))
                    + " {";
        case op_kind_t::loop_end:
            return "} // " + r(op.dst) + " -= 1, repeat while > 0";
        case op_kind_t::label: return label_str(op.label_id) + ":";
        case op_kind_t::jmp: return k + label_str(op.label_id);
        case op_kind_t::jz: return k + r(op.s0) + ", " + label_str(op.label_id);
    }
    assert(!"unknown op kind");

    return k;
}

} // namespace

std::string to_string(const ir_t &ir) {
    ostringstream_t ss;

    // Indentation follows the loop nesting. A loop's closing line is indented
    // like its opening line, so the body stands out between the two.
    int depth = 0;
    for (int i = 0; i < ir.n_ops(); i++) {
        const op_t &op = ir.ops()[i];
        if (op.kind == op_kind_t::loop_end) depth--;

        ss << std::setw(5) << i << " | "
           << std::string(2 * std::max(depth, 0), ' ') << op_str(ir, op)
           << "\n";

        if (op.kind == op_kind_t::loop_begin) depth++;
    }

    return ss.str();
}

bool has_x64ir_token(const std::string &verbose_value) {
    // Tokens are split on `,` with no trimming, the same way the rest of
    // `ONEDNN_VERBOSE` is parsed.
    size_t pos = 0;
    while (true) {
        const size_t end = verbose_value.find(',', pos);
        if (verbose_value.compare(pos, end - pos, "x64ir") == 0) return true;
        if (end == std::string::npos) return false;
        pos = end + 1;
    }
}

std::string kernel_dump_str(const jit_generator_t &gen, const ir_t &ir,
        const data_section_t &data) {
    ostringstream_t ss;

    const char *name = gen.name();
    cpu_isa_t isa = gen.max_cpu_isa();
    size_t code_size = gen.getSize();
    size_t data_size = code_size - data.begin_offset;

    ss << "begin x64ir " << name << " isa=" << isa2str(isa) << "\n";
    ss << "code: " << code_size << " bytes (instructions "
       << code_size - data_size << " bytes, static data " << data_size
       << " bytes)\n";
    ss << "\n" << to_string(ir);
    ss << "end x64ir\n";

    return ss.str();
}

void print_kernel_dump(const jit_generator_t &gen, const ir_t &ir,
        const data_section_t &data) {
    if (!is_dev_mode()) return;

    static const bool enabled = has_x64ir_token(getenv_string_user("VERBOSE"));
    if (!enabled) return;

    printf("%s", kernel_dump_str(gen, ir, data).c_str());
    fflush(stdout);
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
