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

#include <memory>
#include <vector>

#include "cpu/x64/ir/codegen.hpp"
#include "cpu/x64/ir/dump.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/postops_injector.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"
#include "cpu/x64/ir/reg_config.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

void generate_kernel(
        jit_generator_t &gen, const ir_t &ir, const postops_config_t &postops) {
    const cpu_isa_t isa = gen.max_cpu_isa();
    const bool with_postops = postops.post_ops != nullptr;

    // AVX-512 opmasks reserved for the post-ops injector, which writes both
    // and restores neither (see `postops_injector_t`).
    const int eltwise_opmask = 6, binary_tail_opmask = 7;
    const std::vector<int> reserved_masks = with_postops
            ? std::vector<int> {eltwise_opmask, binary_tail_opmask}
            : std::vector<int> {};

    // Build the register configuration for code emission.
    const reg_config_t reg_cfg = make_reg_config(
            isa, abi_param1.getIdx(), Xbyak::Operand::RSP, reserved_masks);

    // Allocate registers.
    const reg_alloc_result_t alloc = allocate_registers(ir, reg_cfg.pools);

    // The injector is created here, not in the emitter, because it lives
    // through the whole code generation. It emits code during `emit()` and
    // writes its table after the postamble. The emitter drives it through the
    // `inject_postops` operation.
    std::unique_ptr<postops_injector_t> injector;
    if (with_postops) {
        injector.reset(new postops_injector_t(gen, isa, *postops.post_ops,
                *postops.dst_md, abi_param1, postops.rhs_arg_offset,
                postops.dst_orig_off, postops.tail_elems, eltwise_opmask,
                binary_tail_opmask));
    }

    gen.preamble();

    // Set up the stack frame for spilled values.
    if (alloc.frame_bytes > 0) gen.sub(gen.rsp, (uint32_t)alloc.frame_bytes);

    // Lower the IR. `emit()` dispatches to the ISA-specific backend based on
    // `gen.max_cpu_isa()`. The emitter may accumulate static data (e.g. the
    // AVX2 mask table) that is written after the postamble.
    data_section_t data;
    emit(gen, ir, alloc, reg_cfg, data, injector.get());

    // Tear down the stack frame.
    if (alloc.frame_bytes > 0) gen.add(gen.rsp, (uint32_t)alloc.frame_bytes);

    gen.postamble();

    // Emit any static data the emitter accumulated.
    emit_data_section(gen, data);

    // Emit the injector's constant table (a no-op unless the chain has eltwise
    // or sum).
    if (injector) injector->maybe_prepare_table();

    // Debug output (see `ir/dump.hpp`). Prints nothing unless enabled.
    print_kernel_dump(gen, ir, data);
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
