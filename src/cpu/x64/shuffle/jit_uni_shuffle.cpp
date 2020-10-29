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

#include <array>
#include <cassert>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/shuffle/jit_uni_shuffle.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
using namespace format_tag;

static constexpr int bf16_size_bytes = sizeof(bfloat16_t);
static constexpr int f32_size_bytes = sizeof(float);
static constexpr int gp_regs = 4; // number of used gp regs

#define GET_OFF(field) offsetof(jit_shuffle_args_t, field)
struct jit_shuffle_args_t {
    const void *src = nullptr;
    void *dst = nullptr;
    const int32_t *input_off_ptr = nullptr;
};

template <int data_type_size>
struct reg_type_base_t {};

template <>
struct reg_type_base_t<bf16_size_bytes> {
    using reg_type_t = Reg16;
};
template <>
struct reg_type_base_t<f32_size_bytes> {
    using reg_type_t = Reg32;
};

static constexpr std::array<int, gp_regs> gprs = {{
        Operand::Code::EBX,
        Operand::Code::EAX,
        Operand::Code::EDX,
        Operand::Code::ESI,
}};

// jit kernel
template <int data_type_size>
struct jit_uni_shuffle_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_shuffle_kernel_t)

    using jit_uni_shuffle_pd_t =
            typename jit_uni_shuffle_t<data_type_size>::pd_t;
    using data_reg_type_t =
            typename reg_type_base_t<data_type_size>::reg_type_t;

    jit_uni_shuffle_kernel_t(const jit_uni_shuffle_pd_t *pd) : pd_(pd) {}

    void uni_pinsr(int reg_num, Reg64 load_reg, int data_size, int xmm_off);

    void store(int dst_off, int reg_num);

    template <typename T>
    T get_reg(int i) {
        assert(i >= 0 && i < gp_regs);
        return T(gprs[i]);
    }

    void generate() override {
        preamble();

        static constexpr int offset_data_type_size
                = 4; // input_off_ array data type size
        static constexpr int step_size
                = 4; // loop increment and offset calculations constant
        // equal to number of f32 elements in Xmm
        const int blk_size = pd_->blk_size_;
        const int steps_in_block = blk_size / step_size;
        const int group_size = pd_->group_size();
        const size_t C = pd_->C();
        const auto C_over_grps = utils::div_up(C, group_size);
        const auto stride = C_over_grps * data_type_size;

        for (int i = 0; i < gp_regs; i++)
            xor_(get_reg<Reg64>(i), get_reg<Reg64>(i));

        mov(input_off_reg, ptr[abi_param1 + GET_OFF(input_off_ptr)]);

        mov(src_reg, ptr[abi_param1 + GET_OFF(src)]);
        mov(dst_reg, ptr[abi_param1 + GET_OFF(dst)]);

        const int SP = pd_->SP_;

        static constexpr int xmm_id = 0;
        const int group_elems = C / group_size;
        const auto stride_mod = SP - 1;

        const auto calculate_output_off = [&](int elem, int elem_blks, int gr) {
            const auto current_4_elems = (elem - gr * group_elems) / step_size;
            const auto current_blk = current_4_elems / steps_in_block;
            const auto current_blk_rem = current_4_elems % steps_in_block;
            return (current_blk * blk_size + current_blk_rem * step_size
                           + elem_blks * stride_mod * blk_size)
                    * data_type_size
                    + gr * stride;
        };

        const auto shuffle_one_by_one = [&](int elem, int gr,
                                                int num_elements) {
            const auto elem_blks = elem / blk_size;
            const auto output_off = calculate_output_off(elem, elem_blks, gr);
            for (int s = 0; s < num_elements; s++) {
                mov(get_reg<Reg32>(0),
                        ptr[input_off_reg
                                + (elem + s) * offset_data_type_size]);
                mov(get_reg<data_reg_type_t>(1),
                        ptr[src_reg + get_reg<Reg64>(0) * data_type_size]);
                const auto elem_blks_mod = (elem + s) / blk_size - elem_blks;
                mov(ptr[dst_reg + output_off + s * data_type_size
                            + elem_blks_mod * stride_mod * blk_size
                                    * data_type_size],
                        get_reg<data_reg_type_t>(1));
            }
        };

        const auto shuffle_vectorized = [&](int elem, int gr) {
            const auto elem_blks = elem / blk_size;
            const auto output_off = calculate_output_off(elem, elem_blks, gr);
            for (int i = 0; i < step_size; i++)
                mov(get_reg<Reg32>(i),
                        ptr[input_off_reg
                                + (elem + i) * offset_data_type_size]);
            for (int i = 0; i < step_size; i++)
                uni_pinsr(xmm_id, get_reg<Reg64>(i), data_type_size, i);

            store(output_off, xmm_id);
        };

        for (int gr = 0; gr < group_size; ++gr)
            // iterate over output elements
            for (int elem = gr * group_elems; elem < group_elems * (gr + 1);
                    elem += step_size) {
                // tail check
                if (group_elems * (gr + 1) - elem < step_size) {
                    shuffle_one_by_one(elem, gr, group_elems * (gr + 1) - elem);
                    // check if processed elements contain end of block
                } else if (elem / blk_size
                        != (elem + step_size - 1) / blk_size) {
                    shuffle_one_by_one(elem, gr, step_size);
                    // general case, load elements and store
                } else {
                    shuffle_vectorized(elem, gr);
                }
            }

        postamble();
    }

    const jit_uni_shuffle_pd_t *pd_;

    const Reg64 src_reg = r9;
    const Reg64 dst_reg = r8;
    const Reg64 input_off_reg = r15;
};

template <int data_type_size>
void jit_uni_shuffle_kernel_t<data_type_size>::uni_pinsr(
        int reg_num, Reg64 load_reg, int data_size, int xmm_off) {
    const auto ins_reg = Xmm(reg_num);
    uni_vpinsrd(ins_reg, ins_reg, ptr[src_reg + load_reg * data_size], xmm_off);
}

template <>
void jit_uni_shuffle_kernel_t<bf16_size_bytes>::uni_pinsr(
        int reg_num, Reg64 load_reg, int data_size, int xmm_off) {
    const auto ins_reg = Xmm(reg_num);
    vpinsrw(ins_reg, ins_reg, ptr[src_reg + load_reg * data_size], xmm_off);
}

template <int data_type_size>
void jit_uni_shuffle_kernel_t<data_type_size>::store(int dst_off, int reg_num) {
    uni_vmovups(ptr[dst_reg + dst_off], Xmm(reg_num));
}

template <>
void jit_uni_shuffle_kernel_t<bf16_size_bytes>::store(
        int dst_off, int reg_num) {
    vmovsd(ptr[dst_reg + dst_off], Xmm(reg_num));
}

template struct jit_uni_shuffle_kernel_t<f32_size_bytes>;
template struct jit_uni_shuffle_kernel_t<bf16_size_bytes>;

#undef GET_OFF

template <int data_type_size>
status_t jit_uni_shuffle_t<data_type_size>::precompute_offsets() {
    const int axis_size = pd()->axis_size();
    const int group_size = pd()->group_size();
    const int transpose_row
            = pd()->is_fwd() ? group_size : axis_size / group_size;
    const int transpose_col
            = pd()->is_fwd() ? axis_size / group_size : group_size;
    std::vector<int> rev_transposed_(axis_size);

    // Precompute transposed axis helper array
    parallel_nd(transpose_col, transpose_row, [&](int i, int j) {
        rev_transposed_[j * transpose_col + i] = i * transpose_row + j;
    });

    const int C = pd()->C();
    const int blk_size = pd()->blk_size_;
    const dim_t CB = utils::div_up(C, blk_size);
    const int SP = pd()->SP_;
    input_off_
            = (int *)malloc(C * sizeof(int), platform::get_cache_line_size());
    if (input_off_ == nullptr) return dnnl_out_of_memory;

    // Precompute input offsets using transposed axis
    parallel_nd(CB, [&](int cb) {
        const int blk_end = nstl::min(blk_size, C - cb * blk_size);
        PRAGMA_OMP_SIMD()
        for (int cc = 0; cc < blk_end; ++cc) {
            const int off = cb * blk_size + cc;
            const int &input_c = rev_transposed_[off];
            input_off_[off]
                    = input_c / blk_size * SP * blk_size + input_c % blk_size;
        }
    });
    return status::success;
}

template <int data_type_size>
status_t jit_uni_shuffle_t<data_type_size>::init(engine_t *engine) {
    CHECK(precompute_offsets());
    CHECK(safe_ptr_assign(
            kernel_, new jit_uni_shuffle_kernel_t<data_type_size>(pd())));
    CHECK(kernel_->create_kernel());
    return status::success;
}

template <int data_type_size>
inline jit_uni_shuffle_t<data_type_size>::jit_uni_shuffle_t(const pd_t *apd)
    : primitive_t(apd) {}

template <int data_type_size>
jit_uni_shuffle_t<data_type_size>::~jit_uni_shuffle_t() {
    free(this->input_off_);
}

template <int data_type_size>
status_t jit_uni_shuffle_t<data_type_size>::execute(
        const exec_ctx_t &ctx) const {
    using namespace prop_kind;
    using namespace utils;

    const memory_desc_wrapper data_d(pd()->data_md());

    const auto i_arg = pd()->is_fwd() ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST;
    const auto o_arg = pd()->is_fwd() ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC;
    auto input = CTX_IN_MEM(const data_t *, i_arg);
    auto output = CTX_OUT_MEM(data_t *, o_arg);

    const int MB = pd()->MB();
    const int SP = pd()->SP_;
    const int blk_size = pd()->blk_size_;
    const size_t stride_mb = data_d.blocking_desc().strides[0];
    parallel_nd(MB, SP, [&](int mb, int sp) {
        const int off = mb * stride_mb + sp * blk_size;

        jit_shuffle_args_t args;
        args.src = input + off;
        args.dst = output + off;

        args.input_off_ptr = this->input_off_;

        (*kernel_)(&args);
    });

    return status::success;
}

template struct jit_uni_shuffle_t<f32_size_bytes>;
template struct jit_uni_shuffle_t<bf16_size_bytes>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
