/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <cassert>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/matmul/jit_uni_sparse_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::data_type;
using namespace Xbyak;

struct sparse_matmul_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(sparse_matmul_kernel_t);

    struct call_params_t {
        const int32_t *src_indices;
        const float *src_values, *wei, *dst;
        size_t block_size;
        size_t nnz;
    };

    sparse_matmul_kernel_t(size_t vlen, const matmul_pd_t *pd)
        : jit_generator(jit_name()), vlen_(vlen) {
        simd_w_ = vlen_ / data_type_size();
        N_ = pd->dst_md()->dims[1];
        tail_block_size_ = N() % block_size();
        tail_size_ = tail_block_size() % simd_w();
    }

    ~sparse_matmul_kernel_t() override = default;

    void operator()(const call_params_t *p) {
        return jit_generator::operator()(p);
    }

    size_t simd_w() const { return simd_w_; }
    size_t vlen() const { return vlen_; }
    size_t tail_block_size() const { return tail_block_size_; }
    size_t tail_size() const { return tail_size_; }

    int data_type_size() const { return sizeof(float); }
    int index_type_size() const { return sizeof(int32_t); }

    int block_size() const { return vlen(); }

    int unroll_factor() const { return 2; }

    int N() const { return N_; }

protected:
    size_t N_;
    size_t vlen_;
    size_t simd_w_;
    size_t tail_block_size_;
    size_t tail_size_;
};

template <cpu_isa_t isa>
struct jit_uni_sparse_matmul_kernel_t : public sparse_matmul_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_sparse_matmul_kernel_t)

    using sparse_matmul_kernel_t::data_type_size;
    using sparse_matmul_kernel_t::simd_w;
    using sparse_matmul_kernel_t::tail_block_size;
    using sparse_matmul_kernel_t::tail_size;
    using sparse_matmul_kernel_t::vlen;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    Reg64 reg_param = abi_param1;

    Reg64 reg_nnz_count = rax;
    Reg64 reg_blocks_count = rsi;
    Reg64 reg_src_indices = rbx;
    Reg64 reg_src_col_idx = rdx;

    Reg64 reg_src_values = r8;
    Reg64 reg_wei = r9;
    Reg64 reg_dst = r10;
    Reg64 reg_offset_n = r11;
    Reg64 reg_nnz_divided_by_2 = r12;
    Reg64 reg_block_offset = r13;
    Reg64 reg_tmp = r14;
    Reg64 reg_nnz = r15;

    Opmask tail_opmask = Opmask(2);
    Vmm tail_vmask = Vmm(0);

    Vmm vreg_src_val = Vmm(isa == avx512_core ? 19 : 11);
    Xmm xreg_src_val = Xmm(11);

    void load_kernel_params() {
#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_src_indices, ptr[reg_param + PARAM_OFF(src_indices)]);
        mov(reg_src_values, ptr[reg_param + PARAM_OFF(src_values)]);
        mov(reg_wei, ptr[reg_param + PARAM_OFF(wei)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        mov(reg_nnz, ptr[reg_param + PARAM_OFF(nnz)]);
#undef PARAM_OFF
    }

    Address wei_ptr(size_t offt = 0) {
        imul(reg_tmp, reg_src_col_idx, N());
        add(reg_tmp, reg_block_offset);
        return ptr[reg_wei + reg_tmp * data_type_size() + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return ptr[reg_dst + reg_block_offset * data_type_size() + offt];
    }

    Address src_values_ptr(size_t offt = 0) {
        return ptr[reg_src_values + reg_nnz_count * data_type_size() + offt];
    }

    Address src_indices_ptr(size_t offt = 0) {
        return dword[reg_src_indices + reg_nnz_count * index_type_size()
                + offt];
    }

    void load_tail(const Zmm &dst, const Address &src) {
        uni_vmovups_tail(dst, tail_opmask, src);
    }

    void load_tail(const Ymm &dst, const Address &src) {
        uni_vmovups_tail(dst, tail_vmask, src);
    }

    void store_tail(const Address &dst, const Zmm &src) {
        uni_vmovups_tail(dst, tail_opmask, src);
    }

    void store_tail(const Address &dst, const Ymm &src) {
        uni_vmovups_tail(dst, tail_vmask, src);
    }

    void prepare_tail_mask();

    Vmm get_dst_reg(int index) const {
        // Vmm(0) is reserved for mask.
        return Vmm(index + 1);
    }

    Vmm get_wei_reg(int index, bool is_tail_block) {
        // Vmm(0) is reserved for mask.
        const int nloads = is_tail_block
                ? utils::div_up(tail_block_size(), simd_w())
                : block_size() / simd_w();
        return Vmm(nloads + index + 1);
    }

    void loop_within_block_row(
            Vmm vreg_src_val, Reg64 reg_src_col_idx, bool is_tail_block) {
        const int nloads = is_tail_block
                ? utils::div_up(tail_block_size(), simd_w())
                : block_size() / simd_w();
        for (int i_load = 0; i_load < nloads; i_load++) {
            Vmm vreg_tmp_wei = get_wei_reg(i_load, is_tail_block);
            // Load a row of weights.
            if (is_tail_block && tail_size() > 0 && i_load == nloads - 1) {
                load_tail(vreg_tmp_wei, wei_ptr(vlen() * i_load));
            } else {
                uni_vmovups(vreg_tmp_wei, wei_ptr(vlen() * i_load));
            }
            // Multiply the broadcasted value with the row of weights
            // and accumulate result in dst.
            Vmm vreg_tmp_dst = get_dst_reg(i_load);
            uni_vfmadd231ps(vreg_tmp_dst, vreg_src_val, vreg_tmp_wei);
        }
    }

    void loop_within_block(int unroll_factor, bool is_tail_block) {
        Label loop_within_block_begin, loop_within_block_end;
        xor_(reg_nnz_count, reg_nnz_count);
        L(loop_within_block_begin);
        {
            cmp(reg_nnz_count, reg_nnz_divided_by_2);
            je(loop_within_block_end, T_NEAR);

            for (int uf = 0; uf < unroll_factor; uf++) {
                // Load src values to broadcast.
                uni_vbroadcastss(
                        vreg_src_val, src_values_ptr(uf * data_type_size()));
                // Load an index.
                movsxd(reg_src_col_idx,
                        src_indices_ptr(uf * index_type_size()));
                loop_within_block_row(
                        vreg_src_val, reg_src_col_idx, is_tail_block);
            }
            add(reg_nnz_count, unroll_factor);
            jmp(loop_within_block_begin, T_NEAR);
        }
        L(loop_within_block_end);

        // process tail over K (due to unrolling).
        // Check if tail process is needed.
        Label skip_row_tail;
        test(reg_nnz, 1);
        jz(skip_row_tail, T_NEAR);

        // Load src values to broadcast.
        uni_vbroadcastss(vreg_src_val, src_values_ptr());
        // Load an index.
        movsxd(reg_src_col_idx, src_indices_ptr());
        loop_within_block_row(vreg_src_val, reg_src_col_idx, is_tail_block);

        L(skip_row_tail);
    }

    void loop_over_blocks(bool is_tail_block) {
        const size_t n_full_blocks = N() / block_size();
        const size_t nblocks = n_full_blocks + is_tail_block;
        // Divide number of non-zero elements in the row by 2 (unroll factor).
        assert(unroll_factor() == 2);
        mov(reg_nnz_divided_by_2, reg_nnz);
        and_(reg_nnz_divided_by_2, -2);

        if (is_tail_block) {
            mov(reg_blocks_count, n_full_blocks);
        } else {
            xor_(reg_blocks_count, reg_blocks_count);
        }

        Label loop_over_blocks_begin, loop_over_blocks_end;
        L(loop_over_blocks_begin);
        {
            cmp(reg_blocks_count, nblocks);
            je(loop_over_blocks_end, T_NEAR);

            mov(reg_block_offset, reg_blocks_count);
            shl(reg_block_offset, math::ilog2q(block_size()));

            const int nloads = is_tail_block
                    ? utils::div_up(tail_block_size(), simd_w())
                    : block_size() / simd_w();
            std::vector<Vmm> vregs_dst(nloads);
            for (int i_load = 0; i_load < nloads; i_load++) {
                vregs_dst[i_load] = get_dst_reg(i_load);
                uni_vpxor(vregs_dst[i_load], vregs_dst[i_load],
                        vregs_dst[i_load]);
            }

            loop_within_block(unroll_factor(), is_tail_block);

            for (int i_load = 0; i_load < nloads; i_load++) {
                if (is_tail_block && tail_size() > 0 && i_load == nloads - 1) {
                    store_tail(dst_ptr(vlen() * i_load), vregs_dst[i_load]);
                } else {
                    uni_vmovups(dst_ptr(vlen() * i_load), vregs_dst[i_load]);
                }
            }
            add(reg_blocks_count, 1);
            jmp(loop_over_blocks_begin, T_NEAR);
        }
        L(loop_over_blocks_end);
    }

    void compute() {
        const size_t n_full_blocks = N() / block_size();
        if (n_full_blocks != 0) { loop_over_blocks(/* is_tail_block */ false); }
        if (tail_block_size() > 0) loop_over_blocks(/* is_tail_block */ true);
    }

    void generate() override {
        preamble();
        prepare_tail_mask();
        load_kernel_params();
        compute();
        postamble();
    }

    jit_uni_sparse_matmul_kernel_t(const matmul_pd_t *pd)
        : sparse_matmul_kernel_t(cpu_isa_traits<isa>::vlen, pd) {}
    ~jit_uni_sparse_matmul_kernel_t() override = default;
};

template <>
void jit_uni_sparse_matmul_kernel_t<avx512_core>::prepare_tail_mask() {
    if (tail_size() == 0) return;

    const int mask_f32 = (1 << tail_size()) - 1;

    Reg32 regw_tmp = reg_tmp.cvt32();
    mov(regw_tmp, mask_f32);
    kmovd(tail_opmask, regw_tmp);
}

template <>
void jit_uni_sparse_matmul_kernel_t<avx2>::prepare_tail_mask() {
    if (tail_size() == 0) return;

    static const uint32_t mask_f32[]
            = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                    0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

    mov(reg_tmp, reinterpret_cast<size_t>(&mask_f32[7 - tail_size()]));
    vmovups(tail_vmask, ptr[reg_tmp]);
}

status_t jit_uni_sparse_matmul_t::init(engine_t *engine) {
    if (mayiuse(avx512_core)) {
        using kernel_t = jit_uni_sparse_matmul_kernel_t<avx512_core>;
        kernel_ = std::unique_ptr<kernel_t> {new kernel_t(pd())};
    } else if (mayiuse(avx2)) {
        using kernel_t = jit_uni_sparse_matmul_kernel_t<avx2>;
        kernel_ = std::unique_ptr<kernel_t> {new kernel_t(pd())};
    }
    if (!kernel_) return status::runtime_error;

    CHECK(kernel_->create_kernel());
    return status::success;
}

jit_uni_sparse_matmul_t::jit_uni_sparse_matmul_t(const pd_t *apd)
    : primitive_t(apd) {}
jit_uni_sparse_matmul_t::~jit_uni_sparse_matmul_t() = default;

status_t jit_uni_sparse_matmul_t::execute(const exec_ctx_t &ctx) const {
    const auto *weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    const auto *src_values = CTX_IN_MEM(const float *, DNNL_ARG_SRC, 0);
    const auto *src_indices = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 1);
    const auto *src_pointers = CTX_IN_MEM(const int32_t *, DNNL_ARG_SRC, 2);

    status_t status = status::success;
    auto dst = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const dim_t M = dst_d.dims()[0];
    const dim_t N = dst_d.dims()[1];

    // TODO: Implement a load balancing mechanism that would distribute
    // rows between threads based on the number of non-zero elements in those
    // rows.
    parallel_nd(M, [&](dim_t m) {
        const int row_begin = src_pointers[m];
        const int row_end = src_pointers[m + 1];
        const int nnz = row_end - row_begin;

        sparse_matmul_kernel_t::call_params_t p;
        p.nnz = nnz;
        p.src_values = src_values + row_begin;
        p.src_indices = src_indices + row_begin;
        p.wei = weights;
        p.dst = dst + (m * N);
        p.block_size = kernel_->block_size();
        (*kernel_)(&p);
    });
    return status::success;
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
