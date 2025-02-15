/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "cpu/x64/jit_brgemm_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

#define GET_OFF(field) offsetof(brgemm_kernel_diff_bias_t, field)

// This version is used from BRGEMM-based Inner Product for weights tensor.
template <typename Vmm>
dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::
        jit_brgemm_kernel_diff_bias_t(const jit_brgemm_primitive_conf_t &ajbgp,
                const brgemm_desc_t &abrg)
    : jit_generator(jit_name())
    , brg_(abrg)
    , reduce_kind_(matmul_reduce_kind::undef)
    , ddst_dt_((ajbgp.isa == avx512_core_fp16 && ajbgp.use_buffer_b)
                      ? data_type::f32
                      : ajbgp.dst_dt)
    , bia_dt_(ajbgp.bia_dt)
    , acc_dt_(ajbgp.acc_dt)
    , ddst_typesize_(types::data_type_size(ddst_dt_))
    , bia_typesize_(types::data_type_size(bia_dt_))
    , acc_typesize_(types::data_type_size(acc_dt_))
    , mult_(data_type_vnni_granularity(ddst_dt_)) {}

// This version is used from MatMul for src tensor.
template <typename Vmm>
dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::
        jit_brgemm_kernel_diff_bias_t(const matmul::brgemm_matmul_conf_t &bgmmc,
                const brgemm_desc_t &abrg)
    : jit_generator(jit_name())
    , brg_(abrg)
    , reduce_kind_(bgmmc.reduce_kind)
    // MatMul `src`.
    , ddst_dt_((bgmmc.isa == avx512_core_fp16 && bgmmc.use_buffer_a)
                      ? data_type::f32
                      : bgmmc.src_dt)
    // MatMul `reduce` buffer.
    , bia_dt_(bgmmc.reduce_dt)
    , acc_dt_(bgmmc.acc_dt)
    , ddst_typesize_(types::data_type_size(ddst_dt_))
    , bia_typesize_(types::data_type_size(bia_dt_))
    , acc_typesize_(types::data_type_size(acc_dt_))
    , mult_(0) { // Unused.
    // This kernel must be called after the copy A routine because it assumes
    // that fp16 data has already been upconverted to f32.

    // Only reduction for `src` is supported.
    assert(reduce_kind_ == matmul_reduce_kind::src);
    // `src` matrix is assumed to have a row major layout.
    assert(bgmmc.treat_A_as_plain || bgmmc.use_buffer_a);
}

template <typename Vmm>
Vmm dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::vmm_mask(
        const Vmm vmm_in, bool mask_flag, bool store,
        Xbyak::Opmask ktail_mask) {
    return mask_flag && isa_has_masks(brg_.isa_impl)
            ? (store ? vmm_in | ktail_mask : vmm_in | ktail_mask | T_z)
            : vmm_in;
}

// Loads from ddst and adds it to bias accumulator. Used when ddst
// is a matrix B.
template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::accumulate_bias(
        int idx, bool mask_flag) {
    auto vddst = get_ddst_reg(idx);
    auto vddst_load = vmm_mask(vddst, mask_flag, false, k_tail_mask);
    auto vbias = get_bias_reg(idx);
    if (ddst_dt_ == data_type::f16) {
        // As we do not have fp16_vnni, we add twice to accumulate
        // adjacent elements.
        for (int i = 0; i < 2; ++i) {
            auto addr = ptr[aux_reg_ddst
                    + ddst_typesize_ * mult_ * idx * brg_.ld_block + i * 2];
            vmovups(vddst_load, addr);
            vpermw(vddst | k_f16_perm_mask | T_z, vreg_perm, vddst);
            vcvtph2psx(vddst, Vmm_lower_t(vddst.getIdx()));
            vaddps(vbias, vbias, vddst);
        }
    } else {
        auto addr = ptr[aux_reg_ddst
                + ddst_typesize_ * mult_ * idx * brg_.ld_block];
        if (IMPLICATION(mask_flag, isa_has_masks(brg_.isa_impl)))
            vmovups(vddst_load, addr);
        else
            vmaskmovps(vddst_load, vmm_tail_mask, addr);
        if (ddst_dt_ == data_type::bf16)
            vdpbf16ps(vbias, vreg_unit, vddst);
        else
            vaddps(vbias, vbias, vddst);
    }
}

// Loads from ddst and adds it to bias accumulator. Used when ddst
// is a matrix A.
template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::accumulate_bias(
        bool mask_flag) {

    auto vddst = get_ddst_reg(0);
    auto vbias_acc = get_bias_reg(0);

    auto vddst_load = vmm_mask(vddst, mask_flag, false, k_tail_mask);
    auto addr_ddst = ptr[aux_reg_ddst];

    if (ddst_dt_ == data_type::f16) {
        vpmovzxwd(vddst_load, addr_ddst);
        vpermw(vddst | k_f16_perm_mask | T_z, vreg_perm, vddst);
        vcvtph2psx(vddst, Vmm_lower_t(vddst.getIdx()));
        vaddps(vbias_acc, vbias_acc, vddst);
    } else if (ddst_dt_ == data_type::bf16) {
        vpmovzxwd(vddst_load, addr_ddst);
        vdpbf16ps(vbias_acc, vreg_unit, vddst);
    } else if (ddst_dt_ == data_type::f32) {
        if (IMPLICATION(mask_flag, isa_has_masks(brg_.isa_impl)))
            vmovups(vddst_load, addr_ddst);
        else
            vmaskmovps(vddst_load, vmm_tail_mask, addr_ddst);
        vaddps(vbias_acc, vbias_acc, vddst);
    } else
        assert(!"Unsupported data type");
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::store(
        int idx, bool mask_flag) {
    auto addr = ptr[reg_bias + bia_typesize_ * idx * brg_.ld_block];
    auto vbias = get_bias_reg(idx);
    auto vbias_lower = get_bias_reg_lower(idx);
    switch (bia_dt_) {
        case data_type::bf16:
            vcvtneps2bf16(vbias_lower, vbias);
            if (mask_flag) {
                vmovdqu16(addr, vmm_mask(vbias, mask_flag, true, k_tail_mask));
            } else {
                vmovups(addr, vbias_lower);
            }
            break;
        case data_type::f16:
            vcvtps2ph(vbias_lower, vbias, 0x4);
            if (mask_flag) {
                vmovdqu16(addr, vmm_mask(vbias, mask_flag, true, k_tail_mask));
            } else {
                vmovups(addr, vbias_lower);
            }
            break;
        case data_type::f32:
            if (IMPLICATION(mask_flag, isa_has_masks(brg_.isa_impl)))
                vmovups(addr, vmm_mask(vbias, mask_flag, true, k_tail_mask));
            else
                vmaskmovps(addr, vmm_tail_mask, vbias);
            break;
        default: assert("Unsupported bias data type");
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::horizontal_sum(
        Xbyak::Xmm src) {
    vhaddps(src, src, src);
    vhaddps(src, src, src);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::horizontal_sum(
        Xbyak::Ymm src, Xbyak::Ymm workspace) {
    const Xbyak::Xmm xmm_ws {workspace.getIdx()};
    const Xbyak::Xmm xmm_src {src.getIdx()};

    vextractf128(xmm_ws, src, 1);
    vaddps(xmm_src, xmm_src, xmm_ws);
    horizontal_sum(xmm_src);
}
template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::horizontal_sum(
        Xbyak::Zmm src, Xbyak::Zmm workspace) {
    const Xbyak::Ymm ymm_ws {workspace.getIdx()};
    const Xbyak::Ymm ymm_src {src.getIdx()};

    vextractf64x4(ymm_ws, src, 1);
    vaddps(ymm_src, ymm_src, ymm_ws);
    horizontal_sum(ymm_src, ymm_ws);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::loop_by_K() {
    Xbyak::Label k_loop, init_zero, init_done, store_final, store_done;

    mov(aux_reg_ddst, reg_ddst);

    test(reg_flag, FLAG_REDUCE_FIRST);
    jnz(init_zero, T_NEAR);

    // Load data from bias acc when reg_flag != FLAG_REDUCE_FIRST.
    auto vbias_acc = get_bias_reg(0);
    auto addr_bias_acc = ptr[reg_bias_acc];
    uni_vmovss(vbias_acc, addr_bias_acc);
    jmp(init_done, T_NEAR);

    // Zero out bias acc register.
    L(init_zero);
    uni_vxorps(vbias_acc, vbias_acc, vbias_acc);
    L(init_done);

    const auto k_size = brg_.reduce_dim / brg_.ld_block;
    const auto k_tail = brg_.reduce_dim % brg_.ld_block;

    // Do reduction over K.
    if (k_size > 0) {
        mov(reg_k_iter, brg_.reduce_dim / brg_.ld_block);
        L(k_loop);
        {
            accumulate_bias(false);
            add(aux_reg_ddst, ddst_typesize_ * brg_.ld_block);
            sub(reg_k_iter, 1);
            jnz(k_loop, T_NEAR);
        }
    }

    if (k_tail > 0) accumulate_bias(true);

    // Do horizontal reduction.
    horizontal_sum(vbias_acc, get_workspace_reg());

    test(reg_flag, FLAG_REDUCE_LAST);
    jnz(store_final, T_NEAR);

    // Store intermediate results to accumulator.
    uni_vmovss(addr_bias_acc, vbias_acc);
    jmp(store_done, T_NEAR);

    L(store_final);

    // Convert and store final results.
    auto addr_bias = ptr[reg_bias];
    auto vbias_acc_lower = get_bias_reg_lower(0);
    switch (bia_dt_) {
        case data_type::bf16:
            vcvtneps2bf16(vbias_acc_lower, vbias_acc);
            vmovdqu16(addr_bias, vbias_acc | k_store_mask);
            break;
        case data_type::f16:
            vcvtps2ph(vbias_acc_lower, vbias_acc, 0x4);
            vmovdqu16(addr_bias, vbias_acc | k_store_mask);
            break;
        case data_type::f32: uni_vmovss(addr_bias, vbias_acc); break;
        default: assert("Unsupported bias data type");
    }
    L(store_done);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::loop_by_N(
        int n_loop, int nb_tail) {
    mov(aux_reg_ddst, reg_ddst);

    int n_iters = n_loop;
    if (nb_tail > 0) n_iters--;
    Xbyak::Label k_loop, init_zero, init_done;
    int n_ = 0;

    test(reg_flag, FLAG_REDUCE_FIRST);
    jnz(init_zero, T_NEAR); // FLAG_REDUCE_FIRST is set

    for (; n_ < n_iters; n_++) {
        auto vbias = get_bias_reg(n_);
        auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
        vmovups(vbias, addr);
    }
    if (nb_tail > 0) {
        auto vbias = vmm_mask(get_bias_reg(n_), true, false, k_tail_mask);
        auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
        if (isa_has_masks(brg_.isa_impl))
            vmovups(vbias, addr);
        else
            vmaskmovps(vbias, vmm_tail_mask, addr);
    }
    jmp(init_done, T_NEAR);
    L(init_zero);

    for (int n_ = 0; n_ < n_loop; n_++) {
        uni_vxorps(get_bias_reg(n_), get_bias_reg(n_), get_bias_reg(n_));
    }
    L(init_done);

    mov(reg_k_iter, utils::div_up(brg_.reduce_dim, mult_));
    L(k_loop);
    {
        int n_ = 0;
        for (; n_ < n_iters; n_++)
            accumulate_bias(n_, false);

        if (nb_tail > 0) accumulate_bias(n_, true);

        add(aux_reg_ddst, ddst_typesize_ * mult_ * brg_.LDB);

        sub(reg_k_iter, 1);
        jnz(k_loop, T_NEAR);
    }

    Xbyak::Label store_final, store_done;
    test(reg_flag, FLAG_REDUCE_LAST);
    jnz(store_final, T_NEAR); // FLAG_REDUCE_LAST is set

    n_ = 0;
    for (; n_ < n_iters; n_++) {
        auto vbias = get_bias_reg(n_);
        auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
        vmovups(addr, vbias);
    }
    if (nb_tail > 0) {
        auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
        auto vbias = get_bias_reg(n_);
        if (isa_has_masks(brg_.isa_impl)) {
            vbias = vmm_mask(vbias, true, true, k_tail_mask);
            vmovups(addr, vbias);
        } else {
            vmaskmovps(addr, vmm_tail_mask, vbias);
        }
    }
    jmp(store_done, T_NEAR);

    L(store_final);
    n_ = 0;

    for (; n_ < n_iters; n_++)
        store(n_, false);

    if (nb_tail > 0) store(n_, true);

    L(store_done);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::init_masks(
        int tail_length) {
    if (ddst_dt_ == data_type::f16) {
        const auto half_mask = size_t((1 << 16) - 1);
        mov(reg_mask, half_mask);
        kmovq(k_f16_perm_mask, reg_mask);

        vmovups(vreg_perm | k_f16_perm_mask | T_z, ptr[rip + f16_perm_table_]);
    }

    if (reduce_kind_ == matmul_reduce_kind::src
            && utils::one_of(bia_dt_, data_type::f16, data_type::bf16)) {
        mov(reg_mask, 1);
        kmovq(k_store_mask, reg_mask);
    }

    if (tail_length == 0) return;
    if (isa_has_masks(brg_.isa_impl)) {
        const auto full_mask = size_t {0xffffffffffffffff};
        const auto tail_mask = size_t((1 << tail_length) - 1);
        mov(reg_mask, full_mask);
        kmovq(k_full_mask, reg_mask);
        mov(reg_mask, tail_mask);
        kmovq(k_tail_mask, reg_mask);

    } else {
        vmovups(vmm_tail_mask, ptr[rip + mask_label_]);
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<
        Vmm>::generate_for_b() {

    int nb = utils::div_up(brg_.load_dim, brg_.ld_block);
    int nb_tail = brg_.load_dim % brg_.ld_block;

    int n_loop = nb / n_max_regs_;
    int n_loop_tail = nb % n_max_regs_;
    if (n_loop_tail == 0 && nb_tail > 0) {
        n_loop--;
        n_loop_tail = n_max_regs_;
    }

    init_masks(nb_tail);

    mov(reg_ddst, ptr[param1 + GET_OFF(ptr_diff_dst)]);
    mov(reg_bias_acc, ptr[param1 + GET_OFF(ptr_diff_bias_acc)]);
    mov(reg_bias, ptr[param1 + GET_OFF(ptr_diff_bias)]);
    mov(reg_flag, ptr[param1 + GET_OFF(flags)]);

    for (int nb_ = 0; nb_ < n_loop; nb_++) {
        loop_by_N(n_max_regs_, 0);

        add(reg_ddst, ddst_typesize_ * mult_ * n_max_regs_ * brg_.ld_block);
        add(reg_bias, bia_typesize_ * n_max_regs_ * brg_.ld_block);
        add(reg_bias_acc, acc_typesize_ * n_max_regs_ * brg_.ld_block);
    }

    if (n_loop_tail > 0) loop_by_N(n_loop_tail, nb_tail);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<
        Vmm>::generate_for_a() {

    mov(reg_ddst, ptr[param1 + GET_OFF(ptr_diff_dst)]);
    mov(reg_bias_acc, ptr[param1 + GET_OFF(ptr_diff_bias_acc)]);
    mov(reg_bias, ptr[param1 + GET_OFF(ptr_diff_bias)]);
    mov(reg_flag, ptr[param1 + GET_OFF(flags)]);

    const int k_tail = brg_.reduce_dim % brg_.ld_block;
    init_masks(k_tail);

    for (int m = 0; m < brg_.load_dim; m++) {
        loop_by_K();
        add(reg_ddst, ddst_typesize_ * brg_.LDA);
        add(reg_bias, bia_typesize_);
        add(reg_bias_acc, acc_typesize_);
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_diff_bias_t<Vmm>::generate() {
    preamble();

    if (ddst_dt_ == data_type::bf16) {
        auto reg_tmp = rax;
        auto reg_unit_val = reg_tmp.cvt16();
        mov(reg_unit_val, 0x3f80); // bf16 values of 1.
        vpbroadcastw(vreg_unit, reg_unit_val);
    }

    int tail = 0;

    // Currently, `reduce_kind` is `undef` when this kernel is used from
    // BRGEMM-based Inner Product.
    if (reduce_kind_ == matmul_reduce_kind::undef) {
        tail = brg_.load_dim % brg_.ld_block;
        generate_for_b();
    } else if (reduce_kind_ == matmul_reduce_kind::src) {
        tail = brg_.reduce_dim % brg_.ld_block;
        generate_for_a();
    } else {
        assert(!"Unsupported reduce kind");
    }

    postamble();

    if (ddst_dt_ == data_type::f16) {
        // convert interleaved vnni data with holes to packed.
        const uint16_t f16_prm_array[16]
                = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
        align(64);
        L(f16_perm_table_);
        for (int i = 0; i < 16; ++i)
            dw(f16_prm_array[i]);
    }

    if (!isa_has_masks(brg_.isa_impl) && tail > 0) {
        align(32);
        L(mask_label_);
        for (int i = 0; i < tail; ++i)
            dd(~uint32_t(0));
        for (int i = tail; i < brg_.ld_block; ++i)
            dd(0);
    }
}

#undef GET_OFF

template struct jit_brgemm_kernel_diff_bias_t<Xbyak::Ymm>;
template struct jit_brgemm_kernel_diff_bias_t<Xbyak::Zmm>;

jit_brgemm_kernel_post_ops_base_t *jit_brgemm_kernel_post_ops_base_t::create(
        cpu_isa_t isa, const brgemm_desc_t &abrg,
        const primitive_attr_t &aattr) {
    if (utils::one_of(isa, avx2, avx2_vnni, avx2_vnni_2)) {
        return new jit_brgemm_kernel_post_ops_t<Xbyak::Ymm>(abrg, aattr);
    } else {
        return new jit_brgemm_kernel_post_ops_t<Xbyak::Zmm>(abrg, aattr);
    }
}

#define GET_OFF(field) offsetof(brgemm_kernel_post_ops_args_t, field)

template <typename Vmm>
dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::jit_brgemm_kernel_post_ops_t(const brgemm_desc_t &abrg,
        const primitive_attr_t &aattr)
    : jit_generator(jit_name(), abrg.isa_impl)
    , brg_(abrg)
    , attr_(aattr)
    , max_vregs_(isa_num_vregs(brg_.isa_impl))
    , with_binary_non_scalar_bcast_(brg_.with_binary
              && binary_injector::any_binary_postop_rhs_non_scalar_broadcast(
                      attr_.post_ops_, memory_desc_wrapper(brg_.dst_md()))) {
    bool has_f8_e5m2_binary_postops = false;
    bool has_f8_e4m3_binary_postops = false;
    if (brg_.with_binary) {
        const auto &post_ops = attr_.post_ops_;
        for (int i = 0; i < post_ops.len(); i++) {
            const auto &entry = post_ops.entry_[i];
            if (!entry.is_binary()) continue;
            has_f8_e5m2_binary_postops
                    = entry.binary.src1_desc.data_type == data_type::f8_e5m2;
            has_f8_e4m3_binary_postops
                    = entry.binary.src1_desc.data_type == data_type::f8_e4m3;
        }
    }

    if (brg_.is_bf16_emu)
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this, emu_reserv_1,
                emu_reserv_2, emu_reserv_3, emu_scratch, emu_reserv_4,
                emu_reserv_4);
    if (brg_.is_fp8_via_convert() || has_f8_e5m2_binary_postops
            || has_f8_e4m3_binary_postops) {
        if (utils::one_of(data_type::f8_e5m2, brg_.dt_a, brg_.dt_b, brg_.dt_d)
                || has_f8_e5m2_binary_postops)
            f8_e5m2_emulator_ = utils::make_unique<fp8_emulation_e5m2_t>(this,
                    emu_reserv_1, emu_reserv_2, emu_reserv_3, emu_mask,
                    emu_scratch);
        if (utils::one_of(data_type::f8_e4m3, brg_.dt_a, brg_.dt_b, brg_.dt_d)
                || has_f8_e4m3_binary_postops)
            f8_e4m3_emulator_ = utils::make_unique<fp8_emulation_e4m3_t>(this,
                    emu_reserv_1, emu_reserv_2, emu_reserv_3, emu_reserv_4,
                    emu_reserv_5, emu_scratch);
    }

    if (brg_.beta != 0) {
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = true;
        static constexpr bool use_exact_tail_scalar_bcast = false;

        const binary_injector::rhs_arg_static_params_t rhs_sp {
                static_cast<size_t>(vmm_tmp(4).getIdx()), this->r14, this->r15,
                this->r13, preserve_gpr, preserve_vmm,
                GET_OFF(ptr_binary_post_ops_rhs), GET_OFF(dst_orig),
                memory_desc_wrapper(brg_.dst_md()),
                static_cast<size_t>(brg_.load_dim % brg_.ld_block), k_tail_mask,
                use_exact_tail_scalar_bcast};
        const binary_injector::static_params_t bsp(this->param1,
                binary_injector::get_all_strategies_supported_by_injector(),
                rhs_sp, f8_e5m2_emulator_.get(), f8_e4m3_emulator_.get());

        const bool save_state = brg_.with_eltwise;
        const auto &reserved_eltwise_gpr = reg_reserved_eltwise;
        const auto reserved_eltwise_maskr = Xbyak::Opmask(1);

        const eltwise_injector::static_params_t esp {
                save_state, reserved_eltwise_gpr, reserved_eltwise_maskr};

        auto st = safe_ptr_assign(postops_injector_,
                po_injector_t::create(
                        this, brg_.isa_impl, attr_.post_ops_, bsp, esp));
        if (st != status::success) {
            assert(!"postops_injector creation failed");
        }
    }

    const auto &wei_scales = attr_.scales_.get(DNNL_ARG_WEIGHTS);
    // per_oc: conv: 1 << 0, (1 << 1) + (1 << 0) (with groups)
    // per_oc: ip: 1 << 0
    is_oc_scale_
            = utils::one_of(wei_scales.get_mask(), 1 << 0, (1 << 1) + (1 << 0));

    inp_dt_ = brg_.dt_c;
    out_dt_ = brg_.dt_d;
    bia_dt_ = brg_.dt_bias;

    inp_typesize_ = brg_.typesize_C;
    out_typesize_ = brg_.typesize_D;
    bia_typesize_ = brg_.typesize_bias;
}

template <typename Vmm>
int dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::zp_c_values_offset(
        int n, bool is_tail /*= false*/) const noexcept {
    if (brg_.zp_type_c == brgemm_broadcast_t::per_n) {
        return (is_tail) ? sizeof(int32_t) * brg_.ldb_tail
                         : sizeof(int32_t) * n * brg_.ld_block;
    }

    return 0;
}

template <typename Vmm>
int dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::zp_comp_a_vpad_offset(int n, int m,
        bool is_tail /*= false*/) const noexcept {
    return (is_tail) ? sizeof(int32_t) * (brg_.ldb_tail + m * brg_.LDB)
                     : sizeof(int32_t) * (n * brg_.ld_block + m * brg_.LDB);
}

template <typename Vmm>
int dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::mb_zp_comp_a_offset(int m_block) const noexcept {
    return sizeof(int32_t) * m_block * brg_.LDB;
}

template <typename Vmm>
int dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::compensation_vpad_offset(int n, int m,
        bool is_tail /*= false*/) const noexcept {
    return (is_tail) ? sizeof(int32_t) * (brg_.ldb_tail + m * brg_.LDB)
                     : sizeof(int32_t) * (n * brg_.ld_block + m * brg_.LDB);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::cvt2ps(
        data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
        int tail_size, bool store, Xbyak::Opmask ktail_mask,
        bool skip_cvt2ps /*= false*/) {
    const bool is_tail = op.isMEM()
            && tail_size != vreg_traits<Vmm>::vlen / sizeof(float)
            // The current kernel is written such that tail_size = 0 implies
            // no tail and full vmm must be processed.
            && tail_size > 0;

    if (IMPLICATION(is_tail, isa_has_masks(brg_.isa_impl))) {
        const Vmm vmm = maybe_mask(vmm_in, is_tail, store, ktail_mask);
        switch (type_in) {
            case data_type::f32:
            case data_type::s32: vmovups(vmm, op); break;
            case data_type::s8: vpmovsxbd(vmm, op); break;
            case data_type::u8: vpmovzxbd(vmm, op); break;
            case data_type::bf16:
                vpmovzxwd(vmm, op);
                vpslld(vmm, vmm, 16);
                break;
            case data_type::f16: vcvtph2ps(vmm, op); break;
            case data_type::f8_e5m2:
                if (brg_.is_fp8_via_convert())
                    f8_e5m2_emulator_->vcvt_f8_to_f32(vmm, op);
                else
                    assert(!"Not supported yet");
                break;
            case data_type::f8_e4m3:
                if (brg_.is_fp8_via_convert())
                    f8_e4m3_emulator_->vcvt_f8_to_f32(vmm, op);
                else
                    assert(!"Not supported yet");
                break;
            default: assert(!"unsupported data type");
        }
    } else {
        load_data(type_in, vmm_in, op.getAddress(), tail_size);
    }
    if (!skip_cvt2ps && types::is_integral_dt(type_in))
        uni_vcvtdq2ps(vmm_in, vmm_in);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::inject_attr_postops(int m_block, int n_block, int tail /*= 0*/) {
    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const auto k_mask = tail == 0 ? k_full_mask : k_tail_mask;
    const auto sum_dt = p.get_sum_dt(out_dt_);

    const auto sum_injector = [&] {
        const float *p_sum_scale = &p.entry_[sum_idx].sum.scale;
        const int32_t *p_sum_zp = &p.entry_[sum_idx].sum.zero_point;
        if (*p_sum_scale != 1.f) mov(reg_ptr_sum_scale, (size_t)p_sum_scale);
        auto vmm_sum_zp = vmm_tmp(1);
        if (*p_sum_zp != 0) {
            mov(reg_ptr_sum_zp, (size_t)p_sum_zp);
            if (is_superset(brg_.isa_impl, avx512_core)) {
                vcvtdq2ps(vmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
            } else {
                vpbroadcastd(vmm_sum_zp, ptr[reg_ptr_sum_zp]);
                uni_vcvtdq2ps(vmm_sum_zp, vmm_sum_zp);
            }
        }

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            const auto vmm = vector(m, n, n_block);
            const auto addr = ptr[aux_reg_out
                    + out_typesize_ * (m * brg_.LDD + n * brg_.ld_block)];

            const auto vmm_prev_dst = vmm_tmp(0);
            cvt2ps(sum_dt, vmm_prev_dst, addr, tail, false, k_mask);
            if (*p_sum_zp != 0)
                uni_vsubps(vmm_prev_dst, vmm_prev_dst, vmm_sum_zp);
            if (*p_sum_scale == 1.f)
                uni_vaddps(vmm, vmm, vmm_prev_dst);
            else {
                if (is_superset(brg_.isa_impl, avx512_core)) {
                    vfmadd231ps(vmm, vmm_prev_dst, ptr_b[reg_ptr_sum_scale]);
                } else {
                    auto vmm_sum_scale = vmm_tmp(2);
                    vpbroadcastd(vmm_sum_scale, ptr[reg_ptr_sum_scale]);
                    vfmadd231ps(vmm, vmm_prev_dst, vmm_sum_scale);
                }
            }
        }
    };

    if (brg_.with_sum) {
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }

    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    if (with_binary_non_scalar_bcast_) {
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            const auto vmm_idx = vector(m, n, n_block).getIdx();
            const size_t aux_output_offset
                    = out_typesize_ * (m * brg_.LDD + n * brg_.ld_block);

            rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, aux_reg_out);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                    vmm_idx, aux_output_offset);
            if (tail) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
        }
    }

    postops_injector_->compute_vector_range(
            0, m_block * n_block, rhs_arg_params);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::apply_comp(
        int m_block, int n_block, int tail /*= 0*/) {
    auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;
    const bool has_tail = tail > 0;
    if (brg_.zp_type_a != brgemm_broadcast_t::none) {
        auto vmm_zp_a_val = vmm_tmp(1);
        mov(reg_zp_a_val, ptr[rsp + reg_zp_a_val_offs_]);
        uni_vpbroadcastd(vmm_zp_a_val, reg_zp_a_val.cvt32());

        mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
        const auto vmm_zp_comp_a = vmm_tmp(0);
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            const size_t zp_comp_offset
                    = sizeof(int32_t) * (n * brg_.ld_block + m * brg_.LDB);
            auto zp_comp_a_addr = is_superset(brg_.isa_impl, avx512_core)
                    ? EVEX_compress_addr(aux_reg_zp_a_comp, zp_comp_offset)
                    : ptr[aux_reg_zp_a_comp + zp_comp_offset];
            if (IMPLICATION(has_tail, isa_has_masks(brg_.isa_impl))) {
                auto vmm_zp_comp_a_masked
                        = maybe_mask(vmm_zp_comp_a, has_tail, false, k_mask);
                vmovups(vmm_zp_comp_a_masked, zp_comp_a_addr);
            } else {
                load_data(data_type::s32, vmm_zp_comp_a, zp_comp_a_addr, tail);
            }
            uni_vpmulld(vmm_zp_comp_a, vmm_zp_a_val, zp_comp_a_addr);

            auto vmm = vector(m, n, n_block);
            uni_vpaddd(vmm, vmm, vmm_zp_comp_a);
        }
    }

    if (brg_.req_s8s8_compensation) {
        mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
        const auto vmm_comp = vmm_tmp(0);
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            const size_t s8s8_comp_offset
                    = sizeof(int32_t) * (n * brg_.ld_block + m * brg_.LDB);

            auto comp_addr = is_superset(brg_.isa_impl, avx512_core)
                    ? EVEX_compress_addr(aux_reg_s8s8_comp, s8s8_comp_offset)
                    : ptr[aux_reg_s8s8_comp + s8s8_comp_offset];
            if (IMPLICATION(tail > 0, isa_has_masks(brg_.isa_impl))) {
                auto vmm_comp_masked
                        = maybe_mask(vmm_comp, tail > 0, false, k_mask);
                vmovups(vmm_comp_masked, comp_addr);
            } else
                load_data(data_type::s32, vmm_comp, comp_addr, tail);

            auto vmm = vector(m, n, n_block);
            uni_vpaddd(vmm, vmm, vmm_comp);
        }
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::maybe_apply_comp(
        int m_block, int n_block, int tail /*= 0*/) {
    Xbyak::Label label_apply_without_comp;
    mov(reg_apply_comp, ptr[rsp + reg_apply_comp_offs_]);
    cmp(reg_apply_comp, 0);
    je(label_apply_without_comp, T_NEAR);
    apply_comp(m_block, n_block, tail);
    L_aligned(label_apply_without_comp);

    for_(int m = 0; m < m_block; m++)
    for (int n = 0; n < n_block; n++) {
        uni_vcvtdq2ps(vector(m, n, n_block), vector(m, n, n_block));
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::apply_post_ops(
        int m_block, int n_block, int tail /*= 0*/) {
    const auto vector = [=](int m, int n) { return Vmm(m * n_block + n); };
    auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;
    const auto req_comp = brg_.is_int8 && brg_.beta != 0
            && (brg_.req_s8s8_compensation
                    || brg_.zp_type_a != brgemm_broadcast_t::none);

    // brg_.alpha == 0 means initialize registers, 1 means read from input
    // brg_.beta == 0 means skip postwork, 1 means do postwork
    // req_comp == true -> convert accumulated values to f32 after applying
    // compensation to avoid the loss of accuracy when converting s32 to f32
    for_(int m = 0; m < m_block; m++)
    for (int n = 0; n < n_block; n++) {
        if (brg_.alpha == 0) {
            // have to init vmm each time because vectors may have been
            // changed in the previous iterations
            uni_vpxor(vector(m, n), vector(m, n), vector(m, n));
        } else {
            auto inp_addr = ptr[aux_reg_in
                    + inp_typesize_ * (m * brg_.LDC + n * brg_.ld_block)];
            cvt2ps(inp_dt_, vector(m, n), inp_addr, tail, false, k_mask,
                    req_comp);
        }
    }

    if (req_comp) maybe_apply_comp(m_block, n_block, tail);

    if (brg_.beta != 0) {
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            const auto addr = ptr[aux_reg_scales
                    + is_oc_scale_ * sizeof(float) * (n * brg_.ld_block)];
            auto vmm = vector(m, n);
            if (IMPLICATION(tail > 0, isa_has_masks(brg_.isa_impl))) {
                vmm = maybe_mask(vector(m, n), tail > 0, false, k_mask);
                vmulps(vmm, vmm, addr);
            } else {
                auto vmm_scales = vmm_tmp(0);
                load_data(data_type::f32, vmm_scales, addr, tail);
                vmulps(vmm, vmm, vmm_scales);
            }
        }
    }

    if (brg_.beta != 0 && brg_.with_bias) {
        for (int n = 0; n < n_block; n++) {
            auto vmm_bias = vmm_tmp(0);
            auto bias_addr
                    = ptr[aux_reg_bias + bia_typesize_ * (n * brg_.ld_block)];
            cvt2ps(bia_dt_, vmm_bias, bias_addr, tail, false, k_mask);
            for (int m = 0; m < m_block; m++) {
                vaddps(vector(m, n), vmm_bias);
            }
        }
    }

    if (postops_injector_) inject_attr_postops(m_block, n_block, tail);

    if (brg_.beta != 0 && brg_.with_dst_scales) {
        mov(aux_reg_dst_scales, ptr[rsp + reg_dst_scales_offs_]);
        const auto addr = ptr[aux_reg_dst_scales];
        auto vmm_scales = vmm_tmp(0);
        if (!isa_has_masks(brg_.isa_impl)) vmovups(vmm_scales, addr);

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto vmm = vector(m, n);
            if (isa_has_masks(brg_.isa_impl)) {
                vmm = maybe_mask(vector(m, n), tail > 0, false, k_mask);
                vmulps(vmm, vmm, addr);
            } else {
                vmulps(vmm, vmm, vmm_scales);
            }
        }
    }

    if (brg_.beta != 0 && brg_.zp_type_c != brgemm_broadcast_t::none) {
        mov(aux_reg_zp_c_values, ptr[rsp + aux_reg_zp_c_values_offs_]);
        auto vmm_zp_c = vmm_tmp(0);
        if (brg_.zp_type_c == brgemm_broadcast_t::per_tensor) {
            if (is_superset(brg_.isa_impl, avx512_core))
                vcvtdq2ps(vmm_zp_c,
                        EVEX_compress_addr(aux_reg_zp_c_values, 0, true));
            else {
                uni_vbroadcastss(vmm_zp_c, ptr[aux_reg_zp_c_values]);
                uni_vcvtdq2ps(vmm_zp_c, vmm_zp_c);
            }
        }
        for (int n = 0; n < n_block; n++) {
            if (brg_.zp_type_c == brgemm_broadcast_t::per_n) {
                int zp_c_off = zp_c_values_offset(n);
                auto zp_c_addr = is_superset(brg_.isa_impl, avx512_core)
                        ? EVEX_compress_addr(aux_reg_zp_c_values, zp_c_off)
                        : ptr[aux_reg_zp_c_values + zp_c_off];
                cvt2ps(data_type::s32, vmm_zp_c, zp_c_addr, tail, false,
                        k_mask);
            }
            for (int m = 0; m < m_block; m++) {
                const auto vmm = vector(m, n);
                uni_vaddps(vmm, vmm, vmm_zp_c);
            }
        }
    }

    const bool dt_requires_saturation = types::is_integral_dt(out_dt_);

    const reg64_t reg_tmp_gpr = reg_tmp;
    auto vmm_lbound = vmm_tmp(0);
    auto vmm_ubound = vmm_tmp(1);
    if (dt_requires_saturation) {
        init_saturate_f32(
                vmm_lbound, vmm_ubound, reg_tmp_gpr, data_type::f32, out_dt_);
    }

    if (brg_.is_bf16_emu) bf16_emu_->init_vcvtneps2bf16();

    for_(int m = 0; m < m_block; m++)
    for (int n = 0; n < n_block; n++) {
        // incase of tail, stores are unconditionally masked, regardless
        // of `n`, implying n_block must be equal to `1`.
        assert(IMPLICATION(tail > 0, n_block == 1));
        auto vmm = vector(m, n);
        const size_t offset
                = out_typesize_ * (m * brg_.LDD + n * brg_.ld_block);
        const auto addr = ptr[aux_reg_out + offset];

        if (dt_requires_saturation) {
            saturate_cvt_f32(vmm, vmm_lbound, vmm_ubound, out_dt_);
        }

        if (is_superset(brg_.isa_impl, avx512_core)) {
            auto vmm_masked = maybe_mask(vmm, tail > 0, true, k_mask);
            Vmm_lower_t vmm_low = Vmm_lower_t(vmm.getIdx());
            Vmm_lower2_t vmm_low2 = Vmm_lower2_t(vmm_low.getIdx());
            auto vmm_low_masked = maybe_mask(vmm_low, tail > 0, true, k_mask);
            auto vmm_low2_masked = maybe_mask(vmm_low2, tail > 0, true, k_mask);
            switch (out_dt_) {
                case data_type::f32:
                case data_type::s32: uni_vmovups(addr, vmm_masked); break;
                case data_type::bf16:
                    if (brg_.is_bf16_emu) {
                        bf16_emu_->vcvtneps2bf16(vmm_low, vmm);
                        vmovdqu16(addr, vmm_low_masked);
                    } else {
                        vcvtneps2bf16(vmm_low, vmm);
                        vmovdqu16(addr, vmm_low_masked);
                    }
                    break;
                case data_type::f16:
                    vcvtps2ph(vmm_low, vmm, _op_mxcsr);
                    vmovdqu16(addr, vmm_low_masked);
                    break;
                case data_type::f8_e5m2:
                    if (brg_.is_fp8_via_convert()) {
                        f8_e5m2_emulator_->vcvt_f32_to_f8(vmm_low2, vmm);
                        vmovdqu8(addr, vmm_low2_masked);
                    } else
                        assert(!"Not supported yet");
                    break;
                case data_type::f8_e4m3:
                    if (brg_.is_fp8_via_convert()) {
                        f8_e4m3_emulator_->vcvt_f32_to_f8(vmm_low2, vmm);
                        vmovdqu8(addr, vmm_low2_masked);
                    } else
                        assert(!"Not supported yet");
                    break;
                case data_type::s8: vpmovsdb(addr, vmm_masked); break;
                case data_type::u8: vpmovusdb(addr, vmm_masked); break;
                default: assert(!"unknown dst_dt");
            }
        } else {
            const int simd_w = vreg_traits<Vmm>::vlen / sizeof(float);
            const int nelems = tail > 0 ? tail : simd_w;
            store_data(out_dt_, vmm, aux_reg_out, offset, nelems);
        }
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::loop_by_N(
        int m_block, int nb2, int nb2_tail, int nb_tail) {
    if (brg_.alpha) { mov(aux_reg_in, reg_in); }
    if (brg_.beta != 0) {
        if (brg_.with_bias) mov(aux_reg_bias, reg_bias);
        if (brg_.zp_type_c != brgemm_broadcast_t::none) {
            mov(aux_reg_zp_c_values, ptr[rsp + reg_zp_c_values_offs_]);
            mov(ptr[rsp + aux_reg_zp_c_values_offs_], aux_reg_zp_c_values);
        }
        if (brg_.zp_type_a != brgemm_broadcast_t::none) {
            mov(aux_reg_zp_a_comp, ptr[rsp + reg_zp_a_comp_offs_]);
            mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
        }
        if (brg_.req_s8s8_compensation) {
            mov(aux_reg_s8s8_comp, ptr[rsp + reg_s8s8_comp_offs_]);
            mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
        }
        mov(aux_reg_scales, reg_scales);
    }
    mov(aux_reg_out, reg_out);

    for (int n_loop_ = 0; n_loop_ < nb2; n_loop_++) {
        apply_post_ops(m_block, n_block2_);

        const auto oc_l_offset = n_block2_ * brg_.ld_block;

        add(aux_reg_out, out_typesize_ * oc_l_offset);
        if (brg_.alpha != 0) { add(aux_reg_in, inp_typesize_ * oc_l_offset); }
        if (brg_.beta != 0) {
            if (brg_.with_bias) add(aux_reg_bias, bia_typesize_ * oc_l_offset);
            if (brg_.zp_type_c != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_c_values, ptr[rsp + aux_reg_zp_c_values_offs_]);
                add(aux_reg_zp_c_values, zp_c_values_offset(n_block2_));
                mov(ptr[rsp + aux_reg_zp_c_values_offs_], aux_reg_zp_c_values);
            }
            if (brg_.zp_type_a != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                add(aux_reg_zp_a_comp, sizeof(int32_t) * oc_l_offset);
                mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
            }
            if (brg_.req_s8s8_compensation) {
                mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                add(aux_reg_s8s8_comp, sizeof(int32_t) * oc_l_offset);
                mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
            }

            add(aux_reg_scales, is_oc_scale_ * sizeof(float) * oc_l_offset);
        }
    }
    if (nb2_tail > 0) {
        apply_post_ops(m_block, nb2_tail);
        const auto oc_l_offset = nb2_tail * brg_.ld_block;

        add(aux_reg_out, out_typesize_ * oc_l_offset);
        if (brg_.alpha != 0) { add(aux_reg_in, inp_typesize_ * oc_l_offset); }
        if (brg_.beta != 0) {
            if (brg_.with_bias) add(aux_reg_bias, bia_typesize_ * oc_l_offset);
            if (brg_.zp_type_c != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_c_values, ptr[rsp + aux_reg_zp_c_values_offs_]);
                add(aux_reg_zp_c_values, zp_c_values_offset(nb2_tail));
                mov(ptr[rsp + aux_reg_zp_c_values_offs_], aux_reg_zp_c_values);
            }
            if (brg_.zp_type_a != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                add(aux_reg_zp_a_comp, sizeof(int32_t) * oc_l_offset);
                mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
            }
            if (brg_.req_s8s8_compensation) {
                mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                add(aux_reg_s8s8_comp, sizeof(int32_t) * oc_l_offset);
                mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
            }

            add(aux_reg_scales, is_oc_scale_ * sizeof(float) * oc_l_offset);
        }
    }
    if (nb_tail > 0) {
        apply_post_ops(m_block, 1, nb_tail);

        if (brg_.alpha != 0) { add(aux_reg_in, inp_typesize_ * (nb_tail)); }
        if (brg_.beta != 0) {
            if (brg_.with_bias) add(aux_reg_bias, bia_typesize_ * (nb_tail));
            if (brg_.zp_type_c != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_c_values, ptr[rsp + aux_reg_zp_c_values_offs_]);
                add(aux_reg_zp_c_values, zp_c_values_offset(1, nb_tail));
                mov(ptr[rsp + aux_reg_zp_c_values_offs_], aux_reg_zp_c_values);
            }
            if (brg_.zp_type_a != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                add(aux_reg_zp_a_comp, sizeof(int32_t) * nb_tail);
                mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
            }
            if (brg_.req_s8s8_compensation) {
                mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                add(aux_reg_s8s8_comp, sizeof(int32_t) * nb_tail);
                mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
            }
            add(aux_reg_scales, is_oc_scale_ * bia_typesize_ * (nb_tail));
        }
        add(aux_reg_out, out_typesize_ * (nb_tail));
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::generate() {
    preamble();

    sub(rsp, stack_space_needed_);

    int nb = brg_.load_dim / brg_.ld_block;
    int nb_tail = brg_.load_dim % brg_.ld_block;

    int nb2 = nb / n_block2_;
    int nb2_tail = nb % n_block2_;
    int n_block = (nb2 == 0) ? nstl::max(1, nb2_tail) : n_block2_;

    int m_max_regs = (brg_.is_bf16_emu
                    ? 24
                    : (brg_.is_fp8_via_convert() ? 23 : max_vregs_ - 4));
    m_max_regs /= n_block;

    int m_block = nstl::min(brg_.bcast_dim, m_max_regs);

    int mb = brg_.bcast_dim / m_block;
    int mb_tail = brg_.bcast_dim % m_block;

    if (isa_has_masks(brg_.isa_impl)) {
        const auto full_mask = size_t {0xffffffffffffffff};
        const auto tail_mask = size_t((1 << nb_tail) - 1);

        reg64_t reg_mask = reg_tmp;
        mov(reg_mask, full_mask);
        kmovq(k_full_mask, reg_mask);
        mov(reg_mask, tail_mask);
        kmovq(k_tail_mask, reg_mask);
    }

    if (brg_.alpha != 0) { mov(reg_in, ptr[param1 + GET_OFF(ptr_in)]); }
    if (brg_.beta != 0) {
        mov(reg_scales, ptr[param1 + GET_OFF(ptr_scales)]);
        mov(reg_apply_comp, ptr[param1 + GET_OFF(apply_comp)]);
        mov(ptr[rsp + reg_apply_comp_offs_], reg_apply_comp);

        if (brg_.with_bias) mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
        if (brg_.zp_type_c != brgemm_broadcast_t::none) {
            mov(reg_zp_c_values, ptr[param1 + GET_OFF(c_zp_values)]);
            mov(ptr[rsp + reg_zp_c_values_offs_], reg_zp_c_values);
        }
        if (brg_.zp_type_a != brgemm_broadcast_t::none) {
            mov(reg_zp_a_comp, ptr[param1 + GET_OFF(a_zp_compensation)]);
            mov(ptr[rsp + reg_zp_a_comp_offs_], reg_zp_a_comp);

            mov(reg_zp_a_val, ptr[param1 + GET_OFF(a_comp_val)]);
            mov(ptr[rsp + reg_zp_a_val_offs_], reg_zp_a_val);
        }
        if (brg_.req_s8s8_compensation) {
            mov(reg_s8s8_comp, ptr[param1 + GET_OFF(s8s8_compensation)]);
            mov(ptr[rsp + reg_s8s8_comp_offs_], reg_s8s8_comp);
        }
        if (brg_.with_dst_scales) {
            mov(reg_dst_scales, ptr[param1 + GET_OFF(ptr_dst_scales)]);
            mov(ptr[rsp + reg_dst_scales_offs_], reg_dst_scales);
        }
    }
    mov(reg_out, ptr[param1 + GET_OFF(ptr_out)]);

    for (int mb_ = 0; mb_ < mb; mb_++) {
        loop_by_N(m_block, nb2, nb2_tail, nb_tail);

        if (brg_.alpha != 0) add(reg_in, inp_typesize_ * (m_block * brg_.LDC));
        if (brg_.beta != 0) {
            if (brg_.zp_type_a != brgemm_broadcast_t::none) {
                mov(reg_zp_a_comp, ptr[rsp + reg_zp_a_comp_offs_]);
                add(reg_zp_a_comp, mb_zp_comp_a_offset(m_block));
                mov(ptr[rsp + reg_zp_a_comp_offs_], reg_zp_a_comp);
            }
            if (brg_.req_s8s8_compensation) {
                mov(reg_s8s8_comp, ptr[rsp + reg_s8s8_comp_offs_]);
                add(reg_s8s8_comp, mb_compensation_offset(m_block));
                mov(ptr[rsp + reg_s8s8_comp_offs_], reg_s8s8_comp);
            }
        }
        add(reg_out, out_typesize_ * (m_block * brg_.LDD));
    }
    if (mb_tail > 0) loop_by_N(mb_tail, nb2, nb2_tail, nb_tail);

    add(rsp, stack_space_needed_);

    postamble();

    if (postops_injector_)
        postops_injector_->prepare_table(/* generate = */ true);
    if (brg_.is_fp8_via_convert()) {
        if (f8_e5m2_emulator_) f8_e5m2_emulator_->prepare_table();
        if (f8_e4m3_emulator_) f8_e4m3_emulator_->prepare_table();
    }
}

#undef GET_OFF

template struct jit_brgemm_kernel_post_ops_t<Xbyak::Ymm>;
template struct jit_brgemm_kernel_post_ops_t<Xbyak::Zmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
