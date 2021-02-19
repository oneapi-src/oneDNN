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

#include "common/c_types_map.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_resampling_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
using namespace format_tag;

#define GET_OFF(field) offsetof(jit_resampling_call_s, field)

template <cpu_isa_t isa>
jit_uni_resampling_kernel<isa>::jit_uni_resampling_kernel(
        const jit_resampling_conf_t conf, const memory_desc_t *dst_md)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, isa)
    , conf_(conf)
    , io_(this, conf_.isa, conf_.data_type,
              {conf_.is_data_size_bigger_than_L3 && conf_.tail == 0},
              io::io_tail_conf_t<Vmm> {conf_.simd_w, conf_.tail, k_tail_mask_,
                      vmm_tail_mask_, reg_tmp_},
              io::io_emu_bf16_conf_t {}, utils::null_opt,
              io::io_gather_conf_t<Vmm> {conf_.simd_w, k_full_mask_,
                      vmm_full_mask_, reg_tmp_, reg_tmp1_}) {
    if (conf_.with_postops) {
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr bool use_exact_tail_scalar_bcast = true;

        const binary_injector::rhs_arg_static_params_t rhs_sp {
                static_cast<size_t>(vmm_post_op_helper_.getIdx()), reg_src_,
                reg_tmp_, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(*dst_md), conf_.tail, k_tail_mask_,
                use_exact_tail_scalar_bcast};

        const binary_injector::static_params_t bsp {
                reg_param, get_supported_bcast_strategies(), rhs_sp};

        postops_injector_
                = utils::make_unique<injector::jit_uni_postops_injector_t<isa>>(
                        this, conf_.post_ops, bsp);
    }
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::apply_sum(
        const int data_idx, const bool is_tail) {
    if (conf_.with_sum) {
        assert(!conf_.sum_scales.empty()
                && "No scales for sum post operation.");
        const auto sum_injector = [this, data_idx, is_tail]() {
            io_.load(ptr[reg_dst_], vmm_tmp_, is_tail);
            const float sum_scale = conf_.sum_scales.front();
            if (sum_scale == 1.f)
                uni_vaddps(Vmm(data_idx), Vmm(data_idx), vmm_tmp_);
            else {
                mov(reg_tmp1_.cvt32(), float2int(sum_scale));
                vmovd(Xmm(vmm_sum_scale_.getIdx()), reg_tmp1_.cvt32());
                vbroadcastss(vmm_sum_scale_, Xmm(vmm_sum_scale_.getIdx()));
            }
        };
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::apply_postops(
        const int data_idx, const bool is_tail, const Reg64 *reg_c) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    if (conf_.with_sum) apply_sum(data_idx, is_tail);

    if (conf_.with_binary) {
        if (conf_.tag_kind == jit_memory_tag_kind_t::blocked) {
            if (isa == sse41) add(reg_c_offset, *reg_c);
            rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                    data_idx, reg_c_offset);
            if (isa == sse41) sub(reg_c_offset, *reg_c);
        } else if (conf_.tag_kind == jit_memory_tag_kind_t::ncsp) {
            rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                    data_idx, reg_c_offset);
        } else {
            rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(data_idx, *reg_c);
        }
        if (is_tail) { rhs_arg_params.vmm_tail_idx_.emplace(data_idx); }
    }

    postops_injector_->compute_vector(data_idx, rhs_arg_params);
}


template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::nearest_ncsp_format() {
    const Reg64 &reg_indices_h = reg_aux_src_0_;
    const Reg64 &reg_indices_w = reg_aux_src_1_;
    const Reg64 &reg_src_shifted = reg_aux_src_2_;
    const Reg64 &reg_oh = reg_tmp1_;

    auto nearest_interpolation = ([&](bool is_tail) {
        uni_vmovdqu(vmm_indices_, ptr[reg_indices_w]);
        io_.gather(reg_src_shifted, vmm_indices_, vmm_src_, is_tail);
        if (conf_.with_postops) apply_postops(vmm_src_.getIdx(), is_tail);
        io_.store(vmm_src_, ptr[reg_dst_], is_tail);
    });

    mov(reg_indices_h, reg_indices_);
    mov(reg_indices_w, reg_indices_);
    add(reg_indices_w, conf_.oh * conf_.el_size_of_indices);

    Label oh_loop_begin, oh_loop_end;
    Label ow_loop_begin, ow_loop_end;
    mov(reg_oh, 0);

    L(oh_loop_begin);
    {
        cmp(reg_oh, conf_.oh);
        jge(oh_loop_end, T_NEAR);
        push(reg_oh);

        mov(reg_work_, conf_.ow);
        mov(reg_src_shifted, reg_src_);
        mov(reg_tmp_, 0);
        mov(reg_tmp_.cvt32(), dword[reg_indices_h]);
        add(reg_src_shifted, reg_tmp_);

        push(reg_indices_w);

        L(ow_loop_begin);
        {
            cmp(reg_work_, conf_.simd_w);
            jl(ow_loop_end, T_NEAR);

            nearest_interpolation(false);

            add(reg_dst_, conf_.simd_w * conf_.dt_size);
            add(reg_indices_w, conf_.simd_w * conf_.el_size_of_indices);
            sub(reg_work_, conf_.simd_w);

            jmp(ow_loop_begin, T_NEAR);
        }
        L(ow_loop_end);

        if (conf_.tail > 0) {
            nearest_interpolation(true);
            add(reg_dst_, conf_.tail * conf_.dt_size);
        }

        add(reg_indices_h, conf_.el_size_of_indices);
        pop(reg_indices_w);
        pop(reg_oh);
        add(reg_oh, 1);
        jmp(oh_loop_begin);
    }
    L(oh_loop_end);
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::nearest_c_oriented_format() {
    const unsigned c_to_compute_without_tail
            = (conf_.inner_stride / conf_.simd_w) * conf_.simd_w;
    const Reg64 &reg_c = reg_tmp_;
    const Reg64 &reg_src_shifted = reg_aux_src_0_;

    Label loop_begin, loop_end;

    L(loop_begin);
    {
        cmp(reg_work_, 1);
        jl(loop_end, T_NEAR);

        mov(reg_src_shifted, reg_src_);
        mov(reg_tmp1_.cvt32(), dword[reg_indices_]);
        add(reg_src_shifted, reg_tmp1_);

        Label c_loop_begin, c_loop_end;
        mov(reg_c, 0);
        L(c_loop_begin);
        {
            cmp(reg_c, c_to_compute_without_tail);
            je(c_loop_end, T_NEAR);

            io_.load(ptr[reg_src_shifted], vmm_src_, false);
            if (conf_.with_postops)
                apply_postops(vmm_src_.getIdx(), false, &reg_c);
            io_.store(vmm_src_, ptr[reg_dst_], false);
            add(reg_src_shifted, conf_.simd_w * conf_.dt_size);
            add(reg_dst_, conf_.simd_w * conf_.dt_size);

            add(reg_c, conf_.simd_w);
            jmp(c_loop_begin, T_NEAR);
        }
        L(c_loop_end);

        if (conf_.tail > 0) {
            io_.load(ptr[reg_src_shifted], vmm_src_, true);
            if (conf_.with_postops)
                apply_postops(vmm_src_.getIdx(), true, &reg_c);
            io_.store(vmm_src_, ptr[reg_dst_], true);
            add(reg_dst_, conf_.tail * conf_.dt_size);
        }

        add(reg_indices_, conf_.el_size_of_indices);

        dec(reg_work_);
        jmp(loop_begin, T_NEAR);
    }
    L(loop_end);
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::linear_ncsp_format() {
    const unsigned indices_stride
            = conf_.ow * conf_.oh * conf_.od * conf_.el_size_of_indices;
    const unsigned weights_stride
            = conf_.ow * conf_.oh * conf_.od * sizeof(float);

    auto linear_interpolation = ([&](const bool is_tail) {
        const Vmm vmm_dst(vmm_idx(0));

        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            uni_vmovdqu(vmm_indices_, ptr[reg_indices_ + i * indices_stride]);
            io_.gather(reg_src_, vmm_indices_, Vmm(vmm_idx(i)), is_tail);
        }

        uni_vmovups(vmm_weights_, ptr[reg_weights]);
        uni_vmulps(vmm_dst, vmm_dst, vmm_weights_);
        for (unsigned i = 1; i < conf_.number_of_corners; i++) {
            uni_vmovups(vmm_weights_, ptr[reg_weights + i * weights_stride]);
            uni_vfmadd231ps(vmm_dst, Vmm(vmm_idx(i)), vmm_weights_);
        }

        if (conf_.with_postops) apply_postops(vmm_idx(0), is_tail);

        io_.store(vmm_dst, ptr[reg_dst_], is_tail);
    });

    Label loop_begin, loop_end;

    L(loop_begin);
    {
        cmp(reg_work_, conf_.simd_w);
        jl(loop_end, T_NEAR);

        linear_interpolation(false);

        add(reg_dst_, conf_.simd_w * conf_.dt_size);
        add(reg_weights, conf_.simd_w * sizeof(float));
        add(reg_indices_, conf_.simd_w * conf_.el_size_of_indices);
        sub(reg_work_, conf_.simd_w);

        jmp(loop_begin, T_NEAR);
    }
    L(loop_end);

    if (conf_.tail > 0) linear_interpolation(true);
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::linear_c_oriented_format() {
    const unsigned c_to_compute_without_tail
            = (conf_.inner_stride / conf_.simd_w) * conf_.simd_w;

    const Reg64 &reg_c = reg_tmp_;
    const Reg64 &reg_index_left = reg_tmp_;
    const Reg64 &reg_index_right = reg_tmp_;

    const std::vector<std::reference_wrapper<const Reg64>> src_regs
            = {reg_src_ftl_, reg_src_ftr_, reg_src_fbl_, reg_src_fbr_,
                    reg_src_btl_, reg_src_btr_, reg_src_bbl_, reg_src_bbr_};
    const std::vector<std::reference_wrapper<const Vmm>> src_vmms
            = {src_ftl_, src_ftr_, src_fbl_, src_fbr_, src_btl_, src_btr_,
                    src_bbl_, src_bbr_};

    assert(src_regs.size() >= conf_.number_of_corners
            && src_vmms.size() >= conf_.number_of_corners);

    auto linear_interpolation = ([&](const Reg64 &reg_c,
                                         const bool is_tail) {
        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            io_.load(ptr[src_regs[i].get()], src_vmms[i].get(), is_tail);
        }

        // w_d[0]*(w_h[0]*(src[0][0][0]*w_w[0] + src[0][0][1]*w_w[1]) +
        //         w_h[1]*(src[0][1][0]*w_w[0] + src[0][1][1]*w_w[1]))
        // +
        // w_d[1]*(w_h[0]*(src[1][0][0]*w_w[0] + src[1][0][1]*w_w[1]) +
        //         w_h[1]*(src[1][1][0]*w_w[0] + src[1][1][1]*w_w[1]))
        uni_vmulps(src_ftl_, src_ftl_, weight_left_);
        uni_vfmadd231ps(src_ftl_, src_ftr_, weight_right_);
        if (conf_.ndims == 4 || conf_.ndims == 5) {
            uni_vmulps(src_fbl_, src_fbl_, weight_left_);
            uni_vfmadd231ps(src_fbl_, src_fbr_, weight_right_);
            uni_vmulps(src_ftl_, src_ftl_, weight_top_);
            uni_vfmadd231ps(src_ftl_, src_fbl_, weight_bottom_);
        }
        if (conf_.ndims == 5) {
            uni_vmulps(src_btl_, src_btl_, weight_left_);
            uni_vfmadd231ps(src_btl_, src_btr_, weight_right_);
            uni_vmulps(src_bbl_, src_bbl_, weight_left_);
            uni_vfmadd231ps(src_bbl_, src_bbr_, weight_right_);
            uni_vmulps(src_btl_, src_btl_, weight_top_);
            uni_vfmadd231ps(src_btl_, src_bbl_, weight_bottom_);
            uni_vmulps(src_ftl_, src_ftl_, weight_front_);
            uni_vfmadd231ps(src_ftl_, src_btl_, weight_back_);
        }

        if (conf_.with_postops)
            apply_postops(src_ftl_.getIdx(), is_tail, &reg_c);
        io_.store(src_ftl_, ptr[reg_dst_], is_tail);
    });

    mov(reg_index_left, 0);

    Label loop_begin, loop_end;
    L(loop_begin);
    {
        cmp(reg_work_, 1);
        jl(loop_end, T_NEAR);

        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            push(src_regs[i]);
        }

        mov(reg_index_left.cvt32(), dword[reg_indices_]);
        for (unsigned i = 0; i < conf_.number_of_corners / 2; i++) {
            add(src_regs[2 * i], reg_index_left);
        }
        mov(reg_index_right.cvt32(),
                dword[reg_indices_ + conf_.el_size_of_indices]);
        for (unsigned i = 0; i < conf_.number_of_corners / 2; i++) {
            add(src_regs[2 * i + 1], reg_index_right);
        }

        uni_vbroadcastss(weight_left_, ptr[reg_weights]);
        uni_vbroadcastss(weight_right_, ptr[reg_weights + sizeof(float)]);

        Label c_loop_begin, c_loop_end;
        mov(reg_c, 0);
        L(c_loop_begin);
        {
            cmp(reg_c, c_to_compute_without_tail);
            je(c_loop_end, T_NEAR);

            linear_interpolation(reg_c, false);
            add(reg_dst_, conf_.simd_w * conf_.dt_size);

            for (unsigned i = 0; i < conf_.number_of_corners; i++)
                add(src_regs[i], conf_.simd_w * conf_.dt_size);

            add(reg_c, conf_.simd_w);
            jmp(c_loop_begin, T_NEAR);
        }
        L(c_loop_end);

        if (conf_.tail > 0) {
            linear_interpolation(reg_c, true);
            add(reg_dst_, conf_.tail * conf_.dt_size);
        }

        // During one loop cycle are read two values for left and
        // right corners from both the weights and indices tables.
        // These two values occurs one after the other in memory,
        // so the address should be shifted by two elements.
        add(reg_indices_, 2 * conf_.el_size_of_indices);
        add(reg_weights, 2 * sizeof(float));

        for (unsigned i = 0; i < conf_.number_of_corners; i++) {
            pop(src_regs[(conf_.number_of_corners - 1) - i]);
        }

        dec(reg_work_);
        jmp(loop_begin, T_NEAR);
    }
    L(loop_end);
}

template <cpu_isa_t isa>
void jit_uni_resampling_kernel<isa>::generate() {
    preamble();

    io_.init_bf16();
    if (conf_.tail > 0) io_.prepare_tail_mask();
    if (is_superset(conf_.isa, avx2)
            && conf_.tag_kind == jit_memory_tag_kind_t::ncsp) {
        io_.init_full_mask();
        io_.prepare_full_mask();
    }

    mov(reg_dst_, ptr[reg_param + GET_OFF(dst)]);
    mov(reg_work_, ptr[reg_param + GET_OFF(batch_of_sp_points_to_process)]);
    mov(reg_indices_, ptr[reg_param + GET_OFF(indices)]);
    mov(reg_c_offset, ptr[reg_param + GET_OFF(c_offset)]);

    if (conf_.alg == alg_kind::resampling_nearest) {
        mov(reg_src_, ptr[reg_param + GET_OFF(src)]);
        if (conf_.tag_kind == jit_memory_tag_kind_t::ncsp)
            nearest_ncsp_format();
        else if (conf_.tag_kind == jit_memory_tag_kind_t::nspc
                || conf_.tag_kind == jit_memory_tag_kind_t::blocked)
            nearest_c_oriented_format();
    } else if (conf_.alg == alg_kind::resampling_linear) {
        mov(reg_weights, ptr[reg_param + GET_OFF(weights)]);
        if (conf_.tag_kind == jit_memory_tag_kind_t::ncsp) {
            mov(reg_src_, ptr[reg_param + GET_OFF(src)]);
            linear_ncsp_format();
        } else if (conf_.tag_kind == jit_memory_tag_kind_t::nspc
                || conf_.tag_kind == jit_memory_tag_kind_t::blocked) {
            mov(reg_src_ftl_, ptr[reg_param + GET_OFF(src)]);
            add(reg_src_ftl_, ptr[reg_param + GET_OFF(src_offset_front)]);
            add(reg_src_ftl_, ptr[reg_param + GET_OFF(src_offset_top)]);
            mov(reg_src_ftr_, reg_src_ftl_);

            if (conf_.ndims == 4 || conf_.ndims == 5) {
                uni_vbroadcastss(
                        weight_top_, ptr[reg_param + GET_OFF(weight_top)]);
                uni_vbroadcastss(weight_bottom_,
                        ptr[reg_param + GET_OFF(weight_bottom)]);
                mov(reg_src_fbl_, ptr[reg_param + GET_OFF(src)]);
                add(reg_src_fbl_, ptr[reg_param + GET_OFF(src_offset_front)]);
                add(reg_src_fbl_, ptr[reg_param + GET_OFF(src_offset_bottom)]);
                mov(reg_src_fbr_, reg_src_fbl_);
            }
            if (conf_.ndims == 5) {
                uni_vbroadcastss(
                        weight_front_, ptr[reg_param + GET_OFF(weight_front)]);
                uni_vbroadcastss(
                        weight_back_, ptr[reg_param + GET_OFF(weight_back)]);
                mov(reg_src_btl_, ptr[reg_param + GET_OFF(src)]);
                add(reg_src_btl_, ptr[reg_param + GET_OFF(src_offset_back)]);
                add(reg_src_btl_, ptr[reg_param + GET_OFF(src_offset_top)]);
                mov(reg_src_btr_, reg_src_btl_);

                mov(reg_src_bbl_, ptr[reg_param + GET_OFF(src)]);
                add(reg_src_bbl_, ptr[reg_param + GET_OFF(src_offset_back)]);
                add(reg_src_bbl_, ptr[reg_param + GET_OFF(src_offset_bottom)]);
                mov(reg_src_bbr_, reg_src_bbl_);
            }
            linear_c_oriented_format();
        }
    }

    postamble();

    if (conf_.with_eltwise && postops_injector_)
        postops_injector_->prepare_table();
}

template struct jit_uni_resampling_kernel<avx512_common>;
template struct jit_uni_resampling_kernel<avx>;
template struct jit_uni_resampling_kernel<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
