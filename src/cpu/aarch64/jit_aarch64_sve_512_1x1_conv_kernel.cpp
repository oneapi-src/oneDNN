/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include <assert.h>
#include <float.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/aarch64/cpu_barrier.hpp"

#include "cpu/aarch64/jit_aarch64_sve_512_1x1_conv_kernel.hpp"
#include "cpu/aarch64/jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) static_cast<int32_t>(offsetof(jit_1x1_conv_call_s, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::utils;

using namespace Xbyak;

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;


void jit_aarch64_sve_512_1x1_conv_kernel::bcast_loop(int load_loop_blk) {

    CGA64::mov(aux1_reg_bcast_data, reg_bcast_data);
    CGA64::mov(aux_reg_bcast_data, reg_bcast_data);

    CGA64::mov(aux_reg_output_data, reg_output_data);
    CGA64::mov(reg_bcast_loop_iter, reg_bcast_loop_work);

    xa::LabelAArch64 bcast_loop;
    xa::LabelAArch64 bcast_loop_tail;
    xa::LabelAArch64 large_tail;

    CGA64::cmp(reg_bcast_loop_iter, jcp.bcast_block);
    CGA64::b(xa::LT, bcast_loop_tail);

    CGA64::L_aarch64(bcast_loop); {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            if (i + 1 == num_substeps) CGA64::L_aarch64(large_tail);
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                CGA64::add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                                jcp.bcast_loop_bcast_substep, reg_tmp_imm);
                CGA64::add_imm(aux_reg_output_data, aux_reg_output_data,
                                jcp.bcast_loop_output_substep, reg_tmp_imm);
            } else {
                CGA64::add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_bcast_substep,
                        reg_tmp_imm);
                CGA64::add_imm(aux_reg_output_data, aux_reg_output_data,
                        jcp.bcast_loop_output_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_output_substep,
                        reg_tmp_imm);
            }
            CGA64::subs_imm(reg_bcast_loop_iter, reg_bcast_loop_iter,
                            jcp.ur, reg_tmp_imm);
        }
        CGA64::cmp(reg_bcast_loop_iter, jcp.bcast_block);
        CGA64::b(xa::GE, bcast_loop);
    }

    CGA64::L_aarch64(bcast_loop_tail);
    if (jcp.ur_tail) {
        xa::LabelAArch64 bcast_loop_tail_out;
        if (jcp.ur_tail >= jcp.ur) {
            CGA64::cmp(reg_bcast_loop_iter, jcp.ur);
            CGA64::b(xa::GE, large_tail);
        }
        if (jcp.ur_tail % jcp.ur) {
            CGA64::cmp(reg_bcast_loop_iter, 0);
            CGA64::b(xa::LE, bcast_loop_tail_out);
            reduce_loop(load_loop_blk, jcp.ur_tail % jcp.ur, 0, true);
            CGA64::L_aarch64(bcast_loop_tail_out);
        }
    }

}

void jit_aarch64_sve_512_1x1_conv_kernel::reduce_loop(
        int load_loop_blk, int ur, int substep, bool wraparound) {

    const bool out_layout_nxc = is_out_layout_nxc(jcp);
    const bool load_layout_nxc = is_load_layout_nxc(jcp);
    const bool bcast_layout_nxc = is_bcast_layout_nxc(jcp);
    const int reduce_dim_tail = jcp.reduce_dim % jcp.reduce_block;
    const int load_dim_tail = jcp.load_dim % jcp.load_block;

    auto vreg_bcast_s = [=](int idx) {
        return xa::ZRegS(idx);
    };

    auto vreg_sum = [=]() {
        return xa::ZReg(31);
    };
    auto vreg_sum_s = [=]() {
        return xa::ZRegS(31);
    };

    auto vreg_load = [=](int i_load, int i_fma) {
        return xa::ZReg(utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                    + jcp.fma_step * i_load + i_fma);
    };
    auto vreg_load_s = [=](int i_load, int i_fma) {
        return xa::ZRegS(utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                    + jcp.fma_step * i_load + i_fma);
    };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return xa::ZReg(i_ur * load_loop_blk + i_load);
    };
    auto vreg_accum_s = [=](int i_load, int i_ur) {
        return xa::ZRegS(i_ur * load_loop_blk + i_load);
    };

    auto bias_load = [=](int i_load, int i_ur){
        int ofs = jcp.typesize_out * jcp.oc_block * i_load;
        if((VL_OFS(ofs) <= LDRMAX) &&
           (VL_OFS(ofs) >= (-1 * LDRMAX)) &&
           ((ofs&0x3f)==0)){

            CGA64::ldr(vreg_accum(i_load, i_ur),
                       xa::ptr(reg_bias_data, static_cast<int32_t>(VL_OFS(ofs))));
        }else{
            CGA64::add_imm(reg_tmp_ofs, reg_bias_data, ofs, reg_tmp_imm);
            CGA64::ldr(vreg_accum(i_load, i_ur), xa::ptr(reg_tmp_ofs));
        }
    };

    auto bcast_load = [=](int i_reduce, int i_ur,
                              int prev_ofs, int bcast_idx) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        int ofs;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                    backward_data)) {
            assert(jcp.reduce_loop_unroll == jcp.reduce_block);
            const int reduce_mul = bcast_layout_nxc ? jcp.reduce_dim
                                                    : jcp.reduce_loop_unroll;
            ofs = (i_reduce == jcp.reduce_loop_unroll)
                    ? (jcp.bcast_dim + i_ur) * reduce_mul
                    : i_ur * reduce_mul + i_reduce;
        } else {
            if (jcp.transpose_src) {
                const int reduce_group = i_reduce / 4;
                const int reduce_shift = i_reduce % 4;
                ofs = 4 * (reduce_group * jcp.ic_block + i_ur) + reduce_shift;
            } else {
                int rmul = bcast_layout_nxc ? jcp.ic : jcp.ic_block;
                ofs = i_reduce * rmul + i_ur;
            }
        }

        ofs = jcp.typesize_in * ofs;
        int tmp_ofs = ofs;
        if( ((ofs&0x3) == 0) && (ofs <= LDRWMAX) && (ofs >= 0)){
          CGA64::ld1rw(vreg_bcast_s(bcast_idx), reg_p_all_ones,
                        xa::ptr(aux_reg_bcast_data, static_cast<int32_t>(ofs)));
        }else{
          if((prev_ofs != -1) && ((ofs - prev_ofs)>=0)
              &&((ofs - prev_ofs) <= LDRWMAX) && (((ofs-prev_ofs)&0x3) == 0)){
            CGA64::ld1rw(vreg_bcast_s(bcast_idx), reg_p_all_ones,
                          xa::ptr(reg_prev_bcast_addr, static_cast<int32_t>((ofs-prev_ofs))));
          }else{
            if((prev_ofs != -1) && ((ofs - prev_ofs)>=0)){
              ofs = ofs - prev_ofs;
              CGA64::add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs, reg_tmp_imm);
            }else{
              CGA64::add_imm(reg_prev_bcast_addr, aux_reg_bcast_data, ofs, reg_tmp_imm);
            }
            prev_ofs = tmp_ofs;

            CGA64::ld1rw(vreg_bcast_s(bcast_idx), reg_p_all_ones, xa::ptr(reg_prev_bcast_addr));
          }
        }
        return prev_ofs;
    };

    auto load_load = [=]( int i_reduce, int i_load, int i_fma ){
        int ofs;
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;
        int lmul = jcp.load_block
                * (load_layout_nxc ? 1
                                   : utils::rnd_up(
                                           jcp.reduce_dim, jcp.reduce_block));
        int rmul = load_layout_nxc ? jcp.load_dim : jcp.load_block;
        ofs = i_load * lmul + u0 * rmul;
        ofs = u1 * jcp.reduce_loop_load_step + jcp.typesize_in * ofs;

        if((VL_OFS(ofs) <= LDRMAX) && (VL_OFS(ofs) >= (-1* LDRMAX)) &&
           ((ofs&0x3f)==0)){
          ofs = VL_OFS(ofs);
          CGA64::ldr(vreg_load(i_load, i_fma), 
                    xa::ptr(aux_reg_load_data, static_cast<int32_t>(ofs)));
        }else{
          CGA64::add_imm(reg_tmp_ofs, aux_reg_load_data, ofs, reg_tmp_imm);
          CGA64::ldr(vreg_load(i_load, i_fma), xa::ptr(reg_tmp_ofs));
        }
    };

    auto out_load = [=](int i_load, int i_ur, int prev_ofs) {

        int ofs, ofs_tmp;
        int bwd_iload = (i_load != 0) && one_of(jcp.prop_kind, backward_weights);
        auto r = (bwd_iload) ? reg_tmp_ofs : aux_reg_output_data;
 
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                    backward_data)) {
            int i_load_shift = out_layout_nxc
                    ? jcp.load_block
                    : (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim)
                            * jcp.load_block;
            int i_ur_shift = out_layout_nxc ? jcp.load_dim : jcp.load_block;
            ofs = (i_load * i_load_shift + i_ur * i_ur_shift)
                    * jcp.typesize_out;
        } else {
            ofs = jcp.typesize_out * jcp.load_block * i_ur;
        }

        ofs_tmp = ofs;

        if(bwd_iload) CGA64::mov(r, i_load);
        if((VL_OFS(ofs) <= LDRMAX) &&
           (VL_OFS(ofs) >= (-1 * LDRMAX)) &&
           ((ofs&0x3f)==0)){
          if(bwd_iload) CGA64::madd(r, r, reg_output_stride, aux_reg_output_data);
          CGA64::ldr(vreg_sum(), xa::ptr(r, static_cast<int32_t>(VL_OFS(ofs))));
        }else{
          if((prev_ofs != -1) &&
              ((ofs - prev_ofs)>0) &&
              (VL_OFS(ofs - prev_ofs) <= LDRMAX)){
            if(bwd_iload) CGA64::madd(r, r, reg_output_stride, reg_prev_out_addr);
            else          r = reg_prev_out_addr;
            CGA64::ldr(vreg_sum(), xa::ptr(r, static_cast<int32_t>(VL_OFS(ofs - prev_ofs))));
          }else{
            if((prev_ofs != -1) && ((ofs - prev_ofs)>0)){
              ofs = ofs - prev_ofs;
              CGA64::add_imm(reg_prev_out_addr, reg_prev_out_addr, ofs, reg_tmp_imm);
            }else{
              CGA64::add_imm(reg_prev_out_addr, aux_reg_output_data, ofs, reg_tmp_imm);
            }
            if(bwd_iload) CGA64::madd(r, r, reg_output_stride, reg_prev_out_addr);
            else          r = reg_prev_out_addr;
            CGA64::ldr(vreg_sum(), xa::ptr(r));

            prev_ofs = ofs_tmp;

          }
        }
        return prev_ofs;
    };

    auto out_str = [=](int i_load, int i_ur, int prev_ofs){
        int ofs, ofs_tmp;
        int bwd_iload = (i_load != 0) && one_of(jcp.prop_kind, backward_weights);
        auto r = (bwd_iload) ? reg_tmp_ofs : aux_reg_output_data;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                   backward_data)){
          ofs = (i_load * jcp.bcast_dim + i_ur) * jcp.load_block * jcp.typesize_out;
        }else{
          ofs = jcp.typesize_out * jcp.load_block * i_ur;
        }
        ofs_tmp = ofs;

        if(bwd_iload) CGA64::mov(r, i_load);
        if((VL_OFS(ofs) <= LDRMAX) &&
           (VL_OFS(ofs) >= (-1 * LDRMAX)) &&
           ((ofs&0x3f)==0)){
          if(bwd_iload) CGA64::madd(r, r, reg_output_stride, aux_reg_output_data);
          CGA64::str(vreg_accum(i_load, i_ur), xa::ptr(r, static_cast<int32_t>(VL_OFS(ofs))));
        }else{
          if((prev_ofs != -1) &&
             ((ofs - prev_ofs)>0) &&
             ((VL_OFS(ofs - prev_ofs)) <= LDRMAX)){
            if(bwd_iload)  CGA64::madd(r, r, reg_output_stride, reg_prev_out_addr);
            else            r = reg_prev_out_addr;
            CGA64::str(vreg_accum(i_load, i_ur), xa::ptr(r, static_cast<int32_t>(VL_OFS(ofs-prev_ofs))));
          }else{
            if((prev_ofs != -1) && ((ofs - prev_ofs)>0)){
              ofs = ofs - prev_ofs;
              CGA64::add_imm(reg_prev_out_addr, reg_prev_out_addr, ofs, reg_tmp_imm);
            }else{
              CGA64::add_imm(reg_prev_out_addr, aux_reg_output_data, ofs, reg_tmp_imm);
            }
            if(bwd_iload) CGA64::madd(r, r, reg_output_stride, reg_prev_out_addr);
            else          r = reg_prev_out_addr;
            CGA64::str(vreg_accum(i_load, i_ur), xa::ptr(r));

            prev_ofs = ofs_tmp;
          }
        }
        return prev_ofs;
    };


    auto prefetch_output = [=](int i_load, int i_ur) {
      int ofs;
      int bwd_iload = (i_load != 0) && one_of(jcp.prop_kind, backward_weights);
      auto r = (bwd_iload) ? reg_tmp_ofs : aux_reg_output_data;
      if (one_of(jcp.prop_kind, forward_training, forward_inference,
                 backward_data)){
        ofs = (i_load * jcp.bcast_dim + i_ur) * jcp.load_block * jcp.typesize_out;
      }else{
        ofs = jcp.typesize_out * jcp.load_block * i_ur;
      }
      std::string op = "LD";
      prefetch(op, 2, r, ofs);
    };

    auto init = [=]() {
        xa::LabelAArch64 init_done;
        xa::LabelAArch64 init_zero;

        if (jcp.with_sum) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    prefetch_output(i_load, i_ur);
                }
            }
        }

        if (jcp.with_bias
                && one_of(jcp.prop_kind, forward_training, forward_inference)) {

            CGA64::tst(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            CGA64::b(xa::EQ, init_zero);

            for (int i_load = 0; i_load < load_loop_blk; i_load++)
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
#if 0
                 if (i_load + 1 == load_loop_blk && load_dim_tail)
                    r = r | k_load_dim_mask | T_z;
#endif
                    bias_load(i_load, i_ur); 
                }
            CGA64::b(init_done);
        }

        CGA64::L_aarch64(init_zero);
        /* Zero clear */
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                CGA64::fmov(vreg_accum_s(i_load, i_ur));
            }
        CGA64::L_aarch64(init_done);
    };

    auto store = [=]() {
        xa::LabelAArch64 store_noadd;
        if (!jcp.with_sum) {
            CGA64::tst(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            CGA64::b(xa::NE, store_noadd);
        }

        int prev_ofs = -1;
        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto r = vreg_accum_s(i_load, i_ur);
#if 0
                if (i_load + 1 == load_loop_blk && load_dim_tail)
                    r = r | k_load_dim_mask | T_z;
#endif
                prev_ofs = out_load(i_load, i_ur, prev_ofs);
                CGA64::fadd(r, r, vreg_sum_s());
            }

        CGA64::L_aarch64(store_noadd);
        if (jcp.with_eltwise) {
            assert(NULL);
            xa::LabelAArch64 store_noeltwise;
            CGA64::tst(reg_reduce_pos_flag, FLAG_REDUCE_LAST);
            CGA64::b(xa::EQ, store_noeltwise);
#if 0
            eltwise_injector_->compute_vector_range(0, ur * load_loop_blk);
#endif
            CGA64::L_aarch64(store_noeltwise);
        }

        prev_ofs = -1;
        for (int i_ur = 0; i_ur < ur; ++i_ur) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                // for nxc_layout-bwd_w, weights are still padded and the
                // output_ptr here can be uninitialized scratchpad.
                // To ensure final output (after reduction) is zero-padded,
                // here we zero-pad output by omitting the mask.
#if 0
                if (jcp.prop_kind != backward_weights
                        && i_load + 1 == load_loop_blk && load_dim_tail) {
                    vreg_acc = vreg_acc | k_load_dim_mask;
                }
#endif
                prev_ofs = out_str(i_load, i_ur, prev_ofs);
            }
        }
    };

#if 0
    auto prefetch_callback = [=](int ur, int i_reduce, int i_ur, int i_load,
                                     bool last_block, bool wraparound,
                                     int reduce_step) {
        bool pf_ker_l1 = true;
        bool pf_ker_l2 = wraparound;
        int n_ops = (jcp.reduce_loop_unroll / reduce_step) * ur * load_loop_blk;
        int i_op = (i_reduce / reduce_step) * ur * load_loop_blk
                + i_ur * load_loop_blk + i_load;

        int n_pf_ker_l1 = pf_ker_l1 ? jcp.reduce_block : 0;
        int n_pf_ker_l2 = pf_ker_l2 && wraparound ? jcp.reduce_block : 0;
        int n_pf_out_l1 = jcp.use_vmovntps ? 0 : ur;

        int pf_inp_ops = n_ops / 2; // # of operations during which to pf input
        int pf_inp_trigger;
        if (jcp.prop_kind == backward_weights)
            pf_inp_trigger = nstl::max(1, pf_inp_ops / jcp.reduce_block);
        else
            pf_inp_trigger = nstl::max(1, pf_inp_ops / ur);

        int n_other_pf
                = load_loop_blk * (n_pf_ker_l1 + n_pf_ker_l2 + n_pf_out_l1);
        int n_other_pf_ops = n_ops - pf_inp_ops;
        int other_pf_trigger
                = n_other_pf ? nstl::max(1, n_other_pf_ops / n_other_pf) : 0;

        if (i_op < pf_inp_ops && i_op % pf_inp_trigger == 0) {
            // input prefetches have the highest priority b/c the
            // first iteration of the kernel block touches all the
            // cache lines
            int i_pf = i_op / pf_inp_trigger;
            auto pf_reg = wraparound && last_block
                    ? reg_bcast_data
                    : (last_block ? aux1_reg_bcast_data : aux_reg_bcast_data);
            int offt = i_pf;
            if (jcp.prop_kind == backward_weights) {
                offt += wraparound && last_block
                        ? 0
                        : (last_block ? jcp.is : jcp.reduce_block);
                offt *= jcp.bcast_block;
            } else {
                offt += wraparound && last_block
                        ? 0
                        : (last_block ? jcp.ur : jcp.bcast_dim);
                offt *= jcp.reduce_block;
            }
            mic_prefetcht0(ptr[pf_reg + offt * jcp.typesize_in]);
        } else if (i_op >= pf_inp_ops && n_other_pf) {
            // remaining prefetches are spread among the rest of the
            // operations; prefetches for output take priority
            // TODO: spread L2 prefetches among L1 prefetches
            i_op -= pf_inp_ops;
            if (i_op % other_pf_trigger == 0) {
                int i_pf = i_op / (load_loop_blk * other_pf_trigger);
                if (i_pf < n_pf_ker_l2) {
                    int offt = (i_pf + (i_load + 1) * jcp.reduce_dim)
                            * jcp.load_block;
                    mic_prefetcht1(
                            ptr[aux_reg_load_data + offt * jcp.typesize_in]);
                } else if (i_pf < n_pf_ker_l2 + n_pf_ker_l1) {
                    i_pf -= n_pf_ker_l2;
                    auto pf_reg
                            = last_block ? reg_load_data : aux_reg_load_data;
                    int offt = (i_pf + i_load * jcp.reduce_dim
                                       + (last_block ? (wraparound
                                                          ? jcp.reduce_dim
                                                          : 0)
                                                     : jcp.reduce_block))
                            * jcp.load_block;
                    mic_prefetcht0(ptr[pf_reg + offt * jcp.typesize_in]);
                } else if (i_pf < n_pf_ker_l1 + n_pf_ker_l2 + n_pf_out_l1) {
                    i_pf -= n_pf_ker_l1 + n_pf_ker_l2;
                    int offt = i_pf * jcp.load_block;
                    mic_prefetcht0(
                            ptr[aux_reg_output_data + offt * jcp.typesize_out]);
                }
            }
        }
    };
#endif

    auto fma_block = [=](bool last_block) {
        assert(jcp.reduce_loop_unroll % jcp.fma_step == 0);

        int reduce_step = jcp.fma_step;
        int prev_bcast_ofs = -1;
        assert(reduce_dim_tail % reduce_step == 0);

        const int i_reduce_end = reduce_dim_tail && last_block
                ? reduce_dim_tail
                : jcp.reduce_loop_unroll;

        int bcast_reg_ofs = utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                               + jcp.fma_step * load_loop_blk;
        int num_bcast_regs = 32 - bcast_reg_ofs;

        for (int i_reduce = 0; i_reduce < i_reduce_end;
                i_reduce += reduce_step) { // IC
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) { // OC
                // if transposed input data used and if spatial size is
                // not divided by transpose step (4) then for last reduce step
                // we should load only needed load_registers data
                // and clear remaining
                if (jcp.transpose_src && jcp.is % jcp.fma_step && last_block
                        && i_reduce == jcp.reduce_loop_unroll - reduce_step) {
                    xa::LabelAArch64 load_all;
                    xa::LabelAArch64 load_finish;
                    CGA64::tst(reg_reduce_pos_flag, FLAG_SP_LAST);
                    CGA64::b(xa::EQ, load_all);

                    const int n_loads = jcp.is % jcp.fma_step;
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        if (i_fma < n_loads)
                            load_load(i_reduce + i_fma, i_load, i_fma);
                        else
                            CGA64::fmov(vreg_load_s(i_load, i_fma));
                    }
                    CGA64::b(load_finish);

                    CGA64::L_aarch64(load_all);
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        load_load(i_reduce + i_fma, i_load, i_fma);
                    }
                    CGA64::L_aarch64(load_finish);
                } else {
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
#if 0
                        if (i_load + 1 == load_loop_blk && load_dim_tail)
                            vreg = vreg | k_load_dim_mask | T_z;
#endif
                        load_load(i_reduce + i_fma, i_load, i_fma);
                    }
                }
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                prev_bcast_ofs = bcast_load(i_reduce, i_ur, prev_bcast_ofs, bcast_reg_ofs + (i_ur % num_bcast_regs));
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
#if 0
                    if (i_load + 1 == load_loop_blk && load_dim_tail)
                        vreg_acc = vreg_acc | k_load_dim_mask | T_z;
#endif
                    CGA64::fmla(vreg_accum_s(i_load, i_ur), reg_p_all_ones,
                            vreg_load_s(i_load, 0),
                            vreg_bcast_s(bcast_reg_ofs + (i_ur % num_bcast_regs)));
 
                }
            }
        }
    };
    xa::LabelAArch64 reduce_loop;
    xa::LabelAArch64 reduce_loop_tail;

    CGA64::mov(aux_reg_load_data, reg_load_data);

    CGA64::mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    CGA64::mov(reduce_loop_iter, reg_reduce_loop_work);
    CGA64::subs_imm(reduce_loop_iter, reduce_loop_iter, 
                    jcp.reduce_loop_unroll, reg_tmp_imm);
    CGA64::b(xa::LE, reduce_loop_tail);

    CGA64::L_aarch64(reduce_loop);
    {
        fma_block(false);
        CGA64::add_imm(aux_reg_bcast_data, aux_reg_bcast_data,
                        jcp.reduce_loop_bcast_step, reg_tmp_imm);
        CGA64::add_imm(aux_reg_load_data, aux_reg_load_data,
                        jcp.reduce_loop_load_step, reg_tmp_imm);
        CGA64::subs_imm(reduce_loop_iter, reduce_loop_iter,
                        jcp.reduce_loop_unroll, reg_tmp_imm);
        CGA64::b(xa::GT, reduce_loop);
    }

    CGA64::L_aarch64(reduce_loop_tail);
    fma_block(true);

    store();

}

void jit_aarch64_sve_512_1x1_conv_kernel::generate() {
    preamble();

    /* All 1 predicate register */
    CGA64::ptrue( reg_p_all_ones.b ); 

    /* Pointers indicate weight, input, and output data */
    CGA64::ldr(reg_bcast_data,  xa::ptr(abi_param1_aarch64, GET_OFF(bcast_data)));  // Input 
    CGA64::ldr(reg_load_data,   xa::ptr(abi_param1_aarch64, GET_OFF(load_data)));   // Weight
    CGA64::ldr(reg_output_data, xa::ptr(abi_param1_aarch64, GET_OFF(output_data))); // Output

    /* Pointer indicates bias data if the layer has bias option */
    if (jcp.with_bias) CGA64::ldr(reg_bias_data, xa::ptr(abi_param1_aarch64, GET_OFF(bias_data)));

    /* Get workloads of each loop */
    CGA64::ldr(reg_load_loop_work,   xa::ptr(abi_param1_aarch64, GET_OFF(load_dim)));
    CGA64::ldr(reg_bcast_loop_work,  xa::ptr(abi_param1_aarch64, GET_OFF(bcast_dim)));
    CGA64::ldr(reg_reduce_loop_work, xa::ptr(abi_param1_aarch64, GET_OFF(reduce_dim)));

    /* A flag for controlling reduce loop */
    CGA64::ldr(reg_reduce_pos_flag, xa::ptr(abi_param1_aarch64, GET_OFF(first_last_flag)));

    if (one_of(jcp.prop_kind, forward_training, forward_inference))
        CGA64::mov(reg_relu_ns, reinterpret_cast<size_t>(&jcp.eltwise.alpha));

    if (jcp.prop_kind == backward_weights)
        CGA64::ldr(reg_output_stride, xa::ptr(abi_param1_aarch64, GET_OFF(output_stride)));

    const int load_dim_tail = jcp.load_dim % jcp.load_block;

    if (load_dim_tail) {
#if 1
        assert(NULL);
#else
        Reg32 reg_tail_32 = reg_load_dim_tail_mask.cvt32();
        mov(reg_tail_32, (1 << load_dim_tail) - 1);
        kmovw(k_load_dim_tail_mask, reg_tail_32);
#endif
    }

    auto load_loop_body = [=](int load_loop_blk) {
#if 0
        if (load_dim_tail)
            kxnorw(k_load_dim_mask, k_load_dim_mask, k_load_dim_mask);
#endif
        CGA64::subs_imm(reg_load_loop_work, reg_load_loop_work,
                      load_loop_blk * jcp.load_loop_iter_step, reg_tmp_imm);

        if (load_dim_tail) {
#if 1
            assert(NULL);
#else
            Label no_update_mask;
            jge(no_update_mask, T_NEAR);
            kmovw(k_load_dim_mask, k_load_dim_tail_mask);
            L(no_update_mask);
#endif
        }
        bcast_loop(load_loop_blk);
        CGA64::add_imm(reg_load_data, reg_load_data, 
                        load_loop_blk * jcp.load_loop_load_step, reg_tmp_imm);
        switch (jcp.prop_kind) {
            case forward_training:
            case forward_inference:
                CGA64::add_imm(reg_bias_data, reg_bias_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out, reg_tmp_imm);
                CGA64::add_imm(reg_output_data, reg_output_data, 
                        load_loop_blk * jcp.load_block * jcp.typesize_out
                                * (is_out_layout_nxc(jcp)
                                                ? 1
                                                : (jcp.with_dw_conv
                                                                ? jcp.ow
                                                                : jcp.bcast_dim)),
                        reg_tmp_imm);
                break;
            case backward_data:
                CGA64::add_imm(reg_output_data, reg_output_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out
                                * (is_out_layout_nxc(jcp) ? 1 : jcp.bcast_dim),
                        reg_tmp_imm);
                break;
            case backward_weights:
                for (int i_load = 0; i_load < load_loop_blk; i_load++)
                    CGA64::add(reg_output_data, reg_output_data, reg_output_stride);
                break;
            default: assert(!"invalid prop_kind");
        }
    };

    const int simd_w = 16;

    xa::LabelAArch64 load_loop_blk[7];

    // with an implicit load_loop_block          {6, 5, 4, 3, 2,  1}
    static const int ur_cases_bcast[] = {2, 5, 6, 9, 14, 32};

    const int size_ur_cases = sizeof(ur_cases_bcast);

    const int *ur_cases = ur_cases_bcast;
    const int num_ur_cases = size_ur_cases / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.nb_load > label_idx && jcp.ur <= ur_cases[ur_idx]) {
            CGA64::cmp(reg_load_loop_work, simd_w * (label_idx + 1));
            CGA64::b(xa::LE, load_loop_blk[label_idx]);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.nb_load > label_idx && jcp.ur <= ur_cases[ur_idx]) {
            CGA64::L_aarch64(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    CGA64::cmp(reg_load_loop_work, 0);
                    CGA64::b(xa::LE, load_loop_blk[num_ur_cases]);
                }
                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    CGA64::cmp(reg_load_loop_work, 2 * label_idx * simd_w);
                    CGA64::b(xa::EQ, load_loop_blk[label_idx - 1]);
                }
                CGA64::cmp(reg_load_loop_work, label_idx * simd_w);
                CGA64::b(xa::GT, load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                CGA64::cmp(reg_load_loop_work, simd_w * (idx + 1));
                CGA64::b(xa::EQ, load_loop_blk[idx]);
            }
            if (ur_idx < num_ur_cases - 2) {
                CGA64::cmp(reg_load_loop_work, simd_w);
                CGA64::b(xa::LE, load_loop_blk[0]);
            }
        }
    }
    CGA64::L_aarch64(load_loop_blk[num_ur_cases]);
    
    postamble();
#if 0
    if (jcp.with_eltwise) eltwise_injector_->prepare_table();
#endif
}

bool jit_aarch64_sve_512_1x1_conv_kernel::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {

    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };
    auto is_convolution
            = [&](int idx) { return p.entry_[idx].is_convolution(); };

    int dw_idx = p.find(primitive_kind::convolution);
    int len = dw_idx != -1 ? dw_idx + 1 : p.len_;

    switch (len) {
        case 0: return true; // no post_ops
        case 1: // eltwise OR sum OR Convolution
            return is_eltwise(0) || is_sum(0) || is_convolution(0);
        case 2: // sum -> eltwise OR eltwise -> convolution
            return (is_sum(0) && is_eltwise(1))
                    || (is_eltwise(0) && is_convolution(1));
        default: return false;
    }

    return false;
}

status_t jit_aarch64_sve_512_1x1_conv_kernel::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, int nthreads, bool reduce_src) {

    /* arch check */
    if (!mayiuse(sve)) return status::unimplemented;

    jcp.nthr = nthreads;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int simd_w = cpu_isa_traits<sve>::vlen / sizeof(float);
    const int ndims = src_d.ndims();
    /* Forward_[training, inference], backward_[data, weight] */
    jcp.prop_kind = cd.prop_kind;

    /* Check group option */
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    /* Batchsize */
    jcp.mb = src_d.dims()[0];
    /* Channel */
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc = jcp.oc_without_padding;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.ic = jcp.ic_without_padding;
    /* D, H, W*/
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    /* Kernel size */
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    /* padding params */
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    /* stride params */
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];
    /* bias info */
    jcp.with_bias = pick_by_prop_kind(jcp.prop_kind, cd.bias_desc.format_kind,
                            format_kind::undef, cd.diff_bias_desc.format_kind)
            != format_kind::undef;

    /* Spatials */
    jcp.os = jcp.od * jcp.oh * jcp.ow;
    jcp.is = jcp.id * jcp.ih * jcp.iw;
    jcp.tr_is = rnd_up(jcp.is, 4);

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;

    // TODO:
    if( jcp.with_dw_conv ) return status::unimplemented;


    /* Post operation check */
    // Using dw_conv_ind as upper-bound below, as post-ops after it will be
    // handled in depthwise convolution.
    jcp.with_sum = p.find(primitive_kind::sum, 0, dw_conv_ind) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise, 0, dw_conv_ind);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
#if 1
        return status::unimplemented;
#else
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
#endif
    }
    /* Data format check */
    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
    auto required_dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1
            && src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    bool args_ok = true && jcp.ngroups == 1 && jcp.src_tag == required_dat_tag
            && jcp.dst_tag == required_dat_tag
            && IMPLICATION(!is_data_layout_nxc,
                    jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0)
            && jcp.f_pad == 0 && jcp.t_pad == 0 && jcp.l_pad == 0
            && jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1
            && jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1 && jcp.ow == jcp.iw
            && jcp.oh == jcp.ih && jcp.od == jcp.id; // enforce rpad=0
    if (!args_ok) return status::unimplemented;

    jcp.ic_block = jcp.oc_block = simd_w;
    jcp.transpose_src = false;
    jcp.use_vmovntps = false;

    if (everyone_is(data_type::f32, src_d.data_type(), weights_d.data_type(),
                dst_d.data_type())) {
        const int is_bwd_d = jcp.prop_kind == backward_data;
        format_tag_t wei_tag = with_groups
                ? pick(2 * ndims - 6 + is_bwd_d, gOIw16i16o, gIOw16o16i,
                        gOIhw16i16o, gIOhw16o16i, gOIdhw16i16o, gIOdhw16o16i)
                : pick(2 * ndims - 6 + is_bwd_d, OIw16i16o, IOw16o16i,
                        OIhw16i16o, IOhw16o16i, OIdhw16i16o, IOdhw16o16i);

        jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
        if (jcp.wei_tag != wei_tag) return status::unimplemented;

        jcp.ver = ver_sve;
        jcp.fma_step = 1;
        jcp.typesize_in = sizeof(prec_traits<data_type::f32>::type);
        jcp.typesize_out = sizeof(prec_traits<data_type::f32>::type);
    } else {
        return status::unimplemented;
    }

    /* once all the formats are set, check the padding consistency */
    if (!is_data_layout_nxc) {
        args_ok = true && jcp.ic <= src_d.padded_dims()[1]
                && jcp.oc <= dst_d.padded_dims()[1]
                && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
                && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
        if (!args_ok) return status::unimplemented;
    }

    const int SMALL_SPATIAL = 10;
    const int BIG_SPATIAL = 28;
    const int BIG_REDUCE_DIM = 1024;
    const int BIG_LOAD_DIM = 256;

    int load_blocking {0};
    int load_blocking_max {0};
    int bcast_blocking {0};
    int bcast_blocking_max {0};
    int reduce_blocking {0};
    int reduce_blocking_max {0};

    jcp.load_grp_count = 1;

    // TODO: mov check funcs into platform files
    const int L1_capacity
            = get_A64FX_cache_size(1, true) / sizeof(float);
    const int L2_size = get_A64FX_cache_size(2, false, nthreads) / sizeof(float);
    const int L2_capacity = (L2_size * 3) / 4;

    /* FWD, BWD data */
    if (one_of(jcp.prop_kind, forward_training, forward_inference,
                backward_data)) {
        if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
            if (jcp.with_dw_conv) jcp.ur = nstl::min(jcp.ow, jcp.ur);
            jcp.reduce_dim = jcp.ic;
            jcp.reduce_block = jcp.ic_block;

            jcp.load_dim = jcp.oc;
            jcp.load_block = jcp.oc_block;

            jcp.bcast_dim = jcp.is;
        } else {
            jcp.reduce_dim = jcp.oc;
            jcp.reduce_block = jcp.oc_block;

            jcp.load_dim = jcp.ic;
            jcp.load_block = jcp.ic_block;

            jcp.bcast_dim = jcp.os;
        }
        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? 1 : jcp.bcast_dim) * jcp.typesize_in;

        jcp.reduce_loop_load_step
                = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;
        jcp.load_loop_load_step
                = (utils::rnd_up(jcp.reduce_dim, jcp.reduce_block))
                * jcp.load_block * jcp.typesize_in;

        // adjusting registry blocking
        int max_regs, min_regs, size_treshold, ur_step;
        const int spatial
                = (one_of(jcp.prop_kind, forward_training, forward_inference))
                ? jcp.od * jcp.oh
                : jcp.id * jcp.ih;
        max_regs = 30;
        min_regs = 9;
        size_treshold = 14;
        ur_step = 1;
        jcp.expl_bcast = false;
        jcp.use_vmovntps = true;

        jcp.ur = 1;

        for (int ur_w = max_regs; ur_w >= min_regs; ur_w -= ur_step) {
            if ((spatial >= size_treshold && spatial % ur_w == 0)
                    || (spatial < size_treshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }
        if (jcp.ur == 1) {
            jcp.ur = nstl::min(max_regs, jcp.os);
            int os_tail = jcp.os % max_regs;
            for (int i = max_regs; i >= min_regs; i -= ur_step) {
                int i_tail = jcp.os % i;
                if (i_tail > os_tail || i_tail == 0) {
                    jcp.ur = i;
                    os_tail = i_tail;
                    if (i_tail == 0) break;
                }
            }
        }
        jcp.bcast_block = jcp.ur;

        /* Number of steps for the dst address to output, used in bcast_loop() */
        jcp.bcast_loop_output_step = jcp.ur * jcp.typesize_out
                * (is_data_layout_nxc ? jcp.load_dim : jcp.load_block);
        jcp.bcast_loop_output_substep = -1; // unused

        /* Number of steps for the src address to be broadcasted in bcast_loop() */
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.typesize_in
                * (is_data_layout_nxc ? jcp.reduce_dim : jcp.reduce_block);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_iter_step = jcp.load_block;

        if (jcp.prop_kind == backward_data)
            jcp.loop_order = loop_lbr;
        else
            jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

        int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
        int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);
        int nb_load = div_up(jcp.load_dim, jcp.load_block);
        if (is_data_layout_nxc) {
            reduce_blocking = jcp.reduce_dim;
        } else {
            reduce_blocking = nb_reduce;
            if (spatial <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
                reduce_blocking = 16;
            else if (spatial > SMALL_SPATIAL
                    && jcp.reduce_dim >= BIG_REDUCE_DIM)
                reduce_blocking = 8;
            // TODO:
            //reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
            reduce_blocking *= jcp.reduce_block;
        }

        // Check input data cache aliasing.
        // For other ISA constants may be updated.
        // 64 * 1024 is chosen due to 1MB L2 16-way cache.
        // 7 is empirical value. It is about half of 16.
        // So we leave about half of the set for other data - weights, dst
        int way_size = (64 * 1024) / jcp.typesize_in;
        int max_hits = 7;
        if (!is_data_layout_nxc
                && jcp.bcast_dim * reduce_blocking > way_size * max_hits) {
            int nrb = reduce_blocking / simd_w;
            int sp = jcp.bcast_dim;
            int wl = way_size / simd_w;
            for (int start_off = 0; start_off < jcp.ur; start_off++) {
                for (int off = start_off, hits = 0; off < sp * nrb; off += wl) {
                    if (off % sp >= jcp.ur || ++hits < max_hits) continue;
                    int max_r_blocking = simd_w * nstl::max(1, (off + wl) / sp);
                    reduce_blocking
                            = nstl::min(reduce_blocking, max_r_blocking);
                    break;
                }
            }
        }

        if (reduce_blocking < jcp.reduce_dim) {
            if (jcp.prop_kind == backward_data)
                jcp.loop_order = reduce_src ? loop_lbr : loop_rlb;
            else
                jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
        }
        load_blocking = jcp.load_dim;

        /* Number of weight elements to be loaded for dest */
        int load_size = jcp.load_dim * jcp.reduce_dim;
        /* Number of elements to be broadcasted from src */
        auto bcast_size
                = (dim_t)jcp.mb * jcp.ngroups * jcp.bcast_dim * jcp.reduce_dim;

        /* 12 cores per CMG */
        if (jcp.ver == ver_sve && jcp.nthr <= 12 && jcp.mb < jcp.nthr
                && nb_load * nb_bcast > jcp.nthr) {
            // Some heuristic here
            float calc_koef = 0.01, best_cost = FLT_MAX;
            int n_lgc = jcp.nthr;
            float ratio = (float)load_size / (float)bcast_size;
            int best_lgc = ratio > 1 ? n_lgc : 1;
            auto calc_job_cost = [&](int lb, int tg, float mem_k) {
                int bb_size = jcp.mb * div_up(nb_bcast, tg);
                float calc_size = (float)(bb_size * jcp.ur)
                        * (lb * jcp.load_block) * jcp.reduce_dim;
                float mem_size = (float)(bb_size * jcp.ur + lb * jcp.load_block)
                        * jcp.reduce_dim;
                return calc_koef * calc_size + mem_k * mem_size;
            };
            for (int lgc, ilgc = 0; ilgc < n_lgc; ilgc++) {
                lgc = ratio > 1 ? n_lgc - ilgc : ilgc + 1;
                int min_lb = nb_load / lgc;
                int max_lb = div_up(nb_load, lgc);
                int min_tg = jcp.nthr / lgc;
                int max_tg = div_up(jcp.nthr, lgc);
                // Some heuristic here
                float mem_koef = (max_tg == 1) ? 1.f : 1.3f;
                float job_cost = 0.;
                if (jcp.nthr % lgc < nb_load % lgc) {
                    job_cost = calc_job_cost(max_lb, min_tg, mem_koef);
                } else {
                    auto job_cost1 = calc_job_cost(max_lb, max_tg, mem_koef);
                    auto job_cost2 = calc_job_cost(min_lb, min_tg, mem_koef);
                    job_cost = nstl::max(job_cost1, job_cost2);
                }

                if (job_cost < best_cost) {
                    best_lgc = lgc;
                    best_cost = job_cost;
                }
            }
            jcp.load_grp_count = best_lgc;
            load_blocking
                    = div_up(nb_load, jcp.load_grp_count) * jcp.load_block;
        } else {
            jcp.load_grp_count
                    = div_up(jcp.nthr, jcp.mb * jcp.ngroups * nb_bcast);
            // TODO:
            //jcp.load_grp_count = best_divider(jcp.nthr, jcp.load_grp_count,
            //        2 * jcp.load_grp_count, false);
        }

        if (jcp.bcast_dim <= 49 && jcp.mb <= jcp.nthr
                && jcp.load_dim > 512 && jcp.load_dim / jcp.reduce_dim >= 4) {
            jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2);
            load_blocking = jcp.load_block;
        }

        bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                                 div_up(jcp.nthr, jcp.load_grp_count))
                * jcp.bcast_block;
        bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
        bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

        int space_for_bcast = (L2_capacity - /* kernel_size - */
                2 * jcp.load_block * reduce_blocking - jcp.ur * reduce_blocking
                - 3 * 1024);
        if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity) space_for_bcast /= 2;

        int bcast_in_cache
                = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
        bcast_blocking = nstl::min(
                bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));
        // NHWC perf
        if (is_data_layout_nxc) bcast_blocking = jcp.bcast_block;

        load_blocking_max = load_blocking;
        bcast_blocking_max = bcast_blocking * 3 / 2;
        reduce_blocking_max = reduce_blocking;

        jcp.ur_tail = (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) % jcp.ur;

    } else if (jcp.prop_kind == backward_weights) { /* BWD weight */

        if (jcp.transpose_src)
            jcp.reduce_dim = jcp.tr_is;
        else
            jcp.reduce_dim = jcp.is;

#if 0
        jcp.reduce_block = best_divider(jcp.reduce_dim, 7, 16, true);
#endif
        if (jcp.reduce_dim % jcp.reduce_block != 0)
            jcp.reduce_block = best_divider(jcp.iw, 4, jcp.iw, false);
        if (jcp.reduce_block > 256) { jcp.reduce_block = 1; }

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        if (jcp.ver == ver_sve && jcp.reduce_block <= 19 &&
                // maskrcnn optimization for nxc; don't reduce ur when ocb<=1
                !(is_data_layout_nxc && jcp.load_dim <= jcp.load_block)) {
            // if reduce_block is big then generated JIT code may be big
            // for small values of ur because reduce_loop_unroll = reduce_block
            jcp.ur = jcp.bcast_block / 2;
            jcp.expl_bcast = true;
        } else {
            jcp.ur = jcp.bcast_block;
            jcp.expl_bcast = false;
        }

        jcp.ur_tail = jcp.bcast_dim % jcp.bcast_block;
        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step = jcp.typesize_in * jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? jcp.ic : jcp.ic_block);
        jcp.reduce_loop_load_step = jcp.typesize_in * jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? jcp.oc : jcp.oc_block);

        jcp.bcast_loop_output_step
                = jcp.oc_block * jcp.ic_block * jcp.typesize_out;
        jcp.bcast_loop_output_substep
                = jcp.oc_block * jcp.ur * jcp.typesize_out;
        jcp.bcast_loop_bcast_step = jcp.ic_block
                * (is_data_layout_nxc ? 1
                                      : utils::rnd_up(
                                              jcp.reduce_dim, jcp.reduce_block))
                * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = jcp.ur * jcp.typesize_in;

        jcp.load_loop_load_step = jcp.typesize_in * jcp.oc_block
                * (is_data_layout_nxc ? 1 : jcp.os);
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */
        balance(jcp);

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        load_blocking = best_divider(load_blocking, 16, load_blocking, false);
        load_blocking *= jcp.load_block;

        load_blocking_max = load_blocking;
        assert(IMPLICATION(
                !is_data_layout_nxc, jcp.load_dim % load_blocking == 0));

        int max_bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        int min_bcast_blocking = 5;

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        bcast_blocking = best_divider(
                bcast_blocking, min_bcast_blocking, max_bcast_blocking, false);
        bcast_blocking *= jcp.bcast_block;
        bcast_blocking_max = bcast_blocking;
        assert(IMPLICATION(
                !is_data_layout_nxc, jcp.bcast_dim % bcast_blocking == 0));

        // for reduction balance
        if (is_data_layout_nxc && jcp.reduce_dim >= BIG_SPATIAL * BIG_SPATIAL
                && jcp.load_dim >= BIG_LOAD_DIM / 2) {
            reduce_blocking = rnd_up(nstl::min(jcp.ow, 256), jcp.reduce_block);
        } else if (jcp.ver == ver_sve) {
            int max_reduce_blocking
                    = nstl::min(L1_capacity / jcp.ur, jcp.reduce_dim);
            int min_reduce_blocking = nstl::min(
                    L1_capacity / jcp.ur, nstl::max(jcp.iw, jcp.ih));
            reduce_blocking = best_divider(jcp.reduce_dim, min_reduce_blocking,
                    max_reduce_blocking, true);
            reduce_blocking
                    = nstl::max(rnd_dn(reduce_blocking, jcp.reduce_block),
                            jcp.reduce_block);
        } else {
            int max_reduce_blocking = L2_capacity
                    / ((bcast_blocking + load_blocking) * jcp.reduce_block);
            max_reduce_blocking = nstl::min(max_reduce_blocking,
                    (L1_capacity / (jcp.bcast_block)) / jcp.reduce_block);

            int num_jobs = div_up(jcp.load_dim, load_blocking)
                    * div_up(jcp.bcast_dim, bcast_blocking);
            int threads_per_job = nstl::max(1, jcp.nthr / num_jobs);
            reduce_blocking = div_up(jcp.mb * jcp.reduce_dim, jcp.reduce_block);
            reduce_blocking = div_up(reduce_blocking, threads_per_job);

            reduce_blocking = best_divider(reduce_blocking,
                    max_reduce_blocking - 2, max_reduce_blocking, true);
            reduce_blocking *= jcp.reduce_block;
        }

        reduce_blocking_max = rnd_dn(reduce_blocking * 3 / 2, jcp.reduce_block);
    } else
        return status::unimplemented;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);

    if (!is_data_layout_nxc) {
        assert(load_blocking % jcp.load_block == 0);
        assert(reduce_blocking % jcp.reduce_block == 0);
        assert(load_blocking_max % jcp.load_block == 0);
        assert(reduce_blocking_max % jcp.reduce_block == 0);
        assert(jcp.reduce_dim % jcp.reduce_block == 0);
    }

    assert(jcp.bcast_block % jcp.ur == 0);

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = utils::div_up(load_blocking, jcp.load_block);
    jcp.nb_load_blocking_max = utils::div_up(load_blocking_max, jcp.load_block);
    jcp.nb_reduce_blocking = utils::div_up(reduce_blocking, jcp.reduce_block);
    jcp.nb_reduce_blocking_max
            = utils::div_up(reduce_blocking_max, jcp.reduce_block);

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    return status::success;
}

void jit_aarch64_sve_512_1x1_conv_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp) {

#if 0 
    using namespace dnnl::impl::memory_tracking::names;

    // Fox nxc layout bias is padded only for bwd_wb direction, as  bias
    // reduction kernels can't handle tails yet.
    if (jcp.with_bias && jcp.prop_kind != backward_data
            && (jcp.oc != jcp.oc_without_padding // blocked layout
                    || (jcp.prop_kind == backward_weights // nxc layout
                            && jcp.oc % jcp.oc_block != 0))) {

        const size_t nelems_padded_bias
                = jcp.ngroups * utils::rnd_up(jcp.oc, jcp.oc_block);
        scratchpad.book(
                key_conv_padded_bias, nelems_padded_bias, jcp.typesize_out);
    }

    if (jcp.prop_kind == backward_weights) {
        const size_t wei_size = (size_t)jcp.ngroups
                * rnd_up(jcp.oc, jcp.oc_block) * rnd_up(jcp.ic, jcp.ic_block);
        scratchpad.book(key_conv_wei_reduction, wei_size * (jcp.nthr_mb - 1),
                jcp.typesize_out);
    }

    if (jcp.transpose_src) {
        const size_t tr_src_size
                = (size_t)jcp.nthr_mb * jcp.ngroups * jcp.ic * jcp.tr_is;
        scratchpad.book(key_conv_tr_src, tr_src_size, jcp.typesize_out);
        scratchpad.book<simple_barrier::ctx_t>(key_conv_tr_src_bctx, jcp.nthr);
    }
#endif
}

void jit_aarch64_sve_512_1x1_conv_kernel::balance(jit_1x1_conv_conf_t &jcp) {
    int nthreads = jcp.nthr;
    // initialize jcp reduction threading properties
    jcp.nthr = jcp.nthr_mb = jcp.nthr_g = jcp.nthr_oc_b = jcp.nthr_ic_b = 1;
    if (nthreads < jcp.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        return;
    }
    const int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    const int nb_load = div_up(jcp.load_dim, jcp.load_block);
    const int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    jcp.nthr_g = jcp.ngroups;
    const int nthr = nthreads / jcp.nthr_g;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level
        * optimizer tries to minimize memory consumption. few notes: (n1)
        * unclear why, but that essentially helps first convolution...
        *  (n2) assuming the reduction over minibatch is always there:
        *    - instead of 8 it should be 5 here (write ~= 2 read):
        *      kernel: temporal workspace 1 write
        *      reduction: 1 read from workspace and 1 write to the diff_wei
        *    - but experiments showed 8 works better than 5 or 6... */
        int bcast_koeff = 1;
        int load_koeff = 1;
        int output_koeff = 12;
        if (jcp.transpose_src) {
            bcast_koeff = 5;
            load_koeff = 1;
            output_koeff = 8;
        }
        return 0
                + (size_t)bcast_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
                * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_bcast, nthr_ic_b)
                * jcp.ic_block * jcp.reduce_block / jcp.stride_h
                / jcp.stride_w /* (n1) */
                + (size_t)load_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
                * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_load, nthr_oc_b)
                * jcp.oc_block * jcp.reduce_block
                + (size_t)output_koeff /* (n2) */
                * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_load, nthr_oc_b)
                * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block * jcp.oc_block;
    };

    int nthr_mb = 1, nthr_oc_b = 1, nthr_ic_b = 1;
    auto best_mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, jcp.mb * nb_reduce);
    for (nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, nb_load);
        for (nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, nb_bcast);
            auto mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                jcp.nthr_mb = nthr_mb;
                jcp.nthr_oc_b = nthr_oc_b;
                jcp.nthr_ic_b = nthr_ic_b;
            }
        }

        const bool ready_for_async
                = utils::one_of(jcp.ver, ver_fma, ver_avx512_core);
        if (!ready_for_async && !dnnl_thr_syncable()) {
            assert(nthr_mb == 1);
            break;
        }
    }
    if (jcp.nthr_mb > nthreads / 2 && jcp.nthr_mb < nthreads)
        jcp.nthr_mb = nstl::min(jcp.mb, nthreads);

    jcp.nthr = jcp.nthr_mb * jcp.nthr_g * jcp.nthr_oc_b * jcp.nthr_ic_b;
    assert(jcp.nthr <= nthreads);
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
