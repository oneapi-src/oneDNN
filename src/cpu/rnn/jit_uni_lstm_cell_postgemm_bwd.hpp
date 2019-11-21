/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CPU_JIT_UNI_LSTM_CELL_POSTGEMM_BWD_HPP
#define CPU_JIT_UNI_LSTM_CELL_POSTGEMM_BWD_HPP

#include "jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_lstm_cell_postgemm_bwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lstm_cell_postgemm_bwd)

    typedef typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type injector_t;

    jit_uni_lstm_cell_postgemm_bwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd) {}

    ~jit_uni_lstm_cell_postgemm_bwd() { delete tanh_injector_; }

    void init(data_type_t sdt) override {
        jit_uni_rnn_postgemm::init(src_data_t);
        // we use rax for both constant tables as they use the same table
        tanh_injector_ = new injector_t(
                this, alg_kind::eltwise_tanh, 0.0f, 0.0f, 1.0f, true, rax);
        generate();
        kernel_ = (kernel_t)this->getCode();
    }

protected:
    injector_t *tanh_injector_;

    // register size in bytes
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    size_t vlen = cpu_isa_traits<isa>::vlen;
    size_t vlen_scratch
            = vlen / (sizeof(float) / types::data_type_size(scratch_data_t));
    size_t cstate_dt_size = sizeof(float);
    size_t hstate_dt_size = sizeof(float);
    size_t gate_dt_size = types::data_type_size(scratch_data_t);
    size_t scratch_dt_size = types::data_type_size(scratch_data_t);

    void generate() {
        using namespace Xbyak;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_end_label;
        Label table_label;

        // Register map
        Reg64 table_reg(rbx); // used to load ones before the loop
        Reg64 loop_cnt(rbx); // loop counter, can be aliased with table_reg
        // We skip vmm0 as it can be used by the injector for masks on sse4.1
        int dG0_idx = 1, dG1_idx = 2, dG2_idx = 3, dG3_idx = 4, tanhCt_idx = 5,
            dHt_idx = 6, dCt_idx = 7, G0_idx = 8, G1_idx = 9, one_idx = 10,
            tmp1_idx = 11, tmp2_idx = 12;
        Vmm one_vmm(one_idx);
        Xmm one_xmm(one_idx);

        // Adress maping
        Address one_addr = ptr[table_reg];

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        auto addr_ws_gates_reg = abi_param1;
        auto addr_scratch_gates_reg = abi_param2;
        auto addr_diff_states_t_lp1_reg = abi_param3;
        auto addr_diff_states_tp1_l_reg = abi_param4;
#ifdef _WIN32
        auto addr_diff_c_states_t_l_reg = r10;
        auto addr_diff_c_states_tp1_l_reg = r11;
        auto addr_c_states_tm1_l_reg = r12;
        auto addr_c_states_t_l_reg = rsi;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        auto base_args = rsp + get_size_of_abi_save_regs() + 40;
        mov(addr_diff_c_states_t_l_reg, ptr[base_args]);
        mov(addr_diff_c_states_tp1_l_reg, ptr[base_args + 8]);
        mov(addr_c_states_tm1_l_reg, ptr[base_args + 16]);
        mov(addr_c_states_t_l_reg, ptr[base_args + 24]);
#else
        auto addr_diff_c_states_t_l_reg = abi_param5;
        auto addr_diff_c_states_tp1_l_reg = abi_param6;
        auto addr_c_states_tm1_l_reg = r10;
        auto addr_c_states_t_l_reg = r11;
        auto base_args = rsp + get_size_of_abi_save_regs() + 8;
        mov(addr_c_states_tm1_l_reg, ptr[base_args]);
        mov(addr_c_states_t_l_reg, ptr[base_args + 8]);
#endif

        // helper lambda to address the gates and biases
        auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dic * scratch_dt_size];
        };

        auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dic * gate_dt_size];
        };

        // initialize registers with addresses and constants
        mov(table_reg, table_label);
        init_regs(vlen);
        uni_vmovups(one_vmm, one_addr);
        tanh_injector_->load_table_addr();

        mov(loop_cnt, rnn_.dic * scratch_dt_size);
        cmp(loop_cnt, vlen_scratch);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            Vmm dG0(dG0_idx), dG1(dG1_idx), dG2(dG2_idx), dG3(dG3_idx),
                    tanhCt(tanhCt_idx), dHt(dHt_idx), dCt(dCt_idx),
                    tmp1(tmp1_idx), G0(G0_idx), G1(G1_idx);

            // TODO: if w_gates are bfloat, we have to convert them to float
            // datatypes summary:
            // - c states are all float
            // - h states are all src_data_t
            // - diff_* are all float
            // - scratch is src_data_t
            // - ws_gates is src_data_t

            // compute tanhCt
            uni_vmovups(tanhCt, ptr[addr_c_states_t_l_reg]);
            tanh_injector_->compute_vector(tanhCt.getIdx());

            // compute dHt
            uni_vmovups(dHt, ptr[addr_diff_states_tp1_l_reg]);
            // assumption: the diff_states_t_lp1 address is already offset by rnn.n_states
            uni_vmovups(tmp1, ptr[addr_diff_states_t_lp1_reg]);
            uni_vaddps(dHt, dHt, tmp1);

            // compute dCt
            uni_vmovups(tmp1, one_vmm);
            uni_vfnmadd231ps(tmp1, tanhCt, tanhCt);
            uni_vmulps(tmp1, tmp1, dHt);
            to_float<scratch_data_t>(dG3, wg_addr(3), vlen);
            uni_vmulps(tmp1, tmp1, dG3);
            uni_vmovups(dCt, ptr[addr_diff_c_states_tp1_l_reg]);
            uni_vaddps(dCt, dCt, tmp1);

            // compute dG0
            // we will reuse G0 and G2 later for dG2
            to_float<src_data_t>(G0, wg_addr(0), vlen);
            to_float<src_data_t>(dG2, wg_addr(2), vlen);
            uni_vmovups(dG0, G0);
            uni_vfnmadd231ps(dG0, dG0, dG0);
            uni_vmulps(dG0, dG0, dCt);
            uni_vmulps(dG0, dG0, dG2);

            // compute dG1
            to_float<src_data_t>(G1, wg_addr(1), vlen);
            uni_vmovups(dG1, G1);
            uni_vfnmadd231ps(dG1, dG1, dG1);
            uni_vmulps(dG1, dG1, dCt);
            uni_vmovups(tmp1, ptr[addr_c_states_tm1_l_reg]);
            uni_vmulps(dG1, dG1, tmp1);

            // compute dG2
            uni_vmovups(tmp1, one_vmm);
            uni_vfnmadd231ps(tmp1, dG2, dG2);
            uni_vmulps(G0, G0, dCt);
            uni_vmulps(tmp1, tmp1, G0);
            uni_vmovups(dG2, tmp1);

            // compute dG3
            to_float<src_data_t>(dG3, wg_addr(3), vlen);
            uni_vfnmadd231ps(dG3, dG3, dG3);
            uni_vmulps(dG3, dG3, dHt);
            uni_vmulps(dG3, dG3, tanhCt);

            // compute diff_state_t_l
            uni_vmulps(dCt, dCt, G1);
            uni_vmovups(ptr[addr_diff_c_states_t_l_reg], dCt);

            to_src<scratch_data_t>(sg_addr(0), dG0, vlen);
            to_src<scratch_data_t>(sg_addr(1), dG1, vlen);
            to_src<scratch_data_t>(sg_addr(2), dG2, vlen);
            to_src<scratch_data_t>(sg_addr(3), dG3, vlen);

            // increment address pointers
            add(addr_ws_gates_reg, vlen_scratch);
            add(addr_scratch_gates_reg, vlen_scratch);
            add(addr_diff_states_t_lp1_reg, vlen);
            add(addr_diff_states_tp1_l_reg, vlen);
            add(addr_diff_c_states_t_l_reg, vlen);
            add(addr_diff_c_states_tp1_l_reg, vlen);
            add(addr_c_states_tm1_l_reg, vlen);
            add(addr_c_states_t_l_reg, vlen);
            inc_regs(vlen);

            // increment loop counter
            sub(loop_cnt, vlen_scratch);
            cmp(loop_cnt, vlen_scratch);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use vmovss for accessing inputs
        L(rem_loop_start_label);
        {
            Xmm dG0(dG0_idx), dG1(dG1_idx), dG2(dG2_idx), dG3(dG3_idx),
                    tanhCt(tanhCt_idx), dHt(dHt_idx), dCt(dCt_idx),
                    tmp1(tmp1_idx), tmp2(tmp2_idx), G0(G0_idx), G1(G1_idx);

            // compute tanhCt
            uni_vmovss(tanhCt, ptr[addr_c_states_t_l_reg]);
            tanh_injector_->compute_vector(tanhCt.getIdx());

            // compute dHt
            uni_vmovss(dHt, ptr[addr_diff_states_tp1_l_reg]);
            // assumption: the diff_states_t_lp1 address is already offset by rnn.n_states
            uni_vmovss(tmp1, ptr[addr_diff_states_t_lp1_reg]);
            uni_vaddss(dHt, dHt, tmp1);

            // compute dCt
            uni_vmovss(tmp1, one_xmm);
            // This overrides tanhCt when using Xmm
            uni_vmovss(tmp2, tanhCt);
            uni_vfnmadd231ps(tmp1, tmp2, tmp2);
            uni_vmulss(tmp1, tmp1, dHt);
            to_float<scratch_data_t>(dG3, wg_addr(3), hstate_dt_size);
            uni_vmulss(tmp1, tmp1, dG3);
            uni_vmovss(dCt, ptr[addr_diff_c_states_tp1_l_reg]);
            uni_vaddss(dCt, dCt, tmp1);

            // compute dG0
            // we will reuse G0 and G2 later for dG2
            to_float<src_data_t>(G0, wg_addr(0), hstate_dt_size);
            to_float<src_data_t>(dG2, wg_addr(2), hstate_dt_size);
            uni_vmovss(dG0, G0);
            uni_vmovss(tmp1, G0);
            uni_vfnmadd231ps(dG0, tmp1, tmp1);
            uni_vmulss(dG0, dG0, dCt);
            uni_vmulss(dG0, dG0, dG2);

            // compute dG1
            to_float<src_data_t>(G1, wg_addr(1), hstate_dt_size);
            uni_vmovss(dG1, G1);
            uni_vmovss(tmp1, G1);
            uni_vfnmadd231ps(dG1, tmp1, tmp1);
            uni_vmulss(dG1, dG1, dCt);
            uni_vmovss(tmp1, ptr[addr_c_states_tm1_l_reg]);
            uni_vmulss(dG1, dG1, tmp1);

            // compute dG2
            uni_vmovss(tmp1, one_xmm);
            uni_vmovss(tmp2, dG2);
            uni_vfnmadd231ps(tmp1, tmp2, tmp2);
            uni_vmulss(G0, G0, dCt);
            uni_vmulss(tmp1, tmp1, G0);
            uni_vmovss(dG2, tmp1);

            // compute dG3
            to_float<src_data_t>(dG3, wg_addr(3), hstate_dt_size);
            uni_vmovss(tmp1, dG3);
            uni_vfnmadd231ps(dG3, tmp1, tmp1);
            uni_vmulss(dG3, dG3, dHt);
            uni_vmulss(dG3, dG3, tanhCt);

            // compute diff_state_t_l
            uni_vmulss(dCt, dCt, G1);
            uni_vmovss(ptr[addr_diff_c_states_t_l_reg], dCt);

            to_src<scratch_data_t>(sg_addr(0), dG0, hstate_dt_size);
            to_src<scratch_data_t>(sg_addr(1), dG1, hstate_dt_size);
            to_src<scratch_data_t>(sg_addr(2), dG2, hstate_dt_size);
            to_src<scratch_data_t>(sg_addr(3), dG3, hstate_dt_size);

            // increment address pointers
            add(addr_ws_gates_reg, scratch_dt_size);
            add(addr_scratch_gates_reg, scratch_dt_size);
            add(addr_diff_states_t_lp1_reg, hstate_dt_size);
            add(addr_diff_states_tp1_l_reg, hstate_dt_size);
            add(addr_diff_c_states_t_l_reg, cstate_dt_size);
            add(addr_diff_c_states_tp1_l_reg, cstate_dt_size);
            add(addr_c_states_tm1_l_reg, cstate_dt_size);
            add(addr_c_states_t_l_reg, cstate_dt_size);
            inc_regs(hstate_dt_size);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);
        }
        L(rem_loop_end_label);

        postamble();

        tanh_injector_->prepare_table();
        init_table(vlen);
        L(table_label);
        {
            for (size_t i = 0; i < vlen / sizeof(float); i++)
                dd(float2int(1.0f));
        }
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
