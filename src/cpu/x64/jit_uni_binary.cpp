/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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
#include <functional>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

#define PARAM_OFF(x) offsetof(call_params_t, x)

template <data_type_t src_type>
bool jit_uni_binary_t<src_type>::post_ops_ok(
        const primitive_attr_t *attr, const memory_desc_wrapper &dst_d) {
    using namespace primitive_kind;

    const auto &p = attr->post_ops_;
    const auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    const auto is_binary = [&](int idx) { return p.entry_[idx].is_binary(); };

    for (int i = 0; i < p.len(); i++) {
        if (p.contain(primitive_kind::sum, i)) {
            if (i > 0) return false;
        } else if (!(is_eltwise(i) || is_binary(i)))
            return false;
    }

    const int vlen = mayiuse(avx512_common)
            ? cpu_isa_traits<avx512_common>::vlen
            : cpu_isa_traits<avx2>::vlen;

    const bool postops_per_oc_broadcast_exists
            = binary_injector::any_binary_postop_rhs_per_oc_broadcast(p, dst_d);
    const int blksize = vlen / sizeof(float);

    const bool blocked_format = !dst_d.is_plain() && dst_d.is_blocking_desc();

    if (postops_per_oc_broadcast_exists && blocked_format) {
        /*
         * check blocking_desc consistency, currently when among postops exists
         * per_oc broadcast, binary kernel doesn't support situations when blocked
         * format size is smaller then vlen. example: sse41 vlen size is 4 and format
         * is nChw8c - not supported, avx2 vlen size is 8 and format is
         * nChw8c - supported.
         */
        const auto blocking_desc = dst_d.blocking_desc();
        if (blocking_desc.inner_nblks != 1
                || blocking_desc.inner_blks[0] != blksize
                || blocking_desc.inner_idxs[0] != 1)
            return false;
    }

    return binary_injector::binary_args_broadcast_supported(p, dst_d)
            && IMPLICATION(postops_per_oc_broadcast_exists,
                    binary_injector::all_binary_postop_rhs_per_oc_broadcast(p,
                            dst_d,
                            [&dst_d](const memory_desc_wrapper &rhs_arg_md) {
                                return IMPLICATION(!mayiuse(avx2),
                                        dst_d.consistent_with(rhs_arg_md)
                                                || dst_d.is_plain());
                            }));
}

using namespace Xbyak;

enum class op_t : unsigned {
    none,
    tensor,
    bcast_c_blocked,
    bcast_n_spatial_c,
    bcast_n_c_spatial
};

static op_t get_bcast_per_c(const memory_desc_wrapper &src0_d) {
    const auto &strides = src0_d.blocking_desc().strides;
    const auto ndims = src0_d.ndims();

    if (!src0_d.is_plain())
        return op_t::bcast_c_blocked;
    else if (strides[1] == 1)
        return op_t::bcast_n_spatial_c;
    else if (strides[0] >= strides[1]
            && IMPLICATION(ndims >= 3, strides[1] >= strides[2]))
        return op_t::bcast_n_c_spatial;
    return op_t::none;
}

struct binary_kernel_t : public jit_generator, public c_compatible {

    using c_compatible::operator new;
    using c_compatible::operator delete;
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        const void *src0, *src1, *dst;
        size_t spat_offt_count;
        const void *post_ops_binary_rhs_arg_vec;
        size_t oc_l_off;
    };

    binary_kernel_t(int vlen) : vlen_(vlen), simd_w_(vlen / sizeof(float)) {}
    ~binary_kernel_t() override = default;

    void operator()(binary_kernel_t::call_params_t *p) {
        jit_generator::operator()(p);
    }

    int simd_w() const noexcept { return simd_w_; }
    int vlen() const noexcept { return vlen_; }
    op_t op_type() const noexcept { return op_type_; }
    op_t bcast_per_oc() const noexcept { return bcast_per_oc_; }

protected:
    const int vlen_;
    const size_t simd_w_;

    op_t op_type_ = op_t::none;
    op_t bcast_per_oc_ = op_t::none;
};

template <cpu_isa_t isa>
struct jit_uni_binary_kernel_t : public binary_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_binary_kernel_t)

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword
            = (isa == sse41) ? xword : ((isa == avx2) ? yword : zword);

    const binary_pd_t *pd_;
    bool is_bf16_;
    const bool is_avx512 = utils::one_of(isa, avx512_core, avx512_core_bf16);
    const Reg64 &reg_param_ = abi_param1;
    const Reg64 &reg_src0_ = r8;
    const Reg64 &reg_src1_ = r9;
    const Reg64 &reg_dst_ = r10;
    const Reg64 &reg_offt_src0_ = r11;
    const Reg64 &reg_offt_src0_count_ = r12;
    const Reg64 &reg_offt_src1_ = rax;
    const Reg64 &reg_reverse_spat_offt_ = r13;
    const Reg64 &reg_tmp_ = r14;
    const Reg64 &reg_elt_inj_table_ = r15;
    const Reg64 &reg_off_rhs_postops_ = rdx;
    const Opmask tail_opmask_ = Opmask(2);
    const Xmm xsum_scale_ = Xmm(15);
    const Vmm vbcast_src1_ = Vmm(is_avx512 ? 30 : 14);
    const Vmm vsum_scale_ = Vmm(is_avx512 ? 31 : 15);

    size_t unroll_regs_ = is_avx512 ? 8 : 4;
    size_t tail_size_ = 0;
    size_t data_type_size_ = 0;
    bool do_sum_ = false;
    bool with_eltwise_ = false;
    float sum_scale_ = 0.f;
    size_t offt_src0_ = 0;
    size_t offt_src1_ = 0;
    bool use_stride_src1_ = false;
    bool broadcast_src1_value_ = false;
    bool use_stride_rhs_postops_ = false;

    static constexpr cpu_isa_t inject_isa
            = isa == avx512_core_bf16 ? avx512_core : isa;
    bool is_tail_kernel_ = false;
    std::unique_ptr<injector::jit_uni_postops_injector_t<inject_isa>>
            postops_injector_;
    const Opmask elt_inj_opmask_ = Opmask(1);

    void init() {
        const memory_desc_wrapper src0_d(pd_->src_md(0));
        const memory_desc_wrapper src1_d(pd_->src_md(1));
        bcast_per_oc_ = get_bcast_per_c(src0_d);
        op_type_ = pd_->is_tensor_op() ? op_t::tensor : bcast_per_oc_;
        assert(op_type_ != op_t::none);
        is_bf16_ = src0_d.data_type() == data_type::bf16;
        data_type_size_ = is_bf16_ ? sizeof(bfloat16_t) : sizeof(float);

        const auto &po = pd_->attr()->post_ops_;
        const bool postops_per_oc_broadcast_exists
                = binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                        po, src0_d);
        broadcast_src1_value_
                = op_type_ == op_t::bcast_n_c_spatial || src1_d.nelems() == 1;
        use_stride_src1_ = !broadcast_src1_value_
                && (op_type_ == op_t::tensor
                        || op_type_ == op_t::bcast_n_spatial_c);
        use_stride_rhs_postops_ = postops_per_oc_broadcast_exists
                && bcast_per_oc_ == op_t::bcast_n_spatial_c;

        tail_size_ = get_tail_size(src0_d, postops_per_oc_broadcast_exists);
        offt_src0_ = vlen_ / (is_bf16_ ? 2 : 1);
        offt_src1_ = use_stride_src1_ ? offt_src0_ : 0;
        do_sum_ = po.contain(primitive_kind::sum, 0)
                && po.entry_[0].sum.scale != 0.f;
        sum_scale_ = do_sum_ ? po.entry_[0].sum.scale : 0.f;
        with_eltwise_ = po.find(primitive_kind::eltwise) != -1;
        const bool with_binary = po.find(primitive_kind::binary) != -1;
        const bool with_postops = with_binary || with_eltwise_;

        if (with_postops) init_post_ops_injector();
    }

    std::size_t get_tail_size(const memory_desc_wrapper &src0_d,
            bool postops_per_oc_broadcast_exists) const {
        const auto &dims = src0_d.dims();
        const auto &ndims = src0_d.ndims();

        dim_t nelems = 0;

        if (bcast_per_oc_ == op_t::bcast_c_blocked && is_tail_kernel_)
            nelems = dims[1];
        else if (op_type_ == op_t::tensor && !postops_per_oc_broadcast_exists)
            nelems = src0_d.nelems(true);
        else {
            if (bcast_per_oc_ == op_t::bcast_n_spatial_c)
                nelems = dims[1];
            else if (bcast_per_oc_ == op_t::bcast_n_c_spatial && ndims >= 3)
                nelems = utils::array_product(dims + 2, ndims - 2);
        }
        // it's float due to for bfloat16 we still load 16 elements, not 32.
        return nelems % simd_w_;
    }

    void init_post_ops_injector() {
        const memory_desc_wrapper src0_d(pd_->src_md(0));
        const auto &po = pd_->attr()->post_ops_;

        const eltwise_injector::static_params_t esp(true /*save_state*/,
                reg_elt_inj_table_, elt_inj_opmask_, true /*is_fwd*/,
                false /*use_dst*/);
        const binary_injector::rhs_arg_static_params_t rhs_arg_bsp {10,
                reg_tmp_, reg_elt_inj_table_, true /*preserve gpr*/,
                true /*preserve vmm*/, PARAM_OFF(post_ops_binary_rhs_arg_vec),
                src0_d, tail_size_, tail_opmask_,
                false /*use_exact_tail_scalar_bcast*/};
        const binary_injector::static_params_t bsp(this->param1, rhs_arg_bsp);

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<inject_isa>>(
                this, po, bsp, esp);
    }

    void apply_postops(int unroll, bool tail) {
        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
        for (int vmm_idx = 1; vmm_idx < unroll + 1; vmm_idx++) {
            if (bcast_per_oc_ == op_t::bcast_c_blocked
                    || bcast_per_oc_ == op_t::bcast_n_c_spatial) {
                rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
                        vmm_idx, ptr[param1 + PARAM_OFF(oc_l_off)]);
            } else if (bcast_per_oc_ == op_t::bcast_n_spatial_c) {
                rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                        vmm_idx, reg_off_rhs_postops_);
                rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                        vmm_idx, (vmm_idx - 1) * static_cast<int>(simd_w_));
            }
            if (tail) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
        }
        postops_injector_->compute_vector_range(1, unroll + 1, rhs_arg_params);
    }

    void load_kernel_params() {
        mov(reg_tmp_, float2int(sum_scale_));
        uni_vmovq(xsum_scale_, reg_tmp_);
        uni_vbroadcastss(vsum_scale_, xsum_scale_);
        mov(reg_offt_src0_count_, ptr[reg_param_ + PARAM_OFF(spat_offt_count)]);
        mov(reg_src0_, ptr[reg_param_ + PARAM_OFF(src0)]);
        mov(reg_src1_, ptr[reg_param_ + PARAM_OFF(src1)]);
        mov(reg_dst_, ptr[reg_param_ + PARAM_OFF(dst)]);
    }

    Address src0_ptr(size_t offt = 0) {
        return vmmword[reg_src0_ + reg_offt_src0_ + offt];
    }

    Address src1_ptr(size_t offt = 0) {
        return vmmword[reg_src1_ + reg_offt_src1_ + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst_ + reg_offt_src0_ + offt];
    }

    void perform_op(const Vmm &v0, const Vmm &v1) {
        using namespace alg_kind;
        const auto alg = pd_->desc()->alg_kind;
        if (alg == binary_add)
            uni_vaddps(v0, v0, v1);
        else if (alg == binary_mul)
            uni_vmulps(v0, v0, v1);
        else if (alg == binary_max)
            uni_vmaxps(v0, v0, v1);
        else if (alg == binary_min)
            uni_vminps(v0, v0, v1);
        else if (alg == binary_div)
            uni_vdivps(v0, v0, v1);
        else
            assert(!"not supported operation!");
    }

    virtual void prepare_isa_subkernel() = 0;
    virtual void compute_bcast(bool tail) = 0;
    virtual void compute_dst(int unroll, bool tail) = 0;

    void forward() {
        Label unroll_loop, unroll_loop_tail, nelems_tail, end;

        // reverse spat_offt to dispatch between labels
        mov(reg_reverse_spat_offt_, reg_offt_src0_count_);
        xor_(reg_offt_src0_,
                reg_offt_src0_); // offt_src0 to get addr of src0/dst
        xor_(reg_offt_src1_, reg_offt_src1_); // offt_src1 to get addr of src1
        if (use_stride_rhs_postops_)
            xor_(reg_off_rhs_postops_, reg_off_rhs_postops_);
        const size_t vec_size = simd_w_ * data_type_size_;

        compute_bcast(false); // bcast/load vreg just one time per a kernel call

        // used in bcast_c_blocked strategy for last blocked if tail exists
        const bool treat_each_compute_step_as_tail
                = is_tail_kernel_ && tail_size_;

        L(unroll_loop);
        {
            const size_t offt = unroll_regs_ * vec_size;
            const size_t offt_elems = unroll_regs_ * simd_w_;
            cmp(reg_reverse_spat_offt_, offt);
            jl(unroll_loop_tail, T_NEAR);

            compute_dst(unroll_regs_, treat_each_compute_step_as_tail);
            sub(reg_reverse_spat_offt_, offt);
            add(reg_offt_src0_, offt);
            if (use_stride_src1_) add(reg_offt_src1_, offt);
            if (use_stride_rhs_postops_) add(reg_off_rhs_postops_, offt_elems);
            jmp(unroll_loop);
        }

        L(unroll_loop_tail);
        {
            cmp(reg_reverse_spat_offt_, vec_size);
            jl(nelems_tail, T_NEAR);

            compute_dst(1, treat_each_compute_step_as_tail);
            sub(reg_reverse_spat_offt_, vec_size);
            add(reg_offt_src0_, vec_size);
            if (use_stride_src1_) add(reg_offt_src1_, vec_size);
            if (use_stride_rhs_postops_) add(reg_off_rhs_postops_, simd_w_);

            jmp(unroll_loop_tail);
        }

        L(nelems_tail);
        {
            cmp(reg_reverse_spat_offt_, 1);
            jl(end, T_NEAR);

            compute_dst(1, true);
        }

        L(end);
    }

    void generate() override {
        preamble();
        load_kernel_params();
        prepare_isa_subkernel();
        forward();
        postamble();

        if (with_eltwise_ && postops_injector_)
            postops_injector_->prepare_table();
    }

    jit_uni_binary_kernel_t(const binary_pd_t *pd, bool tail_kernel = false)
        : binary_kernel_t(cpu_isa_traits<isa>::vlen)
        , pd_(pd)
        , is_tail_kernel_(tail_kernel) {
        init();
    }
    ~jit_uni_binary_kernel_t() override = default;
};

template <cpu_isa_t isa, data_type_t src_type>
struct jit_uni_binary_subkernel_t;

template <data_type_t src_type>
struct jit_uni_binary_subkernel_t<avx512_core_bf16, src_type>
    : public jit_uni_binary_kernel_t<avx512_core_bf16> {
    Opmask bf16_bcast_opmask = Opmask(3);

    void prepare_tail_mask() {
        if (!tail_size_) return;

        const int mask_f32 = (1 << tail_size_) - 1;
        const Reg32 regw_tmp = reg_tmp_.cvt32();
        mov(regw_tmp, mask_f32);
        kmovd(tail_opmask_, regw_tmp);
    }

    void prepare_bf16_bcast_mask() {
        if (is_bf16_ && op_type_ != op_t::tensor) {
            const Reg32 regw_tmp = reg_tmp_.cvt32();
            mov(regw_tmp, 1);
            kmovd(bf16_bcast_opmask, regw_tmp);
        }
    }

    void prepare_isa_subkernel() override {
        prepare_tail_mask();
        prepare_bf16_bcast_mask();
    }

    void load_no_tail(const Vmm &dst, const Address &src, data_type_t dt) {
        switch (dt) {
            case data_type::f32: uni_vmovups(dst, src); break;
            case data_type::bf16:
                vpmovzxwd(dst, src);
                vpslld(dst, dst, 0x10);
                break;
            default: assert(!"unreachable");
        }
    }

    void load_tail(const Vmm &dst, const Opmask &opmask, const Address &src,
            data_type_t dt) {
        switch (dt) {
            case data_type::f32: uni_vmovups_tail(dst, opmask, src); break;
            case data_type::bf16:
                vpmovzxwd(dst | opmask, src);
                vpslld(dst, dst, 0x10);
                break;
            default: assert(!"unreachable");
        }
    }

    void store_no_tail(const Address &dst, const Vmm &src, data_type_t dt) {
        const Ymm ymm_src = Ymm(src.getIdx());
        switch (dt) {
            case data_type::f32: uni_vmovups(dst, src); break;
            case data_type::bf16:
                vcvtneps2bf16(ymm_src, src);
                vmovdqu16(dst, ymm_src);
                break;
            default: assert(!"unreachable");
        }
    }

    void store_tail(const Address &dst, const Opmask &opmask, const Vmm &src,
            data_type_t dt) {
        const Ymm ymm_src = Ymm(src.getIdx());
        switch (dt) {
            case data_type::f32: uni_vmovups_tail(dst, opmask, src); break;
            case data_type::bf16:
                vcvtneps2bf16(ymm_src, src);
                vmovdqu16(dst | opmask, ymm_src);
                break;
            default: assert(!"unreachable");
        }
    }

    void load(const Vmm &dst, const Address &src, data_type_t dt, bool tail) {
        if (!tail)
            load_no_tail(dst, src, dt);
        else
            load_tail(dst, tail_opmask_, src, dt);
    }

    void store(const Address &dst, const Vmm &src, data_type_t dt, bool tail) {
        if (!tail)
            store_no_tail(dst, src, dt);
        else
            store_tail(dst, tail_opmask_, src, dt);
    }

    void bcast(const Vmm &dst, const Address &src, data_type_t dt) {
        switch (dt) {
            case data_type::f32: uni_vbroadcastss(dst, src); break;
            case data_type::bf16:
                vpmovzxwd(dst | bf16_bcast_opmask, src);
                vpslld(dst, dst, 0x10);
                uni_vbroadcastss(dst, Xmm(dst.getIdx()));
                break;
            default: assert(!"unreachable");
        }
    }

    void compute_bcast(bool tail) override {
        if (broadcast_src1_value_)
            bcast(vbcast_src1_, src1_ptr(), src_type);
        else if (offt_src1_ == 0)
            load(vbcast_src1_, src1_ptr(), src_type, tail);
    }

    void compute_dst(int unroll, bool tail) override {
        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + 1);
            const Vmm vreg_tmp = Vmm(unroll + i + 1);
            const Vmm vreg_tmp_src1 = offt_src1_ ? vreg_tmp : vbcast_src1_;
            load(vreg_tmp_src0, src0_ptr(i * offt_src0_), src_type, tail);

            if (offt_src1_) {
                load(vreg_tmp_src1, src1_ptr(i * offt_src1_), src_type, tail);
            }
            perform_op(vreg_tmp_src0, vreg_tmp_src1);

            if (do_sum_) {
                load(vreg_tmp, dst_ptr(i * offt_src0_), src_type, tail);
                uni_vfmadd231ps(vreg_tmp_src0, vreg_tmp, vsum_scale_);
            }
        }

        if (postops_injector_) apply_postops(unroll, tail);

        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + 1);
            store(dst_ptr(i * offt_src0_), vreg_tmp_src0, src_type, tail);
        }
    }

    jit_uni_binary_subkernel_t(const binary_pd_t *pd, bool tail_kernel)
        : jit_uni_binary_kernel_t(pd, tail_kernel) {}
};

template <data_type_t src_type>
struct jit_uni_binary_subkernel_t<avx512_core, src_type>
    : public jit_uni_binary_kernel_t<avx512_core> {
    Opmask bf16_bcast_opmask = Opmask(3);

    // FP32->BF16 emulation
    std::unique_ptr<bf16_emulation_t> bf16_emu_ {nullptr};
    const Reg64 &reg_bf16_tmp = reg_tmp_;
    const Vmm bf16_emu_reserved_1 = Vmm(26);
    const Vmm bf16_emu_reserved_2 = Vmm(27);
    const Vmm bf16_emu_reserved_3 = Vmm(28);
    const Vmm bf16_emu_reserved_4 = Vmm(29);

    void prepare_tail_mask() {
        if (!tail_size_) return;

        const int mask_f32 = (1 << tail_size_) - 1;
        const Reg32 regw_tmp = reg_tmp_.cvt32();
        mov(regw_tmp, mask_f32);
        kmovd(tail_opmask_, regw_tmp);
    }

    void prepare_bf16_emulator() {
        if (is_bf16_) { // init emulation of bfloat16 operations
            bf16_emu_.reset(new bf16_emulation_t(this, bf16_emu_reserved_1,
                    bf16_emu_reserved_2, bf16_emu_reserved_3, reg_bf16_tmp,
                    bf16_emu_reserved_4, bf16_emu_reserved_4));
            (*bf16_emu_).init_vcvtneps2bf16();
        }
    }

    void prepare_bf16_bcast_mask() {
        if (is_bf16_ && op_type_ != op_t::tensor) {
            const Reg32 regw_tmp = reg_tmp_.cvt32();
            mov(regw_tmp, 1);
            kmovd(bf16_bcast_opmask, regw_tmp);
        }
    }

    void prepare_isa_subkernel() override {
        prepare_tail_mask();
        prepare_bf16_emulator();
        prepare_bf16_bcast_mask();
    }

    void load_no_tail(const Vmm &dst, const Address &src, data_type_t dt) {
        switch (dt) {
            case data_type::f32: uni_vmovups(dst, src); break;
            case data_type::bf16:
                vpmovzxwd(dst, src);
                vpslld(dst, dst, 0x10);
                break;
            default: assert(!"unreachable");
        }
    }

    void load_tail(const Vmm &dst, const Opmask &opmask, const Address &src,
            data_type_t dt) {
        switch (dt) {
            case data_type::f32: uni_vmovups_tail(dst, opmask, src); break;
            case data_type::bf16:
                vpmovzxwd(dst | opmask, src);
                vpslld(dst, dst, 0x10);
                break;
            default: assert(!"unreachable");
        }
    }

    void store_no_tail(const Address &dst, const Vmm &src, data_type_t dt) {
        const Ymm ymm_src = Ymm(src.getIdx());
        switch (dt) {
            case data_type::f32: uni_vmovups(dst, src); break;
            case data_type::bf16:
                (*bf16_emu_).vcvtneps2bf16(ymm_src, src);
                vmovdqu16(dst, ymm_src);
                break;
            default: assert(!"unreachable");
        }
    }

    void store_tail(const Address &dst, const Opmask &opmask, const Vmm &src,
            data_type_t dt) {
        const Ymm ymm_src = Ymm(src.getIdx());
        switch (dt) {
            case data_type::f32: uni_vmovups_tail(dst, opmask, src); break;
            case data_type::bf16:
                (*bf16_emu_).vcvtneps2bf16(ymm_src, src);
                vmovdqu16(dst | opmask, ymm_src);
                break;
            default: assert(!"unreachable");
        }
    }

    void load(const Vmm &dst, const Address &src, data_type_t dt, bool tail) {
        if (!tail)
            load_no_tail(dst, src, dt);
        else
            load_tail(dst, tail_opmask_, src, dt);
    }

    void store(const Address &dst, const Vmm &src, data_type_t dt, bool tail) {
        if (!tail)
            store_no_tail(dst, src, dt);
        else
            store_tail(dst, tail_opmask_, src, dt);
    }

    void bcast(const Vmm &dst, const Address &src, data_type_t dt) {
        switch (dt) {
            case data_type::f32: uni_vbroadcastss(dst, src); break;
            case data_type::bf16:
                vpmovzxwd(dst | bf16_bcast_opmask, src);
                vpslld(dst, dst, 0x10);
                uni_vbroadcastss(dst, Xmm(dst.getIdx()));
                break;
            default: assert(!"unreachable");
        }
    }

    void compute_bcast(bool tail) override {
        if (broadcast_src1_value_)
            bcast(vbcast_src1_, src1_ptr(), src_type);
        else if (offt_src1_ == 0)
            load(vbcast_src1_, src1_ptr(), src_type, tail);
    }

    void compute_dst(int unroll, bool tail) override {
        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + 1);
            const Vmm vreg_tmp = Vmm(unroll + i + 1);
            const Vmm vreg_tmp_src1 = offt_src1_ ? vreg_tmp : vbcast_src1_;
            load(vreg_tmp_src0, src0_ptr(i * offt_src0_), src_type, tail);
            if (offt_src1_) {
                load(vreg_tmp_src1, src1_ptr(i * offt_src1_), src_type, tail);
            }
            perform_op(vreg_tmp_src0, vreg_tmp_src1);

            if (do_sum_) {
                load(vreg_tmp, dst_ptr(i * offt_src0_), src_type, tail);
                uni_vfmadd231ps(vreg_tmp_src0, vreg_tmp, vsum_scale_);
            }
        }

        if (postops_injector_) apply_postops(unroll, tail);

        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + 1);
            store(dst_ptr(i * offt_src0_), vreg_tmp_src0, src_type, tail);
        }
    }

    jit_uni_binary_subkernel_t(const binary_pd_t *pd, bool tail_kernel)
        : jit_uni_binary_kernel_t(pd, tail_kernel) {}
};

template <data_type_t src_type>
struct jit_uni_binary_subkernel_t<avx2, src_type>
    : public jit_uni_binary_kernel_t<avx2> {
    const Vmm tail_vmask_ = Vmm(0);

    void prepare_tail_mask() {
        if (!tail_size_) return;

        static const uint32_t mask_f32[14]
                = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                        0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

        mov(reg_tmp_, reinterpret_cast<size_t>(&mask_f32[7 - tail_size_]));
        vmovups(tail_vmask_, ptr[reg_tmp_]);
    }

    void prepare_isa_subkernel() override { prepare_tail_mask(); }

    void load(const Vmm &dst, const Address &src, bool tail) {
        if (!tail)
            uni_vmovups(dst, src);
        else
            uni_vmovups_tail(dst, tail_vmask_, src);
    }

    void store(const Address &dst, const Vmm &src, bool tail) {
        if (!tail)
            uni_vmovups(dst, src);
        else
            uni_vmovups_tail(dst, tail_vmask_, src);
    }

    void compute_bcast(bool tail) override {
        if (broadcast_src1_value_)
            uni_vbroadcastss(vbcast_src1_, src1_ptr());
        else if (offt_src1_ == 0)
            load(vbcast_src1_, src1_ptr(), tail);
    }

    void compute_dst(int unroll, bool tail) override {
        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + 1);
            const Vmm vreg_tmp = Vmm(unroll + i + 1);
            const Vmm vreg_tmp_src1 = offt_src1_ ? vreg_tmp : vbcast_src1_;
            load(vreg_tmp_src0, src0_ptr(i * offt_src0_), tail);
            if (offt_src1_) load(vreg_tmp_src1, src1_ptr(i * offt_src1_), tail);

            perform_op(vreg_tmp_src0, vreg_tmp_src1);

            if (do_sum_) {
                load(vreg_tmp, dst_ptr(i * offt_src0_), tail);
                uni_vfmadd231ps(vreg_tmp_src0, vreg_tmp, vsum_scale_);
            }
        }

        if (postops_injector_) apply_postops(unroll, tail);

        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + 1);
            store(dst_ptr(i * offt_src0_), vreg_tmp_src0, tail);
        }
    }

    jit_uni_binary_subkernel_t(const binary_pd_t *pd, bool tail_kernel)
        : jit_uni_binary_kernel_t(pd, tail_kernel) {}
};

template <data_type_t src_type>
struct jit_uni_binary_subkernel_t<sse41, src_type>
    : public jit_uni_binary_kernel_t<sse41> {

    void prepare_isa_subkernel() override {}

    Address get_address(const int off, const int arg_num) {
        switch (arg_num) {
            case DNNL_ARG_SRC_0: return src0_ptr(off);
            case DNNL_ARG_SRC_1: return src1_ptr(off);
            case DNNL_ARG_DST: return dst_ptr(off);
            default: assert(!"unsupported arg_num"); break;
        }
        return Address(0);
    }

    void load(const Vmm &dst, const int off, const int arg_num, bool tail) {
        if (!tail)
            movups(dst, get_address(off, arg_num));
        else
            for (size_t i = 0; i < tail_size_; i++)
                pinsrd(dst, get_address(i * data_type_size_ + off, arg_num), i);
    }

    void store(const Vmm &src, const int off, bool tail) {
        if (!tail)
            movups(get_address(off, DNNL_ARG_DST), src);
        else
            for (size_t i = 0; i < tail_size_; i++)
                pextrd(get_address(i * data_type_size_ + off, DNNL_ARG_DST),
                        src, i);
    }

    void compute_bcast(bool tail) override {
        if (broadcast_src1_value_)
            uni_vbroadcastss(vbcast_src1_, src1_ptr());
        else if (offt_src1_ == 0)
            load(vbcast_src1_, 0, DNNL_ARG_SRC_1, tail);
    }

    void compute_dst(int unroll, bool tail) override {
        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + 1);
            const Vmm vreg_tmp = Vmm(unroll + i + 1);
            const Vmm vreg_tmp_src1 = offt_src1_ ? vreg_tmp : vbcast_src1_;
            load(vreg_tmp_src0, i * offt_src0_, DNNL_ARG_SRC_0, tail);
            if (offt_src1_)
                load(vreg_tmp_src1, i * offt_src1_, DNNL_ARG_SRC_1, tail);

            perform_op(vreg_tmp_src0, vreg_tmp_src1);
            if (do_sum_) {
                load(vreg_tmp, i * offt_src0_, DNNL_ARG_DST, tail);
                mulps(vreg_tmp, vsum_scale_);
                addps(vreg_tmp_src0, vreg_tmp);
            }
        }

        if (postops_injector_) apply_postops(unroll, tail);

        for (int i = 0; i < unroll; i++) {
            const Vmm vreg_tmp_src0 = Vmm(i + 1);
            store(vreg_tmp_src0, i * offt_src0_, tail);
        }
    }

    jit_uni_binary_subkernel_t(const binary_pd_t *pd, bool tail_kernel)
        : jit_uni_binary_kernel_t(pd, tail_kernel) {}
};

#undef PARAM_OFF

template <data_type_t src_type>
binary_kernel_t *create_binary_kernel(const binary_pd_t *pd, bool tail_kernel) {
    if (mayiuse(avx512_core_bf16)) {
        using subkernel_t
                = jit_uni_binary_subkernel_t<avx512_core_bf16, src_type>;
        return new subkernel_t(pd, tail_kernel);
    } else if (mayiuse(avx512_core)) {
        using subkernel_t = jit_uni_binary_subkernel_t<avx512_core, src_type>;
        return new subkernel_t(pd, tail_kernel);
    } else if (mayiuse(avx2)) {
        using subkernel_t = jit_uni_binary_subkernel_t<avx2, src_type>;
        return new subkernel_t(pd, tail_kernel);
    } else {
        using subkernel_t = jit_uni_binary_subkernel_t<sse41, src_type>;
        return new subkernel_t(pd, tail_kernel);
    }
}

template <data_type_t src_type>
jit_uni_binary_t<src_type>::jit_uni_binary_t(const pd_t *apd)
    : primitive_t(apd) {}

template <data_type_t src_type>
jit_uni_binary_t<src_type>::~jit_uni_binary_t() = default;

template <data_type_t src_type>
status_t jit_uni_binary_t<src_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_,
            create_binary_kernel<src_type>(pd(), false /*tail_kernel*/)));

    const memory_desc_wrapper src0_d(pd_->src_md(0));
    const auto &simd_w = kernel_->simd_w();
    const auto oc = src0_d.ndims() >= 2 ? src0_d.dims()[1] : 1;

    if (op_t::bcast_c_blocked == get_bcast_per_c(src0_d) && oc % simd_w) {
        CHECK(safe_ptr_assign(kernel_tail_,
                create_binary_kernel<src_type>(pd(), true /*tail_kernel*/)));
        CHECK(kernel_tail_->create_kernel());
    }

    return kernel_->create_kernel();
}

template <data_type_t src_type>
status_t jit_uni_binary_t<src_type>::execute(const exec_ctx_t &ctx) const {
    using data_t = typename prec_traits<src_type>::type;

    const auto src0 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_0);
    const auto src1 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    const auto &post_ops = pd()->attr()->post_ops_;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(post_ops, ctx);

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));

    const auto ndims = src0_d.ndims();
    const auto &dims = src0_d.dims();
    const dim_t MB = dims[0];
    const dim_t C = ndims >= 2 ? dims[1] : 1;
    const dim_t D = ndims >= 5 ? dims[ndims - 3] : 1;
    const dim_t H = ndims >= 4 ? dims[ndims - 2] : 1;
    const dim_t W = ndims >= 3 ? dims[ndims - 1] : 1;
    const dim_t SP = D * H * W;
    const bool postops_per_oc_broadcast_exists
            = binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                    post_ops, src0_d);
    const bool no_broadcast = pd()->is_tensor_op();
    const bool with_postops = !post_ops.entry_.empty();
    const auto per_oc_bcast = get_bcast_per_c(src0_d);
    const auto &simd_w = kernel_->simd_w();
    const bool has_oc_tail = C % simd_w;
    const bool point_broadcast = src1_d.nelems() == 1;
    const bool point_broadcast_no_oc_tail = point_broadcast && !has_oc_tail;
    const bool blocked_oc_tail = per_oc_bcast == op_t::bcast_c_blocked
            && has_oc_tail && (with_postops || point_broadcast);
    const auto kernel = kernel_.get();
    const auto kernel_tail = kernel_tail_.get();

    if ((no_broadcast || point_broadcast_no_oc_tail)
            && !postops_per_oc_broadcast_exists && !blocked_oc_tail) {
        const dim_t nelems0 = src0_d.nelems(true);
        const dim_t nelems0_simd = nelems0 / simd_w;
        const dim_t nelems0_tail = nelems0 % simd_w;
        const bool has_tail = nelems0_tail > 0;

        // Compute strategy:
        // Compute number of vectors, divide it equally between all threads.
        // Last one will also handle a tail if present.
        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems0_simd + has_tail, nthr, ithr, start, end);
            if (start >= end) return;

            const bool ithr_does_tail
                    = has_tail && end == nelems0_simd + has_tail;
            const dim_t n_simd_to_do = (end - start - ithr_does_tail) * simd_w;
            const dim_t tail_to_do = ithr_does_tail * nelems0_tail;

            binary_kernel_t::call_params_t p;
            p.spat_offt_count = (n_simd_to_do + tail_to_do) * sizeof(data_t);
            p.src0 = src0 + start * simd_w;
            p.src1 = src1 + (point_broadcast ? 0 : (start * simd_w));
            p.dst = dst + start * simd_w;
            p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
            (*kernel)(&p);
        });
    } else {
        const auto &bcast_dims = pd()->broadcast_dims();
        const dim_t nelems_slice_src0
                = utils::array_product(src0_d.padded_dims() + 1, ndims - 1);
        const dim_t nelems_slice_src1 = no_broadcast
                ? nelems_slice_src0
                : ((bcast_dims[0] == 0) ? utils::array_product(
                           src1_d.padded_dims() + 1, ndims - 1)
                                        : 0);

        if (per_oc_bcast == op_t::bcast_c_blocked) {
            const dim_t C_blocks = std::ceil(src0_d.padded_dims()[1] / simd_w);
            // Compute strategy:
            // Each block is individual - parallel over MB and C_blocks safely.
            const std::function<void(binary_kernel_t::call_params_t *, dim_t)>
                    kernel_blocked_no_tail
                    = [&](binary_kernel_t::call_params_t *p, dim_t C_blk) {
                          (*kernel)(p);
                      };
            const std::function<void(binary_kernel_t::call_params_t *, dim_t)>
                    kernel_blocked_tail
                    = [&](binary_kernel_t::call_params_t *p, dim_t C_blk) {
                          if (C_blk == (C_blocks - 1))
                              (*kernel_tail)(p);
                          else
                              (*kernel)(p);
                      };
            const auto &kernel_blocked = blocked_oc_tail
                    ? kernel_blocked_tail
                    : kernel_blocked_no_tail;

            parallel_nd(MB, C_blocks, [&](dim_t mb, dim_t C_blk) {
                binary_kernel_t::call_params_t p;
                p.spat_offt_count = SP * simd_w * sizeof(data_t);
                const dim_t off = mb * nelems_slice_src0 + C_blk * SP * simd_w;
                p.dst = dst + off;
                p.src0 = src0 + off;
                const dim_t src1_off = point_broadcast
                        ? mb * nelems_slice_src1
                        : (no_broadcast ? off
                                        : mb * nelems_slice_src1
                                                + C_blk * simd_w);
                p.src1 = src1 + src1_off;
                p.oc_l_off = C_blk * simd_w;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                kernel_blocked(&p, C_blk);
            });
        } else if (per_oc_bcast == op_t::bcast_n_spatial_c) {
            // Compute strategy:
            // Each line of channels is individual, parallel over MB and spatial
            parallel_nd(MB, SP, [&](dim_t mb, dim_t sp) {
                binary_kernel_t::call_params_t p;
                p.spat_offt_count = C * sizeof(data_t);
                const auto off = mb * nelems_slice_src0 + sp * C;
                p.dst = dst + off;
                p.src0 = src0 + off;
                const auto src1_off
                        = no_broadcast ? off : mb * nelems_slice_src1;
                p.src1 = src1 + src1_off;
                p.oc_l_off = 0;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                (*kernel)(&p);
            });

        } else if (per_oc_bcast == op_t::bcast_n_c_spatial) {
            // Compute strategy:
            // Each line of spatial is individual, parallel over MB and C. Use a
            // kernel which broadcasts c_i value into a vector register.
            parallel_nd(MB, C, [&](dim_t mb, dim_t c) {
                binary_kernel_t::call_params_t p;
                p.spat_offt_count = SP * sizeof(data_t);
                const auto off = mb * nelems_slice_src0 + c * SP;
                p.dst = dst + off;
                p.src0 = src0 + off;
                const dim_t src1_off = point_broadcast
                        ? mb * nelems_slice_src1
                        : (no_broadcast ? off : mb * nelems_slice_src1 + c);
                p.src1 = src1 + src1_off;
                p.oc_l_off = c;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                (*kernel)(&p);
            });
        }
    }
    return status::success;
}

using namespace data_type;

template struct jit_uni_binary_t<f32>;
template struct jit_uni_binary_t<bf16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
