/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
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

#include "dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/aarch64/cpu_reducer.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace memory_tracking::names;

void reduce_balancer_t::balance() {
    using namespace nstl;
    using namespace utils;

    assert(nthr_ > 0 && job_size_ > 0 && njobs_ > 0 && reduction_size_ > 0);

    const int job_complexity = 1;

    const int min_njobs_per_group = max(1, njobs_ / nthr_);
    const int max_njobs_per_group
            = max(1, static_cast<int>(max_buffer_size_ / (nthr_ * job_size_)));

    /* initial guess */
    int ngroups = min(njobs_ / min_njobs_per_group, nthr_);
    int nthr_per_group
            = allow_nthr_in_group_ ? min(nthr_ / ngroups, reduction_size_) : 1;
    int njobs_per_group_ub = div_up(njobs_, ngroups);

    /* rough upper-bound estimation, will be fixed during brute force */
    size_t thread_complexity_ub = (size_t)njobs_ * job_size_ * reduction_size_;

    /* brute force parameters for the best balance... */
    for (int c_njobs_per_group = min_njobs_per_group;
            c_njobs_per_group < njobs_; ++c_njobs_per_group) {
        /* current assumption */
        int c_ngroups = min(njobs_ / c_njobs_per_group, nthr_);
        int c_nthr_per_group = allow_nthr_in_group_
                ? min(nthr_ / c_ngroups, reduction_size_)
                : 1;
        int c_njobs_per_group_ub = div_up(njobs_, c_ngroups);

        if (c_nthr_per_group > 1 && c_njobs_per_group_ub > max_njobs_per_group)
            continue;

        int c_thread_reduction_ub = div_up(reduction_size_, c_nthr_per_group);
        size_t c_group_size_ub = (size_t)job_size_ * c_njobs_per_group_ub;
        size_t c_thread_complexity_ub = c_group_size_ub
                * (job_complexity * c_thread_reduction_ub
                        + (c_nthr_per_group != 1));

        if (c_thread_complexity_ub < thread_complexity_ub) {
            ngroups = c_ngroups;
            nthr_per_group = c_nthr_per_group;
            njobs_per_group_ub = c_njobs_per_group_ub;
            thread_complexity_ub = c_thread_complexity_ub;
        }
    }

    assert(njobs_per_group_ub <= max_njobs_per_group || nthr_per_group == 1);
    assert(ngroups * nthr_per_group <= nthr_);
    assert((size_t)njobs_per_group_ub * job_size_ * nthr_ <= max_buffer_size_
            || nthr_per_group == 1); /* no reduction buffer overflow */
    assert(IMPLICATION(!allow_nthr_in_group_, nthr_per_group == 1));

    ngroups_ = ngroups;
    nthr_per_group_ = nthr_per_group;
    njobs_per_group_ub_ = njobs_per_group_ub;
}

/* reducer jit-ted driver */

using namespace Xbyak_aarch64;

template <impl::data_type_t data_type, cpu_isa_t isa>
struct reducer_2d_driver_t : public jit_generator {
    using data_t = typename prec_traits<data_type>::type;

    reducer_2d_driver_t(int n_src, size_t src_ld, size_t src_step,
            size_t dst_step, bool nullify_dst)
        : n_src_(n_src)
        , src_ld_(src_ld)
        , src_step_(src_step)
        , dst_step_(dst_step)
        , nullify_dst_(nullify_dst) {}
    virtual void operator()(
            data_t *dst, const data_t *srcs, size_t ny, size_t nx)
            = 0;

protected:
    int n_src_;
    size_t src_ld_, src_step_, dst_step_;
    bool nullify_dst_;
};

template <impl::data_type_t data_type, cpu_isa_t isa>
struct reducer_2d_driver_f_s_32_t : public reducer_2d_driver_t<data_type, isa> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(reducer_2d_driver_f_s_32_t)

    using data_t = typename prec_traits<data_type>::type;

    void operator()(
            data_t *dst, const data_t *srcs, size_t ny, size_t nx) override {
        jit_generator::operator()(dst, srcs, ny, nx);
    }

    /* cpu specific part */
    using Vmm = Xbyak_aarch64::ZRegS;

    const int vlen = cpu_isa_traits<isa>::vlen;
    const int typesize
            = sizeof(typename dnnl::impl::prec_traits<data_type>::type);
    XReg reg_dst = abi_param1;
    XReg reg_src = abi_param2;
    XReg reg_ny = abi_param3;
    XReg reg_nx = abi_param4;

    XReg reg_x = this->x19;
    XReg reg_src_id = this->x20;
    XReg reg_long_offt = this->x21;

    XReg reg_tmp_imm = this->x29;
    XReg reg_tmp_ptr = this->x30;

    PReg preg_one = this->p3;
    PReg preg_all = this->p4;

    reducer_2d_driver_f_s_32_t(int n_src, size_t src_ld, size_t src_step,
            size_t dst_step, bool nullify_dst)
        : reducer_2d_driver_t<data_type, isa>(
                n_src, src_ld, src_step, dst_step, nullify_dst) {}

    void uni_load(const Vmm &z1, const XReg &src, size_t off, int load_len) {
        auto src_ptr = (off == 0) ? src : reg_tmp_ptr;
        if (off != 0) this->add_imm(src_ptr, src, off, reg_tmp_imm);

        if (load_len == typesize)
            this->ld1w(z1, preg_one.s, ptr(src_ptr));
        else if (load_len == vlen)
            this->ld1w(z1, preg_all.s, ptr(src_ptr));
        else
            assert(!"unsupported");
    }

    void uni_store(const Vmm &z1, const XReg &dst, size_t off, int load_len) {
        auto dst_ptr = (off == 0) ? dst : reg_tmp_ptr;
        if (off != 0) this->add_imm(dst_ptr, dst, off, reg_tmp_imm);

        if (load_len == typesize)
            this->st1w(z1, preg_one.s, ptr(dst_ptr));
        else if (load_len == vlen)
            this->st1w(z1, preg_all.s, ptr(dst_ptr));
        else
            assert(!"unsupported");
    }

    void nullify_dst(int nloads, int load_len) {
        UNUSED(load_len);
        for (int i = 0; i < nloads; ++i)
            this->fmov(Vmm(i)); // Zero clear
        /* prefetches[dst] ? */
    }

    void load_dst(int nloads, int load_len) {
        for (int i = 0; i < nloads; ++i)
            uni_load(Vmm(i), reg_dst, i * load_len, load_len);
    }

    void store_dst(int nloads, int load_len) {
        for (int i = 0; i < nloads; ++i)
            uni_store(Vmm(i), reg_dst, i * load_len, load_len);
    }

    void accumulate(int nloads, int load_len, size_t base_off) {
        for (int i = 0; i < nloads; ++i) {
            size_t off = base_off + i * load_len;
            uni_load(Vmm(cpu_isa_traits<isa>::n_vregs - 1), reg_src, off,
                    load_len);
            if (data_type == data_type::f32)
                this->fadd(
                        Vmm(i), Vmm(i), Vmm(cpu_isa_traits<isa>::n_vregs - 1));
            else
                this->add(
                        Vmm(i), Vmm(i), Vmm(cpu_isa_traits<isa>::n_vregs - 1));
        }
    }

    void loop_x() {
        const int nloads[] = {cpu_isa_traits<isa>::n_vregs - 1, 1, 1};
        const int nbranches = sizeof(nloads) / sizeof(nloads[0]);

        const int load_len[nbranches] = {vlen, vlen, typesize};
        Label loop_x_label[nbranches + 1];

        switch (isa) {
            case sve_256: this->ptrue(preg_all.b, VL32); break;
            case sve_512: this->ptrue(preg_all.b, VL64); break;
            default: assert(!"Unsupported ISA"); break;
        }
        if (typesize == 4)
            this->ptrue(preg_one.s, VL1);
        else
            assert(!"Unsupported typesize");

        this->mov(reg_x, reg_nx);

        for (int id = 0; id < nbranches; ++id) {
            this->L(loop_x_label[id]);

            this->cmp(reg_x, nloads[id] * load_len[id]);
            this->b(LT, loop_x_label[id + 1]);

            if (this->nullify_dst_)
                nullify_dst(nloads[id], load_len[id]);
            else
                load_dst(nloads[id], load_len[id]);

            if (nloads[id] > 1) {
                Label loop_srcs;
                this->mov_imm(reg_src_id, this->n_src_);
                this->L(loop_srcs);

                accumulate(nloads[id], load_len[id], 0);
                this->add_imm(reg_src, reg_src, this->src_ld_ * typesize,
                        reg_tmp_imm);

                this->subs(reg_src_id, reg_src_id, 1); // dec
                this->b(NE, loop_srcs);

                size_t base_off
                        = (size_t)this->n_src_ * this->src_ld_ * typesize;
                this->sub_imm(reg_src, reg_src, base_off, reg_tmp_imm);
            } else {
                for (int src_id = 0; src_id < this->n_src_; ++src_id) {
                    const size_t base_off
                            = (size_t)src_id * this->src_ld_ * typesize;
                    accumulate(nloads[id], load_len[id], base_off);
                }
            }

            store_dst(nloads[id], load_len[id]);

            this->add_imm(
                    reg_src, reg_src, nloads[id] * load_len[id], reg_tmp_imm);
            this->add_imm(
                    reg_dst, reg_dst, nloads[id] * load_len[id], reg_tmp_imm);

            this->sub_imm(reg_x, reg_x, nloads[id] * load_len[id], reg_tmp_imm);

            this->b(loop_x_label[id]);
        }

        this->L(loop_x_label[nbranches]);

        /* restore address registers */
        this->sub(reg_src, reg_src, reg_nx);
        this->sub(reg_dst, reg_dst, reg_nx);
    }

    void generate() override {
        assert(isa == sve_512 || isa == sve_256);

        this->preamble();

        this->lsl(reg_nx, reg_nx, 2);

        Label ny_loop;
        this->L(ny_loop);

        loop_x();

        this->add_imm(
                reg_dst, reg_dst, this->dst_step_ * typesize, reg_tmp_imm);
        this->add_imm(
                reg_src, reg_src, this->src_step_ * typesize, reg_tmp_imm);

        this->subs(reg_ny, reg_ny, 1); //dec(reg_ny);
        this->b(NE, ny_loop); // jnz

        this->postamble();
    }
};

template <impl::data_type_t data_type, cpu_isa_t isa>
inline reducer_2d_driver_t<data_type, isa> *create_reduce_2d_drv(int n_src,
        size_t src_ld, size_t src_step, size_t dst_step, bool nullify_dst) {
    if (mayiuse(isa))
        return new reducer_2d_driver_f_s_32_t<data_type, isa>(
                n_src, src_ld, src_step, dst_step, nullify_dst);
    assert(!"unimplemented");
    return nullptr;
}

/* cpu_reducer_t */

template <impl::data_type_t data_type, cpu_isa_t isa>
void cpu_reducer_t<data_type, isa>::conf_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    if (balancer_.nthr_per_group_ == 1) return;

    const size_t space_size = balancer_.ngroups_
            * (balancer_.nthr_per_group_ - 1)
            * cpu_reducer_t<data_type, isa>::space_per_thread(balancer_);
    scratchpad.book<data_t>(key_reducer_space, space_size, PAGE_4K);
    scratchpad.book<simple_barrier::ctx_t>(
            key_reducer_space_bctx, balancer_.ngroups_);
}

template <impl::data_type_t data_type, cpu_isa_t isa>
cpu_reducer_t<data_type, isa>::cpu_reducer_t(const conf_t &conf)
    : conf_(conf), drv_(nullptr) {
    if (balancer().nthr_per_group_ == 1) return;

    drv_ = create_reduce_2d_drv<data_type, isa>(balancer().nthr_per_group_ - 1,
            space_per_thread(balancer()), 0, 0, false);
}

template <impl::data_type_t data_type, cpu_isa_t isa>
cpu_reducer_t<data_type, isa>::~cpu_reducer_t() {
    delete drv_;
}

template <impl::data_type_t data_type, cpu_isa_t isa>
status_t cpu_reducer_t<data_type, isa>::create_kernel() {
    return (drv_) ? drv_->create_kernel() : status::success;
}

template <impl::data_type_t data_type, cpu_isa_t isa>
typename cpu_reducer_t<data_type, isa>::data_t *
cpu_reducer_t<data_type, isa>::get_local_ptr(int ithr, data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    const int id_in_grp = balancer().id_in_group(ithr);

    /* threads 0 from each group writes directly to the destination */
    if (id_in_grp == 0)
        return dst + balancer().ithr_job_off(ithr) * balancer().job_size_;

    const int grp_id = balancer().group_id(ithr);
    const int offset_factor
            = grp_id * (balancer().nthr_per_group_ - 1) + (id_in_grp - 1);

    auto space = scratchpad.template get<data_t>(key_reducer_space);
    return space + offset_factor * space_per_thread(balancer());
}

template <impl::data_type_t data_type, cpu_isa_t isa>
void cpu_reducer_t<data_type, isa>::reduce_nolock(int ithr, data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    bool redundant_reduction
            = balancer().nthr_per_group_ == 1 || balancer().idle(ithr);
    if (redundant_reduction) return;

#ifdef SIMPLE_IMPL
    if (balancer().id_in_group(ithr) != 0)
        return; /* only threads 0 do the reduction */

    const int njobs_in_grp = balancer().ithr_njobs(ithr);
    data_t *d = get_local_ptr(ithr, dst, scratchpad);
    for (int id_in_grp = 1; id_in_grp < balancer().nthr_per_group_;
            ++id_in_grp) {
        const data_t *space = get_local_ptr(ithr + id_in_grp, dst, scratchpad);
        for (size_t i = 0; i < (size_t)njobs_in_grp * balancer().job_size_; ++i)
            d[i] += space[i];
    }
#else
    using namespace utils;

    const int id_in_grp = balancer().id_in_group(ithr);
    const int njobs_in_grp = balancer().ithr_njobs(ithr);
    const size_t cl = 64 / sizeof(data_t);

    const size_t reduction_size = njobs_in_grp * balancer().job_size_;
    size_t start {0}, end {0};
    balance211(div_up(reduction_size, cl), balancer().nthr_per_group_,
            id_in_grp, start, end);

    if (start == end) return;

    data_t *d = get_local_ptr(ithr - id_in_grp, dst, scratchpad) + start * cl;
    const data_t *space
            = get_local_ptr(ithr - id_in_grp + 1, dst, scratchpad) + start * cl;
    const size_t len = nstl::min(end * cl, reduction_size) - start * cl;

    (*drv_)(d, space, 1, len);
#endif
}

template struct cpu_reducer_t<data_type::f32, sve_512>;
template struct cpu_reducer_t<data_type::f32, sve_256>;
template struct cpu_reducer_t<data_type::s32, sve_512>;
template struct cpu_reducer_t<data_type::s32, sve_256>;

/* cpu_reducer_2d_t */

template <impl::data_type_t data_type, cpu_isa_t isa>
void cpu_reducer_2d_t<data_type, isa>::conf_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    if (balancer_.nthr_per_group_ == 1) return;

    const size_t space_size = balancer_.ngroups_ * balancer_.nthr_per_group_
            * cpu_reducer_2d_t<data_type, isa>::space_per_thread(balancer_);
    scratchpad.book<data_t>(key_reducer_space, space_size);
    scratchpad.book<simple_barrier::ctx_t>(
            key_reducer_space_bctx, balancer_.ngroups_);
}

template <impl::data_type_t data_type, cpu_isa_t isa>
cpu_reducer_2d_t<data_type, isa>::cpu_reducer_2d_t(const conf_t &conf)
    : conf_(conf), drv_(nullptr) {
    if (balancer().nthr_per_group_ == 1) return;

    drv_ = create_reduce_2d_drv<data_type, isa>(balancer().nthr_per_group_,
            space_per_thread(balancer()), conf_.job_size_x_, conf_.dst_x_,
            true);
}

template <impl::data_type_t data_type, cpu_isa_t isa>
cpu_reducer_2d_t<data_type, isa>::~cpu_reducer_2d_t() {
    delete drv_;
}

template <impl::data_type_t data_type, cpu_isa_t isa>
status_t cpu_reducer_2d_t<data_type, isa>::create_kernel() {
    return (drv_) ? drv_->create_kernel() : status::success;
}

template <impl::data_type_t data_type, cpu_isa_t isa>
typename cpu_reducer_2d_t<data_type, isa>::data_t *
cpu_reducer_2d_t<data_type, isa>::get_local_ptr(
        int ithr, const memory_tracking::grantor_t &scratchpad) const {
    const int id_in_grp = balancer().id_in_group(ithr);
    const int grp_id = balancer().group_id(ithr);
    const int offset_factor = grp_id * balancer().nthr_per_group_ + id_in_grp;
    auto space = scratchpad.template get<data_t>(key_reducer_space);
    return space + offset_factor * space_per_thread(balancer());
}

template <impl::data_type_t data_type, cpu_isa_t isa>
int cpu_reducer_2d_t<data_type, isa>::choose_x_blocking(
        int nx, int ny, int nthr_per_grp) const {
    // find x_blocking for better balance reducing work between threads
    assert(conf_.x_block_ > 0 && nx > conf_.x_block_
            && nx % conf_.x_block_ == 0);
    int x_blocking = nx / conf_.x_block_;
    int min_x_blocking
            = utils::div_up(x_blocking, nstl::max(1, nthr_per_grp / ny));
    while (true) {
        if (x_blocking % 2 == 0 && x_blocking >= min_x_blocking * 2)
            x_blocking /= 2;
        else if (x_blocking % 3 == 0 && x_blocking >= min_x_blocking * 3)
            x_blocking /= 3;
        else
            break;
    }
    if (x_blocking >= min_x_blocking * 4) x_blocking = 1;
    x_blocking *= conf_.x_block_;
    return x_blocking;
}

template <impl::data_type_t data_type, cpu_isa_t isa>
void cpu_reducer_2d_t<data_type, isa>::reduce_block(const data_t *space_base,
        data_t *dst, int job, int start_y, int start_x, int ny_start,
        int nx_start, int ny_step, int nx_step) const {
    data_t *d = dst + (start_y + ny_start) * conf_.dst_x_ + start_x + nx_start;
    const data_t *space = space_base + (size_t)job * balancer().job_size_
            + (size_t)ny_start * conf_.job_size_x_ + nx_start;
#ifdef SIMPLE_IMPL
    for (int idg = 0; idg < balancer().nthr_per_group_; ++idg) {
        const data_t *w = &space[idg * space_per_thread(balancer())];
        for (int y = 0; y < ny_step; ++y)
            for (int x = 0; x < nx_step; ++x) {
                d[y * conf_.dst_x_ + x]
                        = (idg == 0 ? 0 : d[y * conf_.dst_x_ + x])
                        + w[y * conf_.job_size_x_ + x];
            }
    }
#else
    (*drv_)(d, space, ny_step, nx_step);
#endif
}

template <impl::data_type_t data_type, cpu_isa_t isa>
void cpu_reducer_2d_t<data_type, isa>::reduce_nolock(int ithr, data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    bool redundant_reduction
            = balancer().nthr_per_group_ == 1 || balancer().idle(ithr);
    if (redundant_reduction) return;

    const int id_in_grp = balancer().id_in_group(ithr);
    const int njobs_in_grp = balancer().ithr_njobs(ithr);
    const int njobs_x = utils::div_up(conf_.dst_x_, conf_.job_size_x_);
    const int global_job_start = balancer().ithr_job_off(ithr);

    const data_t *space_base = get_local_ptr(ithr - id_in_grp, scratchpad);

    const int pr_grps = nstl::min(njobs_in_grp, balancer().nthr_per_group_);
    const int pr_nthr_per_grp = balancer().nthr_per_group_ / pr_grps;

    if (id_in_grp >= pr_grps * pr_nthr_per_grp) return; /* idle */

    const int pr_my_grp = id_in_grp / pr_nthr_per_grp;
    const int pr_my_id = id_in_grp % pr_nthr_per_grp;

    int pr_job_start {0}, pr_job_end {0};
    balance211(njobs_in_grp, pr_grps, pr_my_grp, pr_job_start, pr_job_end);

    for (int j = pr_job_start; j < pr_job_end; ++j) {
        const int global_job = global_job_start + j;
        const int j_y = global_job / njobs_x;
        const int j_x = global_job % njobs_x;
        const int start_y = j_y * conf_.job_size_y_;
        const int start_x = j_x * conf_.job_size_x_;
        const int ny = nstl::min(conf_.dst_y_ - start_y, conf_.job_size_y_);
        const int nx = nstl::min(conf_.dst_x_ - start_x, conf_.job_size_x_);
        int x_blocking = choose_x_blocking(nx, ny, pr_nthr_per_grp);

        int nxy_start {0}, nxy_end {0};
        balance211(ny * nx / x_blocking, pr_nthr_per_grp, pr_my_id, nxy_start,
                nxy_end);
        if (nxy_start == nxy_end) continue;
        nxy_start *= x_blocking;
        nxy_end *= x_blocking;

        int nxy = nxy_start;
        if (nxy % nx != 0) {
            int nx_step = nstl::min(nx - nxy % nx, nxy_end - nxy);
            reduce_block(space_base, dst, j, start_y, start_x, nxy / nx,
                    nxy % nx, 1, nx_step);
            nxy += nx_step;
        }
        if ((nxy_end - nxy) > nx) {
            int ny_step = (nxy_end - nxy) / nx;
            reduce_block(space_base, dst, j, start_y, start_x, nxy / nx,
                    nxy % nx, ny_step, nx);
            nxy += nx * ny_step;
        }
        if ((nxy_end - nxy) > 0) {
            reduce_block(space_base, dst, j, start_y, start_x, nxy / nx,
                    nxy % nx, 1, nxy_end - nxy);
        }
    }
}

template struct cpu_reducer_2d_t<data_type::f32, sve_512>;
template struct cpu_reducer_2d_t<data_type::f32, sve_256>;
template struct cpu_reducer_2d_t<data_type::s32, sve_512>;
template struct cpu_reducer_2d_t<data_type::s32, sve_256>;

/* accumulator section */

template <impl::data_type_t data_type, cpu_isa_t isa>
cpu_accumulator_1d_t<data_type, isa>::cpu_accumulator_1d_t() : drv_(nullptr) {
    drv_ = create_reduce_2d_drv<data_type, isa>(1, 0, 0, 0, false);
}

template <impl::data_type_t data_type, cpu_isa_t isa>
cpu_accumulator_1d_t<data_type, isa>::~cpu_accumulator_1d_t() {
    delete drv_;
}

template <impl::data_type_t data_type, cpu_isa_t isa>
status_t cpu_accumulator_1d_t<data_type, isa>::create_kernel() {
    return drv_->create_kernel();
}

template <impl::data_type_t data_type, cpu_isa_t isa>
void cpu_accumulator_1d_t<data_type, isa>::accumulate(
        data_t *dst, const data_t *src, size_t size) {
    (*drv_)(dst, src, 1, size);
}

template struct cpu_accumulator_1d_t<data_type::f32, sve_512>;
template struct cpu_accumulator_1d_t<data_type::f32, sve_256>;
template struct cpu_accumulator_1d_t<data_type::s32, sve_512>;
template struct cpu_accumulator_1d_t<data_type::s32, sve_256>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
