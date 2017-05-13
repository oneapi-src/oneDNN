/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef CPU_JIT_AVX2_GENERATOR_HPP
#define CPU_JIT_AVX2_GENERATOR_HPP

#include <type_traits>

#define XBYAK64
#define XBYAK_NO_OP_NAMES
/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace mkldnn {
namespace impl {
namespace cpu {

typedef enum {
    isa_any,
    sse42,
    avx2,
    avx512_common,
    avx512_core,
    avx512_mic,
    avx512_mic_4ops,
} cpu_isa_t;

template <cpu_isa_t> struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <> struct cpu_isa_traits<sse42> {
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 16;
};
template <> struct cpu_isa_traits<avx2> {
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 32;
    static constexpr int n_vregs = 16;
};
template <> struct cpu_isa_traits<avx512_common> {
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
};
template <> struct cpu_isa_traits<avx512_core>:
    public cpu_isa_traits<avx512_common> {};

template <> struct cpu_isa_traits<avx512_mic>:
    public cpu_isa_traits<avx512_common> {};

template <> struct cpu_isa_traits<avx512_mic_4ops>:
    public cpu_isa_traits<avx512_common> {};

// TODO: move this to jit_generator class?
namespace {

// TODO: move this somewhere else? Although this is only used by jit kernels
// (Roma)
static inline int float2int(float x) {
    union {
        float vfloat;
        int vint;
    } cvt;
    cvt.vfloat = x;
    return cvt.vint;
}

// TODO: A GPR class that hides ABI details from the JIT kernels and allows
// numbering registers from 0 to 14 (x86_64) / 6 (x32) (gpr0, gpr1, ...) and
// stack register (sr).
//
// This will allow using syntax like this:
//
// param = gpr0;
// reg_input = gpr0;
// reg_output =  gpr1;
// ...
//
// #ifndef XBYAK64
// mov(param, ptr[sr])
// #endif
//
// (Roma)

#ifdef XBYAK64
constexpr Xbyak::Operand::Code abi_save_regs[] = {
    Xbyak::Operand::RBX, Xbyak::Operand::RSP, Xbyak::Operand::RBP,
    Xbyak::Operand::R12, Xbyak::Operand::R13, Xbyak::Operand::R14,
    Xbyak::Operand::R15,
#ifdef _WIN
    Xbyak::Operand::RDI, Xbyak::Operand::RSI,
#endif
};
constexpr size_t num_abi_save_regs
    = sizeof(abi_save_regs) / sizeof(abi_save_regs[0]);

#ifdef _WIN
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RCX),
             abi_param2(Xbyak::Operand::RDX),
             abi_param3(Xbyak::Operand::R8),
             abi_param4(Xbyak::Operand::R9),
             abi_not_param1(Xbyak::Operand::RDI);
#else
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RDI),
             abi_param2(Xbyak::Operand::RSI),
             abi_param3(Xbyak::Operand::RDX),
             abi_param4(Xbyak::Operand::RCX),
             abi_not_param1(Xbyak::Operand::RCX);
#endif
#endif

static inline bool mayiuse(const cpu_isa_t cpu_isa) {
    using namespace Xbyak::util;
    static Cpu cpu;

    switch (cpu_isa) {
    case sse42:
        return cpu.has(Cpu::tSSE42);
    case avx2:
        return cpu.has(Cpu::tAVX2);
    case avx512_common:
        return cpu.has(Cpu::tAVX512F);
    case avx512_core:
        return true
            && cpu.has(Cpu::tAVX512F)
            && cpu.has(Cpu::tAVX512BW)
            && cpu.has(Cpu::tAVX512VL)
            && cpu.has(Cpu::tAVX512DQ);
    case avx512_mic:
        return true
            && cpu.has(Cpu::tAVX512F)
            && cpu.has(Cpu::tAVX512CD)
            && cpu.has(Cpu::tAVX512ER)
            && cpu.has(Cpu::tAVX512PF);
    case avx512_mic_4ops:
        return true
            && mayiuse(avx512_mic)
            && cpu.has(Cpu::tAVX512_4FMAPS)
            && cpu.has(Cpu::tAVX512_4VNNIW);
    case isa_any:
        return true;
    }
    return false;
}

}

// TODO (Roma): move all_same to a more appropriate location

template <typename T, typename U, typename... Us>
struct all_same : std::false_type {};

template <typename T, typename... Us>
struct all_same<T, T, Us...> : all_same<T, Us...> { };

template <typename T>
struct all_same<T, T> : std::true_type {};

template <size_t len = 64>
class jit_tagged_label_base {
public:
    enum { maxlen = len };
    template <size_t n, typename... Tags,
             typename = std::enable_if<all_same<char, Tags...>::value>>
    jit_tagged_label_base(const char (&base)[n], Tags... tags) {
        // XXX: This code is ugly but useful
        constexpr size_t ntags = sizeof...(tags);
        static_assert(n + ntags < maxlen, "resulting label may be too long");
        // paste tags first in case base has unexpected null chars
        paste_tags(tags...);
        for (size_t i = 0; i < n; i++)
            label_name_[ntags + i] = base[i];
        // don't assume that the base string is 0-terminated
        label_name_[ntags + n] = '\0';
    }
    operator const char*() const { return label_name_; }
    const char *c_str() const { return label_name_; }
private:
    char label_name_[maxlen];
    void paste_tags() { }
    template <typename... Tags>
    void paste_tags(char tag, Tags... tags) {
        label_name_[sizeof...(tags)] = tag;
        paste_tags(tags...);
    }
};

typedef jit_tagged_label_base<> jit_tagged_label;

class jit_generator : public Xbyak::CodeGenerator
{
protected:
    Xbyak::Reg64 param1 = abi_param1;
    const int EVEX_max_8b_offt = 0x200;
    const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

    void preamble() {
        for (size_t i = 0; i < num_abi_save_regs; ++i)
            push(Xbyak::Reg64(abi_save_regs[i]));
        if (mayiuse(avx512_common)) {
            mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
        }
    }
    void mic_prefetcht0(Xbyak::Address a) {
        if (mayiuse(avx512_mic))
            prefetcht0(a);
    }

    void mic_prefetcht1(Xbyak::Address a) {
        if (mayiuse(avx512_mic))
            prefetcht1(a);
    }

    void mic_prefetcht2(Xbyak::Address a) {
        if (mayiuse(avx512_mic))
            prefetcht2(a);
    }

    void postamble() {
        for (size_t i = 0; i < num_abi_save_regs; ++i)
            pop(Xbyak::Reg64(abi_save_regs[num_abi_save_regs - 1 - i]));
        ret();
    }

    Xbyak::Address EVEX_compress_addr(Xbyak::Reg64 base,
            int offt, bool bcast = false)
    {
        using Xbyak::Zmm;
        using Xbyak::Reg64;
        using Xbyak::Address;
        using Xbyak::RegExp;

        int scale = 0;

        if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
            offt = offt - 2 * EVEX_max_8b_offt;
            scale = 1;
        } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
            offt = offt - 4 * EVEX_max_8b_offt;
            scale = 2;
        }

        auto re = RegExp() + base + offt;
        if (scale)
            re = re + reg_EVEX_max_8b_offt * scale;

        if (bcast)
            return zword_b [re];
        else
            return zword [re];
    }

    // Provide overrides for custom jit_tagged_label and C strings rather than
    // implement a conversion of jit_tagge_label to std::string to avoid
    // additional C++ runtime dependency

    template <size_t len>
    void L(const jit_tagged_label_base<len> &label) {
        Xbyak::CodeGenerator::L(label.c_str());
    }

    void L(const char *label) { Xbyak::CodeGenerator::L(label); }
    void L(const Xbyak::Label& label) { Xbyak::CodeGenerator::L(label); }

    void uni_vpxor(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                   const Xbyak::Operand& op) {
        assert(x1.getIdx() == x2.getIdx());
        pxor(x2, op);
    }
    void uni_vpxor(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                   const Xbyak::Operand& op) {
        vpxor(x1, x2, op);
    }
    void uni_vpxor(const Xbyak::Zmm& x1, const Xbyak::Zmm& x2,
                   const Xbyak::Operand& op) {
        vpxord(x1, x2, op);
    }

    void uni_vmovdqu(const Xbyak::Address& addr, const Xbyak::Xmm& x) {
        movdqu(addr, x);
    }
    void uni_vmovdqu(const Xbyak::Address& addr, const Xbyak::Ymm& x) {
        vmovdqu(addr, x);
    }
    void uni_vmovdqu(const Xbyak::Address& addr, const Xbyak::Zmm& x) {
        vmovdqu32(addr, x);
    }

    void uni_vmovdqu(const Xbyak::Xmm& x, const Xbyak::Address& addr) {
        movdqu(x, addr);
    }
    void uni_vmovdqu(const Xbyak::Ymm& x, const Xbyak::Address& addr) {
        vmovdqu(x, addr);
    }
    void uni_vmovdqu(const Xbyak::Zmm& x, const Xbyak::Address& addr) {
        vmovdqu32(x, addr);
    }

    void uni_vmovups(const Xbyak::Address& addr, const Xbyak::Xmm& x) {
        movups(addr, x);
    }
    void uni_vmovups(const Xbyak::Address& addr, const Xbyak::Ymm& x) {
        vmovups(addr, x);
    }

    void uni_vmovups(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
        movups(x, op);
    }
    void uni_vmovups(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
        vmovups(x, op);
    }

    void uni_vmovntps(const Xbyak::Address& addr, const Xbyak::Xmm& x) {
        movntps(addr, x);
    }
    void uni_vmovntps(const Xbyak::Address& addr, const Xbyak::Ymm& x) {
        vmovntps(addr, x);
    }

    void uni_vbroadcastss(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
        movss(x, op);
        shufps(x, x, 0x0);
    }
    void uni_vbroadcastss(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
        vbroadcastss(x, op);
    }

    void uni_vpbroadcastd(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
        movsd(x, op);
        pshufd(x, x, 0x0);
    }
    void uni_vpbroadcastd(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
        vpbroadcastd(x, op);
    }

    void uni_vdivps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        divps(x, op2);
    }
    void uni_vdivps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vdivps(x, op1, op2);
    }

    void uni_vaddps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        addps(x, op2);
    }
    void uni_vaddps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vaddps(x, op1, op2);
    }

    void uni_vsubps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        subps(x, op2);
    }
    void uni_vsubps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vsubps(x, op1, op2);
    }

    void uni_vmulps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        mulps(x, op2);
    }
    void uni_vmulps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vmulps(x, op1, op2);
    }

    void uni_vfmadd213ps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                         const Xbyak::Operand& op) {
        mulps(x1, x2);
        addps(x1, op);
    }
    void uni_vfmadd213ps(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                         const Xbyak::Operand& op) {
        vfmadd213ps(x1, x2, op);
    }

    void uni_vfmadd231ps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                         const Xbyak::Operand& op) {
        mulps(x2, op);
        addps(x1, x2);
    }
    void uni_vfmadd231ps(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                         const Xbyak::Operand& op) {
        vfmadd231ps(x1, x2, op);
    }

    void uni_vsqrtps(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
        sqrtps(x, op);
    }
    void uni_vsqrtps(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
        vsqrtps(x, op);
    }

    void uni_vpaddd(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                    const Xbyak::Operand& op) {
        assert(x1.getIdx() == x2.getIdx());
        paddd(x2, op);
    }
    void uni_vpaddd(const Xbyak::Ymm& x1, const Xbyak::Xmm& x2,
                    const Xbyak::Operand& op) {
        vpaddd(x1, x2, op);
    }

public:
    jit_generator(
        void *code_ptr = nullptr,
        size_t code_size = 128 * 1024
        ) : Xbyak::CodeGenerator(code_size, code_ptr)
    {
    }
};

}
}
}

#endif
