#ifndef __XBYAK_UTILS_FOR_MKL_DNN
#define __XBYAK_UTILS_FOR_MKL_DNN

#define XBYAK_VERSION 0x5000

#if XBYAK_VERSION >= 0x5000
    #define ZWORD   zword
    #define ZWORD_b zword_b
    #define YWORD   yword
    #define YWORD_b yzword_b
#else
    #define ZWORD zmmword
    #define YWORD ymmword
#endif

typedef struct {
    int ic, oc;
    uint32_t mb;
    uint32_t ih, iw, oh, ow;
    uint32_t ihp, iwp, ohp, owp;
    int l_pad, t_pad;
    int kh, kw;
    int stride_h, stride_w;
    uint32_t nb_ic, ic_block;
    uint32_t nb_oc, oc_block;
    uint32_t nb_ic_blocking, nb_oc_blocking; // blocking of nb_ic and nb_ic
    uint32_t ur_h, ur_w;
    uint32_t ur_w_tail;
    uint32_t ngroups;
    int SIMD_W;
} jit_convolution_param_t;

typedef struct __attribute__ ((__packed__)) jit_convolution_kernel_s {
    float *src;
    float *dst;
    float *filt;
    float *src_prf;
    float *dst_prf;
    float *filt_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t kw_padding;
}  jit_convolution_kernel_t;

#ifdef XBYAK64
namespace Xbyak { namespace util {
static const Operand::Code reg_to_preserve[] = {
    Operand::RBX, Operand::RSP, Operand::RBP,
    Operand::R12, Operand::R13, Operand::R14, Operand::R15,
#ifdef _WIN
    Operand::RDI, Operand::RSI,
#endif
};
#ifdef _WIN
static const Reg64 cdecl_param1(Operand::RCX), cdecl_param2(Operand::RDX), cdecl_param3(Operand::R8), cdecl_param4(Operand::R9);
#else
static const Reg64 cdecl_param1(Operand::RDI), cdecl_param2(Operand::RSI), cdecl_param3(Operand::RDX), cdecl_param4(Operand::RCX);
#endif
}}
#endif

#endif
