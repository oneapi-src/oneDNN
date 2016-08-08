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
	int ic, oc, mb;
	int ih, iw, oh, ow;
	int ihp, iwp, ohp, owp;
	int l_pad, t_pad;
	int kh, kw;
	int stride_h, stride_w;
	int nb_ic, ic_block;
	int nb_oc, oc_block;
	int nb_ic_blocking, nb_oc_blocking; // blocking of nb_ic and nb_ic 
	int ur_h, ur_w;
	int ur_w_tail;
	int ngroups;
	int SIMD_W;
} hnk_conv_param_t;	

typedef struct hnk_conv_kernel_s {
    float *src, *dst, *filt;
    float *src_prf, *dst_prf, *filt_prf;
    size_t kh_padding, kh_padding_prf;
    size_t kw_padding;
} hnk_conv_kernel_t;

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
