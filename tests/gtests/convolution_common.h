#define EXPAND_SIZES(mb, ng, ic, ih, iw, oc, oh, ow, kh, kw, ph, pw, sh, sw) \
    { mb, ng, ic, ih, iw, oc, oh, ow, kh, kw, ph, pw, sh, sw }
#define EXPAND_FORMATS(src, weights, bias, dst) \
    { memory::format::src, memory::format::weights, \
    memory::format::bias, memory::format::dst }

#define ENGINE engine::kind::cpu
#define ALGORITHM convolution_direct

#ifdef DIRECTION_FORWARD
#define FMT_WEIGHTS_BLOCKED OIhw8i8o
#define FMT_WEIGHTS_BLOCKED_G gOIhw8i8o
#elif defined DIRECTION_BACKWARD_DATA
#define FMT_WEIGHTS_BLOCKED OIhw8o8i
#define FMT_WEIGHTS_BLOCKED_G gOIhw8o8i
#elif defined DIRECTION_BACKWARD_WEIGHTS
#define FMT_WEIGHTS_BLOCKED OIhw8o8i
#define FMT_WEIGHTS_BLOCKED_G gOIhw8o8i
#endif

#define FMT_BIAS x
#define FMT_NO_BIAS undef
#define FMT_DATA_BLOCKED nChw8c

#define PARAMS(src, weights, bias, dst, ...) \
    test_convolution_params_t { ENGINE, ALGORITHM, \
    EXPAND_FORMATS(src, weights, bias, dst), EXPAND_SIZES(__VA_ARGS__) }

#include "convolution_simple_small.h"
#include "convolution_alexnet.h"
#include "convolution_googlenet_v1.h"
#include "convolution_googlenet_v2.h"
#include "convolution_cifar10.h"
