// Referenced from
// https://github.com/opencv/opencv/modules/core/include/opencv2/core/hal/intrin_rvv.hpp

#ifndef INTRIN_RVV_H
#define INTRIN_RVV_H

#if defined(__riscv_v_intrinsic) &&  __riscv_v_intrinsic>10999
#include "intrin_rvv_010_compat_non-policy.h"
#include "intrin_rvv_010_compat_overloaded-non-policy.h"
#endif

#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic>11999
#include "intrin_rvv_011_compat.h"
#endif

#endif