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

#ifndef CPU_ISA_HPP
#define CPU_ISA_HPP

namespace mkldnn {
namespace impl {
namespace cpu {

/* [ejk] Consider moving to mkldnn.hpp or even mkldnn.h, for purposes
 * of benchmarking an I/O -- the name() strings return __FUNCTION__ that
 * is long and ugly, and for a client/test prog to transform it to something
 * pretty, these enums come in handy.   Yes, I know you do not technically
 * need these exposed to have a "complete" API.
 *    Ex. benchdnn : 'shorten(impl_str)' */
typedef enum {
    isa_any,
    sse42,
    avx2,
    avx512_common,
    avx512_core,
    avx512_mic,
    avx512_mic_4ops,
} cpu_isa_t;

}
}
}
#endif // CPU_ISA_HPP
