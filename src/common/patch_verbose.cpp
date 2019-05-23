/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <stdlib.h>
#ifndef _WIN32
#include <sys/time.h>
#endif

#include "mkldnn.h"
#include "mkldnn_version.h"
#include "c_types_map.hpp"
#include "verbose.hpp"
#include "cpu/cpu_isa_traits.hpp"

#include "resize_bilinear_pd.hpp"

/* MKL-DNN CPU ISA info */
#define ISA_ANY "No instruction set specific optimizations"
#define SSE41 "Intel(R) Streaming SIMD Extensions 4.1 (Intel(R) SSE4.1)"
#define AVX "Intel(R) Advanced Vector Extensions (Intel(R) AVX)"
#define AVX2 "Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)"
#define AVX512_COMMON "Intel(R) Advanced Vector Extensions 512 (Intel(R) " \
                      "AVX-512)"
#define AVX512_CORE "Intel(R) Advanced Vector Extensions 512 (Intel(R) " \
                    "AVX-512) with AVX512BW, AVX512VL, and AVX512DQ extensions"
#define AVX512_CORE_VNNI "Intel(R) AVX512-Deep Learning Boost (Intel(R) " \
                         "AVX512-DL Boost)"
#define AVX512_MIC "Intel(R) Advanced Vector Extensions 512 (Intel(R) " \
                   "AVX-512) with AVX512CD, AVX512ER, and AVX512PF extensions"
#define AVX512_MIC_4OPS "Intel(R) Advanced Vector Extensions 512 (Intel(R) " \
                   "AVX-512) with AVX512_4FMAPS and AVX512_4VNNIW extensions"

namespace mkldnn {
namespace impl {

//static verbose_t verbose;
//static bool initialized;
//static bool version_printed = false;

/* init_info section */
namespace {
#if !defined(DISABLE_VERBOSE)
#define MKLDNN_VERBOSE_DAT_LEN 256
#define MKLDNN_VERBOSE_AUX_LEN 384
#define MKLDNN_VERBOSE_PRB_LEN 384

#define DECL_DAT_AUX_PRB_STRS() \
    int dat_written = 0, aux_written = 0, prb_written = 0; \
    MAYBE_UNUSED((dat_written * aux_written * prb_written)); \
    char dat_str[MKLDNN_VERBOSE_DAT_LEN] = {'\0'}; MAYBE_UNUSED(dat_str); \
    char aux_str[MKLDNN_VERBOSE_AUX_LEN] = {'\0'}; MAYBE_UNUSED(aux_str); \
    char prb_str[MKLDNN_VERBOSE_PRB_LEN] = {'\0'}; MAYBE_UNUSED(prb_str)

#define DFMT "%" PRId64

void clear_buf(char *buf, int &written) {
    /* TODO: do it better */
    buf[0] = '#';
    buf[1] = '\0';
    written = 1;
}

#define DPRINT(buf, buf_len, written, ...) do { \
    int l = snprintf(buf + written, buf_len - written, __VA_ARGS__); \
    if (l < 0 || written + l > buf_len) { \
        clear_buf(buf, written); \
    } else { \
        written += l; \
    } \
} while(0)

template <typename pd_t> static void init_info_bilinear(pd_t *s, char *buffer) {
    DECL_DAT_AUX_PRB_STRS();

    if (1) { // src
        auto md = s->is_fwd() ? s->src_md() : s->diff_src_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, "src_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // dst
        auto md = s->is_fwd() ? s->dst_md() : s->diff_dst_md();
        DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " dst_");
        int l = mkldnn_md2fmt_str(dat_str + dat_written,
                MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
        if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
    }
    if (1) { // ws
        auto md = s->workspace_md();
        if (md) {
            DPRINT(dat_str, MKLDNN_VERBOSE_DAT_LEN, dat_written, " ws_");
            int l = mkldnn_md2fmt_str(dat_str + dat_written,
                    MKLDNN_VERBOSE_DAT_LEN - dat_written, md);
            if (l >= 0) dat_written += l; else clear_buf(dat_str, dat_written);
        }
    }

    DPRINT(aux_str, MKLDNN_VERBOSE_AUX_LEN, aux_written, "alg");
}

#undef DPRINT

#else // !defined(DISABLE_VERBOSE)

#define DEFINE_STUB(name) \
    template <typename pd_t> \
    static void CONCAT2(init_info_, name)(pd_t *s, char *buffer) \
    { UNUSED(s); UNUSED(buffer); }

DEFINE_STUB(bilinear);
#undef DEFINE_STUB

#endif // !defined(DISABLE_VERBOSE)
}

void init_info(resize_bilinear_pd_t *s, char *b)
{ init_info_bilinear(s, b); }

}
}

