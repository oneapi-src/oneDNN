/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef _CONV_HPP
#define _CONV_HPP

#include <stdint.h>
#include <limits.h>
#include <assert.h>

#include "mkldnn.h"

#include "common.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

namespace conv {

enum { SRC = 0, WEI = 1, BIA = 2, DST = 3, ACC = 4, DAT_TOTAL };
const char *inp_type2str(int what);

enum alg_t { DIRECT, WINO };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);

enum merge_t { NONE, RELU, };
merge_t str2merge(const char *str);
const char *merge2str(merge_t merge);

struct desc_t {
    int g, mb;
    int ic, ih, iw;
    int oc, oh, ow;
    int kh, kw;
    int sh, sw;
    int ph, pw;
    int dh, dw;

    const char *name;
};
const size_t max_desc_len = 196;
int str2desc(desc_t *desc, const char *str);
void desc2str(const desc_t *d, char *buffer, bool canonical = false);

/** configuration structure, that controls initial data filling + error check
 *
 * dt defines convolution precision
 *
 * for each type (SRC, WEI, BIA, and DST) the values are filled as follows:
 * if (rand() > f_sparsity) then:
 *     v <-- f_base // it is guaranteed each kernel window
 *                  // has at least one non-zero element
 * else:
 *     v <-- f_min + rand() * f_step % (f_max - f_min)
 *
 *
 * on final check the resulting values should be in [min .. max] range, the
 * relative difference should not exceed eps
 */
typedef struct dt_conf_t {
    mkldnn_data_type_t dt;
    int min, max; /* representative */
    int f_min, f_max; /* fill range */
    int f_base; /* fill base, use 0 */
    int f_step; /* fill step, use 1 */
    double f_sparsity; /* amount of non-zeros, default 0.25 */
    double eps; /* acceptable error */
} _dt_conf_t[DAT_TOTAL];

extern const _dt_conf_t conf_f32;
extern const _dt_conf_t conf_f32_full;
extern const _dt_conf_t conf_f32_wino;
extern const _dt_conf_t conf_s16s32;
extern const _dt_conf_t conf_s8s32;

const dt_conf_t *str2cfg(const char *str);
const char *cfg2str(const dt_conf_t *cfg);

struct prb_t: public desc_t {
    prb_t(const desc_t &desc, dir_t dir, const dt_conf_t *cfg, alg_t alg,
            merge_t merge, int mb = 0)
        : desc_t(desc), dir(dir), cfg(cfg), alg(alg), merge(merge) {
        if (mb) this->mb = mb;
    }

    dir_t dir;
    const dt_conf_t *cfg;
    alg_t alg;
    merge_t merge;
};
const size_t max_prb_len = 392;
void prb2str(const prb_t *p, char *buffer, bool canonical = false);

/* some extra control parameters which shouldn't be placed in prb_t */
extern const char *skip_impl; /* NULL or "" means do not skip anything */
extern bool allow_unimpl; /* true means do not treat unimplemented as error */

inline size_t src_off_f(const prb_t *p, int mb, int g, int ic, int ih, int iw)
{
    return ((mb * p->ic + g * p->ic/p->g + ic) * p->ih + ih) * p->iw + iw;
}

inline void inv_src_off_f(const prb_t *p, int off, int &mb, int &g, int &ic,
        int &ih, int &iw) {
    iw = off % p->iw; off /= p->iw;
    ih = off % p->ih; off /= p->ih;
    ic = off % (p->ic / p->g); off /= (p->ic / p->g);
    g = off % p->g; off /= p->g;
    mb = off % p->mb; off /= p->mb;
    assert(off == 0);
}

inline size_t wei_off_f(const prb_t *p, int g, int oc, int ic, int kh, int kw)
{
    return (((g * p->oc / p->g + oc) * p->ic / p->g + ic) * p->kh + kh) * p->kw
        + kw;
}

inline void inv_wei_off_f(const prb_t *p, int off, int &g, int &oc, int &ic,
        int &kh, int &kw) {
    kw = off % p->kw; off /= p->kw;
    kh = off % p->kh; off /= p->kh;
    ic = off % (p->ic / p->g); off /= (p->ic / p->g);
    oc = off % (p->oc / p->g); off /= (p->oc / p->g);
    g = off % p->g; off /= p->g;
    assert(off == 0);
}

inline size_t bia_off_f(const prb_t *p, int g, int oc) {
    return g * p->oc / p->g + oc;
}

inline void inv_bia_off_f(const prb_t *p, int off, int &g, int &oc) {
    oc = off % (p->oc / p->g); off /= (p->oc / p->g);
    g = off % p->g; off /= p->g;
    assert(off == 0);
}

inline size_t dst_off_f(const prb_t *p, int mb, int g, int oc, int oh, int ow)
{
    return ((mb * p->oc + g * p->oc/p->g + oc) * p->oh + oh) * p->ow + ow;
}

inline void inv_dst_off_f(const prb_t *p, int off, int &mb, int &g, int &oc,
        int &oh, int &ow) {
    ow = off % p->ow; off /= p->ow;
    oh = off % p->oh; off /= p->oh;
    oc = off % (p->oc / p->g); off /= (p->oc / p->g);
    g = off % p->g; off /= p->g;
    mb = off % p->mb; off /= p->mb;
    assert(off == 0);
}

void compute_ref_fwd(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &dst_m);
void compute_ref_bwd_d(const prb_t *p, dnn_mem_t &diff_src_m, dnn_mem_t &wei_m,
        dnn_mem_t &diff_dst_m);
void compute_ref_bwd_w(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &diff_wei_m,
        dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m);

bool maybe_skip(const char *impl_str);
int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv);

}

#endif
