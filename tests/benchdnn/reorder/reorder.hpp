/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef _REORDER_HPP
#define _REORDER_HPP

#include <vector>

#include "mkldnn.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

namespace reorder {

using dims_t = std::vector<int64_t>;

enum alg_t { ALG_REF, ALG_BOOT };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);

enum flag_t { FLAG_NONE, FLAG_CONV_S8S8, FLAG_GCONV_S8S8 };
flag_t str2flag(const char *str);
const char *flag2str(flag_t flag);

struct dt_conf_s {
    mkldnn_data_type_t dt;
    int min;
    int range;
};
typedef const dt_conf_s *dt_conf_t;

struct reorder_conf_t {
    dims_t dims;
    mkldnn_format_tag_t tag_in, tag_out;
};

struct q10n_conf_t {
    dt_conf_t conf_in;
    dt_conf_t conf_out;
    /* TODO: add attrs */
    attr_t::scale_t::policy_t policy;
    float scale;
};

struct prb_t {
    prb_t(const reorder_conf_t &r, const dt_conf_t &conf_in,
            const dt_conf_t &conf_out, const attr_t &attr, alg_t alg,
            flag_t oflag, float scale = 0.f)
        : reorder(r)
        , conf_in(conf_in)
        , conf_out(conf_out)
        , attr(attr)
        , alg(alg)
        , oflag(oflag) {
        if (scale != 0.f) this->attr.oscale.scale = scale;
    }

    const reorder_conf_t reorder;
    dt_conf_t conf_in;
    dt_conf_t conf_out;
    attr_t attr;
    alg_t alg;
    flag_t oflag;
};

extern const char *perf_template; /* performance output template */

extern const dt_conf_t conf_f32;
extern const dt_conf_t conf_s8;
extern const dt_conf_t conf_u8;
extern const dt_conf_t conf_s32;
dt_conf_t dt2cfg(mkldnn_data_type_t dt);
mkldnn_data_type_t cfg2dt(dt_conf_t cfg);

const size_t max_dims_len = 20;
const size_t max_prb_len = 392;
dims_t str2dims(const char *str);
void dims2str(const dims_t &dims, char *buffer);
void prb2str(const prb_t *p, const res_t *res, char *buffer);
void perf_report(const prb_t *p, const res_t *r, const char *pstr);

int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv, bool main_bench = true);

}

#endif
