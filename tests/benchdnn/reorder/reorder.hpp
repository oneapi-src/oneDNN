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

#include "mkldnn.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

namespace reorder {

struct dt_conf_t {
    mkldnn_data_type_t dt;
    int min;
    int range;
};

struct reorder_conf_t {
    int ndims;
    mkldnn_dims_t dims;
    mkldnn_memory_format_t fmt_in, fmt_out;
};

struct q10n_conf_t {
    const dt_conf_t &conf_in;
    const dt_conf_t &conf_out;
    /* TODO: add attrs */
    attr_t::round_mode_t irmode;
    attr_t::scale_t::policy_t policy;
    float scale;
};

struct prb_t {
    prb_t(const reorder_conf_t *r, const q10n_conf_t *q)
        : reorder(r), conf_in(q->conf_in), conf_out(q->conf_out) {
            attr.irmode = q->irmode;
            attr.oscale.policy = q->policy;
            attr.oscale.scale = q->scale;
        }

    const reorder_conf_t *reorder;
    const dt_conf_t &conf_in;
    const dt_conf_t &conf_out;
    attr_t attr;
};

const size_t max_prb_len = 392;
void prb2str(const prb_t *p, const res_t *res, char *buffer);
void perf_report(const prb_t *p, const res_t *r, const char *pstr);

inline size_t data_off_f(const prb_t *p, int mb, int ic, int ih, int iw) {
    const auto &dims = p->reorder->dims;
    return ((mb * dims[1] + ic) * dims[2] + ih) * dims[3] + iw;
}

int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv);

}

#endif
