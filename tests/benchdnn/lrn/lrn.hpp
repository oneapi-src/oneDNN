/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#ifndef LRN_HPP
#define LRN_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include <iostream>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace lrn {

enum alg_t { ACROSS, WITHIN };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct desc_t {
    int64_t mb, ic, id, ih, iw;
    int64_t ls;
    float alpha, beta, k;
    const char *name;
    int ndims;
};
int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

struct prb_t : public desc_t {
    prb_t(const desc_t &desc, int64_t mb, dir_t dir, dnnl_data_type_t dt,
            dnnl_format_tag_t tag, alg_t alg)
        : desc_t(desc), dir(dir), dt(dt), tag(tag), alg(alg) {
        if (mb) this->mb = mb;
    }
    ~prb_t() {}

    dir_t dir;
    dnnl_data_type_t dt;
    dnnl_format_tag_t tag;
    alg_t alg;

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(prb_t);
};
std::ostream &operator<<(std::ostream &s, const prb_t &p);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *p, const res_t *r, const char *prb_str) {
        p_ = p;
        base_report(r, prb_str);
    }

    virtual void dump_alg(std::ostream &s) const override {
        s << alg2str(p_->alg);
    }

    virtual void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    virtual void dump_desc_csv(std::ostream &s) const override {
        s << p_->mb << ',' << p_->ic << ',' << p_->id << ',' << p_->ih << ','
          << p_->iw << ',' << p_->ls << ',' << p_->alpha << ',' << p_->beta
          << ',' << p_->k;
    }

    virtual const char *name() const override { return p_->name; }
    virtual const dir_t *dir() const override { return &p_->dir; }
    virtual const dnnl_data_type_t *dt() const override { return &p_->dt; }
    virtual const dnnl_format_tag_t *tag() const override { return &p_->tag; }

private:
    const prb_t *p_ = NULL;
};

/* some extra control parameters which shouldn't be placed in prb_t */
extern const char *skip_impl; /* NULL or "" means do not skip anything */

inline int compute_n_summands(const prb_t *p) {
    if (p->alg == ACROSS) {
        return p->ls;
    } else if (p->alg == WITHIN) {
        int n_summands = 1;
        for (int64_t d = p->ndims - 2; d > 0; --d)
            n_summands *= p->ls;
        return n_summands;
    } else {
        assert(!"unknown algorithm");
        return 1;
    }
}

inline size_t data_off(const prb_t *p, int64_t mb, int64_t c, int64_t d,
        int64_t h, int64_t w) {
    return (((mb * p->ic + c) * p->id + d) * p->ih + h) * p->iw + w;
}

inline void inv_data_off(const prb_t *p, size_t off, int64_t &mb, int64_t &c,
        int64_t &d, int64_t &h, int64_t &w) {
    w = off % p->iw;
    off /= p->iw;
    h = off % p->ih;
    off /= p->ih;
    d = off % p->id;
    off /= p->id;
    c = off % p->ic;
    off /= p->ic;
    mb = off % p->mb;
    off /= p->mb;
    assert(off == 0);
}

void compute_ref_fwd(const prb_t *p, const dnn_mem_t &src, dnn_mem_t &dst);
void compute_ref_bwd(const prb_t *p, const dnn_mem_t &src,
        const dnn_mem_t &d_dst, dnn_mem_t &d_src);

int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv);

} // namespace lrn

#endif
