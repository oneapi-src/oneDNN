/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef _CONCAT_HPP
#define _CONCAT_HPP

#include <iostream>

#include "mkldnn.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"
#include "perf_report.hpp"

namespace concat {

struct prb_t {
    prb_t(const std::vector<dims_t> &sdims, mkldnn_data_type_t sdt,
            mkldnn_data_type_t ddt,
            const std::vector<mkldnn_format_tag_t> &stag,
            mkldnn_format_tag_t dtag, int axis)
        : sdims(sdims), sdt(sdt), ddt(ddt), stag(stag), dtag(dtag), axis(axis) {
        generate_ddims();
    }
    ~prb_t() {}

    std::vector<dims_t> sdims;
    dims_t ddims;
    mkldnn_data_type_t sdt, ddt;
    std::vector<mkldnn_format_tag_t> stag;
    mkldnn_format_tag_t dtag;
    int axis;

    int n_inputs() const { return (int)sdims.size(); }

    int64_t axis_size() const {
        int64_t as = 0;
        for (int i = 0; i < n_inputs(); ++i)
            as += sdims[i].at(axis);
        return as;
    }

    void generate_ddims() {
        const dims_t &sdims0 = sdims[0];
        const int ndims = (int)sdims0.size();

        ddims.resize(ndims);

        for (int i = 0; i < ndims; ++i)
            ddims[i] = sdims0[i];
        ddims[axis] = axis_size();
    }
};
std::ostream &operator<<(std::ostream &s, const std::vector<dims_t> sdims);
std::ostream &operator<<(std::ostream &s, const prb_t &p);

struct perf_report_t: public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *p, const res_t *r, const char *prb_str) {
        p_ = p;
        sdt_ = {p_->sdt};
        base_report(r, prb_str);
    }

    virtual void dump_desc_csv(std::ostream &s) const override {
        s << p_->sdims;
    }

    virtual const int *axis() const override { return &p_->axis; }
    virtual const std::vector<mkldnn_data_type_t> *sdt() const override
    { return &sdt_; }
    virtual const mkldnn_data_type_t *ddt() const override { return &p_->ddt; }
    virtual const std::vector<mkldnn_format_tag_t> *stag() const override
    { return &p_->stag; }
    virtual const mkldnn_format_tag_t *dtag() const override
    { return &p_->dtag; }

private:
    const prb_t *p_ = NULL;
    std::vector<mkldnn_data_type_t> sdt_;
};

void compute_ref(const prb_t *p, const std::vector<dnn_mem_t> &src,
        dnn_mem_t &dst);

int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv);

}

#endif
