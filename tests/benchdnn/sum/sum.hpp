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

#ifndef _SUM_HPP
#define _SUM_HPP

#include <iostream>

#include "mkldnn.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"
#include "perf_report.hpp"

namespace sum {

struct prb_t {
    prb_t(dims_t dims, std::vector<mkldnn_data_type_t> idt,
            mkldnn_data_type_t odt, std::vector<mkldnn_format_tag_t> itag,
            mkldnn_format_tag_t otag, std::vector<float> scales)
        : dims(dims), idt(idt), odt(odt), itag(itag), otag(otag), scales(scales)
    {
        if ((int)scales.size() == 1) // broadcast single scale for each input
            for (int i_input = 1; i_input < n_inputs(); i_input++)
                scales[i_input] = scales[0];
    }
    ~prb_t() {}

    dims_t dims;
    std::vector<mkldnn_data_type_t> idt;
    mkldnn_data_type_t odt;
    std::vector<mkldnn_format_tag_t> itag;
    mkldnn_format_tag_t otag;
    std::vector<float> scales;

    int n_inputs() const { return (int)idt.size(); }
};
std::ostream &operator<<(std::ostream &s, const prb_t &p);

struct perf_report_t: public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *p, const res_t *r, const char *prb_str) {
        p_ = p;
        base_report(r, prb_str);
    }

    virtual void dump_desc_csv(std::ostream &s) const override
    { s << p_->dims; }
    virtual const std::vector<mkldnn_data_type_t> *idt() const override
    { return &p_->idt; }
    virtual const mkldnn_data_type_t *odt() const override { return &p_->odt; }
    virtual const std::vector<mkldnn_format_tag_t> *itag() const override
    { return &p_->itag; }
    virtual const mkldnn_format_tag_t *otag() const override
    { return &p_->otag; }

private:
    const prb_t *p_ = NULL;
};

void compute_ref(const prb_t *p, const std::vector<dnn_mem_t> &src,
        dnn_mem_t &dst);

int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv);

}

#endif
