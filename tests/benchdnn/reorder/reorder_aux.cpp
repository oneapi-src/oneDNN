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

#include "mkldnn_debug.hpp"
#include "reorder/reorder.hpp"

#define DPRINT(...) do { \
    int l = snprintf(buffer, rem_len, __VA_ARGS__); \
    buffer += l; rem_len -= l; \
} while(0)

namespace reorder {

int dims2str(int ndims, const mkldnn_dims_t dims, char **_buffer, int rem_len)
{
    char *buffer = *_buffer;

    for (int d = 0; d < ndims; ++d)
        DPRINT("%d,", dims[d]);

    *_buffer = buffer;
    return rem_len;
}

void prb2str(const prb_t *p, const res_t *res, char *buffer) {
    int rem_len = max_prb_len;

    const auto *r = p->reorder;

    rem_len = dims2str(r->ndims, r->dims, &buffer, rem_len);

    DPRINT("fmts=%s->%s;", fmt2str(r->fmt_in), fmt2str(r->fmt_out));
    DPRINT("dts=%s->%s;", dt2str(p->conf_in.dt), dt2str(p->conf_out.dt));
    attr2str(&p->attr, buffer);
}

void perf_report(const prb_t *p, const res_t *r, const char *pstr) {
    const auto &t = r->timer;
    const int max_len = 400;
    char buf[max_len], *buffer = buf;
    int rem_len = max_len - 1;

    DPRINT("perf,");
    DPRINT("%s,", pstr);
    DPRINT("min_ms=%g,", t.ms(benchdnn_timer_t::min));
    DPRINT("max_ms=%g", t.ms(benchdnn_timer_t::max));

    print(0, "%s\n", buf);
}

}
