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

#ifndef _PERF_REPORT_HPP
#define _PERF_REPORT_HPP

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "mkldnn.h"
#include "mkldnn_memory.hpp"

#if 0
**benchdnn** supports custom performance report. Template is passed via
command line and consists of terminal and nonterminal symbols. Nonterminal
symbols are printed as is. Description of terminal symbols is given below.
There is also a notion of modifiers (marked as @) that change meaning of
terminal symbols, e.g. sign '-' means minimum of (in terms of time). See
table of modifiers below.

> **caution:** threads have to be pinned in order to get consistent frequency

Options supported:
| Syntax        | Primitives               | Description
| :--           | :--                      | :--
| %alg%         | Conv                     | Primitive algorithm
| %attr%        | Bnorm, Conv, IP          | Primitive attributes
| %axis%        | Shuffle, Softmax         | Shuffle and softmax axis
| %@bw%         | All with ops             | Bytes per second (modifier extended)
| %cfg%         | Conv, IP, RNN            | Config, describes data types and filling rules
| %@clocks%     | All                      | Time in clocks (modifier extended)
| %desc%        | All                      | Problem descriptor (dimensions and other options included)
| %DESC%        | All                      | CSV-style problem descriptor (mostly dimensions)
| %dir%         | All, except RNN, Reorder | Primitive direction
| %dt%          | Bnorm, Shuffle, Softmax  | Data type (precision)
| %idt%/%odt%   | Reorder                  | Input/Output data types (precision)
| %engine%      | All                      | Engine kind
| %flags%       | Bnorm                    | Batch normalization flags
| %@flops%      | All with ops             | Ops per second (modifier extended)
| %@freq%       | All                      | Effective cpu frequency computed as clocks[@] / time[@]
| %group%       | Shuffle                  | Shuffle group
| %name%        | All with desc_t          | Problem name
| %@ops%        | All with ops             | Number of ops required (padding is not taken into account)
| %prop%        | RNN                      | RNN properties
| %tag%         | Bnorm, Shuffle, Softmax  | Data format tag (physical memory layout)
| %itag%/%otag% | Reorder                  | Input/Output data format tag (physical memory layout)
| %@time%       | All                      | Time in ms (modifier extended)

Modifiers supported:
| Name  | Description
| :--   | :--
| Time: |
| -     | min (time) -- default
| 0     | avg (time)
| +     | max (time)
|       |
| Unit: |      (1e0) -- default
| K     | Kilo (1e3)
| M     | Mega (1e6)
| G     | Giga (1e9)

Each primitive has its own descriptor type with options supported. Dimensions
description can be found internally at each primitive hpp-file.
#endif

struct base_perf_report_t {
    base_perf_report_t(const char *perf_template) : pt_(perf_template) {}
    virtual ~base_perf_report_t() {}

    // TODO: replace this ugliness with std::string;
    // That big due to dump prb_str of max_prb_len size.
    static constexpr size_t max_dump_len = max_prb_len;

    void dprint(char *buf, const char *str) const {
        snprintf(buf, max_dump_len, "%s", str);
    }

    void dprint(char *buf, double val) const {
        snprintf(buf, max_dump_len, "%g", val);
    }

    void handle_option(char *buf, const char *option, const res_t *r,
            const char *prb_str) const {
        const auto &t = r->timer;
        benchdnn_timer_t::mode_t mode = benchdnn_timer_t::min; (void)mode;
        double unit = 1e0;
        char c = *option;

        if (c == '-' || c == '0' || c == '+') {
            mode = modifier2mode(c);
            c = *(++option);
        }

        if (c == 'K' || c == 'M' || c == 'G') {
            unit = modifier2unit(c);
            c = *(++option);
        }

#       define HANDLE(opt, ...) \
        if (!strncmp(opt "%", option, strlen(opt) + 1)) { \
            __VA_ARGS__; \
            return; \
        }

        HANDLE("alg", dump_alg(buf));
        HANDLE("cfg", dump_cfg(buf));
        HANDLE("DESC", dump_desc_csv(buf));
        HANDLE("flags", dump_flags(buf));

        HANDLE("attr", if (attr() && !attr()->is_def()) attr2str(attr(), buf));
        HANDLE("axis", if (axis()) dprint(buf, *axis()));
        HANDLE("dir", if (dir()) dprint(buf, dir2str(*dir())));
        HANDLE("dt", if (dt()) dprint(buf, dt2str(*dt())));
        HANDLE("group", if (group()) dprint(buf, *group()));
        HANDLE("idt", if (idt()) dprint(buf, dt2str(*idt())));
        HANDLE("itag", if (itag()) dprint(buf, tag2str(*itag())));
        HANDLE("name", if (name()) dprint(buf, name()));
        HANDLE("odt", if (odt()) dprint(buf, dt2str(*odt())));
        HANDLE("otag", if (otag()) dprint(buf, tag2str(*otag())));
        HANDLE("prop", if (prop()) dprint(buf, prop2str(*prop())));
        HANDLE("tag", if (tag()) dprint(buf, tag2str(*tag())));

        HANDLE("bw", dprint(buf, ops() / t.ms(mode) / unit * 1e3));
        HANDLE("flops", dprint(buf, ops() / t.ms(mode) / unit * 1e3));
        HANDLE("clocks", dprint(buf, t.ticks(mode) / unit));
        HANDLE("desc", dprint(buf, prb_str));
        HANDLE("engine", dprint(buf, engine_kind2str(engine_tgt_kind)));
        HANDLE("freq", dprint(buf, t.ticks(mode) / t.ms(mode) / unit * 1e3));
        HANDLE("ops", dprint(buf, ops() / unit));
        HANDLE("time", dprint(buf, t.ms(mode) / unit));

#       undef HANDLE

        SAFE_V(FAIL);
    }

    void base_report(const res_t *r, const char *prb_str) const {
        const int max_buf_len = 2 * max_dump_len; // max num of parsed options
        int rem_buf_len = max_buf_len - 1;
        char buffer[max_buf_len], *buf = buffer;

        dump_perf_footer();

        const char *pt = pt_;
        char c;
        while ((c = *pt++) != '\0') {
            if (c != '%') { *buf++ = c; rem_buf_len--; continue; }

            char opt_dump[max_dump_len] = "", *dump = opt_dump;

            handle_option(dump, pt, r, prb_str);

            int l = snprintf(buf, rem_buf_len, "%s", opt_dump);
            buf += l; rem_buf_len -= l;

            if ((pt = strchr(pt, '%')) == NULL) // check for KW
                break;
            pt++;
        }

        *buf = '\0';
        assert(rem_buf_len >= 0);
        print(0, "%s\n", buffer);
    };

    /* truly common types */
    virtual double ops() const { return 0.; }
    virtual const attr_t *attr() const { return nullptr; }
    virtual const int *axis() const { return nullptr; }
    virtual const char *name() const { return nullptr; }
    virtual const int64_t *group() const { return nullptr; }
    virtual const dir_t *dir() const { return nullptr; }
    virtual const mkldnn_data_type_t *dt() const { return nullptr; }
    virtual const mkldnn_data_type_t *idt() const { return nullptr; }
    virtual const mkldnn_data_type_t *odt() const { return nullptr; }
    virtual const mkldnn_format_tag_t *tag() const { return nullptr; }
    virtual const mkldnn_format_tag_t *itag() const { return nullptr; }
    virtual const mkldnn_format_tag_t *otag() const { return nullptr; }
    virtual const mkldnn_prop_kind_t *prop() const { return nullptr; }

    /* primitive-specific properties (but with common interface) */
    virtual void dump_alg(char *buf) const { err_msg(); }
    virtual void dump_cfg(char *buf) const { err_msg(); }
    virtual void dump_desc_csv(char *buf) const { err_msg(); }
    virtual void dump_flags(char *buf) const { err_msg(); }

private:
    const char *pt_;

    void dump_perf_footer() const {
        static bool footer_printed = false;
        if (!footer_printed) {
            // TODO: improve footer to be more human-readable, not plain dump
            print(0, "Output template: %s\n", pt_);
            footer_printed = true;
        }
    }

    static benchdnn_timer_t::mode_t modifier2mode(char c) {
        if (c == '-') return benchdnn_timer_t::min;
        if (c == '0') return benchdnn_timer_t::avg;
        if (c == '+') return benchdnn_timer_t::max;
        return benchdnn_timer_t::min;
    }

    static double modifier2unit(char c) {
        if (c == 'K') return 1e3;
        if (c == 'M') return 1e6;
        if (c == 'G') return 1e9;
        return 1e0;
    }

    static void err_msg() {
        printf("%s is not supported in base_perf_report_t\n",
                __PRETTY_FUNCTION__);
    };
};

#endif
