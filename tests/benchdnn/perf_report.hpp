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

#include <iostream>
#include <sstream>

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

    void handle_option(std::ostream &s, const char *&option, const res_t *r,
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
            option += strlen(opt) + 1; \
            return; \
        }

        HANDLE("alg", dump_alg(s));
        HANDLE("cfg", dump_cfg(s));
        HANDLE("DESC", dump_desc_csv(s));
        HANDLE("flags", dump_flags(s));

        HANDLE("attr", if (attr() && !attr()->is_def()) s << *attr());
        HANDLE("axis", if (axis()) s << *axis());
        HANDLE("dir", if (dir()) s << dir2str(*dir()));
        HANDLE("dt", if (dt()) s << dt2str(*dt()));
        HANDLE("group", if (group()) s << *group());
        HANDLE("idt", if (idt()) s << dt2str(*idt()));
        HANDLE("itag", if (itag()) s << tag2str(*itag()));
        HANDLE("name", if (name()) s << name());
        HANDLE("odt", if (odt()) s << dt2str(*odt()));
        HANDLE("otag", if (otag()) s << tag2str(*otag()));
        HANDLE("prop", if (prop()) s << prop2str(*prop()));
        HANDLE("tag", if (tag()) s << tag2str(*tag()));

        HANDLE("bw", s << ops() / t.ms(mode) / unit * 1e3);
        HANDLE("flops", s << ops() / t.ms(mode) / unit * 1e3);
        HANDLE("clocks", s << t.ticks(mode) / unit);
        HANDLE("desc", s << prb_str);
        HANDLE("engine", s << engine_kind2str(engine_tgt_kind));
        HANDLE("freq", s << t.ticks(mode) / t.ms(mode) / unit * 1e3);
        HANDLE("ops", s << ops() / unit);
        HANDLE("time", s << t.ms(mode) / unit);

#       undef HANDLE

        SAFE_V(FAIL);
    }

    void base_report(const res_t *r, const char *prb_str) const {
        dump_perf_footer();

        std::stringstream ss;

        const char *pt = pt_;
        char c;
        while ((c = *pt++) != '\0') {
            if (c != '%') { ss << c; continue; }
            handle_option(ss, pt, r, prb_str);
        }

        std::string str = ss.str();
        print(0, "%s\n", str.c_str());
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
    virtual void dump_alg(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_cfg(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_desc_csv(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_flags(std::ostream &) const { SAFE_V(FAIL); }

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
};

#endif
