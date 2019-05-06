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
| Syntax    | Primitives       | Description
|:----------|:-----------------|:-----------
| %alg%     | Conv             | Primitive algorithm
| %attr%    | Bnorm, Conv, IP  | Primitive attributes
| %axis%    | Shuffle, Softmax | Shuffle and softmax axis
| %@bw%     | All with ops     | Bytes per second (modifier extended)
| %cfg%     | Conv, IP, RNN    | Config, describes data types and filling rules
| %@clocks% | All              | Time in clocks (modifier extended)
| %desc%    | All              | Problem descriptor (dimensions and other options included)
| %DESC%    | All              | CSV-style problem descriptor (mostly dimensions)
| %dir%     | All, except RNN, | Primitive direction
|           |   Reorder        |
| %dt%      | Bnorm, Shuffle,  | Data type (precision)
|           |   Softmax        |
| %engine%  | All              | Engine kind
| %flags%   | Bnorm            | Batch normalization flags
| %@flops%  | All with ops     | Ops per second (modifier extended)
| %@freq%   | All              | Effective cpu frequency computed as clocks[@] / time[@]
| %group%   | Shuffle          | Shuffle group
| %name%    | All with desc_t  | Problem name
| %@ops%    | All with ops     | Number of ops required (padding is not taken into account)
| %prop%    | RNN              | RNN properties
| %tag%     | Bnorm, Shuffle,  | Data format tag (physical memory layout)
|           |   Softmax        |
| %@time%   | All              | Time in ms (modifier extended)

Modifiers supported:
| Name  | description
|:----  |:-----------
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

    benchdnn_timer_t::mode_t modifier2mode(char c) const {
        if (c == '-') return benchdnn_timer_t::min;
        if (c == '0') return benchdnn_timer_t::avg;
        if (c == '+') return benchdnn_timer_t::max;
        return benchdnn_timer_t::min;
    };

    double modifier2unit(char c) const {
        if (c == 'K') return 1e3;
        if (c == 'M') return 1e6;
        if (c == 'G') return 1e9;
        return 1e0;
    };

    // TODO: replace this ugliness with std::string;
    // That big due to dump prb_str of max_prb_len size.
    static constexpr size_t max_dump_len = max_prb_len;

    void dprint(char *buf, const char *content) const {
        snprintf(buf, max_dump_len, "%s", content);
    }

    void dprint(char *buf, double content) const {
        snprintf(buf, max_dump_len, "%g", content);
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

        if (!strncmp("alg", option, 3))
            dump_algorithm(buf);
        else if (!strncmp("attr", option, 4))
            dump_attributes(buf);
        else if (!strncmp("axis", option, 4))
            dump_axis(buf);
        else if (!strncmp("bw", option, 2))
            dprint(buf, ops() / t.ms(mode) / unit * 1e3);
        else if (!strncmp("cfg", option, 3))
            dump_config(buf);
        else if (!strncmp("clocks", option, 6))
            dprint(buf, t.ticks(mode) / unit);
        else if (!strncmp("desc", option, 4))
            dprint(buf, prb_str);
        else if (!strncmp("DESC", option, 4))
            dump_descriptor_csv(buf);
        else if (!strncmp("dir", option, 3))
            dump_direction(buf);
        else if (!strncmp("dt", option, 2))
            dump_data_type(buf);
        else if (!strncmp("engine", option, 6))
            dprint(buf, engine_kind2str(engine_tgt_kind));
        else if (!strncmp("flags", option, 5))
            dump_flags(buf);
        else if (!strncmp("flops", option, 5))
            dprint(buf, ops() / t.ms(mode) / unit * 1e3);
        else if (!strncmp("freq", option, 4))
            dprint(buf, t.ticks(mode) / t.ms(mode) / unit * 1e3);
        else if (!strncmp("group", option, 5))
            dump_group_size(buf);
        else if (!strncmp("name", option, 4))
            dump_descriptor_name(buf);
        else if (!strncmp("ops", option, 3))
            dprint(buf, ops() / unit);
        else if (!strncmp("prop", option, 4))
            dump_properties(buf);
        else if (!strncmp("tag", option, 3))
            dump_tag(buf);
        else if (!strncmp("time", option, 4))
            dprint(buf, t.ms(mode) / unit);
        else
            SAFE_V(FAIL);
    }

    virtual void dump_algorithm(char *buf) const { err_msg(); }
    virtual void dump_attributes(char *buf) const { err_msg(); }
    virtual void dump_axis(char *buf) const { err_msg(); }
    virtual void dump_config(char *buf) const { err_msg(); }
    virtual void dump_data_type(char *buf) const { err_msg(); }
    virtual void dump_descriptor_csv(char *buf) const { err_msg(); }
    virtual void dump_descriptor_name(char *buf) const { err_msg(); }
    virtual void dump_direction(char *buf) const { err_msg(); }
    virtual void dump_flags(char *buf) const { err_msg(); }
    virtual void dump_group_size(char *buf) const { err_msg(); }
    virtual void dump_properties(char *buf) const { err_msg(); }
    virtual void dump_tag(char *buf) const { err_msg(); }

    virtual double ops() const { return 0.; }

    void dump_perf_footer() const {
        static bool footer_printed = false;
        if (!footer_printed) {
            // TODO: improve footer to be more human-readable, not plain dump
            print(0, "Output template: %s\n", pt_);
            footer_printed = true;
        }
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

private:
    const char *pt_;

    static void err_msg() {
        printf("%s is not supported in base_perf_report_t\n",
                __PRETTY_FUNCTION__);
    };
};

#endif
