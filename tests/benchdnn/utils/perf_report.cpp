/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "dnn_types.hpp"
#include "dnnl_common.hpp"

#include "utils/perf_report.hpp"

void base_perf_report_t::report(res_t *res, const char *prb_str) const {
    dump_perf_footer();

    std::stringstream ss;

    const char *pt = pt_;
    char c;
    while ((c = *pt++) != '\0') {
        if (c != '%') {
            ss << c;
            continue;
        }
        handle_option(ss, pt, res, prb_str);
    }

    std::string str = ss.str();
    BENCHDNN_PRINT(0, "%s\n", str.c_str());
};

void base_perf_report_t::dump_engine(std::ostream &s) const {
    s << engine_tgt_kind;
}

void base_perf_report_t::handle_option(std::ostream &s, const char *&option,
        res_t *res, const char *prb_str) const {
    // Note: ideally, there should be `unspecified` mode, but there's additional
    // logic around `n_modes` involved which might be affected by adding new
    // but non-functional mode. If such mode existed, there's no need in extra
    // `user_mode` variable to identify if mode was specified or not.
    // It helps to choose between different "default" timer modes for different
    // options.
    timer::timer_t::mode_t mode = timer::timer_t::min;
    timer::timer_t::mode_t user_mode = timer::timer_t::n_modes;
    double unit = 1e0;
    char c = *option;

    if (c == '-' || c == '0' || c == '+') {
        user_mode = modifier2mode(c);
        mode = user_mode;
        c = *(++option);
    }

    if (c == 'K' || c == 'M' || c == 'G') {
        unit = modifier2unit(c);
        c = *(++option);
    }

    auto get_flops = [&](const timer::timer_t &t) -> double {
        if (!t.sec(mode)) return 0;
        return ops() / t.sec(mode) / unit;
    };

    auto get_bw = [&](const timer::timer_t &t) -> double {
        if (!t.sec(mode)) return 0;
        return (res->ibytes + res->obytes) / t.sec(mode) / unit;
    };

    auto get_freq = [&](const timer::timer_t &t) -> double {
        if (!t.sec(mode)) return 0;
        return t.ticks(mode) / t.sec(mode) / unit;
    };

    auto get_create_time = [&](const timer::timer_t &t) -> double {
        // If user didn't ask for mode, choose the maximum one to return time
        // for no-cache-hit creation.
        // Cache-hit creation can be triggered by `min` mode.
        auto create_mode = user_mode == timer::timer_t::n_modes
                ? timer::timer_t::max
                : mode;
        if (!t.sec(create_mode)) return 0;
        return t.ms(create_mode) / unit;
    };

    // Please update doc/knobs_perf_report.md in case of any new options!

#define HANDLE(opt, ...) \
    if (!strncmp(opt "%", option, strlen(opt) + 1)) { \
        __VA_ARGS__; \
        option += strlen(opt) + 1; \
        return; \
    }

    // Options operating on driver specific types, e.g. alg_t.
    HANDLE("alg", dump_alg(s));
    HANDLE("cfg", dump_cfg(s));
    HANDLE("desc", dump_desc(s));
    HANDLE("DESC", dump_desc_csv(s));
    HANDLE("engine", dump_engine(s));
    HANDLE("flags", dump_flags(s));
    HANDLE("activation", dump_rnn_activation(s));
    HANDLE("direction", dump_rnn_direction(s));
    // Options operating on common types, e.g. attr_t.
    HANDLE("attr", if (attr() && !attr()->is_def()) s << *attr());
    HANDLE("axis", if (axis()) s << *axis());
    HANDLE("dir", if (dir()) s << *dir());
    HANDLE("dt", if (dt()) s << *dt());
    HANDLE("group", if (group()) s << *group());
    HANDLE("sdt", if (sdt()) s << *sdt());
    HANDLE("stag", if (stag()) s << *stag());
    HANDLE("mb", if (user_mb()) s << *user_mb());
    HANDLE("name", if (name()) s << *name());
    HANDLE("ddt", if (ddt()) s << *ddt());
    HANDLE("dtag", if (dtag()) s << *dtag());
    HANDLE("prop", if (prop()) s << prop2str(*prop()));
    HANDLE("tag", if (tag()) s << *tag());
    HANDLE("stat_tag", if (stat_tag()) s << *stat_tag());
    HANDLE("wtag", if (wtag()) s << *wtag());
    HANDLE("ctx-init", s << *ctx_init());
    HANDLE("ctx-exe", s << *ctx_exe());
    // Options operating on driver independent objects, e.g. timer values.
    HANDLE("bw", s << get_bw(res->timer_map.perf_timer()));
    HANDLE("driver", s << driver_name);
    HANDLE("flops", s << get_flops(res->timer_map.perf_timer()));
    HANDLE("clocks", s << res->timer_map.perf_timer().ticks(mode) / unit);
    HANDLE("prb", s << prb_str);
    HANDLE("freq", s << get_freq(res->timer_map.perf_timer()));
    HANDLE("ops", s << ops() / unit);
    HANDLE("impl", s << res->impl_name);
    HANDLE("ibytes", s << res->ibytes / unit);
    HANDLE("obytes", s << res->obytes / unit);
    HANDLE("iobytes", s << (res->ibytes + res->obytes) / unit);
    HANDLE("idx", s << benchdnn_stat.tests);
    HANDLE("time", s << res->timer_map.perf_timer().ms(mode) / unit);
    HANDLE("ctime",
            s << get_create_time(res->timer_map.cp_timer())
                            + get_create_time(res->timer_map.cpd_timer()));
    HANDLE("cptime", s << get_create_time(res->timer_map.cp_timer()));
    HANDLE("cpdtime", s << get_create_time(res->timer_map.cpd_timer()));

#undef HANDLE

    auto opt_name = std::string(option);
    opt_name.pop_back();
    BENCHDNN_PRINT(0, "Error: perf report option \"%s\" is not supported\n",
            opt_name.c_str());
    SAFE_V(FAIL);
}
