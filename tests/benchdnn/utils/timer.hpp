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

#ifndef UTILS_TIMER_HPP
#define UTILS_TIMER_HPP

#include <string>
#include <unordered_map>

#define TIME_FUNC(func, res, name) \
    do { \
        if (res) { \
            auto &t = res->timer_map.get_timer(name); \
            t.start(); \
            func; \
            t.stamp(); \
        } else { \
            func; \
        } \
    } while (0)

#define TIME_REF(func) TIME_FUNC(func, res, timer::names::ref_timer)
#define TIME_C_PD(func) TIME_FUNC(func, res, timer::names::cpd_timer)
#define TIME_C_PRIM(func) TIME_FUNC(func, res, timer::names::cp_timer)
// Designated timer to calculate time spent on comparison with reference
#define TIME_COMPARE(func) TIME_FUNC(func, res, timer::names::compare_timer)
// Designated timer to calculate time spent on filling
#define TIME_FILL(func) TIME_FUNC(func, res, timer::names::fill_timer)

namespace timer {

struct timer_t {
    enum mode_t { min = 0, avg = 1, max = 2, sum = 3, n_modes };

    timer_t() { reset(); }

    // Fully reset the measurements
    void reset();
    // Restart timer
    void start();
    // Stop timer & update statistics
    void stop(int add_times, int64_t add_ticks, double add_ms);

    void stamp(int add_times = 1);

    void stamp_with_frequency(int add_times, double add_ms, double freq) {
        uint64_t add_ticks = (uint64_t)(add_ms * freq / 1e3);
        stop(add_times, add_ticks, add_ms);
    }

    int times() const { return times_; }

    double total_ms() const { return ms_[avg]; }

    double ms(mode_t mode = min) const {
        if (!times()) return 0; // nothing to report
        return ms_[mode] / (mode == avg ? times() : 1);
    }

    double sec(mode_t mode = min) const { return ms(mode) / 1e3; }

    uint64_t ticks(mode_t mode = min) const {
        if (!times()) return 0; // nothing to report
        return ticks_[mode] / (mode == avg ? times() : 1);
    }

    timer_t(const timer_t &rhs) = default;
    timer_t &operator=(const timer_t &rhs);
    timer_t &operator=(timer_t &&rhs) = default;

    int times_;
    uint64_t ticks_[n_modes], ticks_start_;
    double ms_[n_modes], ms_start_;
};

// Designated timers to support benchdnn performance reporting and general time
// collection to estimate partial time of certain functional pieces.
namespace names {
// Testing objects execution performance.
const std::string perf_timer = "perf_timer";
// Driver's reference computations.
const std::string ref_timer = "compute_ref_timer";
// Primitive descriptor creation performace.
const std::string cpd_timer = "create_pd_timer";
// Primitive creation performace.
const std::string cp_timer = "create_prim_timer";
// Driver's comparison.
const std::string compare_timer = "compare_timer";
// Driver's memory filling.
const std::string fill_timer = "fill_timer";
} // namespace names

struct timer_map_t {
    timer_t &get_timer(const std::string &name);

    timer_t &perf_timer() { return get_timer(names::perf_timer); }
    timer_t &cpd_timer() { return get_timer(names::cpd_timer); }
    timer_t &cp_timer() { return get_timer(names::cp_timer); }

    std::unordered_map<std::string, timer_t> timers;
};

// Note: in case the tuple below get extended, add enum that will control fields
// and replace std::get<N> to stg::get<enum::value> to change mapping between
// type and variable in a single place and not all over the code.
using service_timers_entry_t = std::tuple</* timer print name = */ std::string,
        /* supported mode = */ mode_bit_t,
        /* timer bench name = */ std::string>;

const std::vector<service_timers_entry_t> &get_global_service_timers();

} // namespace timer

#endif
