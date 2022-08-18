/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include <map>
#include <string>

#define TIME_FUNC(func, res, name) \
    do { \
        auto &t = res->timer_map.get_timer(name); \
        t.start(); \
        func; \
        t.stamp(); \
    } while (0)

// Designated timer to calculate time spent on reference computations
#define TIME_REF(func) TIME_FUNC(func, res, timer::timer_t::ref_timer)

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

    timer_t &operator=(const timer_t &rhs);

    int times_;
    uint64_t ticks_[n_modes], ticks_start_;
    double ms_[n_modes], ms_start_;

    // Section with timer fixed timer names for ease of use
    static const std::string perf_timer;
    static const std::string ref_timer;
    static const std::string prim_create_timer;
    static const std::string par_compl_timer;
};

struct timer_map_t {
    timer_t &get_timer(const std::string &name);

    timer_t &perf_timer();
    timer_t &prim_create_timer();
    timer_t &par_compl_timer();

    std::map<std::string, timer_t> timers;
};

} // namespace timer

#endif
