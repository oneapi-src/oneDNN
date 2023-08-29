/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_UTIL_INVARIANT_INT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_UTIL_INVARIANT_INT_HPP

#include <assert.h>
#include <util/compiler_macros.hpp>
#include <util/utils.hpp>
// if the compiler has not defined uint128, we need to implement
// it on our own. Also, the uint128 implementation is incomplete on OSS
// dpcpp.
#if !defined(__SIZEOF_INT128__) || (SC_IS_CLANG() && defined(_MSC_VER))
#include <util/uint128.hpp>
#define HAS_BUILTIN_INT128 0
#else
#define HAS_BUILTIN_INT128 1
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace invariant_int {

// If __uint128_t not defined by compiler, use own implementation
#if HAS_BUILTIN_INT128
using uint128_t = __uint128_t;
#else
using uint128_t = utils::uint128_t;
#endif
#undef HAS_BUILTIN_INT128

// uint128 pow of 2
inline uint128_t u128_pow_2(const int n) {
    return uint128_t(UINT64_C(1)) << n;
}

// internal struct for choosing multiplier
struct ChooseMultiplier {
    int log_;
    int sft_;
    uint128_t m_low_;
    uint128_t m_hig_;
    /**
     * choose multiplier for invariant usigned div
     * @param d the invariant unsigned int
     * @param N number of calculation bits
     * @param prec bits of precision
     * */
    ChooseMultiplier(uint64_t d, int N, int prec) {
        // log = ceil(log2(d))
        log_ = 64 - utils::clz(d - 1);
        uint128_t _2_pow_N_l = u128_pow_2(N + log_);
        uint128_t _2_pow_N_lp = u128_pow_2(N + log_ - prec);

        sft_ = log_;
        m_low_ = _2_pow_N_l / uint128_t(d);
        m_hig_ = (_2_pow_N_l + _2_pow_N_lp) / uint128_t(d);

        while (((m_low_ >> 1) < (m_hig_ >> 1)) && (sft_ > 0)) {
            m_low_ = m_low_ >> 1;
            m_hig_ = m_hig_ >> 1;
            sft_--;
        }
    }
};

/**
 * multiplier for invariant usigned div optimization
 * */
struct UintDivMultiplier {
    uint64_t magic_;
    int sft_pre_;
    int sft_post_;
    bool compensate_;
    bool power_of_2_;
    /**
     * generate multiplier for invariant usigned div.
     * @param d the invariant unsigned int
     * @param N number of calculation bits
     * */
    UintDivMultiplier(uint64_t d, int N) {
        // Get initial multiplier
        assert(d > 1);
        uint128_t _2_pow_N = u128_pow_2(N);
        uint128_t magic;
        int sft_pre, sft_post;
        const auto mulx = ChooseMultiplier(d, N, N);
        // If initial multiplier larger than _2_pow_N
        // Choose a new multiplier when d stripped to odd
        if (mulx.m_hig_ >= _2_pow_N && (d % 2) == 0) {
            // strip d to odd
            int e = utils::ctz(d);
            uint64_t d_odd = d >> e;
            // Get refined multiplier
            const auto muly = ChooseMultiplier(d_odd, N, N - e);
            magic = muly.m_hig_;
            sft_pre = e;
            sft_post = muly.sft_;
        } else {
            magic = mulx.m_hig_;
            sft_pre = 0;
            sft_post = mulx.sft_;
        }
        // Return final multiplier
        if (d == (UINT64_C(1) << mulx.log_)) {
            magic_ = 0;
            sft_pre_ = mulx.log_;
            sft_post_ = 0;
            compensate_ = false;
            power_of_2_ = true;
        } else if (magic >= _2_pow_N) {
            assert(sft_pre == 0);
            magic_ = uint64_t(magic - _2_pow_N);
            sft_pre_ = 1;
            sft_post_ = sft_post - 1;
            compensate_ = true;
            power_of_2_ = false;
        } else {
            magic_ = uint64_t(magic);
            sft_pre_ = sft_pre;
            sft_post_ = sft_post;
            compensate_ = false;
            power_of_2_ = false;
        }
    }
};

/**
 * multiplier for invariant signed div optimization
 * quotient rounded towards 0
 * */
struct SintDivMultiplier {
    uint64_t magic_;
    int sft_;
    bool negative_;
    bool compensate_;
    bool power_of_2_;
    /**
     * generate multiplier for invariant signed div
     * @param d the invariant signed int
     * @param N number of calculation bits
     * */
    SintDivMultiplier(int64_t d, int N) {
        // Get initial multiplier
        negative_ = (d < 0);
        uint64_t abs_d = std::abs(d);
        assert(abs_d > 1);
        uint128_t _2_pow_N = u128_pow_2(N);
        uint128_t _2_pow_N_m1 = u128_pow_2(N - 1);
        //
        const auto mulx = ChooseMultiplier(abs_d, N, N - 1);
        // Return final multiplier
        if (abs_d == (UINT64_C(1) << mulx.log_)) {
            magic_ = 0;
            sft_ = mulx.log_;
            compensate_ = false;
            power_of_2_ = true;
        } else if (mulx.m_hig_ >= _2_pow_N_m1) {
            magic_ = uint64_t(mulx.m_hig_ - _2_pow_N);
            sft_ = mulx.sft_;
            compensate_ = true;
            power_of_2_ = false;
        } else {
            magic_ = uint64_t(mulx.m_hig_);
            sft_ = mulx.sft_;
            compensate_ = false;
            power_of_2_ = false;
        }
    }
};

} // namespace invariant_int
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
