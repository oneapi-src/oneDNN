/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <stdint.h>

#include "utils.hpp"
#include "nstl.hpp"

namespace mkldnn {
namespace impl {
namespace math {

template <typename data_t, typename acc_t>
inline typename utils::remove_reference<data_t>::type saturate(const acc_t &x)
{
    acc_t v = x;
    if (v < nstl::numeric_limits<data_t>::lowest())
        v = nstl::numeric_limits<data_t>::lowest();
    if (v > nstl::numeric_limits<data_t>::max())
        v = nstl::numeric_limits<data_t>::max();
    return (typename utils::remove_reference<data_t>::type)v;
}

template <> inline int8_t saturate<int8_t, uint8_t>(const uint8_t &x) {
    return x <= 127u ? x : 127;
}

template <typename out_t>
inline typename utils::enable_if<nstl::is_integral<out_t>::value, out_t>::type
out_round(float v, round_mode_t rmode = round_mode::nearest)
{ return (out_t)(rmode == round_mode::down ? floorf(v) : rintf(v)); }

template <typename out_t>
inline typename utils::enable_if<!nstl::is_integral<out_t>::value, out_t>::type
out_round(float v, round_mode_t rmode = round_mode::nearest)
{ UNUSED(rmode); return v; }

inline int gcd(int a, int b) {
	a = impl::nstl::abs(a);
	b = impl::nstl::abs(b);
	if (a < b) { int x = a; a = b; b = x; }

	if (b == 0) return a;

	int r;
	while ((r = a % b) != 0) { a = b; b = r; }

	return b;
}

}
}
}

#endif
