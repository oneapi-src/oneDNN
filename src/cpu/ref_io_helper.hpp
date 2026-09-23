/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_REF_IO_HELPER_HPP
#define CPU_REF_IO_HELPER_HPP

#include <cassert>

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/nibble.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace io {

namespace {
template <data_type_t dt>
ALWAYS_INLINE void store_subbyte(float val, void *ptr, dim_t idx) {
    using _nibble_type = typename prec_traits_t<dt>::nibble_type;
    auto *_ptr = reinterpret_cast<_nibble_type *>(ptr);
    _nibble_type _nibble = _ptr[idx / _nibble_type::nelems()];
    using _type = typename prec_traits_t<dt>::type;
    _type _val(cpu::q10n::saturate_and_round<_type>(val));
    _nibble.set(_val.raw_bits_, idx % _nibble_type::nelems());
    _ptr[idx / _nibble_type::nelems()] = _nibble;
}
} // namespace

ALWAYS_INLINE int load_int_value(data_type_t dt, const void *ptr, dim_t idx) {
    assert(ptr);
#define CASE(dt) \
    case dt: \
        return static_cast<int>( \
                reinterpret_cast<const typename prec_traits_t<dt>::type *>( \
                        ptr)[idx]);

    using namespace data_type;
    switch (dt) {
        CASE(s32);
        CASE(s8);
        CASE(u8);
        case s4: {
            const nibble2_t nibble_pair(
                    reinterpret_cast<const uint8_t *>(ptr)[idx / 2]);
            int4_t val(nibble_pair.get(idx % 2));
            return static_cast<int>(val);
        }
        case u4: {
            const nibble2_t nibble_pair(
                    reinterpret_cast<const uint8_t *>(ptr)[idx / 2]);
            uint4_t val(nibble_pair.get(idx % 2));
            return static_cast<int>(val);
        }
        case u2: {
            const nibble4_t nibble_quartet(
                    reinterpret_cast<const uint8_t *>(ptr)[idx / 4]);
            uint2_t val(nibble_quartet.get(idx % 4));
            return static_cast<int>(val);
        }
        default: assert(!"bad data_type");
    }

#undef CASE
    return INT_MAX;
}

ALWAYS_INLINE int64_t load_int64_value(
        data_type_t dt, const void *ptr, dim_t idx) {
    assert(ptr);
    if (dt == data_type::s64)
        return reinterpret_cast<const int64_t *>(ptr)[idx];
    return static_cast<int64_t>(load_int_value(dt, ptr, idx));
}

ALWAYS_INLINE float load_float_value(
        data_type_t dt, const void *ptr, dim_t idx) {
    assert(ptr);
#define CASE(dt) \
    case dt: \
        return static_cast<float>( \
                reinterpret_cast<const typename prec_traits_t<dt>::type *>( \
                        ptr)[idx]);

    using namespace data_type;
    switch (dt) {
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(e8m0);
        case s4: {
            const nibble2_t nibble_pair(
                    static_cast<const uint8_t *>(ptr)[idx / 2]);
            int4_t val(nibble_pair.get(idx % 2));
            return static_cast<float>(val);
        }
        case u4: {
            const nibble2_t nibble_pair(
                    static_cast<const uint8_t *>(ptr)[idx / 2]);
            uint4_t val(nibble_pair.get(idx % 2));
            return static_cast<float>(val);
        }
        case f4_e2m1: {
            const nibble2_t nibble_pair
                    = reinterpret_cast<const nibble2_t *>(ptr)[idx / 2];
            float4_e2m1_t val(nibble_pair.get(idx % 2), true);
            return static_cast<float>(val);
        }
        case u2: {
            const nibble4_t nibble_quartet(
                    static_cast<const uint8_t *>(ptr)[idx / 4]);
            uint2_t val(nibble_quartet.get(idx % 4));
            return static_cast<float>(val);
        }
        default: assert(!"bad data_type");
    }

#undef CASE
    return NAN;
}

ALWAYS_INLINE void store_float_value(
        data_type_t dt, float val, void *ptr, dim_t idx) {
    assert(ptr);
#define CASE(dt) \
    case dt: { \
        using type_ = typename prec_traits_t<dt>::type; \
        *(reinterpret_cast<type_ *>(ptr) + idx) \
                = cpu::q10n::saturate_and_round<type_>(val); \
    } break;

    using namespace data_type;
    switch (dt) {
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(e8m0);
        case f4_e2m1: store_subbyte<f4_e2m1>(val, ptr, idx); break;
        case s4: store_subbyte<s4>(val, ptr, idx); break;
        case u4: store_subbyte<u4>(val, ptr, idx); break;
        case u2: store_subbyte<u2>(val, ptr, idx); break;
        default: assert(!"bad data_type");
    }

#undef CASE
}

} // namespace io
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
