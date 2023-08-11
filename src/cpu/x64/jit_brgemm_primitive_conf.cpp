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

#include <cassert>

#include "cpu/x64/jit_brgemm_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

int jit_brgemm_primitive_conf_t::ks() const {
    return kd * kh * kw;
}

int jit_brgemm_primitive_conf_t::get_weights_oc_block() const {
    using namespace format_tag;

    assert(wei_tag != undef && "Weights tag not defined!");

    int fwd_oc_block = 0;
    switch (wei_tag) {
        case OI16i64o:
        case OIw16i64o:
        case OwI16i64o:
        case OIhw16i64o:
        case OhwI16i64o:
        case OIdhw16i64o:
        case OdhwI16i64o:
        case OI8i64o2i:
        case OIw8i64o2i:
        case OwI8i64o2i:
        case OIhw8i64o2i:
        case OhwI8i64o2i:
        case OIdhw8i64o2i:
        case OdhwI8i64o2i:
        case OI16i64o2i:
        case OIw16i64o2i:
        case OwI16i64o2i:
        case OIhw16i64o2i:
        case OhwI16i64o2i:
        case OIdhw16i64o2i:
        case OdhwI16i64o2i: fwd_oc_block = 64; break;
        case OI16i48o:
        case OIw16i48o:
        case OwI16i48o:
        case OIhw16i48o:
        case OhwI16i48o:
        case OIdhw16i48o:
        case OdhwI16i48o: fwd_oc_block = 48; break;
        case OI16i32o:
        case OIw16i32o:
        case OwI16i32o:
        case OIhw16i32o:
        case OhwI16i32o:
        case OIdhw16i32o:
        case OdhwI16i32o:
        case OI8i32o2i:
        case OIw8i32o2i:
        case OwI8i32o2i:
        case OIhw8i32o2i:
        case OhwI8i32o2i:
        case OIdhw8i32o2i:
        case OdhwI8i32o2i:
        case OI16i32o2i:
        case OIw16i32o2i:
        case OwI16i32o2i:
        case OIhw16i32o2i:
        case OhwI16i32o2i:
        case OIdhw16i32o2i:
        case OdhwI16i32o2i: fwd_oc_block = 32; break;
        case OI8i24o:
        case OIw8i24o:
        case OwI8i24o:
        case OIhw8i24o:
        case OhwI8i24o:
        case OIdhw8i24o:
        case OdhwI8i24o: fwd_oc_block = 24; break;
        case OI8i16o:
        case OIw8i16o:
        case OwI8i16o:
        case OIhw8i16o:
        case OhwI8i16o:
        case OIdhw8i16o:
        case OdhwI8i16o: fwd_oc_block = 16; break;
        default: fwd_oc_block = simd_w;
    };

    return fwd_oc_block;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
