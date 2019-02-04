/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_INNER_PRODUCT_PD_HPP
#define CPU_INNER_PRODUCT_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "inner_product_pd.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {
inline bool dense_gemm_consitency_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    using namespace memory_format;
    using namespace utils;
    return true
        && IMPLICATION(src_d.format() == nChw8c, wei_d.format() == oIhw8i)
        && IMPLICATION(src_d.format() == nChw16c, wei_d.format() == oIhw16i)
        && IMPLICATION(src_d.format() == nCdhw8c, wei_d.format() == oIdhw8i)
        && IMPLICATION(src_d.format() == nCdhw16c, wei_d.format() == oIdhw16i)
        && IMPLICATION(src_d.format() == nchw, wei_d.format() == oihw)
        && IMPLICATION(src_d.format() == ncdhw, wei_d.format() == oidhw)
        && IMPLICATION(src_d.format() == nhwc, wei_d.format() == hwio)
        && IMPLICATION(src_d.format() == ndhwc, wei_d.format() == dhwio)
        && IMPLICATION(src_d.format() == nc, one_of(wei_d.format(), oi, io))
        && dst_d.format() == nc
        && src_d.only_padded_dim(1)
        && wei_d.only_padded_dim(1)
        && src_d.padded_dims()[1]
            == wei_d.padded_dims()[1]
        && src_d.is_dense(true)
        && dst_d.is_dense()
        && wei_d.is_dense(true);
}
}

struct cpu_inner_product_fwd_pd_t: public inner_product_fwd_pd_t {
    using inner_product_fwd_pd_t::inner_product_fwd_pd_t;
};

struct cpu_inner_product_bwd_data_pd_t: public inner_product_bwd_data_pd_t {
    using inner_product_bwd_data_pd_t::inner_product_bwd_data_pd_t;
};

struct cpu_inner_product_bwd_weights_pd_t: public inner_product_bwd_weights_pd_t {
    using inner_product_bwd_weights_pd_t::inner_product_bwd_weights_pd_t;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
