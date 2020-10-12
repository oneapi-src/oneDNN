/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_BRGEMM_INNER_PRODUCT_UTILS_HPP
#define CPU_X64_BRGEMM_INNER_PRODUCT_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_engine.hpp"
#include "cpu/cpu_inner_product_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/x64/jit_brgemm_primitive_conf.hpp"

#define BRGEMM_IP_BWD_D_GLOBAL_B_TRANSPOSE

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace brgemm_inner_product_utils {

inline format_tag_t get_brgemm_ip_weights_tag(dim_t oc, data_type_t wei_dt) {
    using namespace format_tag;
    if (oc >= 64) {
        switch (wei_dt) {
            case data_type::f32: return OI16i64o;
            case data_type::bf16: return OI8i64o2i;
            case data_type::s8: return OI4i64o4i;
            default: return undef;
        }
    } else if (oc >= 32) {
        switch (wei_dt) {
            case data_type::f32: return OI16i32o;
            case data_type::bf16: return OI8i32o2i;
            case data_type::s8: return OI4i32o4i;
            default: return undef;
        }
    } else {
        switch (wei_dt) {
            case data_type::f32: return OI16i16o;
            case data_type::bf16: return OI8i16o2i;
            case data_type::s8: return OI4i16o4i;
            default: return undef;
        }
    }
}

status_t init_ip_conf(jit_brgemm_primitive_conf_t &jbgp,
        const inner_product_desc_t &ipd, const memory_desc_wrapper &src_md,
        const memory_desc_wrapper &weights_md,
        const memory_desc_wrapper &dst_md, const primitive_attr_t &attr,
        int nthreads);

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_primitive_conf_t &jbgp);

} // namespace brgemm_inner_product_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
