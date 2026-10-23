/*******************************************************************************
* Copyright 2017 Intel Corporation
* Copyright 2020 FUJITSU LIMITED
* Copyright 2023, 2025 Arm Ltd. and affiliates
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

#include "common/impl_list_item.hpp"

#include "cpu/cpu_engine.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_reorder.hpp"
#include "cpu/x64/jit_uni_reorder_direct_copy.hpp"
#include "cpu/x64/matmul/brgemm_matmul_reorders.hpp"
#if DNNL_X64_USE_ZEN
#include "cpu/x64/zen64/reorder/zen_reorder.hpp"
#endif
#elif DNNL_AARCH64
#include "cpu/aarch64/matmul/brgemm_matmul_reorders.hpp"
#include "cpu/aarch64/reorder/jit_blk_reorder.hpp"
#include "cpu/aarch64/reorder/jit_uni_reorder.hpp"
#if defined(DNNL_AARCH64_USE_ACL)
#include "cpu/aarch64/reorder/acl_reorder.hpp"
#endif
#elif DNNL_PPC64
#include "cpu/ppc64/ppc64_gemm_reorder.hpp"
#elif DNNL_RV64
#include "cpu/rv64/reorder/jit_blk_reorder.hpp"
#include "cpu/rv64/reorder/jit_uni_reorder.hpp"
#endif

#include "cpu/reorder/ref_reorder.hpp"
#include "cpu/reorder/simple_reorder.hpp"
#include "cpu/reorder/simple_sparse_reorder.hpp"
#include "cpu/rnn/rnn_reorders.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;

const impl_list_item_t *get_reorder_impl_list(const reorder_desc_t *) {
    // clang-format off
    static const impl_list_item_t the_list[] = REG_REORDER_P({
            CPU_INSTANCE_X64_ZEN(x64::zen::reorder::zen_reorder_t)
            CPU_INSTANCE_X64(x64::brgemm_matmul_copy_reorder_t)
            CPU_INSTANCE_X64(x64::jit_uni_reorder_direct_copy_t)
            CPU_INSTANCE_X64(x64::jit_blk_reorder_t)
            CPU_INSTANCE_X64(x64::jit_uni_reorder_t)

            CPU_INSTANCE_AARCH64_ACL(aarch64::acl_reorder_fwd_t)
            CPU_INSTANCE_AARCH64(aarch64::brgemm_matmul_copy_reorder_t)
            CPU_INSTANCE_AARCH64(aarch64::jit_blk_reorder_t)
            CPU_INSTANCE_AARCH64(aarch64::jit_uni_reorder_t)

            CPU_INSTANCE_PPC64(ppc64::ppc64_matrixA_reorder_t)

            CPU_INSTANCE_RV64(rv64::jit_blk_reorder_t)
            CPU_INSTANCE_RV64(rv64::jit_uni_reorder_t)

            CPU_INSTANCE(rnn_data_reorder_t<f32, s8>)
            CPU_INSTANCE(rnn_data_reorder_t<f32, u8>)
            CPU_INSTANCE(rnn_weights_reorder_t<f32, f32>)
            CPU_INSTANCE(rnn_weights_reorder_t<f32, bf16>)
            CPU_INSTANCE(rnn_weights_reorder_t<bf16, bf16>)
            CPU_INSTANCE(rnn_weights_reorder_s8_t<f32>)
            CPU_INSTANCE(rnn_weights_reorder_s8_t<s8>)
            CPU_INSTANCE(rnn_brgemm_weights_reorder_s8_t<f32, s8>)
            CPU_INSTANCE(rnn_brgemm_weights_reorder_s8_t<s8, s8>)

            CPU_INSTANCE(simple_reorder_t)

            CPU_INSTANCE(ref_reorder_t)

            CPU_INSTANCE_X64(simple_sparse_reorder_t<f32, impl::format_tag_t, any, s8, impl::format_tag_t, any>)
            CPU_INSTANCE_X64(simple_sparse_reorder_t<s8, impl::format_tag_t, any, s8, impl::format_tag_t, any>)

            nullptr,
    });
    // clang-format on
    return the_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
