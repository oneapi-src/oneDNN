/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
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
#include "cpu/aarch64/brgemm/brgemm_types.hpp"
#if defined(DNNL_EXPERIMENTAL_UKERNEL) && defined(DNNL_AARCH64_USE_KAI)
#include "cpu/aarch64/kleidiai/kai.hpp"
#endif
namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

#if defined(DNNL_EXPERIMENTAL_UKERNEL) && defined(DNNL_AARCH64_USE_KAI)
template <typename KernelType>
kai_kernel_t<KernelType>::kai_kernel_t(const brgemm_desc_t abrd) {
    kai_kernel_ = new KernelType(abrd);
}
template <typename KernelType>
void kai_kernel_t<KernelType>::operator()(
        brgemm_kernel_params_t *params) const {
    (*kai_kernel_)(params);
}
template <typename KernelType>
std::string kai_kernel_t<KernelType>::to_string() const {
    return kai_kernel_->to_string();
};
template <typename KernelType>
kai_kernel_t<KernelType>::~kai_kernel_t() {
    delete kai_kernel_;
}
template struct kai_kernel_t<kai_f32_qa8dxp_qs4c32p_kernel_packet_t>;
template struct kai_kernel_t<kai_f32_qa8dxp_qs4cxp_kernel_packet_t>;
template struct kai_kernel_t<kai_f32_f32_f32p_kernel_packet_t>;
#endif

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
