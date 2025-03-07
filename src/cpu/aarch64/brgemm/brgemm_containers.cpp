/*******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#include "cpu/aarch64/brgemm/brgemm_containers.hpp"
#include "cpu/aarch64/brgemm/jit_brdgmm_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;

namespace brgemm_containers {

#ifdef BRGEMM_KERNEL_GLOBAL_STORAGE
std::set<std::shared_ptr<brgemm_kernel_t>,
        decltype(brgemm_kernel_container_t::brgemm_kernel_cmp) *>
        brgemm_kernel_container_t::set_
        = std::set<std::shared_ptr<brgemm_kernel_t>,
                decltype(brgemm_kernel_container_t::brgemm_kernel_cmp) *>(
                brgemm_kernel_container_t::brgemm_kernel_cmp);
#endif

bool brgemm_desc_container_t::insert(int idx, brgemm_desc_t &brg,
        const std::vector<char> &bd_mask,
        const std::vector<brgemm_batch_element_t> &static_offsets) {
    bd_mask_list_.push_back(bd_mask);
    brg.brgattr.bd_mask = bd_mask_list_.back().data();

    static_offsets_list_.push_back(static_offsets);
    brg.brgattr.static_offsets = static_offsets_list_.back().data();

    const auto ret = set_.insert(brg);
    refs_[idx] = &(*ret.first);
    // if there was no insertion then clean bd_mask and static_offsets
    if (!ret.second) {
        bd_mask_list_.pop_back();
        static_offsets_list_.pop_back();
    }
    return ret.second;
}

bool brgemm_kernel_container_t::brgemm_kernel_cmp(
        const std::shared_ptr<brgemm_kernel_t> &lhs,
        const std::shared_ptr<brgemm_kernel_t> &rhs) {
    const auto lsz = lhs->get_jit_generator()->getSize();
    const auto rsz = rhs->get_jit_generator()->getSize();
    if (lsz != rsz) return (lsz < rsz);
    const auto lcode = lhs->get_jit_generator()->CodeGenerator::getCode();
    const auto rcode = rhs->get_jit_generator()->CodeGenerator::getCode();
    return (std::memcmp(lcode, rcode, lsz) < 0);
}

status_t brgemm_kernel_container_t::insert(int idx, const brgemm_desc_t *brg) {
    // Use two level hashing of brgemm kernels:
    // 1. Try to find entry in local brgemm_map_ using brgemm descriptor as a
    // key (we can check if brgemm descriptor is unique inside brgemm primitive)
    // 2. Only if we do not find entry in local brgemm_map_  then try to find
    // entry in kernel storage using kernel code as key
    const auto brgemm_it = brgemm_map_.find(brg);
    if (brgemm_it == brgemm_map_.end()) {
        brgemm_kernel_t *brg_kernel = nullptr;
        status_t s = brgemm_kernel_create(&brg_kernel, *brg);
        if (s != status::success) {
            delete brg_kernel;
            return s;
        }
        std::shared_ptr<brgemm_kernel_t> sptr(brg_kernel);
        lock_write();
        const auto kernel_ret = set_.insert(sptr);
        refs_[idx] = kernel_ret.first->get();
        unlock_write();
        const auto brgemm_ret = brgemm_map_.insert({brg, refs_[idx]});
        if (!brgemm_ret.second) return status::runtime_error;
    } else {
        refs_[idx] = brgemm_it->second;
    }
    return status::success;
}

} // namespace brgemm_containers
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
