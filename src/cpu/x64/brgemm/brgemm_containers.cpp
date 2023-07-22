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

#include "cpu/x64/brgemm/brgemm_containers.hpp"
#include "cpu/x64/brgemm/jit_brdgmm_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

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

bool brgemm_desc_container_t::insert(int idx, brgemm_t &brg,
        const std::vector<char> &bd_mask,
        const std::vector<brgemm_batch_element_t> &static_offsets) {
    bd_mask_list_.push_back(bd_mask);
    brg.brgattr.bd_mask = bd_mask_list_.back().data();

    static_offsets_list_.push_back(static_offsets);
    brg.brgattr.static_offsets = static_offsets_list_.back().data();

    const auto ret = map_.insert({brg, idx});
    refs_[idx] = &(ret.first->first);
    // if there was no insertion then clean bd_mask and static_offsets
    if (!ret.second) {
        bd_mask_list_.pop_back();
        static_offsets_list_.pop_back();
    }
    return ret.second;
}

int brgemm_desc_container_t::insert(brgemm_t &brg,
        const std::vector<char> &bd_mask,
        const std::vector<brgemm_batch_element_t> &static_offsets) {
    bd_mask_list_.push_back(bd_mask);
    brg.brgattr.bd_mask = bd_mask_list_.back().data();

    static_offsets_list_.push_back(static_offsets);
    brg.brgattr.static_offsets = static_offsets_list_.back().data();
    const int ref_size = refs_.size();
    const auto ret = map_.insert({brg, -1});
    if (!ret.second) {
        // if there was no insertion then clean bd_mask and static_offsets
        bd_mask_list_.pop_back();
        static_offsets_list_.pop_back();
        return ret.first->second;
    }

    int idx = map_.size() - 1;
    if (idx > ref_size - 1) {
        if (ref_size == 0)
            refs_.resize(1);
        else
            refs_.resize(2 * ref_size);
    }
    refs_[idx] = &(ret.first->first);
    ret.first->second = idx;
    return idx;
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

status_t brgemm_kernel_container_t::insert(int idx, const brgemm_t *brg) {
    // Use two level hashing of brgemm kernels:
    // 1. Try to find entry in local brgemm_map_ using brgemm descriptor as a
    // key (we can check if brgemm descriptor is unique inside brgemm primitive)
    // 2. Only if we do not find entry in local brgemm_map_  then try to find
    // entry in kernel storage using kernel code as key
    const auto brgemm_it = brgemm_map_.find(brg);
    if (brgemm_it == brgemm_map_.end()) {
        brgemm_kernel_t *brg_kernel = nullptr;
        CHECK(brgemm_kernel_create(&brg_kernel, *brg));
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

bool brgemm_palette_container_t::insert(int idx, const brgemm_t *brg) {
    S_t kernel_palette;
    CHECK(brgemm_init_tiles(*brg, kernel_palette.data()));
    const auto ret = set_.insert(kernel_palette);
    refs_[idx] = &(*ret.first);
    return ret.second;
}

} // namespace brgemm_containers
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
