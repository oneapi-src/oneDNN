/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_CODEGEN_REGISTER_ALLOCATOR_HPP
#define GPU_INTEL_JIT_CODEGEN_REGISTER_ALLOCATOR_HPP

#include "common/z_magic.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "ngen/ngen.hpp"
#include "ngen/ngen_register_allocator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Register Allocator Wrapper to allow for custom checks.
class reg_allocator_t {
public:
    // Based on nGEN limitation where subregisters are allocated in dword chunks.
    static const int granularity = 4;

    reg_allocator_t(ngen::HW hw, const std::string &kernel_name_) : ra(hw) {
#ifdef DNNL_DEV_MODE
        kernel_name = kernel_name_;
#endif
        MAYBE_UNUSED(kernel_name_);
    }
    ~reg_allocator_t()
#ifdef DNNL_DEV_MODE
    {
        gpu_assert(!is_speculate) << "Speculative allocation never finished\n";
    }
#else
            = default;
#endif

    ngen::HW hardware() const { return ra.hardware(); }

    ngen::GRFRange alloc_range(int nregs,
            ngen::Bundle base_bundle = ngen::Bundle(),
            ngen::BundleGroup bundle_mask = ngen::BundleGroup::AllBundles()) {
        auto ret = ra.alloc_range(nregs, base_bundle, bundle_mask);
        update_peak_grf_usage();
        return ret;
    }
    ngen::GRF alloc(ngen::Bundle bundle = ngen::Bundle()) {
        auto ret = ra.alloc(bundle);
        update_peak_grf_usage();
        return ret;
    }

    ngen::FlagRegister alloc_flag(bool sub = true) {
        return ra.alloc_flag(sub);
    }

    ngen::GRFRange try_alloc_range(int nregs,
            ngen::Bundle base_bundle = ngen::Bundle(),
            ngen::BundleGroup bundle_mask = ngen::BundleGroup::AllBundles()) {
        auto ret = ra.try_alloc_range(nregs, base_bundle, bundle_mask);
        update_peak_grf_usage();
        return ret;
    }
    ngen::GRF try_alloc(ngen::Bundle bundle = ngen::Bundle()) {
        auto ret = ra.try_alloc(bundle);
        update_peak_grf_usage();
        return ret;
    }

    ngen::Subregister alloc_sub(
            ngen::DataType type, ngen::Bundle bundle = ngen::Bundle()) {
        auto ret = ra.alloc_sub(type, bundle);
        update_peak_grf_usage();
        return ret;
    }

    template <typename T>
    ngen::Subregister alloc_sub(ngen::Bundle bundle = ngen::Bundle()) {
        auto ret = ra.alloc_sub<T>(bundle);
        update_peak_grf_usage();
        return ret;
    }

    ngen::Subregister try_alloc_sub(
            ngen::DataType type, ngen::Bundle bundle = ngen::Bundle()) {
        auto ret = ra.try_alloc_sub(type, bundle);
        update_peak_grf_usage();
        return ret;
    }
    template <typename T>
    ngen::Subregister try_alloc_sub(ngen::Bundle bundle = ngen::Bundle()) {
        auto ret = ra.try_alloc_sub<T>(bundle);
        update_peak_grf_usage();
        return ret;
    }
    template <typename RD>
    void safeRelease(RD &reg) {
        ra.safeRelease(reg);
    }
    template <typename RD>
    void release(RD reg) {
        ra.release(reg);
    }
    template <typename RD>
    void claim(RD reg) {
        ra.claim(reg);
        update_peak_grf_usage();
    }

    void setRegisterCount(int rcount) { ra.setRegisterCount(rcount); }

#ifdef DNNL_DEV_MODE
    int get_peak_regs() const { return peak_regs; }
    int get_alloced_regs() const { return ra.countAllocedRegisters(); }

    // For performing speculative allocations that may not be used in the final
    // register allocation
    void start_speculate() { is_speculate = true; }
    void finish_speculate() {
        is_speculate = false;
        update_peak_grf_usage();
    }
#else
    void start_speculate() {}
    void finish_speculate() {}
#endif

protected:
#ifdef DNNL_DEV_MODE
    void update_peak_grf_usage() {
        if (is_speculate) return;
        int register_count = get_alloced_regs();
        if (peak_regs < register_count) peak_regs = register_count;
    }
#else
    void update_peak_grf_usage() {}
#endif

#ifdef DNNL_DEV_MODE
    int peak_regs = 0;
    bool is_speculate = false;
    std::string kernel_name;
#endif

private:
    ngen::RegisterAllocator ra;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
