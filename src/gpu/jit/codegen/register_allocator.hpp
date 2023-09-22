/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GPU_JIT_CODEGEN_REGISTER_ALLOCATOR_HPP
#define GPU_JIT_CODEGEN_REGISTER_ALLOCATOR_HPP

#include "common/z_magic.hpp"
#include "gpu/jit/ngen/ngen.hpp"
#include "gpu/jit/ngen/ngen_register_allocator.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Register Allocator Wrapper to allow for custom checks.
class reg_allocator_t {
public:
    static const uint64_t warn_none = 0;
    static const uint64_t warn_large_grf = 1;
    static const uint64_t warn_all = warn_large_grf;
    static const uint64_t warn_default = warn_none;

    // Based on nGEN limitation where subregisters are allocated in dword chunks.
    static const int granularity = 4;

    reg_allocator_t(ngen::HW hw, const std::string &kernel_name_,
            int warn_flags_ = warn_default)
        : ra(hw) {
#ifdef DNNL_DEV_MODE
        kernel_name = kernel_name_;
        warn_flags = warn_flags_;
#endif
        MAYBE_UNUSED(kernel_name_);
        MAYBE_UNUSED(warn_flags_);
    }
    ~reg_allocator_t() {
#ifdef DNNL_DEV_MODE
        if ((warn_flags & warn_large_grf) && (peak_regs <= 128)
                && (ra.getRegisterCount() > 128))
            ir_warning() << kernel_name
                         << " uselessly enables large grf mode as " << peak_regs
                         << " registers were used\n";
        ir_assert(!is_speculate) << "Speculative allocation never finished\n";
#endif
    }

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
    int warn_flags;
    bool is_speculate = false;
    std::string kernel_name;
#endif

private:
    ngen::RegisterAllocator ra;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
