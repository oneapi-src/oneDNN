/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_GEMM_GEN_GEMM_KERNEL_HPP
#define GPU_INTEL_JIT_GEMM_GEN_GEMM_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/kernel_arg_list.hpp"
#include "gpu/intel/jit/gemm/gen_gemm_kernel_generator.hpp"
#include "gpu/intel/jit/gemm/kernel_catalog.hpp"
#include "gpu/intel/jit/gemm/kernel_evaluator.hpp"
#include "gpu/intel/jit/jit_generator_base.hpp"
#include "gpu/intel/kernel_cache.hpp"
#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

static inline Type convert_dnnl_to_kernel_type(data_type_t type) {
    switch (type) {
        default: assert(!"Unknown type");
        case data_type::f64: return Type::f64;
        case data_type::f32: return Type::f32;
        case data_type::f16: return Type::f16;
        case data_type::bf16: return Type::bf16;
        case data_type::f8_e5m2: return Type::bf8;
        case data_type::f8_e4m3: return Type::hf8;
        case data_type::s32: return Type::s32;
        case data_type::u8: return Type::u8;
        case data_type::s8: return Type::s8;
        case data_type::u4: return Type::u4;
        case data_type::s4: return Type::s4;
        case data_type::undef: return Type::invalid;
    }
}

struct gen_gemm_kernel_desc_t {
    friend struct gen_gemm_kernel_t;

    const GEMMProblem *problem() const { return &problem_; };
    const GEMMStrategy *strategy() const { return &strategy_; };

    const CommonDriverInfo *driver_info() const { return &driver_info_; };
    const EvaluateAuxOutput *aux_params() const { return &aux_params_; };

    compute::scalar_type_t scalar_type() const {
        switch (problem_.Ts) {
            case Type::s4: return compute::scalar_type_t::_int4;
            case Type::u4: return compute::scalar_type_t::_uint4;
            case Type::s8: return compute::scalar_type_t::_char;
            case Type::u8: return compute::scalar_type_t::_uchar;
            case Type::s16: return compute::scalar_type_t::_short;
            case Type::u16: return compute::scalar_type_t::_ushort;
            case Type::s32: return compute::scalar_type_t::_int;
            case Type::u32: return compute::scalar_type_t::_uint;
            case Type::s64: return compute::scalar_type_t::_long;
            case Type::u64: return compute::scalar_type_t::_ulong;
            case Type::bf8: return compute::scalar_type_t::_bfloat8;
            case Type::hf8: return compute::scalar_type_t::_hfloat8;
            case Type::bf16: return compute::scalar_type_t::_bfloat16;
            case Type::f16: return compute::scalar_type_t::_half;
            case Type::f32: return compute::scalar_type_t::_float;
            case Type::f64: return compute::scalar_type_t::_double;
            default: return compute::scalar_type_t::undef;
        }
    }

    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_t &kernel) const;

    serialized_t serialize() const { return serialized_t(problem_, strategy_); }
    compute::gpu_arch_t arch() const { return arch_; }

    const kcatalog::Entry &entry() const {
        assert(entry_ != nullptr);
        return *entry_;
    };

protected:
    compute::gpu_arch_t arch_;
    ngen::HW hw_ = ngen::HW::Unknown;
    int stepping_ = 0;
    GEMMProblem problem_ = {};
    GEMMStrategy strategy_ = {};
    const kcatalog::Entry *entry_ = nullptr;
    EvaluateAuxOutput aux_params_;
    CommonDriverInfo driver_info_;

    /* optional information to fine-tune kernel */
    int m_ = -1, n_ = -1, k_ = -1;
    int eu_count_ = -1;
    bool disable_systolic_ = false;

    status_t transfer_post_ops(gpu_post_ops_t &&post_ops, bool swap_ab);

    status_t finalize(const char *tags);
    void update_driver_info();
};

struct gen_gemm_nocopy_kernel_desc_t : public gen_gemm_kernel_desc_t {
    enum compute_mode {
        mode_default = 0,
        mode_tf32 = 0x1,
        mode_bf16x1 = 0x2,
        mode_w_decomp = 0x4,
        mode_deterministic = 0x8000
    };

    friend void set_mode(compute_mode &mode, compute_mode flag) {
        mode = static_cast<compute_mode>(mode | flag);
    }

    status_t select_kernel(compute::gpu_arch_t arch, int stepping, int eu_count,
            bool has_systolic, compute_mode mode, int batch_dims, bool trans_a,
            bool trans_b, bool trans_co, bool swap_ab, int ao_dims, int bo_dims,
            bool wei_scale_2d, int wei_q2d_group_k, bool c_offset, bool bias,
            sum_ab_t reduce_ab, float alpha, float beta, data_type_t a_type,
            data_type_t b_type, data_type_t c_type, data_type_t ao_type,
            data_type_t bo_type, data_type_t wei_scales_type,
            data_type_t co_type, data_type_t acc_type, int align_a, int align_b,
            int align_c, dim_t m, dim_t n, dim_t k, dim_t lda, dim_t ldb,
            dim_t ldc, dim_t batch, gpu_post_ops_t &&post_ops);
};

struct gen_gemm_xe_systolic_kernel_desc_t : public gen_gemm_kernel_desc_t {
    status_t select_kernel(compute::gpu_arch_t arch, int stepping, int eu_count,
            int batch_dims, bool packed_c, bool trans_co, bool a_offset,
            bool b_offset, bool c_offset, bool bias, float alpha, float beta,
            data_type_t a_type, data_type_t b_type, data_type_t c_type,
            data_type_t ao_type, data_type_t bo_type, data_type_t co_type,
            data_type_t acc_type, dim_t m, dim_t n, dim_t k, dim_t batch,
            int unroll_m, int unroll_n, bool alt, gpu_post_ops_t &&post_ops);

    static void choose_unrolls(compute::gpu_arch_t arch, int eu_count,
            data_type_t a_type, data_type_t b_type, data_type_t c_type, dim_t m,
            dim_t n, dim_t k, dim_t batch, int &unroll_m, int &unroll_n,
            bool &alt);

    static int min_block_k(data_type_t a_type) { return 2048; }
};

struct gen_gemm_kernel_t : public jit_generator_base {

    explicit gen_gemm_kernel_t(const gen_gemm_kernel_desc_t &desc)
        : desc_(desc) {}

    const char *kernel_name() const override { return "gemm_kernel"; }
    xpu::binary_t get_binary(cl_context context, cl_device_id device) override;

    const gen_gemm_kernel_desc_t *desc() const { return &desc_; }

protected:
    const gen_gemm_kernel_desc_t &desc_;
    ngen::NEOInterfaceHandler interface_ {ngen::HW::Unknown};

    void init_interface();
};

} // namespace jit

template <>
struct trivial_key_validator_t<jit::gen_gemm_kernel_desc_t> {
    static bool is_valid(const jit::gen_gemm_kernel_desc_t &) { return true; }
};

template <>
struct trivial_key_validator_t<jit::gen_gemm_nocopy_kernel_desc_t> {
    static bool is_valid(const jit::gen_gemm_nocopy_kernel_desc_t &derived) {
        return trivial_key_validator_t<jit::gen_gemm_kernel_desc_t>::is_valid(
                derived);
    }
};

template <>
struct trivial_key_validator_t<jit::gen_gemm_xe_systolic_kernel_desc_t> {
    static bool is_valid(
            const jit::gen_gemm_xe_systolic_kernel_desc_t &derived) {
        return trivial_key_validator_t<jit::gen_gemm_kernel_desc_t>::is_valid(
                derived);
    }
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
