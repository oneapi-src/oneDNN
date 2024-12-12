/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_CONV_ZERO_OUT_HPP
#define GPU_INTEL_JIT_CONV_ZERO_OUT_HPP

#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/codegen/register_scope.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/ngen/ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class zero_out_kernel_desc_t : public kernel_desc_base_t {
public:
    static const size_t bytes_per_thr;

    zero_out_kernel_desc_t() = default;
    zero_out_kernel_desc_t(int regs, int simd, bool dpas)
        : regs_(regs), simd_(simd), dpas_(dpas) {}
    std::string kernel_name() const override;
    exec_config_t exec_cfg(const impl::engine_t *engine) const override;
    bool with_dpas() const override { return dpas_; }
    compute::range_t local_range() const override;
    void init_kernel_iface(kernel_iface_t &kernel_iface) const override;
    void init_kernel_info(kernel_info_t &kernel_info,
            const kernel_params_base_t &params,
            const impl::engine_t *engine) const override;
    status_t create_kernel(compute::kernel_t &kernel,
            gpu_primitive_t *primitive, impl::engine_t *engine) const override;
    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_t &kernel) const;
    serialized_t serialize() const override;
    static zero_out_kernel_desc_t deserialize(const serialized_t &s);

    static compute::nd_range_t nd_range(int simd, size_t size);

private:
    int regs_ = 0;
    int simd_ = 0;
    bool dpas_ = false;
};

class zero_out_kernel_params_t : public kernel_params_base_t {
public:
    zero_out_kernel_params_t() = default;
    zero_out_kernel_params_t(size_t size) : size(size) {}

    size_t size = 0;
};

template <ngen::HW hw = ngen::HW::Unknown>
class zero_out_kernel_t : public ir_kernel_t<hw> {
public:
    IR_KERNEL_FORWARD(hw)

    zero_out_kernel_t(const exec_config_t &exec_cfg,
            const kernel_info_t &kernel_info, bool require_dpas,
            const impl::engine_t *engine)
        : zero_out_kernel_t<hw>(zero_out_kernel_desc_t(exec_cfg.regs(),
                                        exec_cfg.simd(), require_dpas),
                engine) {}

    zero_out_kernel_t(
            const kernel_desc_base_t &_desc, const impl::engine_t *engine)
        : ir_kernel_t<hw>(_desc, engine, {GENERATOR_NAME, GENERATOR_LINE}) {
        setup_interface();
        generate_prologue();

        int simd_size = getSIMD();
        bool use_lsc = (hw >= ngen::HW::XeHPG);

        auto size = getArgument(kernel_iface().arg_name(0));
        auto ptr = getArgument(kernel_iface().arg_name(1));
        auto global_id = ra_.template alloc_sub<uint32_t>();
        auto off0 = ra_.template alloc_sub<uint32_t>();
        const int bytes_per_thr
                = into<int>(zero_out_kernel_desc_t::bytes_per_thr);

        mul(1, global_id, r0.ud(1), getLocalSize(0).uw());
        add(1, global_id, global_id, getLocalID(0));
        shl(1, off0, global_id, math::ilog2q(bytes_per_thr / simd_size));

        int grf_size = ngen::GRF::bytes(hw);
        int bytes_per_store = 16;
        int uw_size = sizeof(uint16_t);
        int ud_size = sizeof(uint32_t);
        int uq_size = sizeof(uint64_t);

        auto zero = ra_.alloc_range(bytes_per_store * ud_size / grf_size);
        auto off_vec = ra_.alloc_range(bytes_per_thr * ud_size / grf_size);
        auto off_vec_q_strided
                = ra_.alloc_range(bytes_per_thr * uq_size / grf_size);
        auto ptr_vec = ra_.alloc_range(bytes_per_thr * uq_size / grf_size);

        for (int i = 0; i < bytes_per_store * ud_size; i += 64) {
            auto z = get_subregister(hw, ngen::DataType::ud, zero, i);
            mov(16, z, 0);
        }

        auto idx_vec = ra_.alloc().uw(0);
        mov(8, idx_vec(1), ngen::Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        mov(8, idx_vec(2), idx_vec(1));
        for (int i = 16; i < grf_size / uw_size; i += 16) {
            mov(8, idx_vec.uw(i)(2), idx_vec(2));
        }

        reg_buf_t dst, src0, src1;
        for (int i = 0; i < bytes_per_thr; i += 8) {
            auto off_sub_vec
                    = get_subregister(hw, ngen::DataType::ud, off_vec, i)(1);
            this->eadd3(8, ngen_operand_t(reg_buf_data_t(dst, off_sub_vec)),
                    ngen_operand_t(reg_buf_data_t(
                            src1, idx_vec.uw((i % grf_size) * 2)(2))),
                    ngen_operand_t(reg_buf_data_t(src0, off0)),
                    ngen_operand_t(i));
            auto ptr_sub_vec
                    = get_subregister(hw, ngen::DataType::uq, ptr_vec, i)(1);
            auto off_sub_vec_q_strided = get_subregister(
                    hw, ngen::DataType::ud, off_vec_q_strided, i * 2)(2);
            emov(8, off_sub_vec_q_strided, off_sub_vec);
            eadd(8, ptr_sub_vec, ptr, off_sub_vec_q_strided);
        }

        for (int i = 0; i < bytes_per_thr; i += bytes_per_store) {
            auto off_sub_vec
                    = get_subregister(hw, ngen::DataType::ud, off_vec, i)(1);
            cmp(16 | lt | f0[0], off_sub_vec, size);
            auto h = get_subregister(hw, ngen::DataType::uq, ptr_vec, i);
            if (use_lsc) {
                std::unique_ptr<ngen::DataSpecLSC> lsc_spec;
                lsc_spec = utils::make_unique<ngen::DataSpecLSC>(
                        ngen::scattered(ngen::DataSizeLSC::D8U32, 1));
                store.ugm(16 | f0[0], *lsc_spec, A64, h, zero[0]);
            } else {
                store(16 | f0[0], ngen::scattered_byte(), A64, h, zero[0]);
            }
        }

        generate_epilogue();
    }
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
