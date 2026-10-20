/*******************************************************************************
* Copyright 2026 openKylin community
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

#ifndef CPU_RV64_JIT_UNI_RESAMPLING_KERNEL_HPP
#define CPU_RV64_JIT_UNI_RESAMPLING_KERNEL_HPP

#include "common/c_types_map.hpp"

#include "cpu/cpu_resampling_pd.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

template <cpu_isa_t isa, data_type_t d_type>
struct jit_uni_resampling_kernel_t : public jit_generator_t {

    jit_uni_resampling_kernel_t(const jit_resampling_conf_t &conf);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_resampling_kernel_t)

    // Populate conf and select the memory layout (nspc/ncsp); returns
    // status::unimplemented for any layout/dtype/alg this kernel does not handle.
    static status_t init_conf(
            jit_resampling_conf_t &conf, const resampling_pd_t *pd);

    void operator()(const jit_resampling_args_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    void generate_f32();
    // Plain ncsp f32: one output row per call, vectorized along OW.
    void generate_f32_w();
    // f16 (Zvfh) and bf16 (Zvfbfmin) share one widen-compute-narrow body; only
    // the convert mnemonics differ.
    void generate_xf16();

    jit_resampling_conf_t conf_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
