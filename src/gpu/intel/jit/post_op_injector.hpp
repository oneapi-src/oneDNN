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

#ifndef GPU_INTEL_JIT_POST_OP_INJECTOR_HPP
#define GPU_INTEL_JIT_POST_OP_INJECTOR_HPP

#include "common/primitive_attr.hpp"
#include "gpu/intel/gpu_post_ops.hpp"
#include "gpu/intel/jit/eltwise_injector.hpp"
#include "gpu/intel/jit/generator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

inline bool post_op_injector_is_supported(
        const post_ops_t &post_ops, bool skip_sum) {
    bool is_supported = true;
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        const auto &po = post_ops.entry_[idx];
        if (po.is_binary())
            is_supported &= false;
        else if (po.is_convolution())
            is_supported &= false;
        else if (po.is_eltwise())
            is_supported &= eltwise_injector_f32_is_supported(po.eltwise.alg);
        else if (po.is_sum(false, false))
            is_supported &= skip_sum;
    }
    return is_supported;
}

template <gpu_gen_t hw>
struct post_op_injector_t {
    post_op_injector_t(generator_t<hw> *host, data_type_t accumulator_type,
            const post_ops_t &post_ops, int eu_count,
            const ngen::GRFRange &scratch = ngen::GRFRange(),
            bool is_fwd = true)
        : is_fwd_(is_fwd), scratch_(scratch) {
        assert(accumulator_type == data_type_t::dnnl_f32);
        workers_.reserve(post_ops.len());
        for (int idx = 0; idx < post_ops.len(); ++idx) {
            const auto &po = post_ops.entry_[idx];
            if (po.is_eltwise())
                workers_.emplace_back(host, po.eltwise.alg, po.eltwise.alpha,
                        po.eltwise.beta, po.eltwise.scale, eu_count, scratch,
                        is_fwd);
        }
    }

    post_op_injector_t(generator_t<hw> *host, data_type_t accumulator_type,
            const gpu_post_ops_t &post_ops, int eu_count,
            const ngen::GRFRange &scratch = ngen::GRFRange(),
            bool is_fwd = true)
        : is_fwd_(is_fwd), scratch_(scratch) {
        assert(accumulator_type == data_type_t::dnnl_f32);
        workers_.reserve(post_ops.len());
        for (auto &po : post_ops) {
            if (po.is_eltwise()) {
                auto &e = po.as_eltwise();
                workers_.emplace_back(host, e.alg, e.alpha, e.beta, e.scale,
                        eu_count, scratch, is_fwd);
            }
        }
    }

    int min_scratch_regs();
    int preferred_scratch_regs();
    void set_scratch(const ngen::GRFRange &scratch);

    void compute(const ngen::GRF &reg) { compute(reg - reg); }
    void compute(const ngen::GRFRange &regs);

private:
    std::vector<eltwise_injector_f32_t<hw>> workers_;
    bool is_fwd_;
    ngen::GRFRange scratch_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_JIT_POST_OP_INJECTOR_HPP
