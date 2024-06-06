/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/intel/jit/v2/conv/gen_convolution.hpp"

#include <iostream>
#include <utility>

#include "common/impl_registration.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/conv/gen_convolution.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/jit/v2/conv/bridge.hpp"
#include "gpu/intel/jit/v2/conv/ir_builder.hpp"
#include "gpu/intel/jit/v2/conv/kernel.hpp"
#include "gpu/intel/jit/v2/conv/plan_preset.hpp"
#include "gpu/intel/jit/v2/conv/plan_registry.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

void maybe_init_layout(
        memory_desc_t &md, const layout_raw_tag_t &_tag, bool remove_a_dim) {
    if (md.format_kind != format_kind::any) return;
    auto tag = _tag;
    int non_spatial_ndims = tag.ndims() - 3;
    if (remove_a_dim) {
        tag.remove_dim('a');
        non_spatial_ndims--;
    }
    while (tag.ndims() > md.ndims) {
        tag.remove_dim('a' + non_spatial_ndims);
    }
    jit::layout_t layout(md, tag.str(), /*do_normalize=*/false);
    md = layout.to_dnnl(md.dims);
}

status_t init_layouts(const kernel_desc_t &desc, convolution_pd_t *pd) {
    auto &src_md = *const_cast<memory_desc_t *>(pd->invariant_src_md());
    auto &wei_md = *const_cast<memory_desc_t *>(pd->invariant_wei_md());
    auto &dst_md = *const_cast<memory_desc_t *>(pd->invariant_dst_md());
    maybe_init_layout(src_md, desc.src_tag.raw_tag(), false);
    maybe_init_layout(wei_md, desc.wei_tag.raw_tag(), !pd->with_groups());
    maybe_init_layout(dst_md, desc.dst_tag.raw_tag(), false);
    return status::success;
}

class gen_convolution_t {
public:
    template <typename T>
    static bool is_supported(T *pd, prop_kind_t prop) {
        bool enable_conv_v2 = gpu_utils::dev_getenv("enable_conv_v2", false);
        if (!enable_conv_v2) return false;

        // Non-trivial strides produces non-linear offset arithmetic which is
        // not yet supported.
        if (pd->is_bwd_d()
                && !(pd->KSW() == 1 && pd->KSH() == 1 && pd->KSD() == 1))
            return false;

        if (pd->with_bias()) return false;
        if (!pd->attr()->has_default_values()) return false;
        return true;
    }

    template <typename T>
    static status_t init_pd(T *pd, impl::engine_t *engine, prop_kind_t prop) {
        if (!is_supported(pd, prop)) return status::unimplemented;

        using compute::compute_engine_t;
        auto *compute_engine = utils::downcast<compute_engine_t *>(engine);
        if (!compute_engine->mayiuse_ngen_kernels())
            return status::unimplemented;
        if (!pd->set_default_alg_kind(alg_kind::convolution_direct))
            return status::unimplemented;

        pd->exec_plan = std::make_shared<exec_plan_t>();
        CHECK(init_exec_plan(*pd->exec_plan, pd, engine));
        return status::success;
    }

    gen_convolution_t() = default;

    template <typename T>
    status_t init(T *primitive, impl::engine_t *engine) {
        auto &exec_plan = *primitive->pd()->exec_plan;
        CHECK(exec_plan.create_kernels(kernels_, primitive, engine));
        return status::success;
    }

    template <typename T>
    status_t execute(const T *primitive, const exec_ctx_t &ctx) const {
        auto &exec_plan = *primitive->pd()->exec_plan;
        return exec_plan.execute(primitive, ctx, kernels_);
    }

private:
    template <typename T>
    static status_t init_exec_plan(
            exec_plan_t &exec_plan, T *pd, impl::engine_t *engine) {
        std::shared_ptr<kernel_desc_base_t> desc;
        std::shared_ptr<kernel_params_base_t> params;
        CHECK(init_conv(desc, params, pd, engine));
        exec_plan.add_kernel(desc, params);
        return status::success;
    }

    static status_t init_conv(std::shared_ptr<kernel_desc_base_t> &desc,
            std::shared_ptr<kernel_params_base_t> &params, convolution_pd_t *pd,
            impl::engine_t *engine) {
        auto prb = to_problem(pd, engine);
        kernel_desc_t _desc;
        kernel_params_t _params;
        if (plan_preset_t::instance().is_set()) {
            _desc = plan_preset_t::instance().get();
            _desc.hw = hw_t(engine);
            _desc.spec_reqs.specialize(prb);
            {
                ir_utils::ir_check_log_level_t check_level(ir_utils::LOG_FATAL);
                auto plan = create_conv_plan_and_finalize_desc(_desc);
            }
        } else {
            auto &registry = const_plan_registry();
            _desc = registry.find_best(prb);
            _desc.spec_reqs.specialize(prb);
        }
        if (_desc.is_empty()) return status::unimplemented;
        ir_assert(ir_check_fatal(_desc.fits(prb)));
        CHECK(init_layouts(_desc, pd));
        _params.prb = prb;
        desc = std::make_shared<kernel_desc_t>(_desc);
        params = std::make_shared<kernel_params_t>(_params);
        return status::success;
    }

    std::vector<compute::kernel_t> kernels_;
};

status_t gen_convolution_fwd_t::pd_t::init(impl::engine_t *engine) {
    return gen_convolution_t::init_pd(this, engine, prop_kind::forward);
}

status_t gen_convolution_fwd_t::init(impl::engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_convolution_bwd_data_t::pd_t::init(impl::engine_t *engine) {
    return gen_convolution_t::init_pd(this, engine, prop_kind::backward_data);
}

status_t gen_convolution_bwd_data_t::init(impl::engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_convolution_bwd_weights_t::pd_t::init(impl::engine_t *engine) {
    return gen_convolution_t::init_pd(
            this, engine, prop_kind::backward_weights);
}

status_t gen_convolution_bwd_weights_t::init(impl::engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
