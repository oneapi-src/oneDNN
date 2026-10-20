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

#ifndef CPU_RV64_JIT_UNI_RESAMPLING_HPP
#define CPU_RV64_JIT_UNI_RESAMPLING_HPP

#include <memory>
#include <vector>

#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_resampling_pd.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/rv64/jit_uni_resampling_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

template <cpu_isa_t isa>
struct jit_uni_resampling_fwd_t : public primitive_t {
    static constexpr data_type_t d_type = (isa == zvfh) ? data_type::f16
            : (isa == zvfbfwma)                         ? data_type::bf16
                                                        : data_type::f32;
    using data_t = typename prec_traits_t<d_type>::type;

    struct pd_t : public cpu_resampling_fwd_pd_t {
        using cpu_resampling_fwd_pd_t::cpu_resampling_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", conf_.isa, ""),
                jit_uni_resampling_fwd_t);

        status_t init(const engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            VDISPATCH_RESAMPLING(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_RESAMPLING(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_RESAMPLING(utils::everyone_is(d_type, src_md()->data_type,
                                         dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_RESAMPLING(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_RESAMPLING(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_RESAMPLING(attr()->has_default_values(
                                         sm::post_ops, dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_ATTR);
            // Resolve format_any src1 against dst before post_ops_ok() reads
            // their layout.
            VDISPATCH_RESAMPLING(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_RESAMPLING(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);

            // init_conf fills conf_ and rejects any layout/alg/dtype it cannot
            // serve.
            const status_t conf_status
                    = jit_uni_resampling_kernel_t<isa, d_type>::init_conf(
                            conf_, this);
            VDISPATCH_RESAMPLING(
                    conf_status == status::success, VERBOSE_UNSUPPORTED_TAG);
            init_scratchpad();
            return status::success;
        }

        jit_resampling_conf_t conf_ = {};

    private:
        // The kernel reads the binary rhs origins through a pointer array; book
        // it here so execute() never allocates. post_ops_ok() caps binaries at
        // one, so the array holds a single entry.
        void init_scratchpad() {
            if (!conf_.fuse_binary) return;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.template book<const void *>(
                    memory_tracking::names::key_binary_post_ops_rhs_ptrs, 1);
        }

        // Any number of eltwise ops, at most one binary and one sum. The
        // binary is f32-only (there is no xf16 binary path) and limited to the
        // broadcasts the driver positions host-side: scalar, per-oc and
        // full-dst. Anything else falls back to simple/ref_resampling.
        bool post_ops_ok() const {
            const auto &po = attr()->post_ops_;
            if (po.has_default_values()) return true;
            // The injector rejects sum, so validate it separately.
            int n_sum = 0;
            post_ops_t po_no_sum;
            for (int i = 0; i < po.len(); i++) {
                const auto &e = po.entry_[i];
                if (e.kind != primitive_kind::sum) {
                    po_no_sum.entry_.push_back(e);
                    continue;
                }
                if (++n_sum > 1) return false;
                // A non-zero zero-point needs a subtract the kernel does not
                // emit; sum.dt must match dst on read-back.
                if (e.sum.zero_point != 0) return false;
                if (!utils::one_of(e.sum.dt, data_type::undef, d_type))
                    return false;
            }
            const memory_desc_wrapper dst_d(dst_md());
            // The kernel positions every rhs host-side, so it advertises only
            // the broadcasts it can place. n_vaux = 3: v_aux3/v_aux4 alias the
            // binary rhs scratch, so the heavy eltwise algs stay out.
            if (!injector::post_ops_ok(injector::post_ops_ok_args_t(isa,
                        {injector::eltwise, injector::binary}, po_no_sum,
                        &dst_d, false /*sum_at_pos_0_only*/,
                        false /*sum_requires_scale_one*/,
                        true /*sum_requires_zp_zero*/,
                        true /*sum_requires_same_params*/,
                        bcast_set_t {broadcasting_strategy_t::scalar,
                                broadcasting_strategy_t::per_oc,
                                broadcasting_strategy_t::per_oc_spatial,
                                broadcasting_strategy_t::no_broadcast},
                        /*n_vaux=*/3)))
                return false;
            int n_binary = 0;
            for (int i = 0; i < po.len(); i++) {
                if (!po.entry_[i].is_binary()) continue;
                if (++n_binary > 1) return false;
                if (d_type != data_type::f32) return false;
                const auto &b = po.entry_[i].binary;
                if (b.src1_desc.data_type != data_type::f32) return false;
                const memory_desc_wrapper s1(b.src1_desc);
                const bool scalar = s1.nelems(true) == 1;
                bool per_oc = dst_d.ndims() >= 2 && s1.ndims() == dst_d.ndims()
                        && s1.dims()[1] == dst_d.dims()[1] && s1.is_dense(true);
                for (int k = 0; per_oc && k < dst_d.ndims(); k++)
                    if (k != 1 && s1.dims()[k] != 1) per_oc = false;
                const bool full = s1.similar_to(dst_d, true, false);
                if (!scalar && !per_oc && !full) return false;
            }
            return true;
        }
    };

    jit_uni_resampling_fwd_t(const pd_t *apd);
    ~jit_uni_resampling_fwd_t() override;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_resampling_kernel_t<isa, d_type>> kernel_;
    // conf_.w_vec: per-OW source column byte offsets and 1D linear weights
    // (index 0 left, 1 right), shared by every output row.
    std::vector<int32_t> w_idx_[2];
    std::vector<float> w_wei_[2];
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
