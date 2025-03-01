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

#ifndef GPU_INTEL_OCL_MULTI_CONCAT_HPP
#define GPU_INTEL_OCL_MULTI_CONCAT_HPP

#include "common/concat.hpp"
#include "common/engine.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"
#include "common/stream.hpp"
#include "gpu/gpu_concat_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct multi_concat_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_concat_pd_t {
        static constexpr int batch_failure = -1;

        using gpu_concat_pd_t::gpu_concat_pd_t;

        pd_t(const pd_t &rhs) = default;
        ~pd_t() override = default;

        DECLARE_CONCAT_PD_T("multi:any", multi_concat_t);

        int max_batch_size() const {
            if (n_inputs() > 64) return 64;
            if (n_inputs() > 16) return 16;
            return batch_failure;
        }

        status_t init(impl::engine_t *engine) {
            VDISPATCH_CONCAT(max_batch_size() != batch_failure,
                    VERBOSE_SKIP_PRIMITIVE_IMPL);
            VDISPATCH_CONCAT(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONCAT_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);

            auto n_batches = utils::div_up(n_inputs(), max_batch_size());
            concat_pds_.resize(n_batches);
            dst_chunk_mds_.resize(n_batches);

            dim_t concat_dim_offset = 0;
            const auto ndims = dst_md()->ndims;
            status_t status = status::success;
            for (int i = 0; i < n_batches; ++i) {
                const auto src_offset = max_batch_size() * i;
                const auto remaining = n_inputs() - src_offset;
                const auto batch_size = std::min(max_batch_size(), remaining);
                dim_t batch_width = 0;
                dims_t dims, offsets = {0};
                utils::array_copy(dims, dst_md()->dims, ndims);
                for (int j = 0; j < batch_size; ++j) {
                    const auto &src = src_md(src_offset + j);
                    batch_width += src->dims[concat_dim_];
                }
                dims[concat_dim_] = batch_width;
                offsets[concat_dim_] = concat_dim_offset;
                status = memory_desc_init_submemory(
                        dst_chunk_mds_[i], *dst_md(), dims, offsets);
                if (status != status::success) {
                    concat_pds_.clear();
                    dst_chunk_mds_.clear();
                    VDISPATCH_CONCAT(
                            false, VERBOSE_DESC_CREATION_FAIL, "dst submemory");
                }
                status = concat_primitive_desc_create(concat_pds_[i], engine,
                        &dst_chunk_mds_[i], batch_size, concat_dim_,
                        src_md(src_offset), attr());
                if (status != status::success) {
                    concat_pds_.clear();
                    dst_chunk_mds_.clear();
                    VDISPATCH_CONCAT(
                            false, VERBOSE_PRIMITIVE_CREATION_FAIL, "concat");
                }
                concat_dim_offset += batch_width;
            }
            return status;
        }

        std::vector<std::shared_ptr<primitive_desc_t>> concat_pds_;
        std::vector<memory_desc_t> dst_chunk_mds_;
    };

    status_t init(impl::engine_t *engine) override {
        const auto &pds = pd()->concat_pds_;
        const size_t n = pds.size();
        concats_.resize(n);
        for (size_t i = 0; i < n; ++i)
            CHECK(create_nested_primitive(concats_[i], pds[i], engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        const auto n = pd()->n_inputs();
        const auto max_batch_size = pd()->max_batch_size();
        if (max_batch_size == pd_t::batch_failure) return status::runtime_error;

        auto execute_concat
                = [&](const std::shared_ptr<impl::primitive_t> &concat,
                          int c_num, int n_inputs) {
                      exec_args_t r_args;
                      const auto arg_offset = DNNL_ARG_MULTIPLE_SRC;
                      for (int i = 0; i < n_inputs; ++i)
                          r_args[arg_offset + i] = ctx.args().at(
                                  arg_offset + max_batch_size * c_num + i);
                      r_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);
                      exec_ctx_t r_ctx(ctx, std::move(r_args));

                      r_ctx.set_scratchpad_grantor(ctx.grantor_handle());
                      return concat->execute(r_ctx);
                  };

        const auto n_batches = utils::div_up(n, max_batch_size);
        for (int i = 0; i < n_batches; ++i) {
            const auto remaining = n - max_batch_size * i;
            const auto batch_size = std::min(max_batch_size, remaining);
            CHECK(execute_concat(concats_[i], i, batch_size));
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::shared_ptr<impl::primitive_t>> concats_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
