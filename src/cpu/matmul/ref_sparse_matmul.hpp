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

#ifndef CPU_MATMUL_REF_SPARSE_MATMUL_HPP
#define CPU_MATMUL_REF_SPARSE_MATMUL_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

struct ref_sparse_matmul_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_sparse_matmul_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));

            VDISPATCH_MATMUL(wei_d.is_sparse_desc() || src_d.is_sparse_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(wei_d.is_sparse_desc() ^ src_d.is_sparse_desc(),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            VDISPATCH_MATMUL(IMPLICATION(src_d.is_sparse_desc(),
                                     utils::one_of(src_d.encoding(),
                                             sparse_encoding::csr,
                                             sparse_encoding::coo)),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(IMPLICATION(wei_d.is_sparse_desc(),
                                     utils::one_of(wei_d.encoding(),
                                             sparse_encoding::csr,
                                             sparse_encoding::coo)),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);

            VDISPATCH_MATMUL(
                    utils::everyone_is(f16, src_type, wei_type, dst_type)
                            || utils::everyone_is(
                                    f32, src_type, wei_type, dst_type),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            if (src_d.is_sparse_desc()) {
                sparse_mem_encoding = src_d.encoding();
                VDISPATCH_MATMUL(
                        IMPLICATION(sparse_mem_encoding == sparse_encoding::coo,
                                s32 == src_d.metadata_type(0)),
                        VERBOSE_UNSUPPORTED_SPARSE_CFG);
                VDISPATCH_MATMUL(
                        IMPLICATION(sparse_mem_encoding == sparse_encoding::csr,
                                utils::everyone_is(s32, src_d.metadata_type(0),
                                        src_d.metadata_type(1))),
                        VERBOSE_UNSUPPORTED_SPARSE_CFG);
            }
            if (wei_d.is_sparse_desc()) {
                sparse_mem_encoding = wei_d.encoding();
                VDISPATCH_MATMUL(
                        IMPLICATION(sparse_mem_encoding == sparse_encoding::coo,
                                s32 == wei_d.metadata_type(0)),
                        VERBOSE_UNSUPPORTED_SPARSE_CFG);

                VDISPATCH_MATMUL(
                        IMPLICATION(sparse_mem_encoding == sparse_encoding::csr,
                                utils::everyone_is(s32, wei_d.metadata_type(0),
                                        wei_d.metadata_type(1))),
                        VERBOSE_UNSUPPORTED_SPARSE_CFG);
            }

            VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_MATMUL(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(formats_ok(src_d, wei_d), VERBOSE_UNSUPPORTED_TAG);

            init_scratchpad();
            return status::success;
        }

        bool formats_ok(const memory_desc_wrapper &src_d,
                const memory_desc_wrapper &wei_d) const {
            if (!memory_desc_wrapper(dst_md()).matches_one_of_tag(
                        format_tag::ab))
                return false;
            if (src_d.is_sparse_desc())
                return wei_d.matches_one_of_tag(format_tag::ab);
            if (wei_d.is_sparse_desc())
                return src_d.matches_one_of_tag(format_tag::ab);
            return false;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper wei_d(weights_md());

            if (sparse_mem_encoding == sparse_encoding::coo) {
                auto scratchpad = scratchpad_registry().registrar();
                const bool is_wei_sparse = wei_d.is_sparse_desc();
                const auto ptr_size
                        = src_d.dims()[static_cast<int>(is_wei_sparse)] + 1;
                scratchpad.template book<int32_t>(
                        key_matmul_sparse_tmp_ptr, ptr_size);
            }
        }

        sparse_encoding_t sparse_mem_encoding = sparse_encoding::undef;
    };

    ref_sparse_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    // COO sparse encodings are converted to CSR format by
    // compressing the respective row indices into CSR pointers.
    void cvt_coo_indices_to_csr_pointers(const int32_t *indices,
            int32_t *pointers, const int nnz, const int nrows) const;

    // Executes the matrix mutiplication, C = A x B where one of the input
    // matrices is dense. Operation indices are determined depending on
    // whether the mulitplier or multiplicand is dense
    void run_csr_kernel(const void *dmat, const void *values,
            const int32_t *indices, const int32_t *pointers, void *res,
            const dim_t M, const dim_t N, const dim_t K,
            const data_type_t mm_dt, bool is_src_sparse) const;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
