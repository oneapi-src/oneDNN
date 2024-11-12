/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_OCL_MICRO_SDPA_HPP
#define GPU_OCL_MICRO_SDPA_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/gemm_types.hpp"
#include "common/gemm_utils.hpp"
#include "common/primitive.hpp"
#include "common/sdpa_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/microkernels/shim.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

class mask_iterator {
    int mask_;
    int index_;

public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = int;
    using value_type = int;
    using pointer = value_type *;
    using reference = value_type &;
    mask_iterator() : mask_(0), index_(0) {}
    mask_iterator(int mask) : mask_(mask), index_(0) {
        if ((mask_ & 0x1) == 0) { ++(*this); }
    }
    mask_iterator &begin() { return *this; }
    mask_iterator end() const { return 0; }
    value_type operator*() const { return index_; }
    mask_iterator &operator++() {
        do {
            index_++;
            mask_ >>= 1;
        } while ((mask_ & 0x1) == 0 && mask_ != 0);
        if (mask_ == 0) { index_ = 0; }
        return *this;
    }
    bool operator!=(const mask_iterator &other) const {
        return mask_ != other.mask_ || index_ != other.index_;
    }
};

struct micro_sdpa_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public sdpa_pd_t {
        using sdpa_pd_t::sdpa_pd_t;
        static constexpr int mask_mb_indes = 0;
        static constexpr int mask_q_index = 2;
        static constexpr int mask_k_index = 3;

        DECLARE_COMMON_PD_T("ocl:micro:any", micro_sdpa_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            VDISPATCH_SDPA(
                    utils::everyone_is(4, qry_md()->ndims, key_md()->ndims,
                            val_md()->ndims, dst_md()->ndims),
                    VERBOSE_UNSUPPORTED_TAG);
            if (with_attn_mask()) {
                VDISPATCH_SDPA(
                        attn_mask_md()->ndims == 4, VERBOSE_UNSUPPORTED_TAG);
                VDISPATCH_SDPA(utils::one_of(attn_mask_md()->dims[mask_q_index],
                                       desc()->queries(), 1),
                        VERBOSE_INVALID_BROADCAST, "attn_mask", mask_q_index);
                VDISPATCH_SDPA(
                        attn_mask_md()->dims[mask_k_index] == desc()->keys(),
                        VERBOSE_INVALID_BROADCAST, "attn_mask", mask_k_index);
            }
            VDISPATCH_SDPA(utils::everyone_is(data_type::f16,
                                   qry_md()->data_type, dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SDPA(
                    utils::one_of(key_md()->data_type, f16, u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SDPA(
                    utils::one_of(val_md()->data_type, f16, u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SDPA(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SDPA(desc()->values() == desc()->head_size(),
                    "values does not match head size");

            VDISPATCH_SDPA(qry_md()->dims[1] >= key_md()->dims[1]
                            && qry_md()->dims[1] >= val_md()->dims[1],
                    "number of heads in query tensor(%s) must be greater "
                    "than the number of heads in the key(%s) and value(%s) "
                    "tensors",
                    qry_md()->dims[1], key_md()->dims[1], val_md()->dims[1]);

            int kq_scales_mask = desc()->kq_scales.mask_;
            int kq_zp_mask = desc()->kq_zero_points.get(DNNL_ARG_WEIGHTS);
            if (!desc()->kq_scales.has_default_values()
                    && !desc()->kq_zero_points.has_default_values())
                VDISPATCH_SDPA(kq_scales_mask == kq_zp_mask,
                        "kq scales mask(%d) must equal kq zero point(%d) "
                        "mask",
                        kq_scales_mask, kq_zp_mask);
            VDISPATCH_SDPA(utils::one_of(kq_scales_mask, 0, 1, 3, 11, 15),
                    "unsupported mask for kq matmul(%d). must be 0, 1, 3, 11, "
                    "or 15",
                    kq_scales_mask);
            VDISPATCH_SDPA(utils::one_of(kq_zp_mask, 0, 1, 3, 11, 15),
                    "unsupported mask for kq matmul(%d). must be 0, 1, 3, 11, "
                    "or 15",
                    kq_zp_mask);

            /// NOTE: Limitation of microkernels
            if (utils::one_of(
                        desc()->kq_zero_points.get_data_type(DNNL_ARG_WEIGHTS),
                        s4, u4)) {
                VDISPATCH_SDPA(key_group_size() == 16,
                        "if kq zero points data type is s4 or u4 then the "
                        "group size(%d) must be 16.",
                        key_group_size());
            }

            int vs_scales_mask = desc()->vs_scales.mask_;
            int vs_zp_mask = desc()->vs_zero_points.get(DNNL_ARG_WEIGHTS);
            if (!desc()->vs_scales.has_default_values()
                    && !desc()->vs_zero_points.has_default_values())
                VDISPATCH_SDPA(vs_scales_mask == vs_zp_mask,
                        "vs scales mask(%d) must equal vs zero point(%d) "
                        "mask",
                        vs_scales_mask, vs_zp_mask);
            VDISPATCH_SDPA(utils::one_of(vs_scales_mask, 0, 1, 3, 7, 15),
                    "unsupported mask for vs matmul(%d). must be 0, 1, 3, 7, "
                    "or 15",
                    vs_scales_mask);
            VDISPATCH_SDPA(utils::one_of(vs_zp_mask, 0, 1, 3, 7, 15),
                    "unsupported mask for vs matmul(%d). must be 0, 1, 3, 7, "
                    "or 15",
                    vs_zp_mask);

            /// NOTE: Limitation of microkernels
            if (utils::one_of(
                        desc()->vs_zero_points.get_data_type(DNNL_ARG_WEIGHTS),
                        s4, u4)) {
                VDISPATCH_SDPA(value_group_size() == 16,
                        "if vs zero points data type is s4 or u4 then the "
                        "group size(%d) must be 16.",
                        value_group_size());
            }

            CHECK(init_microkernels(engine));
            return status::success;
        }

        status_t set_default_format(memory_desc_t &md, bool allow_transpose) {
            using namespace format_tag;
            memory_desc_wrapper mdw(md);
            if (mdw.format_any()) return status::unimplemented;
            if (!is_md_gemm_compatible_plain_format(&md))
                return status::unimplemented;
            if (gemm_desc_t::get_trans(md) == dnnl_trans && !allow_transpose)
                return status::unimplemented;
            return status::success;
        }

        status_t set_default_formats() {
            CHECK(set_default_format(desc_.q_desc, false));
            CHECK(set_default_format(desc_.k_desc, true));
            CHECK(set_default_format(desc_.v_desc, false));
            CHECK(set_default_format(desc_.dst_desc, false));
            return status::success;
        }

        const micro::Package &gemm_kq() const { return gemm_kq_; }
        const micro::Package &gemm_vs() const { return gemm_vs_; }

        int sg_size() const { return sg_size_; }

        // Block size for head_size, which must be hard-coded into the kernel.
        int d_max() const {
            int head_size = into<int>(desc()->head_size());
            for (int i = 32; i <= 1024; i *= 2)
                if (head_size <= i) return i;
            return head_size;
        }

        /// If true, dequantize the K tensor using scaling in the KQ matmul
        bool with_key_scales() const {
            return (!desc()->kq_scales.has_default_values());
        }

        /// If true, dequantize the V tensor using scaling in the VS matmul
        bool with_value_scales() const {
            return (!desc()->vs_scales.has_default_values());
        }

        /// If true, dequantize the K tensor with zero points in the KQ matmul
        bool with_key_zp() const {
            return (!desc()->kq_zero_points.has_default_values(
                    DNNL_ARG_WEIGHTS));
        }

        /// If true, dequantize the V tensor with zero points in the VS matmul
        bool with_value_zp() const {
            return (!desc()->vs_zero_points.has_default_values(
                    DNNL_ARG_WEIGHTS));
        }

        /// Returns the data type of the scales tensor for the KQ matmul
        data_type_t key_scales_dt() const {
            return desc()->kq_scales.data_type_;
        }

        /// Returns the data type of the zero points tensor for the KQ matmul
        data_type_t key_zp_dt() const {
            return desc()->kq_zero_points.get_data_type(DNNL_ARG_WEIGHTS);
        }

        /// Returns the data type of the scales tensor for the VS matmul
        data_type_t value_scales_dt() const {
            return desc()->vs_scales.data_type_;
        }

        /// Returns the data type of the zero points tensor for the VS matmul
        data_type_t value_zp_dt() const {
            return desc()->vs_zero_points.get_data_type(DNNL_ARG_WEIGHTS);
        }

        static int scale_group_size(
                const runtime_scales_t &scales, const memory_desc_t &desc) {
            dim_t out = utils::array_product(desc.dims, 4);
            if (scales.has_default_groups()) {
                for (int idx : mask_iterator(scales.mask_)) {
                    out /= desc.dims[idx];
                }
            } else {
                for (int idx : mask_iterator(scales.mask_)) {
                    if (idx < 2) {
                        out /= desc.dims[idx];
                    } else {
                        out /= (desc.dims[idx] / scales.group_dims_[idx - 2]);
                    }
                }
            }
            return static_cast<int>(out);
        }

        static int zp_group_size(
                const zero_points_t &zp, const memory_desc_t &desc) {
            dim_t out = utils::array_product(desc.dims, 4);
            if (zp.has_default_groups(DNNL_ARG_WEIGHTS)) {
                for (int idx : mask_iterator(zp.get(DNNL_ARG_WEIGHTS))) {
                    out /= desc.dims[idx];
                }
            } else {
                auto groups = zp.get_groups(DNNL_ARG_WEIGHTS);
                for (int idx : mask_iterator(zp.get(DNNL_ARG_WEIGHTS))) {
                    if (idx < 2) {
                        out /= desc.dims[idx];
                    } else {
                        out /= (desc.dims[idx] / groups[idx - 2]);
                    }
                }
            }
            return static_cast<int>(out);
        }

        // Returns the group size for the quantization parameters for the KQ matmul
        int key_group_size() const {
            int out = 0;
            if (with_key_scales())
                out = scale_group_size(desc()->kq_scales, *key_md());
            else if (with_key_zp()) {
                out = zp_group_size(desc()->kq_zero_points, *key_md());
            }
            return out;
        }

        // Returns the group size for the quantization parameters for the VS matmul
        int value_group_size() const {
            int out = 0;
            if (with_value_scales())
                out = scale_group_size(desc()->vs_scales, *val_md());
            else if (with_value_zp()) {
                out = zp_group_size(desc()->vs_zero_points, *val_md());
            }
            return out;
        }

        compute::gpu_arch_t arch() const { return arch_; }

    private:
        micro::Package gemm_kq_, gemm_vs_;
        int sg_size_ = 0;
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

        status_t init_microkernels(impl::engine_t *engine);
    };

    status_t init(impl::engine_t *engine) override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute(const exec_ctx_t &ctx) const override;

    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
