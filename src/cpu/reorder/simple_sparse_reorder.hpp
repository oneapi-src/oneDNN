/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef CPU_REORDER_SIMPLE_SPARSE_REORDER_HPP
#define CPU_REORDER_SIMPLE_SPARSE_REORDER_HPP

#include <assert.h>

#include "simple_reorder.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// The following cases can be covered:
//
// - sparse_tag -> sparse_tag
// - encoding -> encoding
//
// - sparse_tag -> dense_tag
// - dense_tag -> sparse_tag
//
// - sparse_tag -> encoding
// - encoding -> sparse_tag
//
// - dense_tag -> encoding
// - encoding -> dense_tag
#define SIMPLE_SPARSE_REORDER_TEMPL_DECL \
    impl::data_type_t type_i, format_tag_t fmt_i, \
            impl::data_type_t type_o, format_tag_t fmt_o, \
            bool order_keep

#define SIMPLE_SPARSE_REORDER_TEMPL_CALL \
    type_i, fmt_i, type_o, fmt_o, order_keep

// TODO: move common code to reorder_utils.hpp.
namespace sparse_spec {
struct reference {};
} // namespace sparse_spec

namespace sparse_inputs_order {
constexpr bool keep = true;
constexpr bool reverse = false;
constexpr bool any = keep;
} // namespace sparse_inputs_order

template <SIMPLE_SPARSE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_sparse_reorder_impl {};

namespace {
template <typename T>
constexpr bool is_format_tag(T) {
    return std::is_same<T, format_tag_t>::value ? true : false;
}
} // namespace

using namespace data_type;

// TODO: think about combining compression reorders with sparse reorders.
/* specific reorders: IP compression */
template <SIMPLE_SPARSE_REORDER_TEMPL_DECL>
struct simple_sparse_reorder_impl<SIMPLE_SPARSE_REORDER_TEMPL_CALL,
        typename utils::enable_if<(is_format_tag(fmt_i)
                                          && (fmt_i == format_tag::oi
                                                  || fmt_i == format_tag::io))
                        && (is_format_tag(fmt_o)
                                && fmt_o == format_tag::OI16i64o4i),
                sparse_spec::reference>::type> {

    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace data_type;
        using namespace utils;
        if (input_d.has_runtime_dims_or_strides()) return false;
        const size_t D_mask = utils::array_product(
                input_d.dims(), math::ilog2q(attr->output_scales_.mask_ + 1));
        const size_t oc = (input_d.dims()[0]);

        return output_d.matches_tag(fmt_o) && input_d.matches_tag(fmt_i)
                && output_d.is_sparse_desc()
                && output_d.sparse_desc().encoding == sparse_encoding::packed
                && one_of(input_d.data_type(), f32, s8)
                && output_d.data_type() == s8 && (D_mask == 1 || D_mask == oc);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        auto input = CTX_IN_MEM(const data_t<type_i> *, DNNL_ARG_FROM);
        auto output = CTX_OUT_MEM(data_t<type_o> *, DNNL_ARG_TO);

        const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd->src_md());
        const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd->dst_md());

        const auto &input_dims = input_d.dims();
        const auto &padded_dims = output_d.padded_dims();
        constexpr int i_outer_blksize = 16;
        constexpr int i_blksize = i_outer_blksize * 4;
        constexpr int o_blksize = 64;

        const int OC = input_dims[0];
        const int NB_OC = padded_dims[0] / o_blksize;
        const int IC = input_dims[1];
        const int NB_IC = padded_dims[1] / i_blksize;
        const int plain_o_stride = input_d.blocking_desc().strides[0];
        const int plain_i_stride = input_d.blocking_desc().strides[1];
        size_t offset = padded_dims[0] * padded_dims[1];

        int total_blocks = offset / 4096;
        int16_t *comp_tile_len_ptr = reinterpret_cast<int16_t *>(output);
        int comp_tile_len_index = 0;
        int cl_length = 0;
        // TODO: why 2 / 64?
        // Wasting memory space due to allocation a buffer for the whole tensor?
        int output_offset = ceil((float)total_blocks * 2 / 64.0);

        size_t offset_2 = static_cast<size_t>(ceil((float)total_blocks * 2 / 64.0)) * 64;
        uint64_t *bitmask_ptr = reinterpret_cast<uint64_t *>(output + offset + offset_2);

        auto outp = &output[output_d.blk_off(0, 0, 0, 0) + output_offset * 64];

        // TODO: add threading.
        for (int O = 0; O < NB_OC; O++) {
            for (int I = 0; I < NB_IC; I++) {
                auto inp
                        = &input[input_d.blk_off(o_blksize * O, i_blksize * I)];
                const int oc_block = nstl::min(o_blksize, OC - O * o_blksize);
                const int ic_block = nstl::min(i_blksize, IC - I * i_blksize);
                int non_zeros = 0, zeros = 0;
                int bitmask_idx = (O * NB_IC + I) * i_blksize;
                comp_tile_len_ptr[comp_tile_len_index] = cl_length;

                for (int ic_base = 0; ic_base < ic_block;
                        ic_base += 4) { // 64, steps of 4
                    bitmask_ptr[bitmask_idx] = 0;
                    int bit = 0;
                    int count = 0;
                    for (int oc = 0; oc < oc_block; oc++) { // 64
                        if (count % 64 == 0) {
                            bitmask_ptr[bitmask_idx] = 0;
                            bit = 0;
                        }
                        int plain_off = oc * plain_o_stride
                                + ic_base * plain_i_stride;
                        int ic_block_here = nstl::min(4, ic_block - ic_base);
                        for (int ic = 0; ic < ic_block_here; ic++) { // 4
                            data_t<type_o> o = inp[plain_off];
                            if (o != 0) {
                                *outp++ = o;
                                bitmask_ptr[bitmask_idx] |= (1UL << bit);
                                non_zeros++;
                            } else {
                                zeros++;
                            }
                            plain_off += plain_i_stride;
                            bit++;
                            count++;
                        }
                        if (count % 64 == 0) { bitmask_idx++; }
                    }
                }
                int16_t cl = (int16_t)ceil(non_zeros / 64.0);
                comp_tile_len_index++;
                cl_length = comp_tile_len_ptr[comp_tile_len_index - 1] + cl;
                int unsed_bytes_in_cl = 64 - (non_zeros % 64);
                if (unsed_bytes_in_cl == 64) { unsed_bytes_in_cl = 0; }
                outp += unsed_bytes_in_cl; // 64: next output starts in new cacheline
            }
        }
        return status::success;
    }
};

template <SIMPLE_SPARSE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_sparse_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("simple_sparse:any", simple_sparse_reorder_t);

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            const bool args_ok = true && src_md->data_type == type_i
                    && dst_md->data_type == type_o && attr->has_default_values()
                    && simple_sparse_reorder_impl<
                            SIMPLE_SPARSE_REORDER_TEMPL_CALL,
                            spec>::is_applicable(src_md, dst_md, attr);
            if (!args_ok) return status::invalid_arguments;

            auto _pd = new pd_t(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                delete _pd;
                return status::unimplemented;
            }

            _pd->init_scratchpad_md();
            return safe_ptr_assign(*reorder_pd, _pd);
        }
        friend dnnl::impl::impl_list_item_t;
    };

    simple_sparse_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return simple_sparse_reorder_impl<SIMPLE_SPARSE_REORDER_TEMPL_CALL,
                spec>::execute(pd(), ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

#undef SIMPLE_SPARSE_REORDER_TEMPL_DECL
#undef SIMPLE_SPARSE_REORDER_TEMPL_CALL

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
