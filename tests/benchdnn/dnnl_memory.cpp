/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <algorithm>
#include <atomic>
#include <memory>
#include <numeric>
#include <string>

#include "oneapi/dnnl/dnnl.h"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "tests/test_thread.hpp"

int execute_reorder(const dnn_mem_t &src, dnn_mem_t &dst,
        const_dnnl_primitive_attr_t attr) {
    std::shared_ptr<const dnn_mem_t> r_src(&src, [](const dnn_mem_t *) {});
    std::shared_ptr<dnn_mem_t> r_dst(&dst, [](dnn_mem_t *) {});

    dnnl_primitive_desc_t r_pd = nullptr;
    dnnl_primitive_t r {};

    // Optimization to reduce testing time for GPU.
    //
    // For CPU <-> GPU reorders, the library creates GPU-side kernels.
    // Benchdnn heavily relies on reorders and this greatly increases execution
    // time because of big overhead on building OpenCL kernels.
    //
    // First, try to create CPU reorder for the requested GPU reorder. If
    // succeeded, then create CPU memory object wrapping mapped pointers of
    // source and destination and execute CPU reorder. If CPU reorder can't be
    // create, then just execute a regular GPU reorder.
    //
    // This optimization is skipped when testing reorder, sum and concat
    // primitives because they are used specifically to test GPU reorders.
#if (DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL) \
        || (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL)
    std::string driver = std::string(driver_name);
    bool is_reorder_related_driver = (driver == std::string("reorder")
            || driver == std::string("sum") || driver == std::string("concat"));
    const auto &cpu_engine = get_cpu_engine();
    if (!is_reorder_related_driver
            && (src.engine_kind() == dnnl_gpu
                    || dst.engine_kind() == dnnl_gpu)) {

        dnnl_status_t status = dnnl_reorder_primitive_desc_create(
                &r_pd, &src.md_, cpu_engine, &dst.md_, cpu_engine, attr);
        if (status == dnnl_success) {
            // Create CPU memory objects wrapping mapped pointers of source and
            // destination
            r_src.reset(new dnn_mem_t(dnn_mem_t::create_from_host_ptr(
                    src.md_, cpu_engine, (void *)src)));
            r_dst.reset(new dnn_mem_t(dnn_mem_t::create_from_host_ptr(
                    dst.md_, cpu_engine, (void *)dst)));
        }
    }
#endif

    if (!r_pd) {
        DNN_SAFE(dnnl_reorder_primitive_desc_create(&r_pd, &src.md_,
                         src.engine(), &dst.md_, dst.engine(), attr),
                CRIT);
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                r_pd, dnnl_query_exec_arg_md, index);
    };
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);
    dnn_mem_t scratchpad(scratchpad_md, src.engine());

    DNN_SAFE(dnnl_primitive_create(&r, r_pd), CRIT);
    dnnl_status_t pd_destroy_status = dnnl_primitive_desc_destroy(r_pd);
    if (pd_destroy_status != dnnl_success) {
        dnnl_primitive_destroy(r);
        DNN_SAFE(pd_destroy_status, CRIT);
    }

    args_t args;
    args.set(DNNL_ARG_FROM, *r_src);
    args.set(DNNL_ARG_TO, *r_dst);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad);

    SAFE(execute_and_wait(r, args), CRIT);
    DNN_SAFE(dnnl_primitive_destroy(r), CRIT);

    return OK;
}
int dnn_mem_t::reorder(const dnn_mem_t &rhs, const_dnnl_primitive_attr_t attr) {
    if (this == &rhs) return OK;
    return execute_reorder(rhs, *this, attr);
}

dnn_mem_t dnn_mem_t::create_from_host_ptr(
        const dnnl_memory_desc_t &md, dnnl_engine_t engine, void *host_ptr) {
    // XXX: allows to construct CPU memory only.
    assert(is_cpu(engine));

    // XXX: assumption that SYCL works fine with native host pointers
    return dnn_mem_t(md, engine, host_ptr);
}

// Returns physical offset by logical one. Logical offset is represented by an
// array pos. If is_pos_padded is true pos represents the position in already
// padded area.
dnnl_dim_t md_off_v(const dnnl_memory_desc_t &md, const dnnl_dims_t pos,
        bool is_pos_padded) {
    assert(md.format_kind == dnnl_blocked);
    const auto &blk = md.format_desc.blocking;

    dnnl_dims_t pos_copy = {0};
    for (int d = 0; d < md.ndims; ++d)
        pos_copy[d] = pos[d] + (is_pos_padded ? 0 : md.padded_offsets[d]);

    dnnl_dim_t phys_offset = md.offset0;

    if (blk.inner_nblks > 0) {
        dnnl_dim_t blk_stride = 1;
        for (int iblk = blk.inner_nblks - 1; iblk >= 0; --iblk) {
            const int d = blk.inner_idxs[iblk];

            dnnl_dim_t p = pos_copy[d] % blk.inner_blks[iblk];
            pos_copy[d] /= blk.inner_blks[iblk];

            phys_offset += p * blk_stride;
            blk_stride *= blk.inner_blks[iblk];
        }
    }

    for (int d = 0; d < md.ndims; ++d) {
        const dnnl_dim_t p = pos_copy[d];
        phys_offset += p * blk.strides[d];
    }

    return phys_offset;
}

// Returns physical offset by logical one. logical offset is represented by a
// scalar l_offset. If is_pos_padded is true, l_offset represents logical
// offset in already padded area.
static dnnl_dim_t md_off_l(dnnl_dims_t _pos, const dnnl_memory_desc_t &md,
        dnnl_dim_t l_offset, bool is_pos_padded = false) {
    dnnl_dims_t pos;
    for (int rd = 0; rd < md.ndims; ++rd) {
        const int d = md.ndims - 1 - rd;
        const dnnl_dim_t cur_dim
                = is_pos_padded ? md.padded_dims[d] : md.dims[d];
        pos[d] = l_offset % cur_dim;
        if (_pos) _pos[d] = pos[d];
        l_offset /= cur_dim;
    }
    return md_off_v(md, pos, is_pos_padded);
}

template <typename T>
int check_zero_padding_impl(const dnn_mem_t &mem, int arg, int *error_count) {
    const int ndims = mem.md_.ndims;
    const auto *dims = mem.md_.dims;
    const auto *pdims = mem.md_.padded_dims;

    if (ndims == 0) return OK;
    if (mem.md_.format_kind != dnnl_blocked) return OK;

    auto product = [](const dnnl_dim_t *beg, const dnnl_dim_t *end) {
        return std::accumulate(
                beg, end, (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
    };

    int errors = 0;
    std::atomic<int> ok(true);

    const T *mem_ptr = (const T *)mem;

    for (int dim_m_idx = 0; dim_m_idx < ndims; ++dim_m_idx) {
        if (dims[dim_m_idx] == pdims[dim_m_idx]) continue;

        auto dim_l = product(pdims, pdims + dim_m_idx);
        auto dim_r = product(pdims + dim_m_idx + 1, pdims + ndims);

        dnnl::impl::parallel_nd(dim_l, dim_r, [&](dnnl_dim_t l, dnnl_dim_t r) {
            for (dnnl_dim_t m = dims[dim_m_idx]; m < pdims[dim_m_idx]; ++m) {
                auto l_idx = (l * pdims[dim_m_idx] + m) * dim_r + r;
                auto idx = md_off_l(nullptr, mem.md_, l_idx, true);
                if (!(mem_ptr[idx] == 0)) ok = false;
            }
        });

        // Run the check one more time to report incorrect elements. This check
        // is sequential.
        if (!ok) {
            for_(dnnl_dim_t l = 0; l < dim_l; ++l)
            for_(dnnl_dim_t m = dims[dim_m_idx]; m < pdims[dim_m_idx]; ++m)
            for (dnnl_dim_t r = 0; r < dim_r; ++r) {
                auto l_idx = (l * pdims[dim_m_idx] + m) * dim_r + r;
                dnnl_dims_t pos = {};
                auto idx = md_off_l(pos, mem.md_, l_idx, true);

                bool idx_ok = (mem_ptr[idx] == 0);
                if (!idx_ok) errors++;

                const bool dump = (!idx_ok && (errors < 10 || verbose >= 10))
                        || (verbose >= 99);
                if (dump) {
                    BENCHDNN_PRINT(0,
                            "[%4ld][arg:%d]"
                            "[" IFMT "," IFMT "," IFMT "," IFMT "," IFMT
                            "," IFMT "] fp:  0.f dt:% 9.6g \n",
                            (long)idx, arg, pos[0], pos[1], pos[2], pos[3],
                            pos[4], pos[5], mem.get_elem(idx));
                }
            }
        }
    }

    if (!ok) {
        BENCHDNN_PRINT(0, "@@@ [arg:%d] check_zero_padding failed\n", arg);
    }

    if (error_count != nullptr) *error_count = errors;

    return ok ? OK : FAIL;
}

int check_zero_padding(const dnn_mem_t &mem, int arg, int *error_count) {
#define CASE(dt, type) \
    case dt: return check_zero_padding_impl<type>(mem, arg, error_count);

    switch (mem.md_.data_type) {
        case dnnl_data_type_undef:
            return OK;

            CASE(dnnl_bf16, bfloat16_t);
            CASE(dnnl_f16, float16_t);
            CASE(dnnl_f32, float);
            CASE(dnnl_s32, int32_t);
            CASE(dnnl_s8, int8_t);
            CASE(dnnl_u8, uint8_t);

        default: assert(!"bad data_type");
    };
#undef CASE

    return FAIL;
}
