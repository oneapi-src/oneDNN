/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <cstring>
#include <random>

#include "utils/dnnl_query.hpp"
#include "utils/fill.hpp"
#include "utils/numeric.hpp"
#include "utils/parallel.hpp"

int fill_scales(
        const attr_t &attr, int arg, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto &e = attr.scales.get(arg);
    return fill_scales(e, mem_dt, mem_fp);
}

int fill_scales(const attr_t::arg_scales_t::entry_t &e, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    if (mem_dt) { assert(mem_dt.nelems() == mem_fp.nelems()); }

    if (e.policy == policy_t::COMMON) {
        assert(nelems == 1);
        mem_fp.set_elem(0, e.scale);
        if (mem_dt) mem_dt.set_elem(0, e.scale);
    } else {
        /* Do fixed partitioning to have same filling for any number of threads */
        static constexpr int64_t chunk_size = 64;
        const int64_t n_chunks = div_up(nelems, chunk_size);
        benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
            int64_t idx_start = idx_chunk * chunk_size;
            int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
            // Note: we use a different seed for each chunk to avoid
            // repeating patterns. We could use discard(idx_start) too but
            // it has a complexity in O(idx_start). We also add 1 to avoid
            // seeding with 0.
            std::minstd_rand int_seed(idx_start + 1);
            int_seed.discard(1);

            std::uniform_int_distribution<> gen(-2, 2);

            for (int64_t idx = idx_start; idx < idx_end; ++idx) {
                int pow2 = gen(int_seed);
                int pow2_shift = 1 << std::abs(pow2);
                const float gen_val
                        = pow2 < 0 ? (1.f / pow2_shift) : pow2_shift;
                const float val = gen_val;
                mem_fp.set_elem(idx, val);
                if (mem_dt) mem_dt.set_elem(idx, val);
            }
        });
    }

    return OK;
}

int fill_zero_points(
        const attr_t &attr, int arg, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    const auto &e = attr.zero_points.get(arg);
    if (e.policy == policy_t::COMMON) {
        assert(nelems == 1);
        mem_fp.set_elem(0, e.value);
        if (mem_dt) mem_dt.set_elem(0, e.value);
    } else {
        /* Do fixed partitioning to have same filling for any number of threads */
        static constexpr int64_t chunk_size = 64;
        const int64_t n_chunks = div_up(nelems, chunk_size);
        benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
            int64_t idx_start = idx_chunk * chunk_size;
            int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
            // Note: we use a different seed for each chunk to avoid
            // repeating patterns. We could use discard(idx_start) too but
            // it has a complexity in O(idx_start). We also add 1 to avoid
            // seeding with 0.
            std::minstd_rand int_seed(idx_start + 1);
            int_seed.discard(1);

            std::uniform_int_distribution<> gen(-2, 2);

            for (int64_t idx = idx_start; idx < idx_end; ++idx) {
                const float zp_val = gen(int_seed);
                mem_fp.set_elem(idx, zp_val);
                if (mem_dt) mem_dt.set_elem(idx, zp_val);
            }
        });
    }

    return OK;
}

int fill_random_real_dense(dnn_mem_t &mem_fp) {
    auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

#ifdef DNNL_EXPERIMENTAL_SPARSE
    // The `nelems()` function returns a product of dims/pdims regardless of
    // whether the tensor is dense or sparse (this is by design). Because of
    // that we need to adjust the `nelems` value for the sparse tensor as the
    // number of elements to fill is equal to `nnz`.
    if (mem_fp.format_kind() == dnnl_format_kind_sparse)
        nelems = query_md_nnz(mem_fp.md_);
#endif

    /* Do fixed partitioning to have same filling for any number of threads */
    static constexpr int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(nelems + idx_start + 1);
        int_seed.discard(1);

        std::uniform_real_distribution<> gen(-16, 16);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float val = gen(int_seed);
            mem_fp.set_elem(
                    idx, round_to_nearest_representable(mem_fp.dt(), val));
        }
    });

    return OK;
}

#ifdef DNNL_EXPERIMENTAL_SPARSE
int fill_random_real_sparse(
        const_dnnl_memory_t dnnl_memory, dnn_mem_t &mem_fp) {
    auto orig_cc_mem_md = query_md(dnnl_memory);
    const int nhandles = query_md_num_handles(orig_cc_mem_md);
    assert(nhandles == 3);
    // Since a sparsity pattern affects performance, it's crucial to keep the
    // pattern intact and only randomize tensor values.
    // The assumption is every sparse format contains three handles and the
    // second and the third are responsible for a sparsity pattern.
    for (int idx = 1; idx < nhandles; idx++) {
        void *dst_ptr = mem_fp.get_mapped_pointer<void>(idx);
        void *src_ptr = nullptr;
        dnnl_memory_get_data_handle_v2(dnnl_memory, &src_ptr, idx);

        const size_t size = dnnl_memory_desc_get_size_v2(orig_cc_mem_md, idx);
        std::memcpy(dst_ptr, src_ptr, size);
    }

    return fill_random_real_dense(mem_fp);
}
#endif

int fill_random_real(const_dnnl_memory_t dnnl_memory, dnn_mem_t &mem_fp) {
#ifdef DNNL_EXPERIMENTAL_SPARSE
    if (mem_fp.format_kind() == dnnl_format_kind_sparse)
        return fill_random_real_sparse(dnnl_memory, mem_fp);
#endif
    return fill_random_real_dense(mem_fp);
}
