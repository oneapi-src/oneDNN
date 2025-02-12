/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <random>
#include <set>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/fill.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

dims_t get_runtime_dims(const dims_t &dims, const dims_mask_t &mask) {
    if (mask.none() || dims.empty()) return dims;
    dims_t runtime_dims;
    runtime_dims.resize(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        runtime_dims[i] = mask[i] ? DNNL_RUNTIME_DIM_VAL : dims[i];
    }
    return runtime_dims;
}

// TODO: Generalize md creation for sparse data when other primitives
// start supporting it.
benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> create_md(const prb_t *prb,
        data_kind_t kind, dnnl_data_type_t dt = dnnl_data_type_undef) {
    if (kind == SRC) {
        if (dt == dnnl_data_type_undef) dt = prb->src_dt();
        const auto &src_rt_dims = get_runtime_dims(
                prb->src_dims(), prb->src_runtime_dim_mask());
#ifdef DNNL_EXPERIMENTAL_SPARSE
        auto src_encoding = prb->sparse_options.get_encoding(DNNL_ARG_SRC);
        auto src_sparsity = prb->sparse_options.get_sparsity(DNNL_ARG_SRC);
        if (src_encoding != dnnl_sparse_encoding_undef) {
            const dnnl_dim_t nnz
                    = std::max(prb->m * prb->k * (1.0f - src_sparsity), 1.0f);
            switch (src_encoding) {
                case dnnl_csr:
                    return dnn_mem_t::init_csr_md(prb->ndims,
                            src_rt_dims.data(), dt, nnz, dnnl_s32, dnnl_s32);
                    break;
                case dnnl_coo:
                    return dnn_mem_t::init_coo_md(
                            prb->ndims, src_rt_dims.data(), dt, nnz, dnnl_s32);
                    break;
                default: assert(!"unsupported encoding"); return nullptr;
            }
        } else
#endif
            return dnn_mem_t::init_md(prb->ndims, src_rt_dims.data(), dt,
                    prb->stag, prb->strides[STRIDES_SRC]);
    }

    if (kind == WEI) {
        if (dt == dnnl_data_type_undef) dt = prb->wei_dt();
        const auto &weights_rt_dims = get_runtime_dims(
                prb->weights_dims(), prb->weights_runtime_dim_mask());
#ifdef DNNL_EXPERIMENTAL_SPARSE
        auto wei_encoding = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);
        auto wei_sparsity = prb->sparse_options.get_sparsity(DNNL_ARG_WEIGHTS);

        if (wei_encoding != dnnl_sparse_encoding_undef) {
            const dnnl_dim_t nnz
                    = std::max(prb->k * prb->n * (1.0f - wei_sparsity), 1.0f);
            switch (wei_encoding) {
                case dnnl_csr:
                    return dnn_mem_t::init_csr_md(prb->ndims,
                            weights_rt_dims.data(), dt, nnz, dnnl_s32,
                            dnnl_s32);
                case dnnl_coo:
                    return dnn_mem_t::init_coo_md(prb->ndims,
                            weights_rt_dims.data(), dt, nnz, dnnl_s32);
                case dnnl_packed:
                    return dnn_mem_t::init_sparse_packed_md(
                            prb->ndims, weights_rt_dims.data(), dt, nnz);
                    break;
                default: assert(!"unsupported encoding"); return nullptr;
            }
        } else
#endif
            return dnn_mem_t::init_md(prb->ndims, weights_rt_dims.data(), dt,
                    prb->wtag, prb->strides[STRIDES_WEI]);
    }

    if (kind == DST) {
        if (dt == dnnl_data_type_undef) dt = prb->dst_dt();
        const auto &dst_rt_dims
                = get_runtime_dims(prb->dst_dims, prb->dst_runtime_dim_mask());
        return dnn_mem_t::init_md(prb->ndims, dst_rt_dims.data(), dt, prb->dtag,
                prb->strides[STRIDES_DST]);
    }
    return nullptr;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    auto src_d = create_md(
            prb, SRC, force_f32_dt ? dnnl_f32 : dnnl_data_type_undef);
    auto wei_d = create_md(
            prb, WEI, force_f32_dt ? dnnl_f32 : dnnl_data_type_undef);
    auto dst_d = create_md(
            prb, DST, force_f32_dt ? dnnl_f32 : dnnl_data_type_undef);

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> bia_d {};
    if (prb->bia_dt != dnnl_data_type_undef) {
        auto bia_dims = get_runtime_dims(
                prb->bia_dims(), prb->bias_runtime_dim_mask());
        bia_d = dnn_mem_t::init_md(prb->ndims, bia_dims.data(),
                force_f32_dt ? dnnl_f32 : prb->bia_dt,
                prb->dst_runtime_dim_mask() != 0 ? tag::abx : tag::any);
    }

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->dst_dims.data());

    const auto overload_quant_mask = [&](policy_t policy, int arg) {
        // Overload PER_OC/PER_OCIC mask definition for batched cases.
        if (policy == policy_t::PER_OC || policy == policy_t::PER_OCIC) {
            int mask = 1 << (prb->ndims - 1);
            if (policy == policy_t::PER_OCIC) mask += 1 << (prb->ndims - 2);
            attr_args.prepare_quant(prb->attr, arg, mask);
        }
    };

    overload_quant_mask(prb->attr.scales.get(DNNL_ARG_SRC).policy,
            DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    overload_quant_mask(prb->attr.scales.get(DNNL_ARG_WEIGHTS).policy,
            DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    overload_quant_mask(prb->attr.scales.get(DNNL_ARG_DST).policy,
            DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    overload_quant_mask(prb->attr.zero_points.get(DNNL_ARG_SRC).policy,
            DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    overload_quant_mask(prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).policy,
            DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    TIME_C_PD(DNN_SAFE_STATUS(dnnl_matmul_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.engine,
            init_pd_args.src_md ? init_pd_args.src_md : src_d, wei_d, bia_d,
            dst_d, dnnl_attr)));

    return dnnl_success;
}

int init_prim_ref(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref,
        const prb_t *prb, res_t *res) {
    if (!(has_bench_mode_bit(mode_bit_t::corr) && fast_ref)) return OK;
    // Create prim_ref if only original prim was successfully created.
    if (res->state != INITIALIZED) return OK;

    // f32 cases should go through reference no matter what.
    if (is_cpu() && (prb->src_dt() == dnnl_f32 && prb->wei_dt() == dnnl_f32))
        return OK;

#ifdef DNNL_EXPERIMENTAL_SPARSE
    if (prb->sparse_options.get_encoding(DNNL_ARG_SRC)
                    != dnnl_sparse_encoding_undef
            || prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS)
                    != dnnl_sparse_encoding_undef)
        return OK;
#endif

    // Create a new copy of prb to avoid potentially corrupting the test by
    // modifying prb in place.
    auto cpu_attr = prb->attr;
    update_cpu_ref_attrs(cpu_attr);
    std::vector<std::vector<dnnl_data_type_t>> prim_ref_dt {
            prb->dt, {dnnl_f32}};
    // If there's no bias, undef data type should be used for prim_ref as well.
    dnnl_data_type_t cpu_bia_dt
            = prb->bia_dt == dnnl_data_type_undef ? prb->bia_dt : dnnl_f32;
    std::vector<dnnl_data_type_t> prim_ref_bia_dt {prb->bia_dt, cpu_bia_dt};
    if (is_cpu()) {
        prim_ref_dt.erase(prim_ref_dt.begin());
        prim_ref_bia_dt.erase(prim_ref_bia_dt.begin());
    }
    dnnl_primitive_t prim_ref_ {};

    for_(const auto &prim_ref_dt_i : prim_ref_dt)
    for (const auto &prim_ref_bia_dt_i : prim_ref_bia_dt) {
        prb_t prb_cpu {*prb, prim_ref_dt_i, tag::any, tag::any, tag::any,
                {vdims_t(STRIDES_SIZE)}, prim_ref_bia_dt_i, prb->bia_mask,
                {0, 0, 0},
#ifdef DNNL_EXPERIMENTAL_SPARSE
                sparse_options_t(),
#endif
                cpu_attr, prb->ctx_init, prb->ctx_exe, prb->impl_filter};

        init_pd_args_t<prb_t> init_pd_args(
                /* res = */ nullptr, get_cpu_engine(), &prb_cpu, prb->dir,
                /* hint = */ nullptr, /* src_md = */ nullptr);
        init_pd(init_pd_args);

        benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pdw;
        // `is_service_prim=true` prevents from filtering the implementation
        // by name which is intended through a `get_prim_ref_impl_filter()`.
        // As `fetch_impl` doesn't have any further logic related to it, it's
        // safe to set it to `false`.
        fetch_impl(pdw, init_pd_args, get_prim_ref_impl_filter(),
                /* res = */ nullptr,
                /* is_service_prim = */ false);

        // Prim desc wasn't created - try the next set...
        if (!pdw) continue;

        auto st = dnnl_primitive_create(&prim_ref_, pdw);
        // Primitive wasn't created - try the next set...
        if (st != dnnl_success) continue;

        BENCHDNN_PRINT(5, "CPU reference oneDNN implementation: %s\n",
                query_impl_info(pdw).c_str());
        res->prim_ref_repro = prb_cpu.str();
        prim_ref.reset(prim_ref_);
        return OK;
    }

    prim_ref.reset(prim_ref_);
    return OK;
}

#ifdef DNNL_EXPERIMENTAL_SPARSE
// The main idea is to generate values and metadata directly without generating
// the dense matrix to avoid excessive memory consumption for large problem
// sizes.
int fill_sparse_data(data_kind_t kind, const prb_t *prb, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res, dnnl_sparse_encoding_t encoding) {
    if (query_md_num_handles(mem_dt.md_) != 3) return FAIL;

    if (kind != SRC && kind != WEI) return FAIL;

    const int64_t dim0 = kind == SRC ? prb->m : prb->k;
    const int64_t dim1 = kind == SRC ? prb->k : prb->n;

    // Coefficient for distribution of nnz per row.
    const int64_t coef = 3;
    const int64_t nnz = query_md_nnz(mem_fp.md_);
    const int64_t avg_nnz_per_row = nnz / dim0;

    int64_t distributed_nnz_cnt = 0;

    std::uniform_int_distribution<> pointers_gen(0, avg_nnz_per_row * coef);
    std::minstd_rand pointers_seed;

    // Distribute nnz across all rows.
    std::vector<int64_t> distributed_nnz(dim0);
    for (int64_t i = 0; i < dim0; i++) {
        int64_t nnz_per_row = std::min(pointers_gen(pointers_seed), (int)dim1);
        nnz_per_row = std::min(nnz_per_row, (nnz - distributed_nnz_cnt));
        distributed_nnz[i] = nnz_per_row;
        distributed_nnz_cnt += nnz_per_row;
    }

    // Distribute remaining nnz.
    int64_t remaining_nnz_cnt = nnz - distributed_nnz_cnt;
    while (remaining_nnz_cnt > 0) {
        const int64_t remaining_nnz_per_row
                = std::max((int)(remaining_nnz_cnt / dim0), 1);
        for (int64_t i = 0; i < dim0; i++) {
            int64_t nnz_to_add = std::min(
                    remaining_nnz_per_row, (dim1 - distributed_nnz[i]));
            nnz_to_add = std::min(nnz_to_add, remaining_nnz_cnt);
            distributed_nnz[i] += nnz_to_add;
            remaining_nnz_cnt -= nnz_to_add;
            distributed_nnz_cnt += nnz_to_add;

            if (remaining_nnz_cnt == 0) break;
        }
    }

    if (remaining_nnz_cnt != 0) return FAIL;

    int values_idx = 0;
    int indices_idx = 1;
    const int pointers_idx = 2;

    if (encoding == dnnl_csr) {
        // fill pointers for CSR encoding
        mem_fp.set_elem(0, 0, pointers_idx);
        mem_dt.set_elem(0, 0, pointers_idx);

        for (int64_t i = 0; i < dim0; i++) {
            const int32_t pointer
                    = mem_fp.get_elem(i, pointers_idx) + distributed_nnz[i];
            mem_fp.set_elem(i + 1, pointer, pointers_idx);
            mem_dt.set_elem(i + 1, pointer, pointers_idx);
        }
    } else if (encoding == dnnl_coo) {
        values_idx = 0;
        indices_idx = 2;
        const int row_indices_idx = 1;

        // fill row indices for COO encoding
        int32_t row_ptr = 0;

        for (int64_t i = 0; i < dim0; i++) {
            for (int32_t j = 0; j < distributed_nnz[i]; j++) {
                mem_fp.set_elem(row_ptr + j, i, row_indices_idx);
                mem_dt.set_elem(row_ptr + j, i, row_indices_idx);
            }
            row_ptr = row_ptr + distributed_nnz[i];
        }
    }

    std::uniform_int_distribution<> indices_gen(0, dim1 - 1);
    std::minstd_rand indices_seed;

    // Generate indices.
    std::vector<int32_t> indices;
    std::set<int32_t> indices_set;
    for (int64_t i = 0; i < dim0; i++) {
        while ((int64_t)indices_set.size() != distributed_nnz[i]) {
            int index = indices_gen(indices_seed);
            if (indices_set.count(index)) continue;
            indices_set.insert(index);
        }
        indices.insert(indices.end(), indices_set.begin(), indices_set.end());
        indices_set.clear();
    }

    benchdnn_parallel_nd((int)indices.size(), [&](int64_t i) {
        const int32_t index = indices[i];
        mem_fp.set_elem(i, index, indices_idx);
        mem_dt.set_elem(i, index, indices_idx);
    });

    // Generate values.
    cfg_t cfg(prb, {SRC, WEI, BIA, DST});

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nnz, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nnz);

        std::uniform_int_distribution<> values_gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));
        std::minstd_rand values_seed(kind * nnz + idx_start + 1);
        values_seed.discard(1);

        for (int64_t i = idx_start; i < idx_end; i++) {
            float val = values_gen(values_seed);
            mem_fp.set_elem(i,
                    round_to_nearest_representable(cfg.get_dt(kind), val),
                    values_idx);
            mem_dt.set_elem(i,
                    round_to_nearest_representable(cfg.get_dt(kind), val),
                    values_idx);
        }
    });

    return OK;
}
#endif

int fill_data(data_kind_t kind, const prb_t *prb, const cfg_t &cfg,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;

    bool is_sparse_packed = false;
    bool is_any_sparse = false;
    std::vector<bool> nnz_mask;
#ifdef DNNL_EXPERIMENTAL_SPARSE
    const auto sparse_encoding = prb->sparse_options.get_encoding(kind);
    const bool is_sparse_csr_coo
            = sparse_encoding == dnnl_csr || sparse_encoding == dnnl_coo;
    is_sparse_packed = sparse_encoding == dnnl_packed;
    is_any_sparse = sparse_encoding != sparse_options_t::def_encoding;

    if (is_sparse_csr_coo) {
        return fill_sparse_data(
                kind, prb, mem_dt, mem_fp, res, sparse_encoding);
    }

    if (is_sparse_packed) {
        nnz_mask.resize(nelems, false);
        const dnnl_dim_t nnz = query_md_nnz(mem_dt.md_);
        assert(nnz > 0);
        for (int i = 0; i < nnz; i++)
            nnz_mask[i] = true;
        std::default_random_engine rng(nnz);
        std::shuffle(nnz_mask.begin(), nnz_mask.end(), rng);
    }
#endif

    // Refer to modes documentation for filling principles.
    // Note: sparse filling is more complex than a general one in a sense that
    // it requires metadata in addition to data. To have reasonable bitwise
    // validation for sparse, only data must be random and indices should remain
    // identical between runs. So far, simply don't support bitwise mode for
    // sparse problems. `CSR`/`COO` will utilize their `fill_sparse_data`
    // function, `packed` will fall back into a regular filling as it involves
    // `nnz_mask`.
    if (has_bench_mode_bit(mode_bit_t::bitwise) && !is_any_sparse) {
        return fill_random_real(mem_dt, mem_fp, res);
    }
    if (has_bench_mode_bit(mode_bit_t::perf) && !is_any_sparse) {
        return fill_random_real(
                mem_dt, mem_fp, res, get_perf_fill_cfg(mem_dt.dt()));
    }

    cfg_t::density_args_t density_args;
    density_args.data_kind = kind;
    density_args.n_acc = prb->k;
    const auto density = cfg.get_density(density_args);

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));
        std::bernoulli_distribution b_dist(density);

        // make sure the first element is positive
        if (idx_start == 0 && !is_sparse_packed) {
            float val = 0;
            while (val <= 0)
                val = gen(int_seed);
            mem_fp.set_elem(
                    0, round_to_nearest_representable(cfg.get_dt(kind), val));
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = density == 1.f ? true : b_dist(b_seed);
            float val = 0.0f;
            if (is_sparse_packed) {
                is_one = nnz_mask[idx];
                while (val == 0.0f)
                    val = gen(int_seed);
                val *= is_one;
            } else {
                val = is_one * gen(int_seed);
            }
            mem_fp.set_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
    });

    SAFE(mem_dt.reorder(mem_fp, cfg.get_swapped_dt(kind)), WARN);

    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(
            {prb->src_dt(), prb->wei_dt(), prb->bia_dt, prb->dst_dt()},
            prb->dir, res);
    skip_unimplemented_sum_po(
            prb->attr, res, dnnl_matmul, prb->src_dt(), prb->dst_dt());
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_matmul);

#ifdef DNNL_EXPERIMENTAL_SPARSE
    if ((is_nvidia_gpu() || is_amd_gpu()) && !prb->sparse_options.is_def()) {
        BENCHDNN_PRINT(2,
                "[SKIP][%s:%d]: oneDNN doesn't support sparse matmul for "
                "NVIDIA and AMD GPUs.\n",
                __FILE__, __LINE__);
        res->state = SKIPPED;
        res->reason = skip_reason::case_not_supported;
        return;
    }

    const auto wei_encoding
            = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);
    bool is_wei_dense = (wei_encoding == dnnl_sparse_encoding_undef);
    bool is_src_coo_sparse
            = (prb->sparse_options.get_encoding(DNNL_ARG_SRC) == dnnl_coo);
    if (!prb->sparse_options.is_def() && is_gpu()
            && (!is_wei_dense || !is_src_coo_sparse)) {
        BENCHDNN_PRINT(2,
                "[SKIP][%s:%d]: GPU sparse matmul only supports COO encoding "
                "for source.\n",
                __FILE__, __LINE__);
        res->state = SKIPPED;
        res->reason = skip_reason::case_not_supported;
        return;
    }

    if (!prb->sparse_options.is_def() && is_cpu() && is_wei_dense
            && prb->wtag != "any" && prb->wtag != "ab") {
        BENCHDNN_PRINT(2,
                "[SKIP][%s:%d]: Only `any` and `ab` tags are supported for "
                "dense weights on CPU.\n",
                __FILE__, __LINE__);
        res->state = SKIPPED;
        res->reason = skip_reason::case_not_supported;
        return;
    }

    if (wei_encoding == dnnl_packed) {
        BENCHDNN_PRINT(2,
                "[SKIP][%s:%d]: Weights argument doesn't support packed "
                "encoding.\n",
                __FILE__, __LINE__);
        res->state = SKIPPED;
        res->reason = skip_reason::case_not_supported;
        return;
    }
#endif

    if (is_cpu()) {
        const bool is_x8s8f16
                = prb->wei_dt() == dnnl_s8 && prb->dst_dt() == dnnl_f16;
        if (is_x8s8f16) {
            BENCHDNN_PRINT(2, "[SKIP][%s:%d]: CPU doesn't support x8s8f16.\n",
                    __FILE__, __LINE__);
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }
        if (!prb->attr.scales.is_def(DNNL_ARG_DST)
                && prb->attr.scales.get(DNNL_ARG_DST).policy
                        != attr_t::COMMON) {
            BENCHDNN_PRINT(2,
                    "[SKIP][%s:%d]: Only Common dst scales are supported "
                    "on CPU.\n",
                    __FILE__, __LINE__);
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }
    }

    if (is_gpu()) {
        const auto &po = prb->attr.post_ops;
        if (prb->dst_dt() == dnnl_f64 && !po.is_def()) {
            BENCHDNN_PRINT(2,
                    "[SKIP][%s:%d]: Post-ops for f64 data type is not "
                    "supported.\n",
                    __FILE__, __LINE__);
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }

        const int sum_idx = po.find(attr_t::post_ops_t::kind_t::SUM);
        if (sum_idx != -1 && po.entry[sum_idx].sum.dt != dnnl_data_type_undef) {
            BENCHDNN_PRINT(2,
                    "[SKIP][%s:%d]: GPU doesn't support non-default sum_dt "
                    "argument.\n",
                    __FILE__, __LINE__);
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }

        // GPU for x8s8bf16 doesn't support:
        // * Destination zero-point.
        // * Any run-time dimensions.
        // * Any batch dimensions.
        const bool is_x8s8bf16
                = prb->wei_dt() == dnnl_s8 && prb->dst_dt() == dnnl_bf16;
        const bool rt_dims_are_none = prb->src_runtime_dim_mask().none()
                && prb->weights_runtime_dim_mask().none()
                && prb->dst_runtime_dim_mask().none();
        const bool x8s8bf16_ok = IMPLICATION(is_x8s8bf16,
                prb->attr.zero_points.get(DNNL_ARG_DST).is_def()
                        && rt_dims_are_none && prb->ndims <= 2);
        if (!x8s8bf16_ok) {
            BENCHDNN_PRINT(2,
                    "[SKIP][%s:%d]: x8s8bf16 configuration on GPU doesn't "
                    "support certain features.\n",
                    __FILE__, __LINE__);
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }

        const bool is_bf16 = prb->src_dt() == dnnl_bf16
                && prb->wei_dt() == dnnl_bf16
                && (prb->dst_dt() == dnnl_bf16 || prb->dst_dt() == dnnl_f32);
        const bool bf16_bias_ok = IMPLICATION(
                prb->bia_dt == dnnl_bf16, prb->ndims <= 2 + is_bf16);
        if (!bf16_bias_ok) {
            BENCHDNN_PRINT(2,
                    "[SKIP][%s:%d]: bf16 bias support is limited to bf16 "
                    "configuration and 2D-matmul.\n",
                    __FILE__, __LINE__);
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }

        if (((prb->src_dt() == dnnl_f8_e4m3 || prb->dst_dt() == dnnl_f8_e4m3)
                    || (prb->src_dt() == dnnl_f8_e5m2
                            || prb->dst_dt() == dnnl_f8_e5m2))
                && (!po.is_def() || !prb->attr.scales.is_def())) {
            BENCHDNN_PRINT(2,
                    "[SKIP][%s:%d]: GPU supports fp8 through ref only for "
                    "f8_e4m3 on all platformas and for f8_e5m2 pre-XeHPC with "
                    "limited post-op support.\n",
                    __FILE__, __LINE__);
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    if (!prb->attr.zero_points.is_def()
            && (prb->wei_dt() != dnnl_s8 && prb->wei_dt() != dnnl_u8
                    && prb->wei_dt() != dnnl_s4 && prb->wei_dt() != dnnl_u4)) {
        BENCHDNN_PRINT(2,
                "[INVALID][%s:%d]: Zero-points applied to a non-integral data "
                "type.\n",
                __FILE__, __LINE__);
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }

    if (!prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def()) {
        const auto &groups = prb->attr.scales.get(DNNL_ARG_WEIGHTS).groups;
        if (!groups.empty()) {
            if (prb->k % groups[0]) {
                BENCHDNN_PRINT(2,
                        "[INVALID][%s:%d]: Weights decompression scales "
                        "require IC ('%d') to be divisible by groups ('%d')\n",
                        __FILE__, __LINE__, (int)prb->k, (int)groups[0]);
                res->state = SKIPPED;
                res->reason = skip_reason::invalid_case;
                return;
            } else if (groups.size() > 2) {
                BENCHDNN_PRINT(2,
                        "[INVALID][%s:%d]: Weights decompression scales groups "
                        "support only two dimensions\n",
                        __FILE__, __LINE__);
                res->state = SKIPPED;
                res->reason = skip_reason::invalid_case;
                return;
            }
        }
    }

    if (!prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).is_def()) {
        const auto &groups = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).groups;
        if (!groups.empty()) {
            if (groups[0] > 0 && (prb->k % groups[0])) {
                BENCHDNN_PRINT(2,
                        "[INVALID][%s:%d]: Weights decompression zero-points "
                        "require IC ('%d') to be divisible by groups ('%d')\n",
                        __FILE__, __LINE__, (int)prb->k, (int)groups[0]);
                res->state = SKIPPED;
                res->reason = skip_reason::invalid_case;
                return;
            } else if (groups.size() > 2) {
                BENCHDNN_PRINT(2,
                        "[INVALID][%s:%d]: Weights decompression zero-points "
                        "groups support only two dimensions\n",
                        __FILE__, __LINE__);
                res->state = SKIPPED;
                res->reason = skip_reason::invalid_case;
                return;
            }
        }
    }

    if ((prb->wei_dt() == dnnl_s4 || prb->wei_dt() == dnnl_u4)
            && (prb->n % 2)) {
        BENCHDNN_PRINT(2,
                "[INVALID][%s:%d]: Int4 Weights decompression requires OC "
                "('%d') to be even.\n",
                __FILE__, __LINE__, (int)prb->n);
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }

    auto src_rt_mask = prb->src_runtime_dim_mask();
    auto wei_rt_mask = prb->weights_runtime_dim_mask();
    auto dst_rt_mask = prb->dst_runtime_dim_mask();

    // Memory layouts must be defined when some dimensions are unknown at pd
    // creation time.
    if ((src_rt_mask.any() && prb->stag == "any")
            || (wei_rt_mask.any() && prb->wtag == "any")
            || (dst_rt_mask.any() && prb->dtag == "any")) {
        BENCHDNN_PRINT(1,
                "[INVALID][%s:%d]: Runtime dimensions require user to specify "
                "a memory format for affected arguments. Consider specifying "
                "`--stag`, `--wtag`, and/or `--dtag`.\n",
                __FILE__, __LINE__);
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }

    const int m_idx = prb->ndims - 2;
    const int k_idx_src = prb->ndims - 1;
    const int k_idx_wei = prb->ndims - 2;
    const int n_idx = prb->ndims - 1;
    if (src_rt_mask[m_idx] != dst_rt_mask[m_idx]
            || src_rt_mask[k_idx_src] != wei_rt_mask[k_idx_wei]
            || wei_rt_mask[n_idx] != dst_rt_mask[n_idx]) {
        BENCHDNN_PRINT(2,
                "[INVALID][%s:%d]: Runtime masks for `m`, `k`, and `n` "
                "dimensions must be consistent.\n",
                __FILE__, __LINE__);
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }

    if (prb->ndims > 2) {
        dims_mask_t batch_rt_mask;
        for (int i = 0; i < prb->ndims - 2; ++i)
            batch_rt_mask[i] = true;
        src_rt_mask &= batch_rt_mask;
        wei_rt_mask &= batch_rt_mask;
        dst_rt_mask &= batch_rt_mask;
        if (src_rt_mask != wei_rt_mask || src_rt_mask != dst_rt_mask) {
            BENCHDNN_PRINT(2,
                    "[INVALID][%s:%d]: Runtime masks for batch dimensions must "
                    "be consistent.\n",
                    __FILE__, __LINE__);
            res->state = SKIPPED;
            res->reason = skip_reason::invalid_case;
            return;
        }
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DST,
    };
    return exec_args;
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, WEI, BIA, DST});

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

#ifdef DNNL_EXPERIMENTAL_SPARSE
        auto src_encoding = prb->sparse_options.get_encoding(DNNL_ARG_SRC);
        auto wei_encoding = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);

        const bool is_sparse_src = exec_arg == DNNL_ARG_SRC
                && src_encoding != dnnl_sparse_encoding_undef;

        const bool is_sparse_wei = exec_arg == DNNL_ARG_WEIGHTS
                && wei_encoding != dnnl_sparse_encoding_undef;
        const bool is_sparse_wei_packed
                = is_sparse_wei && wei_encoding == dnnl_packed;

        if ((is_sparse_src || is_sparse_wei) && !is_sparse_wei_packed) {
            if (is_sparse_src) {
                auto src_fp_d = create_md(prb, SRC);
                ref_mem_map.emplace(exec_arg, dnn_mem_t(src_fp_d, ref_engine));
            }

            if (is_sparse_wei) {
                auto wei_fp_d = create_md(prb, WEI);
                ref_mem_map.emplace(exec_arg, dnn_mem_t(wei_fp_d, ref_engine));
            }
        } else
#endif
        {
            // Scratchpad memory relates to a primitive. If reference needs it,
            // use switch below to define a memory desc for it.
            if (exec_arg != DNNL_ARG_SCRATCHPAD) {
                ref_mem_map.emplace(exec_arg,
                        dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
            }
        }
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_data(SRC, prb, cfg, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_WEIGHTS:
                SAFE(fill_data(WEI, prb, cfg, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_BIAS:
                SAFE(fill_data(BIA, prb, cfg, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DST: {
                const auto &po = prb->attr.post_ops;
                const int sum_idx = po.find(attr_t::post_ops_t::SUM);
                if (sum_idx >= 0) {
                    SAFE(fill_data(DST, prb, cfg, mem, ref_mem, res), WARN);
                    // Bitwise mode for sum requires a copy due to data for
                    // post-op will be overwritten and it must be refreshed.
                    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
                        SAFE(mem_map.at(-exec_arg).reorder(ref_mem), WARN);
                    }
                }
            } break;
            default:
                SAFE(init_ref_memory_args_default_case(
                             exec_arg, mem, ref_mem, prb->attr, res),
                        WARN);
                break;
        }

        update_ref_mem_map_from_prim(prim_ref, mem, ref_mem_map, exec_arg,
                cfg.get_swapped_dt(exec_arg2data_kind(exec_arg)));

        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(2); // regular + cpu_ref
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    // Use CPU prim as the reference in GPU testing to reduce testing time.
    SAFE(init_prim_ref(v_prim[1], prb, res), WARN);
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    SAFE(check_caches(v_prim[0], prb, res), WARN);
    // Don't check caches for CPU prim as the reference.
    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    // TODO: move the regular buffer kinds like SRC or DST to a common function,
    //       e.g. get_kinds_to_check_default_case
    std::vector<data_kind_t> check_kinds = {DST};
    if (!prb->attr.dropout.is_def()) check_kinds.push_back(DROPOUT_MASK);
    return check_kinds;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = v_prim[0];
    const auto &prim_ref = v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(mem_map, prb, prim, supported_exec_args(prb->dir));
    TIME_FILL(SAFE(init_ref_memory_args(
                           ref_mem_map, mem_map, prim, prb, res, prim_ref),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(prim, args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb), args, ref_args, setup_cmp,
            res, prim_ref);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb), args, prb->attr,
                 prb->inplace, res),
            WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace matmul
