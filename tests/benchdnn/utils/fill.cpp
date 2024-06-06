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

#include <cstring>
#include <random>
#include <sstream>

#include "utils/dnnl_query.hpp"
#include "utils/fill.hpp"
#include "utils/numeric.hpp"
#include "utils/parallel.hpp"

fill_cfg_t::fill_cfg_t(dnnl_data_type_t dt, float range_min_val,
        float range_max_val, bool only_integer, attr_t::post_ops_t::kind_t alg,
        const std::string &name)
    : dt_(dt)
    , range_min_val_(MAX2(lowest_dt(dt_), range_min_val))
    , range_max_val_(MIN2(max_dt(dt_), range_max_val))
    , only_integer_(is_integral_dt(dt_) || only_integer)
    , name_(name) {
    if (alg == attr_t::post_ops_t::kind_t::SUB) {
        // Apply range inversion if `alg` is `sub`. This helps to keep output
        // data positive if it was intended to be positive. In rest cases act
        // like for binary `add` algorithm. If `attr` is unavailable in the
        // code, use `attr_t::post_ops_t::kind_t::ADD` as a defulat value.
        float sub_range_min_val_ = -range_min_val_;
        float sub_range_max_val_ = -range_max_val_;
        range_min_val_ = MIN2(sub_range_min_val_, sub_range_max_val_);
        range_max_val_ = MAX2(sub_range_min_val_, sub_range_max_val_);
    } else if (alg == attr_t::post_ops_t::kind_t::MUL) {
        // Reduce the range for multiplication to decrease a computational
        // error magnitute which can lead to rounding to a different output
        // value for low-precision data types.
        // TODO: replace with using specific values instead.
        range_min_val_ /= 8.f;
        range_max_val_ /= 8.f;
    }
}

fill_cfg_t::fill_cfg_t(
        const std::vector<float> &user_set, const std::string &name)
    : dt_(dnnl_data_type_undef)
    , range_min_val_(-FLT_MAX)
    , range_max_val_(FLT_MAX)
    , predefined_set_(user_set)
    , only_integer_(false)
    , name_(name) {
    assert(!predefined_set_.empty());
}

std::string fill_cfg_t::print_verbose() const {
    std::stringstream ss;

    ss << "[FILL_CFG]";
    if (!name_.empty()) ss << " name:" << name_;

    // Predefined set is mutually excluded with a range setting.
    if (!predefined_set_.empty()) {
        ss << " set:[";
        for (const auto &e : predefined_set_) {
            ss << e << ";";
        }
        ss << "]";
    } else {
        ss << " dt:" << dt_;
        ss << " range:[" << range_min_val_ << ";" << range_max_val_ << "]";
        if (only_integer_) ss << " only_integer:true";
    }

    return ss.str();
}

const fill_cfg_t &get_default_fill_cfg() {
    static const fill_cfg_t fill_cfg;
    return fill_cfg;
}

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
        const int min_val = MAX2(-2, static_cast<int>(lowest_dt(mem_dt.dt())));
        benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
            int64_t idx_start = idx_chunk * chunk_size;
            int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
            // Note: we use a different seed for each chunk to avoid
            // repeating patterns. We could use discard(idx_start) too but
            // it has a complexity in O(idx_start). We also add 1 to avoid
            // seeding with 0.
            std::minstd_rand int_seed(idx_start + 1);
            int_seed.discard(1);

            std::uniform_int_distribution<> gen(min_val, 2);

            for (int64_t idx = idx_start; idx < idx_end; ++idx) {
                const float zp_val = gen(int_seed);
                mem_fp.set_elem(idx, zp_val);
                if (mem_dt) mem_dt.set_elem(idx, zp_val);
            }
        });
    }

    return OK;
}

int fill_random_real_dense(dnn_mem_t &mem, dnn_mem_t &mem_ref, res_t *res,
        const fill_cfg_t &fill_cfg) {
    auto nelems = mem_ref.nelems();
    if (nelems == 0) return OK;

    BENCHDNN_PRINT(6, "%s\n", fill_cfg.print_verbose().c_str());

    // This function doesn't handle the predefined set yet.
    assert(fill_cfg.predefined_set_.empty());

#ifdef DNNL_EXPERIMENTAL_SPARSE
    // The `nelems()` function returns a product of dims/pdims regardless of
    // whether the tensor is dense or sparse (this is by design). Because of
    // that we need to adjust the `nelems` value for the sparse tensor as the
    // number of elements to fill is equal to `nnz`.
    if (mem_ref.format_kind() == dnnl_format_kind_sparse)
        nelems = query_md_nnz(mem_ref.md_);
#endif

    // Note: fill_cfg_t drives value distribution, but the final rounding is
    // in compliance with the memory object the values are inserted. Depending
    // on a case, it may or may not benefit to force same data type for filling
    // and final memory object data type.
    const dnnl_data_type_t round_dt = mem ? mem.dt() : mem_ref.dt();

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

        std::uniform_real_distribution<> gen_real(
                fill_cfg.range_min_val_, fill_cfg.range_max_val_);
        std::uniform_int_distribution<> gen_int(
                fill_cfg.range_min_val_, fill_cfg.range_max_val_);

        const auto get_val = [&]() {
            return fill_cfg.only_integer_
                    ? static_cast<float>(gen_int(int_seed))
                    : gen_real(int_seed);
        };

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float val = get_val();
            mem_ref.set_elem(
                    idx, round_to_nearest_representable(round_dt, val));
        }
    });

    // Note: `only_integer_` option is tricky. While it allows to avoid
    // cancellation effect triggering, it doesn't allow to validate loads
    // properly due to the values used. To ensure the library works properly,
    // basically, add fractional part of 0.5f for all floating types regardless
    // the setting, and for all integral types use values not available in other
    // data types to trigger potential overflow.
    if (fill_cfg.only_integer_) {
        const auto adjust_val = [&](float orig_val) {
            if (!is_integral_dt(round_dt)) {
                // Catch faulty integer loads instead fp type.
                return orig_val + 0.5f >= fill_cfg.range_max_val_
                        ? orig_val - 0.5f
                        : orig_val + 0.5f;
            } else if (round_dt == dnnl_s8) {
                ; // s8 is fine.
            } else if (round_dt == dnnl_u8) {
                return 128.f; // catch faulty s8 loads instead of u8.
            } else if (round_dt == dnnl_s32) {
                return 256.f; // catch faulty int8 loads instead of s32.
            } else {
                assert(!"unexpected data type");
            }
            return orig_val;
        };

        const float elem_first_val = adjust_val(mem_ref.get_elem(0));
        mem_ref.set_elem(
                0, round_to_nearest_representable(round_dt, elem_first_val));
    }

    if (mem) {
        // TODO: move `res` inside reorder.
        auto status = mem.reorder(mem_ref);
        if (status != OK) {
            if (res) res->state = FAILED;
            return status;
        }
    }

    return OK;
}

#ifdef DNNL_EXPERIMENTAL_SPARSE
int fill_random_real_sparse(const_dnnl_memory_t dnnl_memory, dnn_mem_t &mem,
        dnn_mem_t &mem_ref, res_t *res, const fill_cfg_t &fill_cfg) {
    auto orig_cc_mem_md = query_md(dnnl_memory);
    const int nhandles = query_md_num_handles(orig_cc_mem_md);
    assert(nhandles == 3);
    // Since a sparsity pattern affects performance, it's crucial to keep the
    // pattern intact and only randomize tensor values.
    // The assumption is every sparse format contains three handles and the
    // second and the third are responsible for a sparsity pattern.
    for (int idx = 1; idx < nhandles; idx++) {
        void *dst_ptr = mem_ref.get_mapped_pointer<void>(idx);
        void *src_ptr = nullptr;
        dnnl_memory_get_data_handle_v2(dnnl_memory, &src_ptr, idx);

        const size_t size = dnnl_memory_desc_get_size_v2(orig_cc_mem_md, idx);
        std::memcpy(dst_ptr, src_ptr, size);
    }

    return fill_random_real_dense(mem, mem_ref, res, fill_cfg);
}
#endif

int fill_random_real(dnn_mem_t &mem, dnn_mem_t &mem_ref, res_t *res,
        const fill_cfg_t &fill_cfg, const_dnnl_memory_t dnnl_memory) {
#ifdef DNNL_EXPERIMENTAL_SPARSE
    if (mem_ref.format_kind() == dnnl_format_kind_sparse) {
        assert(dnnl_memory != nullptr);
        return fill_random_real_sparse(
                dnnl_memory, mem, mem_ref, res, fill_cfg);
    }
#endif
    return fill_random_real_dense(mem, mem_ref, res, fill_cfg);
}

int fill_random_real(dnn_mem_t &mem_ref, const fill_cfg_t &fill_cfg,
        const_dnnl_memory_t dnnl_memory) {
    dnn_mem_t dummy;
    return fill_random_real(dummy, mem_ref, nullptr, fill_cfg, dnnl_memory);
}
