/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "gpu/compute/dispatch.hpp"
#include "gpu/ocl/simple_concat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

/* Returns dimension indices in (our best guess at) nesting order */
std::vector<int> get_ordered_dim_idxs(const memory_desc_wrapper &mdw) {
    const auto ndims = mdw.ndims();
    std::vector<int> idxs(ndims);
    for (int i = 0; i < ndims; ++i)
        idxs[i] = i;

    const auto &strides = mdw.blocking_desc().strides;
    const auto &sizes = mdw.dims();
    const auto cmp = [&](int a, int b) {
        // Dimensions of size 1 have the same stride as the next outer
        // dimension, so only sorting by strides will not necessarily get the
        // correct order. In the case of ties, the dim with the larger size gets
        // sorted second. Permutations of dims of size 1 with the same stride
        // should not effect the correctness of concat.
        return strides[a] < strides[b]
                || (strides[a] == strides[b] && sizes[a] < sizes[b]);
    };
    std::sort(idxs.begin(), idxs.end(), cmp);
    return idxs;
}

// Returns true if two sets of data have the same order of axis (i.e. the
// strides are ordered by the same permutation) and the layout is dense except
// possibly exterior to the concat dimension.
bool is_striding_ok(const std::vector<int> &idxs,
        const memory_desc_wrapper &mdw, int concat_dim) {
    const auto ndims = mdw.ndims();

    // Compute the total size of blocks for each dim to help predict strides
    std::vector<dim_t> blocks(ndims, 1);
    const auto &blkg = mdw.blocking_desc();
    dim_t exp_stride = 1;
    for (int i = 0; i < blkg.inner_nblks; ++i) {
        blocks[blkg.inner_idxs[i]] *= blkg.inner_blks[i];
        exp_stride *= blkg.inner_blks[i];
    }

    // Check that the order specified by idxs matches the src tensor
    const auto &padded_dims = mdw.padded_dims();
    bool allow_larger_stride = false;
    for (auto idx : idxs) {
        auto stride = blkg.strides[idx];
        if (stride < exp_stride) return false;
        if (stride > exp_stride && !allow_larger_stride) return false;
        auto step = utils::div_up(padded_dims[idx], blocks[idx]);
        exp_stride = stride * step;
        allow_larger_stride = idx == concat_dim;
    }
    return true;
}

struct prb_info_t {
    static constexpr dim_t max_block_size = 1024;
    static constexpr int scattered_message_penalty = 4;

    int simd;
    int type_size;
    int block;
    int messages;

    prb_info_t(int simd, int hw_threads, int type_size, dim_t inner_size,
            dim_t outer_elems)
        : simd(simd), type_size(type_size) {
        int best_block = 1;
        int best_messages = subgroup_messages(simd, best_block);
        dim_t inner_elems = inner_size / type_size;
        for (int i = 1; i < simd * 32; ++i) {
            dim_t block_size = i * type_size;
            if (block_size > max_block_size) break;
            if (block_size > inner_size) break;
            if (inner_elems % i) continue;
            // From empirical observations on ATSM-512, reduce max block size if
            // we are already occupying all threads.
            dim_t reqd_threads = outer_elems * (inner_elems / i);
            if (reqd_threads > (dim_t)hw_threads
                    && 2 * block_size > max_block_size)
                continue;
            int messages = subgroup_messages(simd, i);
            if (i * best_messages < best_block * messages) continue;
            if (i * best_messages == best_block * messages && i < best_block)
                continue;
            best_block = i;
            best_messages = messages;
        }
        block = best_block;
        messages = best_messages;
    }

    static int subgroup_messages(int simd, int block) {
        const int set_bits[] = {0, 1, 1, 2, 1, 2, 2, 3};
        int thread_block_elems = block / simd;
        int block_messages
                = (thread_block_elems / 8) + set_bits[thread_block_elems & 0x7];
        int scattered_messages = block % simd ? 1 : 0;
        return block_messages + scattered_messages * scattered_message_penalty;
    }

    bool operator<(const prb_info_t &other) const {
        auto average_bytes = type_size * block * other.messages;
        auto other_average_bytes = other.type_size * other.block * messages;
        if (average_bytes != other_average_bytes)
            return average_bytes > other_average_bytes;
        return type_size * block > other.type_size * other.block;
    }
};

static status_t init_conf_common(
        engine_t *engine, concat_conf_t &conf, const concat_pd_t *pd) {
    using namespace utils;

    const memory_desc_wrapper dst_mdw(pd->dst_md());
    auto ndims = dst_mdw.ndims();
    auto nelems = dst_mdw.nelems(true);
    auto data_type_size = dst_mdw.data_type_size();

    if (nelems == 0) return status::unimplemented;

    const auto &blk = dst_mdw.blocking_desc();
    const auto concat_dim = pd->concat_dim();
    // TODO: refactor to avoid duplication in get_ordered_dim_idxs and
    // is_same_axis_order.
    dim_t extern_axis = 1;
    int extern_dim = -1;
    std::vector<dim_t> blocks(ndims, 1);
    for (int i = 0; i < blk.inner_nblks; ++i)
        blocks[blk.inner_idxs[i]] *= blk.inner_blks[i];
    bool equal_strides_ok
            = dst_mdw.padded_dims()[concat_dim] / blocks[concat_dim] == 1;
    for (int i = 0; i < ndims; ++i) {
        const auto &stride = blk.strides[i];
        if (stride > blk.strides[concat_dim]
                || (equal_strides_ok && stride == blk.strides[concat_dim])) {
            if (extern_dim == -1 || stride < blk.strides[extern_dim])
                extern_dim = i;
            extern_axis *= dst_mdw.padded_dims()[i] / blocks[i];
        }
    }

    int offset = 0;
    const auto dst_dim_order = get_ordered_dim_idxs(dst_mdw);
    const dim_t c_blks = blocks[concat_dim];
    auto concat_dim_size = dst_mdw.padded_dims()[concat_dim] / c_blks;
    auto common_factor = concat_dim_size;
    int nonempty_inputs = 0;
    bool has_padding = false;
    for (int i = 0; i < pd->n_inputs(); ++i) {
        const memory_desc_wrapper src_mdw(pd->src_md(i));

        if (src_mdw.padded_dims()[concat_dim] == 0) continue;

        // check concat dim padding -- allow padding in the last nonempty input
        if (has_padding)
            return status::unimplemented;
        else if (src_mdw.padded_dims()[concat_dim]
                != src_mdw.dims()[concat_dim])
            has_padding = true;

        if (src_mdw.data_type() != dst_mdw.data_type())
            return status::unimplemented;

        if (!types::blocking_desc_is_equal(*pd->dst_md(), *pd->src_md(i), true))
            return status::unimplemented;

        if (!is_striding_ok(dst_dim_order, src_mdw, concat_dim))
            return status::unimplemented;

        const auto &src_blk = src_mdw.blocking_desc();
        const auto step = src_mdw.padded_dims()[concat_dim] / c_blks;
        common_factor = math::gcd(common_factor, step);

        auto src_extern_dim_size = (extern_dim == -1)
                ? src_blk.strides[concat_dim] * step
                : src_blk.strides[extern_dim];
        conf.src_extern_dim_sizes[nonempty_inputs]
                = src_extern_dim_size * data_type_size;
        conf.offset[nonempty_inputs] = offset;
        nonempty_inputs++;
        offset += step;
    }

    conf.dst_extern_dim_size = (extern_dim == -1)
            ? blk.strides[concat_dim] * concat_dim_size
            : blk.strides[extern_dim];
    conf.inner_axis = blk.strides[concat_dim];
    conf.n = nonempty_inputs;

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto *device_info = compute_engine->device_info();

    for (int i = 0; i < conf.n; ++i)
        conf.offset[i] /= common_factor;
    concat_dim_size /= common_factor;
    conf.inner_axis *= common_factor;
    dim_t inner_size = conf.inner_axis * data_type_size;
    dim_t outer_elems = concat_dim_size * extern_axis;

    // TODO: add proper scales support
    const bool has_scales = false;
    const int hw_threads = device_info->hw_threads();
    std::vector<prb_info_t> infos;
    for (int simd : {16, 8, 1}) {
        if (simd > 1 && !compute_engine->mayiuse_sub_group(simd)) continue;
        if (has_scales) {
            infos.emplace_back(simd, hw_threads, (int)data_type_size,
                    inner_size, outer_elems);
            continue;
        }
        for (int bytes : {4, 2, 1}) {
            if (inner_size % bytes) continue;
            infos.emplace_back(
                    simd, hw_threads, bytes, inner_size, outer_elems);
        }
    }
    if (infos.empty()) return status::unimplemented;
    std::sort(infos.begin(), infos.end());
    const auto &info = infos[0];

    conf.simd = info.simd;
    conf.inner_axis = inner_size / info.type_size;
    conf.data_type_size = info.type_size;
    conf.dst_offset0 = dst_mdw.offset0() * data_type_size / info.type_size;
    conf.dst_extern_dim_size
            = conf.dst_extern_dim_size * data_type_size / info.type_size;
    conf.block = info.block;
    conf.gws_d[0] = conf.inner_axis / conf.block * conf.simd;
    conf.gws_d[1] = extern_axis;
    conf.gws_d[2] = concat_dim_size;

    // Bound estimates based on limited empirical evidence
    if (conf.simd == 1 && conf.gws_d[2] > 64) return status::unimplemented;
    if (conf.simd > 1 && inner_size < 32) return status::unimplemented;

    compute::get_optimal_lws(
            conf.gws_d, conf.lws_d, 3, 0, device_info->gpu_arch());
    return status::success;
}

static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const concat_conf_t &conf) {
    kernel_ctx.define_int("DST_EXT_OFFSET", conf.dst_extern_dim_size);
    for (int i = 0; i < conf.n; ++i) {
        kernel_ctx.define_int(utils::format("SRC%d_EXT_OFFSET", i),
                conf.src_extern_dim_sizes[i] / conf.data_type_size);
        kernel_ctx.define_int(utils::format("OFFSET%d", i), conf.offset[i]);
    }
    kernel_ctx.define_int(utils::format("OFFSET%d", conf.n), conf.gws_d[2]);
    kernel_ctx.define_int("INNER_OFFSET", conf.inner_axis);
    kernel_ctx.define_int("BLOCK", conf.block);
    kernel_ctx.define_int("N_INPUTS", conf.n);
    kernel_ctx.define_int("SIMD", conf.simd);
    kernel_ctx.define_int("DATA_TYPE_SIZE", conf.data_type_size);
    return status::success;
}

status_t simple_concat_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(engine, conf, this);
}

status_t simple_concat_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}
status_t simple_concat_t::execute_concat(const exec_ctx_t &ctx) const {

    const auto &conf = pd()->conf;
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dst);
    arg_list.set(1, conf.dst_offset0);
    int next_arg = 2;
    for (int i = 0; i < pd()->n_inputs(); ++i) {
        if (pd()->src_md(i)->dims[pd()->concat_dim()] == 0) continue;
        auto &src = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC + i);
        arg_list.set(next_arg++, src);
    }

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
