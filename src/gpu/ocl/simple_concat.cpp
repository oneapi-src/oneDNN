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

enum class striding_t {
    mismatched, // Formats are incompatible
    sparse, // Dimensions have additional striding
    partitioned, // Striding between concat dim and exterior dimensions
    dense
};

striding_t striding_type(const std::vector<int> &idxs,
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
    dim_t last_stride = exp_stride;
    striding_t striding = striding_t::dense;
    for (auto idx : idxs) {
        auto stride = blkg.strides[idx];
        if (stride < last_stride) return striding_t::mismatched;
        if (stride < exp_stride) return striding_t::sparse;
        if (stride > exp_stride) {
            if (!allow_larger_stride) return striding_t::sparse;
            striding = striding_t::partitioned;
        }
        last_stride = stride;
        exp_stride = stride * (padded_dims[idx] / blocks[idx]);
        allow_larger_stride = idx == concat_dim;
    }
    return striding;
}

// Returns true if two sets of data have the same order of axis (i.e. the
// strides are ordered by the same permutation) and the layout is dense except
// possibly exterior to the concat dimension.
bool is_striding_ok(striding_t striding) {
    return utils::one_of(striding, striding_t::dense, striding_t::partitioned);
}

struct prb_info_t {
    static constexpr int scattered_message_penalty = 2;

    int simd;
    int type_size;
    int block;
    int messages;

    prb_info_t(int simd, int type_size, dim_t max_elems, dim_t max_read_size,
            dim_t inner_size, dim_t outer_elems, compute::gpu_arch_t hw)
        : simd(simd), type_size(type_size) {
        int best_block = 1;
        int best_messages = subgroup_messages(simd, 1, type_size);
        const dim_t inner_elems = inner_size / type_size;
        for (int i = 1; i < simd * 32; ++i) {
            dim_t block_size = i * (dim_t)type_size;
            if (block_size > max_block_size(hw)) break;
            if (block_size > max_read_size) break;
            if (i > max_elems) break;
            if (i < inner_elems && inner_elems % i) continue;
            if (i > inner_elems
                    && (i % inner_elems || max_read_size % block_size))
                continue;
            int messages = subgroup_messages(simd, i, type_size);
            if (i * best_messages < best_block * messages) continue;
            if (i * best_messages == best_block * messages && i < best_block)
                continue;
            best_block = i;
            best_messages = messages;
        }
        block = best_block;
        messages = best_messages;
    }

    static dim_t max_block_size(compute::gpu_arch_t hw) {
        return hw == compute::gpu_arch_t::xe_hpc ? 1024 : 512;
    }

    static int subgroup_messages(int simd, int block, int type_size) {
        const int set_bits[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
        int thread_block_elems = block / simd;
        int max_vec_size = type_size == 1 ? 16 : 8;
        int num_max_vecs = thread_block_elems / max_vec_size;
        int num_small_vecs = set_bits[thread_block_elems & (max_vec_size - 1)];
        int block_messages = num_max_vecs + num_small_vecs;
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
    bool can_read_past_concat_dim = true;
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

        striding_t striding = striding_type(dst_dim_order, src_mdw, concat_dim);
        if (!is_striding_ok(striding)) return status::unimplemented;
        if (striding != striding_t::dense) can_read_past_concat_dim = false;

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
    dim_t max_read_size
            = can_read_past_concat_dim ? inner_size * extern_axis : inner_size;

    // TODO: add proper scales support
    const bool has_scales = false;
    const int eu_count = device_info->eu_count();
    const int max_sg_size = device_info->max_subgroup_size();
    const dim_t total_elems = conf.inner_axis * concat_dim_size * extern_axis;
    const dim_t max_elems = std::max((dim_t)1, total_elems / eu_count);
    std::vector<prb_info_t> infos;
    for (int simd : {32, 16, 8, 1}) {
        if (simd > max_sg_size) continue;
        if (simd > max_elems) continue;
        if (simd > 1 && !compute_engine->mayiuse_sub_group(simd)) continue;
        for (int bytes : {8, 4, 2, 1}) {
            if (has_scales && bytes < (int)data_type_size) break;
            if (inner_size % bytes) continue;
            infos.emplace_back(simd, bytes, max_elems, max_read_size,
                    inner_size, outer_elems, device_info->gpu_arch());
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
    if (conf.block < conf.inner_axis) {
        conf.gws_d[0] = conf.inner_axis / conf.block * conf.simd;
        conf.gws_d[1] = extern_axis;
    } else {
        conf.gws_d[0] = conf.simd;
        conf.gws_d[1] = extern_axis * conf.inner_axis / conf.block;
    }
    conf.gws_d[2] = concat_dim_size;

    // Bound estimates based on limited empirical evidence
    int coalesced_writes = ((inner_size ^ (inner_size - 1)) >> 1) + 1;
    size_t extern_axis_bound = 256 * 512 * std::min(coalesced_writes, 8);
    if (conf.simd == 1 && conf.gws_d[2] > 64) return status::unimplemented;
    if (conf.simd == 1 && conf.gws_d[1] > extern_axis_bound)
        return status::unimplemented;
    if (conf.inner_axis == 1 && 16 * conf.simd <= conf.block
            && (size_t)conf.block < conf.gws_d[2])
        return status::unimplemented;

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
