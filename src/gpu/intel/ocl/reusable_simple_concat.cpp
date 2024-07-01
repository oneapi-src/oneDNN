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
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>

#include "gpu/intel/compute/dispatch.hpp"
#include "gpu/intel/ocl/concat_utils.hpp"
#include "gpu/intel/ocl/reusable_simple_concat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

// returns the total number of bytes kernel parameters will demand for n inputs
size_t kernel_parameter_bytes(
        const size_t n, const size_t n_blocks, const size_t idx_type_size) {
    // zero_pad_offset, zero_pad_concat_axis, read_overlap, gws0_block, inner_offset, must_compute_ext_idx
    const size_t additional_param_bytes
            = 5 * idx_type_size + sizeof(unsigned char);

    // dst, dst_offset0, dst_ext_offset,
    const size_t dst_param_bytes = sizeof(void *) + 2 * idx_type_size;

    // *src##n, src_ext_offset##n, offset##n, padded_offset##n, src_concat_axis##n
    const size_t src_param_bytes = n * (sizeof(void *) + (4 * idx_type_size));

    return dst_param_bytes + src_param_bytes + additional_param_bytes;
}

static status_t init_conf_common(
        impl::engine_t *engine, concat_conf_t &conf, const concat_pd_t *pd) {
    using namespace utils;
    const memory_desc_t &ref_dst_md = *pd->dst_md();
    if (ref_dst_md.format_kind != format_kind::blocked) {
        return status::unimplemented;
    }
    const auto concat_dim = pd->concat_dim();

    normalization_t normalize(ref_dst_md, concat_dim);
    for (int i = 0; i < pd->n_inputs(); ++i) {
        const memory_desc_t &src_md = *pd->src_md(i);
        if (!normalize.add_source(src_md)) { return status::unimplemented; }
    }

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto *device_info = compute_engine->device_info();
    dim_t max_write_size = normalize.max_write_size();
    dim_t max_read_size = normalize.max_read_size();

    // TODO: add proper scales support
    const bool has_scales = false;
    const compute::gpu_arch_t hw = device_info->gpu_arch();
    // TODO: improve performance on xe2
    if (hw != compute::gpu_arch_t::xe_hpc
            && hw != compute::gpu_arch_t::xe_hpg) {
        return status::unimplemented;
    }

    const int register_bytes = prb_info_t::register_bytes(hw);
    const int hw_threads = device_info->hw_threads();
    const int max_sg_size = device_info->max_subgroup_size();
    const auto data_type_size = normalize.data_type_size();
    dim_t total_bytes = data_type_size;
    for (int i = 0; i < pd->dst_md()->ndims; ++i)
        total_bytes *= pd->dst_md()->padded_dims[i];

    std::vector<prb_info_t> infos;
    for (int simd : {32, 16, 8, 1}) {
        if (simd > max_sg_size) continue;
        if (simd > 1 && !compute_engine->mayiuse_sub_group(simd)) continue;
        for (int bytes : {8, 4, 2, 1}) {
            if (has_scales && bytes < (int)data_type_size) break;
            if (max_write_size % bytes) continue;
            const dim_t total_elems = total_bytes / bytes;
            const dim_t concurrent_elems
                    = utils::div_up(simd * total_elems, hw_threads);
            const dim_t elems_per_reg = register_bytes / bytes;
            const dim_t max_elems
                    = utils::rnd_up(concurrent_elems, elems_per_reg);
            if (simd > max_elems) continue;
            infos.emplace_back(simd, bytes, max_elems, max_read_size,
                    max_write_size, device_info->gpu_arch());
        }
    }
    if (infos.empty() || !infos[0].block) { return status::unimplemented; }
    std::sort(infos.begin(), infos.end());
    const auto &info = infos[0];

    memory_desc_t dst_md, src_md;
    int offset = 0, padded_offset = 0, nonempty_inputs = 0;
    dim_t final_padding = 0;
    for (int i = 0; i < pd->n_inputs(); ++i) {
        if (pd->src_md(i)->padded_dims[concat_dim] == 0) continue;
        memcpy(&src_md, pd->src_md(i), sizeof(memory_desc_t));
        normalize(src_md);
        const auto &src_blkg = src_md.format_desc.blocking;
        conf.src_extern_dim_sizes[nonempty_inputs]
                = src_blkg.strides[axis::outer] * data_type_size;
        dim_t concat_dim = src_md.dims[axis::concat];
        dim_t concat_pdim = src_md.padded_dims[axis::concat];
        conf.offset[nonempty_inputs] = offset;
        conf.padded_offset[nonempty_inputs] = padded_offset;
        final_padding = concat_pdim - concat_dim;
        offset += concat_dim;
        padded_offset += concat_pdim;
        nonempty_inputs++;
    }
    memcpy(&dst_md, pd->dst_md(), sizeof(memory_desc_t));
    normalize(dst_md);
    const auto &dst_blkg = dst_md.format_desc.blocking;
    conf.dst_extern_dim_size = dst_blkg.strides[axis::outer] * data_type_size;
    conf.dst_padded_concat_axis = dst_md.padded_dims[axis::concat];
    conf.dst_concat_axis
            = std::min(conf.dst_padded_concat_axis, offset + final_padding);
    dim_t concat_dim_size = padded_offset;

    conf.n_blocks = 0;
    dim_t stride = 1;
    for (int i = dst_blkg.inner_nblks - 1; i >= 0; --i) {
        auto blk = dst_blkg.inner_blks[i];
        auto idx = dst_blkg.inner_idxs[i];
        if (i == dst_blkg.inner_nblks - 1)
            blk = blk * data_type_size / info.type_size;
        if (idx == axis::concat) {
            conf.blocks[conf.n_blocks] = blk;
            conf.strides[conf.n_blocks] = stride;
            conf.n_blocks++;
        }
        stride *= blk;
    }

    dim_t extern_axis = dst_md.dims[axis::outer];
    dim_t inner_axis
            = dst_md.padded_dims[axis::inner] * data_type_size / info.type_size;
    dim_t inner_offset
            = dst_blkg.strides[axis::concat] * data_type_size / info.type_size;
    conf.n = nonempty_inputs;
    conf.simd = info.simd;
    conf.inner_axis = inner_offset;
    conf.data_type_size = info.type_size;
    conf.dst_offset0 = dst_md.offset0 * data_type_size / info.type_size;
    conf.read_block = info.block;
    conf.write_block = std::min(info.block, max_write_size / info.type_size);
    // TODO: Fix math::lcm overflow
    dim_t shared_read = math::gcd(inner_axis, conf.read_block);
    conf.gws0_block = inner_axis * conf.read_block / shared_read;
    conf.read_overlap = conf.gws0_block / inner_axis;
    conf.gws_d[0] = conf.gws0_block * conf.simd / conf.read_block;
    conf.gws_d[1] = extern_axis / conf.read_overlap;
    conf.gws_d[2] = concat_dim_size;

    // Bound estimates based on limited empirical evidence
    int coalesced_writes = ((max_write_size ^ (max_write_size - 1)) >> 1) + 1;
    size_t extern_axis_bound = 256 * 512 * std::min(coalesced_writes, 8);
    if (conf.simd == 1 && conf.gws_d[2] > 64) return status::unimplemented;
    if (conf.simd == 1 && conf.gws_d[1] > extern_axis_bound) {
        return status::unimplemented;
    }
    if (conf.inner_axis == 1 && 16 * conf.simd <= conf.read_block
            && (size_t)conf.read_block < conf.gws_d[2]) {
        return status::unimplemented;
    }
    if (conf.n_blocks && conf.write_block * conf.data_type_size == 1) {
        return status::unimplemented;
    }
    // In this band, this implementation underperforms against either
    // (1) gen9 concat or (2) ref concat.
    if (dst_md.dims[axis::inner] > (1l << 20) // (1)
            && dst_md.dims[axis::inner] <= (1l << 24) // (2)
            && dst_md.dims[axis::concat] != dst_md.padded_dims[axis::concat]
            && dst_md.dims[axis::concat] < 8) {
        return status::unimplemented;
    }

    conf.lws_d
            = compute::get_optimal_lws(conf.gws_d, 0, device_info->gpu_arch());

    conf.use_large_index = (total_bytes > std::numeric_limits<int>::max());
    size_t param_bytes = (conf.use_large_index)
            ? kernel_parameter_bytes(conf.n, conf.n_blocks, sizeof(dim_t))
            : kernel_parameter_bytes(conf.n, conf.n_blocks, sizeof(int));
    if (param_bytes > device_info->max_kernel_param_size()) {
        return status::unimplemented;
    }
    return status::success;
}

static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const concat_conf_t &conf) {
    kernel_ctx.define_int("WRITE_BLOCK", conf.write_block);
    kernel_ctx.define_int("READ_BLOCK", conf.read_block);
    kernel_ctx.define_int("N_INPUTS", conf.n);
    kernel_ctx.define_int("BLOCK_DEPTH", conf.n_blocks);
    for (int i = 0; i < conf.n_blocks; ++i) {
        kernel_ctx.define_int(utils::format("BLOCK_B%d", i), conf.blocks[i]);
        kernel_ctx.define_int(utils::format("BLOCK_S%d", i), conf.strides[i]);
    }
    kernel_ctx.define_int("SIMD", conf.simd);
    kernel_ctx.define_int("DATA_TYPE_SIZE", conf.data_type_size);

    kernel_ctx.define_int("USE_LARGE_INDEX", conf.use_large_index);
    return status::success;
}

status_t reusable_simple_concat_t::pd_t::init_conf(impl::engine_t *engine) {
    return init_conf_common(engine, conf, this);
}

status_t reusable_simple_concat_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

template <typename IDX_T>
void push_idx_kernel_args(compute::kernel_arg_list_t &partial_list,
        const exec_ctx_t &ctx, const concat_conf_t &conf,
        const concat_pd_t *pd) {
    const auto concat_dim = pd->concat_dim();

    bool cutoff = (conf.dst_concat_axis % conf.read_overlap != 0);
    for (int idx = 0, valid_idx = 0; idx < pd->n_inputs(); ++idx) {
        // skip invalid inputs
        if (pd->src_md(idx)->padded_dims[concat_dim] == 0) continue;

        auto &src = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC + idx);
        partial_list.append(src);

        partial_list.append(static_cast<IDX_T>(
                conf.src_extern_dim_sizes[valid_idx] / conf.data_type_size));
        partial_list.append(static_cast<IDX_T>(conf.offset[valid_idx]));
        partial_list.append(static_cast<IDX_T>(conf.padded_offset[valid_idx]));
        dim_t src_concat_axis = valid_idx + 1 < conf.n
                ? conf.offset[valid_idx + 1]
                : conf.dst_concat_axis;
        partial_list.append(static_cast<IDX_T>(src_concat_axis));

        cutoff |= (conf.offset[valid_idx] % conf.read_overlap != 0);
        valid_idx++;
    }
    partial_list.append(static_cast<IDX_T>(conf.dst_concat_axis));
    partial_list.append(static_cast<IDX_T>(conf.dst_padded_concat_axis));

    partial_list.append(static_cast<IDX_T>(conf.read_overlap));
    partial_list.append(static_cast<IDX_T>(conf.gws0_block));
    partial_list.append(static_cast<IDX_T>(conf.inner_axis));
    // Workgroup reads may extend past the concat dimension, so we must also
    // consider the external axis when computing write indices
    bool must_compute_ext_idx
            = (conf.read_overlap * conf.gws0_block > conf.inner_axis) || cutoff;
    partial_list.append(static_cast<std::uint8_t>(must_compute_ext_idx));
}

status_t reusable_simple_concat_t::execute_concat(const exec_ctx_t &ctx) const {
    const auto &conf = pd()->conf;
    if (conf.n == 0) return status::success;

    compute::kernel_arg_list_t arg_list;
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    arg_list.append(dst);
    arg_list.append(static_cast<std::uint64_t>(conf.dst_offset0));
    arg_list.append(static_cast<std::uint64_t>(
            conf.dst_extern_dim_size / conf.data_type_size));

    if (conf.use_large_index) {
        push_idx_kernel_args<std::uint64_t>(arg_list, ctx, conf, pd());
    } else {
        push_idx_kernel_args<int>(arg_list, ctx, conf, pd());
    }

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
