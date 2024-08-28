/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

static status_t init_conf_common(impl::engine_t *engine, const concat_pd_t *pd,
        reusable_simple_concat_params_t &conf,
        reusable_simple_concat_runtime_params_t &rt_conf) {
    using namespace utils;
    const memory_desc_t &ref_dst_md = *pd->dst_md();
    const memory_desc_wrapper ref_dst_mdw = *pd->dst_md();

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
    const int register_bytes = prb_info_t::register_bytes(hw);
    const int hw_threads = device_info->hw_threads();
    const int max_sg_size = device_info->max_subgroup_size();
    const auto data_type_size = normalize.data_type_size();
    dim_t dst_bytes = ref_dst_mdw.size();
    dim_t max_bytes = ref_dst_mdw.size();

    std::vector<prb_info_t> infos;
    for (int simd : {32, 16, 8, 1}) {
        if (simd > max_sg_size) continue;
        if (simd > 1 && !compute_engine->mayiuse_sub_group(simd)) continue;
        for (int bytes : {8, 4, 2, 1}) {
            if (has_scales && bytes < (int)data_type_size) break;
            if (max_write_size % bytes) continue;
            const dim_t total_elems = dst_bytes / bytes;
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
        max_bytes = std::max(max_bytes,
                into<dim_t>(memory_desc_wrapper(pd->src_md(i)).size()));
        memcpy(&src_md, pd->src_md(i), sizeof(memory_desc_t));
        normalize(src_md);
        const auto &src_blkg = src_md.format_desc.blocking;
        rt_conf.src_extern_dim_sizes[nonempty_inputs]
                = src_blkg.strides[axis::outer] * data_type_size;
        dim_t concat_dim = src_md.dims[axis::concat];
        dim_t concat_pdim = src_md.padded_dims[axis::concat];
        rt_conf.offset[nonempty_inputs] = offset;
        rt_conf.padded_offset[nonempty_inputs] = padded_offset;
        final_padding = concat_pdim - concat_dim;
        offset += concat_dim;
        padded_offset += concat_pdim;
        nonempty_inputs++;
    }
    memcpy(&dst_md, pd->dst_md(), sizeof(memory_desc_t));
    normalize(dst_md);
    const auto &dst_blkg = dst_md.format_desc.blocking;
    rt_conf.dst_extern_dim_size
            = dst_blkg.strides[axis::outer] * data_type_size;
    rt_conf.dst_padded_concat_axis = dst_md.padded_dims[axis::concat];
    rt_conf.dst_concat_axis
            = std::min(rt_conf.dst_padded_concat_axis, offset + final_padding);
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
    rt_conf.inner_axis = inner_offset;
    conf.data_type_size = info.type_size;
    rt_conf.dst_offset0 = dst_md.offset0 * data_type_size / info.type_size;
    conf.read_block = info.block;
    conf.write_block = std::min(info.block, max_write_size / info.type_size);
    // TODO: Fix math::lcm overflow
    dim_t shared_read = math::gcd(inner_axis, conf.read_block);
    rt_conf.gws0_block = inner_axis * conf.read_block / shared_read;
    rt_conf.read_overlap = rt_conf.gws0_block / inner_axis;
    rt_conf.gws_d[0] = rt_conf.gws0_block * conf.simd / conf.read_block;
    rt_conf.gws_d[1] = extern_axis / rt_conf.read_overlap;
    rt_conf.gws_d[2] = concat_dim_size;

    // Lots of zero padding byte writes -- very costly in this kernel
    if (conf.write_block * conf.data_type_size == 1
            && 4 * dst_md.dims[axis::concat]
                    <= dst_md.padded_dims[axis::concat]) {
        return status::unimplemented;
    }

    rt_conf.lws_d = compute::get_optimal_lws(
            rt_conf.gws_d, dim_idx::invalid, device_info->gpu_arch());

    conf.use_large_index = (max_bytes > std::numeric_limits<int>::max());
    //conf.use_large_index = (total_bytes > std::numeric_limits<int>::max()); //old?

    // attempt to enable internal padding kernel
    if (normalize.is_internal_padding_concat()) {
        size_t concat2_inner_axis = dst_md.dims[axis::inner];
        size_t concat2_dtsize = dnnl_data_type_size(dst_md.data_type);

        size_t min_bytes_per_thread = 8;
        size_t loads_per_thread = min_bytes_per_thread / concat2_dtsize;

        // heuristic for problem size too small
        size_t min_block_read_elements = conf.simd * loads_per_thread;
        size_t src0_inner_elems = rt_conf.offset[1] * concat2_inner_axis;
        bool src0_size_sufficient = src0_inner_elems > min_block_read_elements;

        // heuristic for data misaligned for subgroup loads/stores
        bool can_subgroup_read_dt = true;
        size_t min_subgroup_alignment_size = 4;
        if (concat2_dtsize < min_subgroup_alignment_size) {
            if (conf.n_blocks > 0) {
                can_subgroup_read_dt = (concat2_dtsize * conf.blocks[0])
                        >= min_subgroup_alignment_size;
            } else {
                can_subgroup_read_dt = false;
            }
        }

        // TODO: generalize "src_bank" calculation for smaller block sizes
        bool supported_block_size = (conf.n_blocks > 0)
                && ((conf.blocks[0] == 8) || (conf.blocks[0] == 16)
                        || (conf.blocks[0] == 32));

        bool can_use_internal_padding_concat2 = (conf.n == 2)
                && can_subgroup_read_dt && src0_size_sufficient
                && supported_block_size;
        if (can_use_internal_padding_concat2) {
            rt_conf.inner_axis = concat2_inner_axis;
            conf.data_type_size = concat2_dtsize;
            conf.use_internal_padding_kernel = true;

            rt_conf.gws_d[0] = utils::div_up(dst_md.padded_dims[axis::concat]
                                               * dst_md.dims[axis::inner],
                                       conf.simd * loads_per_thread)
                    * conf.simd;
            rt_conf.gws_d[1] = dst_md.dims[axis::outer];
            rt_conf.gws_d[2] = 1;

            // TODO: compute::get_optimal_lws( // no emperical diff
            rt_conf.lws_d[0] = conf.simd;
            rt_conf.lws_d[1] = 1;
            rt_conf.lws_d[2] = 1;

            //printf("\n dims[%zu %zu %zu] padded: [%zu %zu %zu]\n gws[%zu %zu %zu] simd:%zu dtsize:%zu\n",
            //dst_md.dims[axis::outer], dst_md.dims[axis::concat], dst_md.dims[axis::inner],
            //dst_md.padded_dims[axis::outer], dst_md.padded_dims[axis::concat], dst_md.padded_dims[axis::inner],
            //rt_conf.gws_d[0], rt_conf.gws_d[1], rt_conf.gws_d[2],
            //conf.simd, conf.data_type_size);
            //for(int i=0; i<conf.n_blocks; ++i){
            //printf("b%d: %d, ", i, conf.blocks[i]);
            //}
        }
    }

    return status::success;
}

compute::kernel_ctx_t reusable_simple_concat_params_t::get_kernel_ctx() const {
    compute::kernel_ctx_t kernel_ctx;
    kernel_ctx.define_int("WRITE_BLOCK", write_block);
    kernel_ctx.define_int("READ_BLOCK", read_block);
    kernel_ctx.define_int("N_INPUTS", n);
    kernel_ctx.define_int("BLOCK_DEPTH", n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        kernel_ctx.define_int(utils::format("BLOCK_B%d", i), blocks[i]);
        kernel_ctx.define_int(utils::format("BLOCK_S%d", i), strides[i]);
    }
    kernel_ctx.define_int("SIMD", simd);
    kernel_ctx.define_int("DATA_TYPE_SIZE", data_type_size);

    kernel_ctx.define_int("USE_LARGE_INDEX", use_large_index);
    return kernel_ctx;
}

status_t reusable_simple_concat_t::pd_t::init_conf(impl::engine_t *engine) {
    return init_conf_common(engine, this, conf, rt_conf);
}

template <typename IDX_T>
void push_idx_kernel_args(compute::kernel_arg_list_t &partial_list,
        const exec_ctx_t &ctx, const reusable_simple_concat_params_t &conf,
        const reusable_simple_concat_runtime_params_t &rt_conf,
        const concat_pd_t *pd) {
    const auto concat_dim = pd->concat_dim();

    bool cutoff = (rt_conf.dst_concat_axis % rt_conf.read_overlap != 0);
    for (int idx = 0, valid_idx = 0; idx < pd->n_inputs(); ++idx) {
        // skip invalid inputs
        if (pd->src_md(idx)->padded_dims[concat_dim] == 0) continue;

        auto &src = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC + idx);
        partial_list.append(src);

        partial_list.append(static_cast<IDX_T>(
                rt_conf.src_extern_dim_sizes[valid_idx] / conf.data_type_size));
        partial_list.append(static_cast<IDX_T>(rt_conf.offset[valid_idx]));
        partial_list.append(
                static_cast<IDX_T>(rt_conf.padded_offset[valid_idx]));
        dim_t src_concat_axis = valid_idx + 1 < conf.n
                ? rt_conf.offset[valid_idx + 1]
                : rt_conf.dst_concat_axis;
        partial_list.append(static_cast<IDX_T>(src_concat_axis));

        cutoff |= (rt_conf.offset[valid_idx] % rt_conf.read_overlap != 0);
        valid_idx++;
    }

    partial_list.append(static_cast<IDX_T>(rt_conf.dst_concat_axis));
    partial_list.append(static_cast<IDX_T>(rt_conf.dst_padded_concat_axis));

    partial_list.append(static_cast<IDX_T>(rt_conf.read_overlap));
    partial_list.append(static_cast<IDX_T>(rt_conf.gws0_block));
    partial_list.append(static_cast<IDX_T>(rt_conf.inner_axis));

    // Workgroup reads may extend past the concat dimension, so we must also
    // consider the external axis when computing write indices
    bool must_compute_ext_idx
            = (rt_conf.read_overlap * rt_conf.gws0_block > rt_conf.inner_axis)
            || cutoff;
    partial_list.append(static_cast<std::uint8_t>(must_compute_ext_idx));
}

template <typename IDX_T>
void push_idx_kernel_args_internal_padding(
        compute::kernel_arg_list_t &partial_list, const exec_ctx_t &ctx,
        const reusable_simple_concat_params_t &conf,
        const reusable_simple_concat_runtime_params_t &rt_conf,
        const concat_pd_t *pd) {
    const auto concat_dim = pd->concat_dim();

    partial_list.append(static_cast<IDX_T>(rt_conf.dst_concat_axis));
    partial_list.append(static_cast<IDX_T>(rt_conf.dst_padded_concat_axis));

    //printf("dst_concat_axis=%ld dst_padded_concat_axis=%ld\n",
    //rt_conf.dst_concat_axis, rt_conf.dst_padded_concat_axis);
    for (int idx = 0, valid_idx = 0; idx < pd->n_inputs(); ++idx) {
        // skip invalid inputs
        if (pd->src_md(idx)->padded_dims[concat_dim] == 0) continue;

        auto &src = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC + idx);
        partial_list.append(src);

        partial_list.append(static_cast<IDX_T>(rt_conf.offset[valid_idx]));
        partial_list.append(
                static_cast<IDX_T>(rt_conf.padded_offset[valid_idx]));
        dim_t src_concat_axis = valid_idx + 1 < conf.n
                ? rt_conf.offset[valid_idx + 1]
                : rt_conf.dst_concat_axis;
        partial_list.append(static_cast<IDX_T>(src_concat_axis));

        partial_list.append(pd->src_md(idx)->padded_dims[concat_dim]);

        //printf("offset%d=%ld padded_offset%d=%ld src_concat_axis%d=%ld "
        //"padded_src_concat_axis%d=%ld\n",
        //valid_idx, rt_conf.offset[valid_idx], valid_idx,
        //rt_conf.padded_offset[valid_idx], valid_idx, src_concat_axis,
        //valid_idx, pd->src_md(idx)->padded_dims[concat_dim]);
        valid_idx++;
    }

    partial_list.append(static_cast<IDX_T>(
            rt_conf.inner_axis)); //INNERDIM add only for internal_pad kernel
    //printf("inner_axis=%ld \n", rt_conf.inner_axis);
}

status_t reusable_simple_concat_t::execute_concat(const exec_ctx_t &ctx) const {
    const auto &conf = pd()->conf;
    const auto &rt_conf = pd()->rt_conf;
    if (conf.n == 0) return status::success;

    compute::kernel_arg_list_t arg_list;
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    arg_list.append(dst);

    auto nd_range = compute::nd_range_t(rt_conf.gws_d, rt_conf.lws_d);

    status_t status;
    if (conf.use_internal_padding_kernel) {
        if (conf.use_large_index) {
            push_idx_kernel_args_internal_padding<std::uint64_t>(
                    arg_list, ctx, conf, rt_conf, pd());
        } else {
            push_idx_kernel_args_internal_padding<int>(
                    arg_list, ctx, conf, rt_conf, pd());
        }
        status = parallel_for(
                ctx, nd_range, internal_padding_kernel_, arg_list);
    } else {
        arg_list.append(static_cast<std::uint64_t>(rt_conf.dst_offset0));
        arg_list.append(static_cast<std::uint64_t>(
                rt_conf.dst_extern_dim_size / conf.data_type_size));

        if (conf.use_large_index) {
            push_idx_kernel_args<std::uint64_t>(
                    arg_list, ctx, conf, rt_conf, pd());
        } else {
            push_idx_kernel_args<int>(arg_list, ctx, conf, rt_conf, pd());
        }
        status = parallel_for(ctx, nd_range, kernel_, arg_list);
    }
    return status;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
