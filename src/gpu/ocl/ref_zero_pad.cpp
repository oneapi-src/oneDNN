/*******************************************************************************
 * Copyright 2020 Intel Corporation
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

#include "gpu/ocl/ref_zero_pad.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_zero_pad_t::execute(const exec_ctx_t &ctx) const {

    compute::kernel_arg_list_t arg_list;

    const memory_t *memory = ctx.input(DNNL_ARG_SRC);
    memory_storage_t *mem_storage = memory->memory_storage();
    memory_desc_wrapper mdw(memory->md());

    const int ndims = mdw.ndims();
    const auto &dims = mdw.dims();
    const auto &pdims = mdw.padded_dims();
    const blocking_desc_t blocking_desc = mdw.blocking_desc();
    const ptrdiff_t nelems = (ptrdiff_t)mdw.nelems(true);
    const unsigned int hw_threads
            = utils::downcast<compute::compute_engine_t *>(
                    ctx.stream()->engine())
                      ->device_info()
                      ->hw_threads();

    // Setup Initial parameters used in opencl kernel computation
    dims_t blk_size;
    for (int i = 0; i < ndims; i++) {
        blk_size[i] = 1;
    }

    cl_ulong step_nelems = 1;
    for (int i = 0; i < blocking_desc.inner_nblks; i++) {
        step_nelems *= blocking_desc.inner_blks[i];
        blk_size[blocking_desc.inner_idxs[i]] *= blocking_desc.inner_blks[i];
    }

    // This constant needs to be the same as DEFAULT_NELEMS_BLOCK in
    // ref_zero_pad.cl
    const int default_nelems_block = 8;

    // This divisibility condition cannot be changed without some modifications
    // to use of DEFAULT_NELEMS_BLOCK in ref_zero_pad.cl
    size_t nelems_block = 1;
    while (nelems_block < default_nelems_block
            && step_nelems % (nelems_block * 2) == 0)
        nelems_block *= 2;

    arg_list.set(0, *mem_storage);
    arg_list.set(1, mdw.data_type_size());
    arg_list.set(2, step_nelems);
    arg_list.set(3, nelems_block);

    for (int i = 0; i < ndims; i++) {
        if (dims[i] == pdims[i]) continue;
        cl_ulong stride = 1;
        cl_ulong step_count = 1;

        step_count = blocking_desc.strides[i] / step_nelems;
        stride = blocking_desc.strides[i] * (pdims[i] / blk_size[i]);
        size_t npsteps = (nelems / stride) * step_count;

        // Balance work unit size with parallelism
        cl_ulong step_block = 1;
        while (step_nelems / nelems_block * step_block < 4 * 1024
                && step_count % (step_block * 2) == 0
                && npsteps / step_block > 2 * hw_threads) {
            step_block *= 2;
        }

        dim_t tail_start = dims[i] % blk_size[i];
        dims_t pos;
        for (int j = 0; j < ndims; j++) {
            pos[j] = 0;
        }

        zero_pad_mask_t bitmask;
        for (unsigned int j = 0; j < ZERO_PAD_MASK_SIZE; j++)
            bitmask.mask[j] = 0;

        bool is_done = false;
        while (!is_done) {
            size_t idx = mdw.off_v(pos, true);
            bitmask.mask[idx / 8]
                    |= (pos[i] >= tail_start ? (1 << (idx % 8)) : 0);

            //Increment position in the block
            is_done = true;
            for (int j = 0; j < ndims; j++) {
                if (blk_size[j] - 1 == pos[j]) continue;
                is_done = false;
                pos[j] = pos[j] + 1;
                for (int k = j - 1; k >= 0; k--)
                    pos[k] = 0;
                break;
            }
        }

        arg_list.set(4, step_block);
        arg_list.set(5, step_count);
        arg_list.set(6, stride);
        arg_list.set(7, bitmask);

        const size_t gws[3]
                = {nelems_block, step_count / step_block, npsteps / step_count};
        const compute::nd_range_t nd_range = compute::nd_range_t(3, gws);
        status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);
        if (status != status::success) return status;
    }
    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
