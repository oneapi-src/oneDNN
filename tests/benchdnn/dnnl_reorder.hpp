/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef DNNL_REORDER_HPP
#define DNNL_REORDER_HPP

#include <memory>

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

int execute_reorder(const dnn_mem_t &src, dnn_mem_t &dst,
        const_dnnl_primitive_attr_t attr) {

    std::shared_ptr<const dnn_mem_t> r_src(&src, [](const dnn_mem_t *) {});
    std::shared_ptr<dnn_mem_t> r_dst(&dst, [](dnn_mem_t *) {});

    dnnl_stream_t r_stream = stream_tgt;

    dnnl_primitive_desc_t r_pd = nullptr;
    dnnl_primitive_t r;

    dnnl_engine_t engine_cpu = nullptr;
    dnnl_stream_t stream_cpu = nullptr;

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
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    std::string driver = std::string(driver_name);
    bool is_reorder_related_driver = (driver == std::string("reorder")
            || driver == std::string("sum") || driver == std::string("concat"));
    if (!is_reorder_related_driver
            && (src.engine_kind() == dnnl_gpu
                    || dst.engine_kind() == dnnl_gpu)) {

        DNN_SAFE(dnnl_engine_create(&engine_cpu, dnnl_cpu, 0), CRIT);
        DNN_SAFE(dnnl_stream_create(
                         &stream_cpu, engine_cpu, dnnl_stream_default_flags),
                CRIT);

        dnnl_status_t status = dnnl_reorder_primitive_desc_create(
                &r_pd, &src.md_, engine_cpu, &dst.md_, engine_cpu, attr);
        if (status == dnnl_success) {
            // Create CPU memory objects wrapping mapped pointers of source and
            // destination
            r_src.reset(new dnn_mem_t(src.md_, engine_cpu, (void *)src));
            r_dst.reset(new dnn_mem_t(dst.md_, engine_cpu, (void *)dst));

            r_stream = stream_cpu;
        }
    }
#endif

    if (!r_pd) {
        DNN_SAFE(dnnl_reorder_primitive_desc_create(&r_pd, &src.md_,
                         src.engine(), &dst.md_, dst.engine(), attr),
                CRIT);
    }

    DNN_SAFE(dnnl_primitive_create(&r, r_pd), CRIT);
    dnnl_engine_t reorder_engine;
    DNN_SAFE(dnnl_primitive_desc_query(
                     r_pd, dnnl_query_engine, 0, &reorder_engine),
            CRIT);
    DNN_SAFE(dnnl_primitive_desc_destroy(r_pd), CRIT);

    args_t args;
    args.set(DNNL_ARG_FROM, *r_src);
    args.set(DNNL_ARG_TO, *r_dst);

    DNN_SAFE(execute_and_wait(r, r_stream, args), CRIT);
    DNN_SAFE(dnnl_primitive_destroy(r), CRIT);

    if (stream_cpu) DNN_SAFE(dnnl_stream_destroy(stream_cpu), CRIT);
    if (engine_cpu) DNN_SAFE(dnnl_engine_destroy(engine_cpu), CRIT);

    return OK;
}

#endif
