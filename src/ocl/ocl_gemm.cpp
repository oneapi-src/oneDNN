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

#include <CL/cl.h>
#include <memory>

#include "mkldnn.h"

#include "common/gemm_utils.hpp"
#include "common/memory_storage.hpp"
#include "common/mkldnn_traits.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"
#include "ocl/jit_gen9_gemm.hpp"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::ocl;

namespace {

template <data_type_t data_type>
mkldnn_status_t gemm_generic(cl_command_queue queue, const char *transa,
        const char *transb, dim_t m, dim_t n, dim_t k, cl_float alpha, cl_mem a,
        dim_t offset_a, dim_t lda, cl_mem b, dim_t offset_b, dim_t ldb,
        cl_float beta, cl_mem c, dim_t offset_c, dim_t ldc) {

    using data_t = typename prec_traits<data_type>::type;

    status_t status;

    // Check inputs
    status = check_gemm_input(
            *transa, *transb, m, n, k, lda, ldb, ldc, alpha, beta);
    if (status != mkldnn_success)
        return status;

    // Create engine
    cl_context ocl_ctx;
    OCL_CHECK(clGetCommandQueueInfo(
            queue, CL_QUEUE_CONTEXT, sizeof(ocl_ctx), &ocl_ctx, nullptr));

    cl_device_id ocl_dev;
    OCL_CHECK(clGetCommandQueueInfo(
            queue, CL_QUEUE_DEVICE, sizeof(ocl_dev), &ocl_dev, nullptr));

    std::unique_ptr<ocl_engine_t> engine;
    engine_t *engine_ptr;

    status = ocl_engine_factory_t().engine_create(
            &engine_ptr, ocl_dev, ocl_ctx);
    if (status != status::success)
        return status;
    engine.reset(utils::downcast<ocl_engine_t *>(engine_ptr));

    // Create stream
    std::unique_ptr<stream_t> s;
    stream_t *s_ptr;
    status = engine->create_stream(&s_ptr, queue);
    if (status != status::success)
        return status;
    s.reset(s_ptr);

    // Create primitive descriptor
    using pd_type = typename jit_gen9_gemm_t<data_type>::pd_t;

    gemm_desc_t op_desc;
    op_desc.primitive_kind = mkldnn_gemm;
    op_desc.transa = (*transa == 'n' || *transa == 'N') ? transpose::notrans
                                                        : transpose::trans;
    op_desc.transb = (*transb == 'n' || *transb == 'N') ? transpose::notrans
                                                        : transpose::trans;
    op_desc.m = m;
    op_desc.n = n;
    op_desc.k = k;
    op_desc.lda = lda;
    op_desc.ldb = ldb;
    op_desc.ldc = ldc;
    op_desc.alpha = alpha;
    op_desc.beta = beta;
    op_desc.a_type = data_type;
    op_desc.b_type = data_type;
    op_desc.c_type = data_type;

    mkldnn_memory_desc_t a_desc, b_desc, c_desc;

    status = create_gemm_memory_desc(&a_desc, &op_desc, 0, data_type);
    assert(status == status::success);
    status = create_gemm_memory_desc(&b_desc, &op_desc, 1, data_type);
    assert(status == status::success);
    status = create_gemm_memory_desc(&c_desc, &op_desc, 2, data_type);
    assert(status == status::success);

    std::unique_ptr<primitive_desc_t> pd;
    primitive_attr_t attr;
    primitive_desc_t *pd_ptr;
    status = primitive_desc_t::create<pd_type>(&pd_ptr,
            reinterpret_cast<const op_desc_t *>(&op_desc), &attr, engine.get(),
            nullptr);
    if (status != status::success)
        return status;
    pd.reset(pd_ptr);

    // Create memory objects
    std::unique_ptr<memory_t> a_mem(new memory_t(
            engine.get(), &a_desc, memory_flags_t::use_backend_ptr, a));
    std::unique_ptr<memory_t> b_mem(new memory_t(
            engine.get(), &b_desc, memory_flags_t::use_backend_ptr, b));
    std::unique_ptr<memory_t> c_mem(new memory_t(
            engine.get(), &c_desc, memory_flags_t::use_backend_ptr, c));
    if (!a_mem || !b_mem || !c_mem)
        return status::out_of_memory;

    a_mem->memory_storage()->set_offset(offset_a * sizeof(data_t));
    b_mem->memory_storage()->set_offset(offset_b * sizeof(data_t));
    c_mem->memory_storage()->set_offset(offset_c * sizeof(data_t));

    // Create primitive
    std::unique_ptr<primitive_t> gemm_prim;
    primitive_t *gemm_prim_ptr;
    status = pd->create_primitive(&gemm_prim_ptr);
    if (status != status::success)
        return status;
    gemm_prim.reset(gemm_prim_ptr);

    exec_args_t args = {
        { MKLDNN_ARG_SRC_0, { a_mem.get(), true } },
        { MKLDNN_ARG_SRC_1, { b_mem.get(), true } },
        { MKLDNN_ARG_DST, { c_mem.get(), false } },
    };

    exec_ctx_t ctx(s.get(), std::move(args));
    status = gemm_prim->execute(ctx);
    if (status != status::success)
        return status;

    return s->wait();
}

} // namespace

extern "C" {
mkldnn_status_t MKLDNN_API mkldnn_ocl_sgemm(cl_command_queue queue,
        char transa, char transb, dim_t m, dim_t n, dim_t k,
        cl_float alpha, cl_mem a, dim_t offset_a, dim_t lda, cl_mem b,
        dim_t offset_b, dim_t ldb, cl_float beta, cl_mem c, dim_t offset_c,
        dim_t ldc) {
    return gemm_generic<data_type::f32>(queue, &transb, &transa, n, m, k, alpha,
            b, offset_b, ldb, a, offset_a, lda, beta, c, offset_c, ldc);
}

mkldnn_status_t MKLDNN_API mkldnn_ocl_hgemm(cl_command_queue queue,
        char transa, char transb, dim_t m, dim_t n, dim_t k,
        cl_float alpha, cl_mem a, dim_t offset_a, dim_t lda, cl_mem b,
        dim_t offset_b, dim_t ldb, cl_float beta, cl_mem c, dim_t offset_c,
        dim_t ldc) {
    return gemm_generic<data_type::f16>(queue, &transb, &transa, n, m, k, alpha,
            b, offset_b, ldb, a, offset_a, lda, beta, c, offset_c, ldc);
}
}
