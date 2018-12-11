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

#include <CL/sycl.hpp>
#include <memory>

#include "mkldnn.hpp"

#include "common/gemm_utils.hpp"
#include "common/memory_storage.hpp"
#include "common/mkldnn_traits.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"
#include "ocl/jit_gen9_gemm.hpp"
#include "sycl/sycl_engine.hpp"
#include "sycl/sycl_stream.hpp"

using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::sycl;

namespace {

template <data_type_t data_type, typename T>
void gemm_generic(cl::sycl::queue &queue, mkldnn::transpose transa,
        mkldnn::transpose transb, dim_t m, dim_t n, dim_t k, float alpha,
        cl::sycl::buffer<T, 1> &a, dim_t offset_a, dim_t lda,
        cl::sycl::buffer<T, 1> &b, dim_t offset_b, dim_t ldb, float beta,
        cl::sycl::buffer<T, 1> &c, dim_t offset_c, dim_t ldc) {

    using data_t = typename prec_traits<data_type>::type;
    static_assert(sizeof(data_t) == sizeof(T), "not expected");

    status_t status;

    // Check inputs
    const char transa_c = (transa == mkldnn::transpose::notrans) ? 'n' : 't';
    const char transb_c = (transb == mkldnn::transpose::notrans) ? 'n' : 't';
    status = check_gemm_input(
            transa_c, transb_c, m, n, k, lda, ldb, ldc, alpha, beta);
    error::wrap_c_api(status, "invalid arguments");

    // Create engine
    cl::sycl::device dev = queue.get_device();
    cl::sycl::context ctx = queue.get_context();
    engine_kind_t eng_kind;
    if (dev.is_cpu() || dev.is_host()) {
        eng_kind = engine_kind::cpu;
        error::wrap_c_api(
                status::unimplemented, "SYCL CPU GEMM not implemented");
    } else {
        assert(dev.is_gpu());
        eng_kind = engine_kind::gpu;
    }

    std::unique_ptr<sycl_engine_base_t> engine;
    engine_t *engine_ptr;
    status = get_engine_factory(eng_kind)->engine_create(&engine_ptr, dev, ctx);
    error::wrap_c_api(status, "invalid queue");

    engine.reset(utils::downcast<sycl_engine_base_t *>(engine_ptr));

    // Create stream
    std::unique_ptr<stream_t> s;
    stream_t *s_ptr;
    status = engine->create_stream(&s_ptr, queue);
    error::wrap_c_api(status, "invalid queue");
    s.reset(s_ptr);

    // Create primitive descriptor
    using pd_type = typename ocl::jit_gen9_gemm_t<data_type>::pd_t;

    gemm_desc_t op_desc;
    op_desc.primitive_kind = mkldnn_gemm;
    op_desc.transa = static_cast<transpose_t>(transa);
    op_desc.transb = static_cast<transpose_t>(transb);

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
    error::wrap_c_api(status, "invalid arguments");
    pd.reset(pd_ptr);

    // Create memory objects
    auto a_buf_u8
            = a.template reinterpret<uint8_t>(cl::sycl::range<1>(a.get_size()));
    auto b_buf_u8
            = b.template reinterpret<uint8_t>(cl::sycl::range<1>(b.get_size()));
    auto c_buf_u8
            = c.template reinterpret<uint8_t>(cl::sycl::range<1>(c.get_size()));

    untyped_sycl_buffer_t a_untyped_buf(a_buf_u8);
    untyped_sycl_buffer_t b_untyped_buf(b_buf_u8);
    untyped_sycl_buffer_t c_untyped_buf(c_buf_u8);

    std::unique_ptr<memory_t> a_mem(
            new memory_t(engine.get(), &a_desc, memory_flags_t::use_backend_ptr,
                    static_cast<void *>(&a_untyped_buf)));
    std::unique_ptr<memory_t> b_mem(
            new memory_t(engine.get(), &b_desc, memory_flags_t::use_backend_ptr,
                    static_cast<void *>(&b_untyped_buf)));
    std::unique_ptr<memory_t> c_mem(
            new memory_t(engine.get(), &c_desc, memory_flags_t::use_backend_ptr,
                    static_cast<void *>(&c_untyped_buf)));
    if (!a_mem || !b_mem || !c_mem)
        error::wrap_c_api(status::out_of_memory, "could not create memory");

    a_mem->memory_storage()->set_offset(offset_a * sizeof(data_t));
    b_mem->memory_storage()->set_offset(offset_b * sizeof(data_t));
    c_mem->memory_storage()->set_offset(offset_c * sizeof(data_t));

    // Create primitive
    primitive_t *gemm_prim;
    status = pd->create_primitive(&gemm_prim);
    error::wrap_c_api(status, "could not create a primitive");

    exec_args_t args = {
        { MKLDNN_ARG_SRC_0, { a_mem.get(), true } },
        { MKLDNN_ARG_SRC_1, { b_mem.get(), true } },
        { MKLDNN_ARG_DST, { c_mem.get(), false } },
    };

    exec_ctx_t exec_ctx(s.get(), std::move(args));
    status = gemm_prim->execute(exec_ctx);
    gemm_prim->release();
    error::wrap_c_api(status, "could not execute a primitive");

    error::wrap_c_api(s->wait(), "could not wait a stream");
}

} // namespace

namespace mkldnn {

void MKLDNN_API gemm(cl::sycl::queue &queue, mkldnn::transpose transa,
        mkldnn::transpose transb, dim_t m, dim_t n, dim_t k, float alpha,
        cl::sycl::buffer<float, 1> &a, dim_t offset_a, dim_t lda,
        cl::sycl::buffer<float, 1> &b, dim_t offset_b, dim_t ldb, float beta,
        cl::sycl::buffer<float, 1> &c, dim_t offset_c, dim_t ldc) {
    return gemm_generic<data_type::f32>(queue, transa, transb, m, n, k, alpha,
            a, offset_a, lda, b, offset_b, ldb, beta, c, offset_c, ldc);
}

void MKLDNN_API gemm(cl::sycl::queue &queue, mkldnn::transpose transa,
        mkldnn::transpose transb, dim_t m, dim_t n, dim_t k, float alpha,
        cl::sycl::buffer<cl::sycl::half, 1> &a, dim_t offset_a, dim_t lda,
        cl::sycl::buffer<cl::sycl::half, 1> &b, dim_t offset_b, dim_t ldb,
        float beta, cl::sycl::buffer<cl::sycl::half, 1> &c, dim_t offset_c,
        dim_t ldc) {
    return gemm_generic<data_type::f16>(queue, transa, transb, m, n, k, alpha,
            a, offset_a, lda, b, offset_b, ldb, beta, c, offset_c, ldc);
}
} // namespace mkldnn
