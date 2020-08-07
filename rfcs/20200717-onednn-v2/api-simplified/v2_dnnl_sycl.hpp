#include <CL/sycl.hpp>

#include "dnnl.hpp"

namespace dnnl {
namespace sycl {

enum class memory_kind { // maybe memory_model is better name?
    usm_device, // default for SYCL-agnostic engine creation
    usm_shared,
    buffer,
};

// ======
// Engine
// ======

// Note that engine::kind is dropped.
engine engine_create(const cl::sycl::device &dev, const cl::sycl::context &ctx);
cl::sycl::device engine_get_device(const engine &e);
cl::sycl::context engine_get_context(const engine &e);

// ======
// Stream
// ======

// Note, that stream_flags are dropped.
// So far seems they could be derived from queue (for DPC++ and OpenCL and
// could be assumed `in_order` for ThreadPool), so could dropped.
stream stream_create(const engine &e, cl::sycl::queue &queue);
cl::sycl::queue stream_get_queue(const stream &s);

// ======
// Memory
// ======

// for mkind == buffer, handle could only be DNNL_MEMORY_{ALLOCATE,NONE}
memory memory_create(const desc &md, const engine &aengine, memory_kind mkind,
        void *handle = DNNL_MEMORY_ALLOCATE);
memory memory_create(const desc &md, const engine &astream, memory_kind mkind,
        void *handle = DNNL_MEMORY_ALLOCATE);

// memory_kind could be changed during the lifetime, by setting the USM handle
// or SYCL buffer
memory_kind memory_get_memory_kind(const memory &amemory);

// memory_kind == buffer implied
template <typename T, int ndims>
memory memory_create(const desc &md, const engine &aengine,
        cl::sycl::buffer<T, ndims> buf, stream &s);

template <typename T, int ndims>
void memory_set_data_handle(memory &m, cl::sycl::buffer<T, ndims> b);
template <typename T, int ndims>
void memory_set_data_handle(memory &m, cl::sycl::buffer<T, ndims> b, stream &s);
template <typename T, int ndims = 1>
cl::sycl::buffer<T, ndims> memory_get_sycl_buffer(const memory &m);

// ==========
// Primitives
// ==========

cl::sycl::event primitive_execute(
        const primitive &p, const stream &s,
        const std::unordered_map<int, memory> &args,
        const std::vector<cl::sycl::event> &dependencies = {});

} // namespace sycl
} // namespace dnnl
