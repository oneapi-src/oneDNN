#include <CL/sycl.hpp>

#include "dnnl.hpp"

namespace dnnl {
namespace sycl_interop {

enum class memory_kind { // maybe memory_model is better name?
    usm_device, // default for SYCL-agnostic engine creation
    usm_shared,
    buffer,
};

// ======
// Engine
// ======

// Note that engine::kind is dropped.
engine make_engine(const cl::sycl::device &dev, const cl::sycl::context &ctx);
cl::sycl::device get_device(const engine &e);
cl::sycl::context get_context(const engine &e);

// ======
// Stream
// ======

// Note, that stream_flags are dropped.
// So far seems they could be derived from queue (for DPC++ and OpenCL and
// could be assumed `in_order` for ThreadPool), so could dropped.
stream make_stream(const engine &e, cl::sycl::queue &queue);
cl::sycl::queue get_queue(const stream &s);

// ======
// Memory
// ======

// for mkind == buffer, handle could only be DNNL_MEMORY_{ALLOCATE,NONE}
memory make_memory(const desc &md, const engine &aengine, memory_kind mkind,
        void *handle = DNNL_MEMORY_ALLOCATE);
memory make_memory(const desc &md, const engine &astream, memory_kind mkind,
        void *handle = DNNL_MEMORY_ALLOCATE);

// memory_kind could be changed during the lifetime, by setting the USM handle
// or SYCL buffer
memory_kind get_memory_kind(const memory &amemory);

// memory_kind == buffer implied
template <typename T, int ndims>
memory make_memory(const desc &md, const engine &aengine,
        cl::sycl::buffer<T, ndims> buf);
template <typename T, int ndims>
memory make_memory(const desc &md, const stream &astream,
        cl::sycl::buffer<T, ndims> buf);

template <typename T, int ndims>
void set_buffer(memory &m, cl::sycl::buffer<T, ndims> b);
template <typename T, int ndims>
void set_buffer(memory &m, cl::sycl::buffer<T, ndims> b, stream &s);
template <typename T, int ndims = 1>
cl::sycl::buffer<T, ndims> get_buffer(const memory &m);

// ==========
// Primitives
// ==========

cl::sycl::event execute(
        const primitive &p, const stream &s,
        const std::unordered_map<int, memory> &args,
        const std::vector<cl::sycl::event> &dependencies = {});

} // namespace sycl_interop
} // namespace dnnl
