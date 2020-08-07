#include <CL/sycl.hpp>

#include "dnnl.hpp"

namespace dnnl {
namespace sycl {

enum class memory_kind { // maybe memory_model is better name?
    usm_device, // default for SYCL-agnostic engine creation
    usm_shared,
    buffer,
};

// api = api::sycl is implied
engine engine_create(
        engine::kind kind, memory_kind model = memory_kind::usm_device);
// Do we actually need kind here?
engine engine_create(engine::kind kind, memory_kind model,
        const cl::sycl::device &dev, const cl::sycl::context &ctx);
cl::sycl::device engine_get_device(const engine &e);
cl::sycl::context engine_get_context(const engine &e);
memory_kind engine_get_memory_kind(const engine &e);

// Do we need flags?
// So far seems they could be derived from queue (for DPC++ and OpenCL and
// could be assumed `in_order` for ThreadPool), so could dropped.
stream stream_create(const engine &e, cl::sycl::queue &queue);
cl::sycl::queue stream_get_queue(const stream &s);

template <typename T, int ndims>
memory memory_create(const desc &md, const engine &aengine,
        cl::sycl::buffer<T, ndims> buf, stream &s);

template <typename T, int ndims>
void memory_set_data_handle(memory &m, cl::sycl::buffer<T, ndims> b, stream &s);
template <typename T, int ndims>
cl::sycl::buffer<T, ndims> memory_get_data_handle(const memory &m);
template <typename T, int ndims = 1>
cl::sycl::buffer<T, ndims> memory_get_sycl_buffer(const memory &m);

cl::sycl::event primitive_execute(const primitive &p, const stream &s,
        const std::unordered_map<int, memory> &args,
        const std::vector<cl::sycl::event> &dependencies = {});

} // namespace sycl
} // namespace dnnl
