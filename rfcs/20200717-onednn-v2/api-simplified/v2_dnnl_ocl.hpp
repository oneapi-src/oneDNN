#include <CL/cl.hpp>

#include "dnnl.hpp"

namespace dnnl {
namespace ocl_interop {

// ======
// Engine
// ======

// Note that engine::kind is dropped.
engine make_engine(cl_device_id &dev, cl_context &ctx);
cl_device_id get_device(const engine &e);
cl_context get_context(const engine &e);

// ======
// Stream
// ======

// Note, that stream_flags are dropped.
stream make_stream(const engine &e, cl_command_queue &queue);
cl_command_queue get_command_queue(const stream &s);

// ======
// Memory
// ======

void set_mem_object(memory &m, cl_mem mem_object);
cl_mem get_mem_object(const memory &m);

} // namespace ocl_interop
} // namespace dnnl
