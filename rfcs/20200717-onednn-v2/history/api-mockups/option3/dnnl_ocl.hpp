#pragma once

#include <utility>
#include <CL/cl.h>

#include "dnnl.hpp"

namespace dnnl {

template <>
struct interop<dnnl::runtime::ocl, engine> {
    // Have to use a tuple to return two values.
    using type = std::pair<cl_device_id, cl_context>;
};

template <>
struct interop<dnnl::runtime::ocl, stream> {
    using type = cl_command_queue;
    using event = cl_event; // XXX not so nice :(
};

template <>
struct interop<dnnl::runtime::ocl, memory> {
    using type = std::pair<cl_mem, size_t>;
};

} // namespace dnnl
