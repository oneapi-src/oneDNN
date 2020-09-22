#pragma once

#include <utility>
#include <CL/cl.h>

#include "dnnl.hpp"

namespace dnnl {

template <>
struct interop<dnnl::runtime_class::ocl> {
    using engine = std::pair<cl_device_id, cl_context>;
    using stream = cl_command_queue;
    using event = cl_event;
    using memory = std::pair<cl_mem, size_t>;
};

} // namespace dnnl
