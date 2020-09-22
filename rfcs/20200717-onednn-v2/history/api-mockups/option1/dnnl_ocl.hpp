#pragma once

#include <vector>

#include <CL/cl.h>

#include "dnnl.hpp"

namespace dnnl {
namespace ocl {

engine engine_create(engine::kind kind, cl_device_id dev, cl_context ctx);
cl_device_id engine_get_device(const engine &e);
cl_context engine_get_context(const engine &e);

stream stream_create(const engine &e, cl_command_queue queue);
cl_command_queue stream_get_queue(const stream &s);

memory memory_create(cl_mem mem_object);
void memory_set_data_handle(memory &m, cl_mem mem_object);
cl_mem memory_get_data_handle(const memory &m);

cl_event execute(const primitive &p, const stream &s,
        const std::unordered_map<int, memory> &args,
        const std::vector<cl_event> &dependencies = {});

} // namespace ocl
} // namespace dnnl
