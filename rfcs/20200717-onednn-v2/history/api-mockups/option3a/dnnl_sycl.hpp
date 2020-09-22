#pragma once

#include <utility>
#include <CL/sycl.hpp>

#include "dnnl.hpp"

namespace dnnl {

template <>
struct interop<dnnl::runtime_class::sycl> {
    using engine = std::pair<cl::sycl::device, cl::sycl::context>;
    using stream = cl::sycl::queue;
    using event = cl::sycl::event;
    template <typename T, int dims>
    using memory = std::pair<cl::sycl::buffer<T, dims>, size_t>;
};

} // namespace dnnl
