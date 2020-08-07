#pragma once

#include <utility>
#include <CL/sycl.hpp>

#include "dnnl.hpp"

namespace dnnl {

template <>
struct interop<dnnl::runtime::sycl, engine> {
    // Have to use a tuple to return two values.
    using type = std::pair<cl::sycl::device, cl::sycl::context>;
};

template <>
struct interop<dnnl::runtime::sycl, stream> {
    using type = cl::sycl::queue;
    using event = cl::sycl::event; // XXX not so nice :(
};

template <>
struct interop<dnnl::runtime::sycl, memory> {
    template <typename T, int dims>
    using typed_type = std::pair<cl::sycl::buffer<T, dims>, size_t>;
};

} // namespace dnnl
