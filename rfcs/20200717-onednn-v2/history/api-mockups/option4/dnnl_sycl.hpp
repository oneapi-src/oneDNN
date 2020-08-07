#pragma once

#include <utility>
#include <CL/sycl.hpp>

#include "dnnl.hpp"

namespace dnnl {

// engine
template<> inline engine::engine(kind k, cl::sycl::device dev, cl::sycl::context ctx);
template<> inline cl::sycl::device engine::get_api_object() const;
template<> inline cl::sycl::context engine::get_api_object() const;

// memory

template <> memory(const desc &d, const engine &e, void *handle = DNNL_MEM_ALLOCATE);
template <> memory(const desc &d, const stream &s, void *handle = DNNL_MEM_ALLOCATE);
template <> memory(const desc &d, const engine &e, unsigned flags, void *handle = DNNL_MEM_ALLOCATE);
template <> memory(const desc &d, const stream &s, unsigned flags, void *handle = DNNL_MEM_ALLOCATE);



// primitive execution
template<> inline cl::sycl::event primitive::execute(const stream &s, const memory &src, const memory &dst, const std::vector<cl::sycl::event> &deps) const;
template<> inline cl::sycl::event primitive::execute(const stream &s, const std::unordered_map<int, memory> &args, const std::vector<cl::sycl::event> &deps) const;

template <>
struct interop<dnnl::runtime_class::sycl> {
    using engine = std::pair<cl::sycl::device, cl::sycl::context>;
    using stream = cl::sycl::queue;
    using event = cl::sycl::event;
    template <typename T, int dims>
    using memory = std::pair<cl::sycl::buffer<T, dims>, size_t>;
};

} // namespace dnnl
