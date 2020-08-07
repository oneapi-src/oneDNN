#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>

namespace dnnl {

enum class api_class { cpp, ocl, sycl, };

struct engine {
    enum class kind { cpu, gpu };

    engine();
    engine(kind k, int idx);
    engine(api_class a, kind k, int idx); // to create pure c++ for dpcpp build

    api_class get_api_class() const;
    kind get_kind() const;

    // API-specific ctor:
    // - engine(kind k, cl::sycl::device, cl::sycl::context)
    template<typename... Args> engine(kind k, Args &&... args);

    // API-specific getter:
    // - engine::get_api_object<cl::sycl::device>()
    template<typename T> T get_api_object() const;
};

struct stream {
    stream();
    stream(const engine &e);
};

using memory_flags_t = dnnl_memory_flags_t; // unsigned
namespace memory_flags {
    unsigned skip_zero_pad = 0x1;
}

struct memory {
    struct desc {};

    memory();

    desc get_desc() const;

    // backwards compatibility:
    // memory(const desc &d, const engine &e, void *handle = DNNL_MEM_ALLOCATE);

    memory(const desc &d, const engine &e, void *handle = DNNL_MEM_ALLOCATE);
    memory(const desc &d, const stream &s, void *handle = DNNL_MEM_ALLOCATE);
    memory(const desc &d, const engine &e, unsigned flags, void *handle = DNNL_MEM_ALLOCATE);
    memory(const desc &d, const stream &s, unsigned flags, void *handle = DNNL_MEM_ALLOCATE);

    // Common API way to retrieve the underlying storage. May fail if the
    // api_class or this particular memory object does not use pointers to
    // store data.
    void *get_data_handle();

    // Common API way to set the underlying storage. May fail if the
    // api_class does not use pointers to store data.
    void set_data_handle(void *handle);
    void set_data_handle(void *handle, const stream &s);

};

struct primitive {
    void execute(const stream &s, const memory &src, const memory &dst) const;
    void execute(const stream &s, const std::unordered_map<int, memory> &args) const;

    template <typename T>
    T execute(const stream &s, const memory &src, const memory &dst, const std::vector<T> &deps) const;
    template <typename T>
    T execute(const stream &s, const std::unordered_map<int, memory> &args, const std::vector<T> &deps) const;
};

#define DNNL_ARG_SRC 0
#define DNNL_ARG_DST 1

} // namespace dnnl
