// Inspiration:
//
// https://github.com/KhronosGroup/SYCL-Shared/blob/master/proposals/sycl_generalization.md
//

#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>

// No runtime_class-specific headers.

namespace dnnl {

enum class runtime_class {
    cpu,
    ocl,
    sycl, // collides with the namespace, unfortunately...
};

// Interop trait that maps a (runtime_class, dnnl type) pair to a
// runtime_class-specific type via ::type
template <runtime_class name>
struct interop;

struct engine {
    enum class kind { cpu, gpu };

    engine();

    // Common API constructor. Interpretation of idx is implementation and
    // kind-specific, but needs to be firmly defined.
    engine(kind k, int idx);

    // Returns runtime_class associated with the engine.
    runtime_class get_runtime();

    kind get_kind();

    // Returns the underlying runtime_class-specific object. Throws if none is
    // available.
    template <runtime_class name>
    typename interop<name>::engine get_native() const;
};

// Runtime_class-specific engine factory.
template <runtime_class name>
engine make_from_native(engine::kind k, typename interop<name>::engine native);

struct stream {
    stream();

    stream(const engine &e);

    // Returns the underlying runtime_class-specific object. Throws if none is
    // available.
    template <runtime_class name>
    typename interop<name>::stream get_native() const;
};

// Runtime_class-specific memory factory.
template <runtime_class name>
stream make_from_native(
        const engine &e, const typename interop<name>::stream native);

struct memory {
    struct desc {};

    memory();

    desc get_desc();

    // Common API constructor that may fail if a runtime_class underlying the
    // engine does not support pointers. SYCL GPU engines default to allocating
    // USM memory.
    memory(const desc &d, const engine &e, const stream &s, void *handle = 0);

    // Common API way to retrieve the underlying storage. May fail if the
    // runtime_class or this parcitular memory object does not use pointers to
    // store data.
    void *get_data_handle();

    // Common API way to set the underlying storage. May fail if the
    // runtime_class does not use pointers to store data.
    void set_data_handle(void *handle, const stream &s);

    // Templated runtime_class-specific way to retrieve the underlying storage.
    // May fail if this particular memory object uses a different runtime_class
    // or uses pointers to store data.
    template <runtime_class name>
    typename interop<name>::memory get_native() const;

    // Templated runtime_class-specific way to retrieve the underlying storage.
    // May fail if this particular memory object uses a different runtime_class
    // or uses pointers to store data.
    template <runtime_class name, typename T = uint8_t, int dims = 1>
    typename interop<name>::template memory<T, dims> get_native() const;

    // Templated and typed runtime_class-specific way to set the underlying
    // storage. May fail if this particular memory object uses a different
    // runtime_class.
    template <runtime_class name>
    void set_native(typename interop<name>::memory handle, const stream &s);

    // Templated and typed runtime_class-specific way to set the underlying
    // storage. May fail if this particular memory object uses a different
    // runtime_class.
    template <runtime_class name, typename T, int dims>
    void set_native(typename interop<name>::template memory<T, dims> handle,
            const stream &s);
};

// Runtime_class-specific memory factory.
template <runtime_class name>
memory make_from_native(const memory::desc &d, const engine &e,
        const typename interop<name>::memory handle);

// Runtime_class-specific typed memory factory.
template <runtime_class name, typename T, int dims>
memory make_from_native(const memory::desc &d, const engine &e,
        const typename interop<name>::template memory<T, dims> handle);

struct primitive {
    void execute(const stream &s, const memory &src, const memory &dst) const;
    void execute(
            const stream &s, const std::unordered_map<int, memory> &args) const;
};

// Runtime_class-specific execute functions.
template <runtime_class name>
typename interop<name>::event execute(const primitive &p, const stream &s,
        const std::unordered_map<int, memory> &args,
        const std::vector<typename interop<name>::event> &dependencies = {});

// Runtime_class-specific reorder execute function.
template <runtime_class name>
typename interop<name>::event execute(const primitive &p, const stream &s,
        const memory &src, const memory &dst,
        const std::vector<typename interop<name>::event> &dependencies = {});

#define DNNL_ARG_SRC 0
#define DNNL_ARG_DST 1

} // namespace dnnl
