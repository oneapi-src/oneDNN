// Inspiration:
//
// https://github.com/KhronosGroup/SYCL-Shared/blob/master/proposals/sycl_generalization.md
//

#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>

// No runtime-specific headers.

namespace dnnl {

enum class runtime {
    seq,
    omp,
    tbb,
    threadpool,
    ocl,
    sycl, // collides with the namespace, unfortunately...
};

// Interop trait that maps a (runtime, dnnl type) pair to a runtime-specific
// type via ::type
template <runtime name, typename object>
struct interop;

struct engine {
    enum class kind { cpu, gpu };

    engine();

    // Common API constructor. Interpretation of idx is implementation and
    // kind-specific, but needs to be firmly defined.
    engine(kind k, int idx);

    // Returns runtime associated with the engine.
    runtime get_runtime();

    kind get_kind();

    // Returns the underlying runtime-specific object. Throws if none is
    // available.
    template <runtime name>
    typename interop<name, engine>::type get_native() const;
};

// Runtime-specific engine factory.
template <runtime name>
engine make_from_native(
        engine::kind k, typename interop<name, engine>::type native);

struct stream {
    stream();

    stream(const engine &e);

    // Returns the underlying runtime-specific object. Throws if none is
    // available.
    template <runtime name>
    typename interop<name, stream>::type get_native() const;
};

// Runtime-specific memory factory.
template <runtime name>
stream make_from_native(
        const engine &e, const typename interop<name, stream>::type native);

struct memory {
    struct desc {};

    memory();

    desc get_desc();

    // Common API constructor that may fail if a runtime underlying the engine
    // does not support pointers. SYCL GPU engines default to allocating USM
    // memory.
    memory(const desc &d, const engine &e, const stream &s, void *handle = 0);

    // Common API way to retrieve the underlying storage. May fail if the
    // runtime or this parcitular memory object does not use pointers to store
    // data.
    void *get_data_handle();

    // Common API way to set the underlying storage. May fail if the runtime
    // does not use pointers to store data.
    void set_data_handle(void *handle, const stream &s);

    // Templated runtime-specific way to retrieve the underlying storage. May
    // fail if this particular memory object uses a different runtime or uses
    // pointers to store data.
    template <runtime name>
    typename interop<name, memory>::type get_native() const;

    // Templated runtime-specific way to retrieve the underlying storage. May
    // fail if this particular memory object uses a different runtime or uses
    // pointers to store data.
    template <runtime name, typename T = uint8_t, int dims = 1>
    typename interop<name, memory>::template typed_type<T, dims>
    get_native() const;

    // Templated and typed runtime-specific way to set the underlying storage.
    // May fail if this particular memory object uses a different runtime.
    template <runtime name>
    void set_native(
            typename interop<name, memory>::type handle, const stream &s);

    // Templated and typed runtime-specific way to set the underlying storage.
    // May fail if this particular memory object uses a different runtime.
    template <runtime name, typename T, int dims>
    void set_native(
            typename interop<name, memory>::template typed_type<T, dims> handle,
            const stream &s);
};

// Runtime-specific memory factory.
template <runtime name>
memory make_from_native(const memory::desc &d, const engine &e,
        const typename interop<name, memory>::type handle);

// Runtime-specific typed memory factory.
template <runtime name, typename T, int dims>
memory make_from_native(const memory::desc &d, const engine &e,
        const typename interop<name, memory>::template typed_type<T, dims>
                handle);

struct primitive {
    void execute(const stream &s, const memory &src, const memory &dst) const;
    void execute(
            const stream &s, const std::unordered_map<int, memory> &args) const;
};

// Runtime-specific execute functions.
template <runtime name>
typename interop<name, stream>::event execute(const primitive &p,
        const stream &s, const std::unordered_map<int, memory> &args,
        const std::vector<typename interop<name, stream>::event> &dependencies
        = {});

// Runtime-specific reorder execute function.
template <runtime name>
typename interop<name, stream>::event execute(const primitive &p,
        const stream &s, const memory &src, const memory &dst,
        const std::vector<typename interop<name, stream>::event> &dependencies
        = {});

#define DNNL_ARG_SRC 0
#define DNNL_ARG_DST 1

} // namespace dnnl
