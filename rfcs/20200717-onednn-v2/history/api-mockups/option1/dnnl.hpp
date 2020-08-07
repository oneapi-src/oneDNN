// Inspiration:
//
// https://github.com/KhronosGroup/SYCL-Shared/blob/master/proposals/sycl_generalization.md
//

#pragma once

#include <cstddef>
#include <unordered_map>

// No runtime-specific headers.

namespace dnnl {

struct engine {
    enum class kind { cpu, gpu };

    engine();

    // Common API constructor. Interpretation of idx is implementation and
    // kind-specific, but needs to be firmly defined.
    engine(kind k, int idx);

    kind get_kind();
};

struct stream {};

struct memory {
    struct desc {};

    memory();

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
};

struct primitive {
    void execute(const stream &s, const memory &src, const memory &dst) const;
    void execute(
            const stream &s, const std::unordered_map<int, memory> &args) const;
};

#define DNNL_ARG_SRC 0
#define DNNL_ARG_DST 1

} // namespace dnnl
