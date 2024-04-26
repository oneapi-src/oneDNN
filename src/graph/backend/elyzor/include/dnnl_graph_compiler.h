#ifndef DNNL_GRAPH_COMPILER_H
#define DNNL_GRAPH_COMPILER_H
#define DNNL_DLL
#define DNNL_DLL_EXPORTS

#include <cinttypes>
#include <cstddef>
#include "dnnl_types.h"

/*
 * Public API for integration with third-party graph compilers.
 */

#ifdef __cplusplus
extern "C" {
#endif

struct dnnl_graph_compiler;
struct dnnl_graph_compiler_executable;

struct dnnl_graph_compiler_context {
    uint32_t num_threads;

    void *(*allocator)(size_t size);

    void (*deallocator)(void *ptr);
};

struct dnnl_graph_compiler_tensor {
    size_t id;
    uint8_t ndims;
    size_t *dims;
    void *data;
};

DNNL_API dnnl_status_t dnnl_graph_compiler_create(
        const struct dnnl_graph_compiler_context *ctx,
        const struct dnnl_graph_compiler **gc);

DNNL_API void dnnl_graph_compiler_destroy(const struct dnnl_graph_compiler *gc);

DNNL_API dnnl_status_t dnnl_graph_compiler_compile(
        const struct dnnl_graph_compiler *gc, const char *graph_json,
        const struct dnnl_graph_compiler_executable **exe);

DNNL_API void dnnl_graph_compiler_destroy_executable(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe);

DNNL_API dnnl_status_t dnnl_graph_compiler_execute(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe,
        dnnl_graph_compiler_tensor *inputs,
        dnnl_graph_compiler_tensor *outputs);

#ifdef __cplusplus
}
#endif
#endif // DNNL_GRAPH_COMPILER_H
