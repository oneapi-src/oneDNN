#ifndef PERF_H
#define PERF_H
/** \file
 * perf_begin/end.
 * To activate the rdpmc performance counters in linux, I snarfed some code from
 * the linux tools [ejk] */
#if defined(__cplusplus)
extern "C" {
#endif

typedef struct { int fd; long page_size; void* addr; } perf_t;
perf_t const* perf_begin();
void perf_end(perf_t const* perf_ctx);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif /* PERF_H */
