#include "perf.h"
#if defined(__cplusplus)
extern "C" {
#endif

#if 1 || defined(_WIN32) || defined(_SX)
static perf_t perf_ctx_unused = {.fd=0, .page_size=0, .addr = (void*)0 };

perf_t const* perf_begin() { return &perf_ctx_unused; }
void perf_end(perf_t const* perf_ctx){ (void)perf_ctx/*unused*/; }
#else // linux ...
#error "OS-specific timing setup has been removed <snip, snip>"
//
// microsecond resolution of <chrono> seems pretty much universal nowadays,
// and suffices for most benchdnn testing.
//
// Fancy things could include:
//    - setting cpu governor for max performance
//    - setting cpu affinity
//    - enabling rdpmc (if avail, a fair chunk of code!)
//    - rdtsc
//
#endif // linux ...
#if defined(__cplusplus)
} // extern "C"
#endif

/* vim: set sw=4 et: */
