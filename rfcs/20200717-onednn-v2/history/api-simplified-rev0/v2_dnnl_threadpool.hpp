#include "dnnl.hpp"
#include "dnnl_threadpool_iface.hpp"

namespace dnnl {
namespace threadpool {

// option 1 (preferable, but breaks backwards compatibility)
// The flags are dropped, the `in_order` is assumed.
stream stream_create(const engine &e, threadpool_iface *threadpool);
threadpool_iface *stream_get_threadpool(const stream &s);

#if 0
// option 1 (preferable, but breaks backwards compatibility)
// the flags are preserved
stream stream_create(const engine &e, threadpool_iface *threadpool,
        stream::flags aflags = flags::default_flags);
threadpool_iface *stream_get_threadpool(const stream &s);


// option 2 (keep stream_attr, mimics current API)
void stream_attr_set_threadpool(stream_attr &sa, threadpool_iface *threadpool,
        stream::flags aflags = flags::default_flags);
threadpool_iface *stream_attr_get_threadpool(const stream_attr &sa);
#endif

} // namespace threadpool
} // namespace dnnl
