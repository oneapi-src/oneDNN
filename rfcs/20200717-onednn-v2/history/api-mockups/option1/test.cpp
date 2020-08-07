#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

int main(int argc, char **argv) {
    dnnl::memory m;
    dnnl::engine e;
    dnnl::stream s;
    dnnl::primitive p;

    float *ptr = new float[100];
    cl::sycl::buffer<float, 1> buf(cl::sycl::range<1>(100));

    m.set_data_handle(ptr, s);
    dnnl::sycl::memory_set_data_handle(m, buf, s);

    buf = dnnl::sycl::memory_get_data_handle<float, 1>(m);
    ptr = static_cast<decltype(ptr)>(m.get_data_handle());

    auto d = dnnl::sycl::engine_get_device(e);
    auto c = dnnl::sycl::engine_get_context(e);

    e = dnnl::sycl::engine_create(e.get_kind(), d, c);

    auto q = dnnl::sycl::stream_get_queue(s);
    s = dnnl::sycl::stream_create(e, q);

    std::unordered_map<int, dnnl::memory> args
            = {{DNNL_ARG_SRC, m}, {DNNL_ARG_DST, m}};
    auto ev = dnnl::sycl::execute(p, s, args);
    ev = dnnl::sycl::execute(p, s, args, {ev});
}
