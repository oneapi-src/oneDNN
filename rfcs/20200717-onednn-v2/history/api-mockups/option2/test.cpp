#include <type_traits>

#include <CL/sycl.hpp> // Only in the user's code

#include "dnnl.hpp"
#include "dnnl_interop.hpp"

int main(int argc, char **argv) {
    dnnl::memory m;
    dnnl::engine e;
    dnnl::stream s;
    dnnl::primitive p;

    float *ptr = new float[100];
    cl::sycl::buffer<float, 1> buf(cl::sycl::range<1>(100));

    m.set_data_handle(ptr, s);
    dnnl::set_native(m, s, buf);

    // XXX: we would need overloads for all the types and dimensions?
    buf = dnnl::get_native<decltype(buf)>(m);

    ptr = static_cast<decltype(ptr)>(m.get_data_handle());
    ptr = dnnl::get_native<decltype(ptr)>(m);

    auto d = dnnl::get_native<cl::sycl::device>(e);
    auto c = dnnl::get_native<cl::sycl::context>(e);

    auto e1 = dnnl::make(e.get_kind(), d, c);
    static_assert(std::is_same<decltype(e1), decltype(e)>::value, "OK");

    auto q = dnnl::get_native<cl::sycl::queue>(s);
    auto s1 = dnnl::make(e, q);
    static_assert(std::is_same<decltype(s1), decltype(s)>::value, "OK");

    std::unordered_map<int, dnnl::memory> args
            = {{DNNL_ARG_SRC, m}, {DNNL_ARG_DST, m}};
    auto ev = dnnl::execute<cl::sycl::event>(p, s, args);
    auto ev1 = dnnl::execute<cl::sycl::event>(p, s, args, {ev});
    auto ev2 = dnnl::execute(p, s, args, std::vector<decltype(ev)>({ev}));
    static_assert(std::is_same<cl::sycl::event, decltype(ev)>::value, "OK");
    static_assert(std::is_same<decltype(ev1), decltype(ev)>::value, "OK");
    static_assert(std::is_same<decltype(ev2), decltype(ev)>::value, "OK");
}
