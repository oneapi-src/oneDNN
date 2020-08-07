#include <utility>
#include <type_traits>

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

int main() {
    dnnl::memory m;
    dnnl::engine e;
    dnnl::stream s;
    dnnl::primitive p;

    float *ptr = new float[100];
    cl::sycl::buffer<float, 2> buf(cl::sycl::range<2>(10, 10));

    m.set_data_handle(ptr, s);
    auto span = std::make_pair(buf, size_t(0));
    m.set_native<dnnl::runtime::sycl>(span, s);

    auto span1 = m.get_native<dnnl::runtime::sycl>();
    static_assert(
            std::is_same<decltype(span1),
                    std::pair<cl::sycl::buffer<uint8_t, 1>, size_t>>::value,
            "OK");

    auto span2 = m.get_native<dnnl::runtime::sycl, float, 2>();
    static_assert(std::is_same<decltype(span2), decltype(span)>::value, "OK");

    // XXX it is not clear that a memory is being created
    m = dnnl::make_from_native<dnnl::runtime::sycl>(m.get_desc(), e, span1);

    auto dc = e.get_native<dnnl::runtime::sycl>();
    static_assert(
            std::is_same<decltype(dc.first), cl::sycl::device>::value, "OK");
    static_assert(
            std::is_same<decltype(dc.second), cl::sycl::context>::value, "OK");

    // XXX it is not clear that an engine is being created
    auto e1 = dnnl::make_from_native<dnnl::runtime::sycl>(e.get_kind(), dc);
    static_assert(std::is_same<decltype(e1), decltype(e)>::value, "OK");

    auto q = s.get_native<dnnl::runtime::sycl>();
    auto s1 = dnnl::make_from_native<dnnl::runtime::sycl>(e, q);
    static_assert(std::is_same<decltype(s1), decltype(s)>::value, "OK");

    std::unordered_map<int, dnnl::memory> args
            = {{DNNL_ARG_SRC, m}, {DNNL_ARG_DST, m}};
    auto ev = dnnl::execute<dnnl::runtime::sycl>(p, s, args);
    auto ev1 = dnnl::execute<dnnl::runtime::sycl>(p, s, args, {ev});
    static_assert(std::is_same<cl::sycl::event, decltype(ev)>::value, "OK");
    static_assert(std::is_same<decltype(ev1), decltype(ev)>::value, "OK");
}
