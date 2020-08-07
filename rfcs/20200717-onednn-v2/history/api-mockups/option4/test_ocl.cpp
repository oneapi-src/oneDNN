#include <utility>
#include <type_traits>

#include "dnnl.hpp"
#include "dnnl_ocl.hpp"

int main() {
    dnnl::memory m;
    dnnl::engine e;
    dnnl::stream s;
    dnnl::primitive p;

    float *ptr = new float[100];
    cl_mem buf;

    m.set_data_handle(ptr, s);
    auto span = std::make_pair(buf, size_t(0));
    m.set_native<dnnl::runtime_class::ocl>(span, s);

    auto span1 = m.get_native<dnnl::runtime_class::ocl>();
    static_assert(
            std::is_same<decltype(span1), std::pair<cl_mem, size_t>>::value,
            "OK");

    // XXX it is not clear that a memory is being created
    m = dnnl::make_from_native<dnnl::runtime_class::ocl>(
            m.get_desc(), e, span1);

    auto dc = e.get_native<dnnl::runtime_class::ocl>();
    static_assert(std::is_same<decltype(dc.first), cl_device_id>::value, "OK");
    static_assert(std::is_same<decltype(dc.second), cl_context>::value, "OK");

    // XXX it is not clear that an engine is being created
    auto e1 = dnnl::make_from_native<dnnl::runtime_class::ocl>(
            e.get_kind(), dc);
    static_assert(std::is_same<decltype(e1), decltype(e)>::value, "OK");

    auto q = s.get_native<dnnl::runtime_class::ocl>();
    auto s1 = dnnl::make_from_native<dnnl::runtime_class::ocl>(e, q);
    static_assert(std::is_same<decltype(s1), decltype(s)>::value, "OK");

    std::unordered_map<int, dnnl::memory> args
            = {{DNNL_ARG_SRC, m}, {DNNL_ARG_DST, m}};
    auto ev = dnnl::execute<dnnl::runtime_class::ocl>(p, s, args);
    auto ev1 = dnnl::execute<dnnl::runtime_class::ocl>(p, s, args, {ev});
    static_assert(std::is_same<cl_event, decltype(ev)>::value, "OK");
    static_assert(std::is_same<decltype(ev1), decltype(ev)>::value, "OK");
}
