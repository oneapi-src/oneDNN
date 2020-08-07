#include <utility>
#include <type_traits>

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

int main() {
    dnnl::memory::desc md;
    dnnl::memory m;
    dnnl::engine e;
    dnnl::stream s;
    dnnl::primitive p;

    float *ptr = new float[100];
    cl::sycl::buffer<float, 2> buf(cl::sycl::range<2>(10, 10));

    m = dnnl::memory(md, e);
    m = dnnl::memory(md, e, ptr); // USM
    m = dnnl::memory(md, e, no_zero_pad | allocate_usm_shared);
    m = dnnl::memory(md, e, no_zero_pad, ptr);
    m = dnnl::memory(md, s, ptr);
    m = dnnl::memory(md, s, no_zero_pad, ptr); // ???

    m = dnnl::memory(md, e, buf); // Buffer

    m.set_data_handle(ptr);
    m.set_data_handle(s, ptr);
    m.set_data_handle(e, no_zero_pad, buf);
}
