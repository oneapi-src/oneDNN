# USM and Buffers Support

## Current State

Currently, `dev-v2` supports both USM and Buffer interfaces for the built
library. On API level the dispatch happens using `DNNL_USE_DPCPP_USM` macro
that changes the behavior of `dnnl::memory` constructor:
``` cpp
dnnl::memory::memory(const desc &md, const engine &engine, void *ahandle)
#ifdef DNNL_USE_DPCPP_USM
        : memory(with_sycl_tag {}, md, engine, ahandle, /* use_usm = */ true) {}
#else
        : memory(with_sycl_tag {}, md, engine, ahandle, /* use_usm = */ false) {}
#endif
```

On the implementation side, the USM and buffers are easily translated to the
OpenCL kernel argument using:
- `cgh.set_arg(..., usm_ptr)` for USM, and
- `cgh.set_arg(..., accessor)` for buffers;

This easy translation, in particular, allows transparently mixing USM and
buffer objects within one kernel. It is worth mentioning, that if we use SYCL
kernels (instead of OpenCL), that translation wouldn't be that trivial, as
memory objects are passed directly to SYCL kernels (as arguments), and cannot
be set independently as in case of attached `cl_mem` as an argument to a
kernel. If we want to make our kernels work with arbitrary combination of USM
and buffers we will:
1. Either have our kernels templated by the kind of each input memory object,
   and instantiate them for each and every possible combination;
2. Or ask compiler team to introduce a method similar to `cfg.set_arg(...)`
   that will allow alleviate the difference between USM and buffers, and allow
   the dispatching at runtime, rather than at compile time.
   - Currently, there is no convenient way of doing that even from the SYCL
     standard point of view.
3. Or implement a wrapper that would allow encapsulating an accessor (buffer)
   and USM, and dispatch between those in the kernel based on the type the
   wrapper was created with.

The 1st option is highly discouraging, as it will increase the time of library
compilation and its size for no good reason. The latter 2nd is unlikely to
happen as well, as I don't think we have good enough driver to convince
compiler team to implement the feature. The 3rd option seems feasible even
though it makes the programming a little more difficult + may have overhead at
runtime. See the [Appendix A](#appendix-a-wrapper-to-handle-usm-and-buffers)
for the possible implementation.

Throughout this section the assumption is that idea in the
[Appendix A](#appendix-a-wrapper-to-handle-usm-and-buffers) will work as
intended. However, original the RFC was written in the assumption that that
won't be the case. To document the discussion for this scenario there will be
hidden snippets of text under "What if
[Appendix A](#appendix-a-wrapper-to-handle-usm-and-buffers) doesn't work" title.

## Options to Support USM and Buffers in v2.0

For some of the options below the following enum is used:
``` cpp
// dnnl_sycl.hpp
namespace dnnl {
namespace sycl {
enum class memory_kind {
    buffer,
    usm_shared,
    usm_device,
};
} // namespace sycl
} // namespace dnnl
```

The options:

1. A choice between USM and buffers happens at compile time.

   The obvious benefits are:
   - Smaller size of the library: kernels work with one type of memory kind;
   - Simpler implementation: macros or templates will allow to write linear
     code with no branches for USM vs buffers;

   The cons:
   - The least flexible option for a user.
   - Yet another configuration. Will need to decide what configuration to ship
     in binary form.
     - If we take this route, probably USM as it is used by the majority of
       our users, namely frameworks.

2. Allow using USM or buffers, but primitive execution should only use one
   flavor (i.e. mixing is not allowed). There are different levels of enforcing
   the policy:
   - Just state the requirement in the documentation, and check that all input
     and output memory objects are of the same kind;
   - Extend engine constructor with the flag that would indicate the flavor
     a user wants to use, e.g.:
     ``` cpp
     auto eng_cpu_buf = dnnl::engine(kind::cpu, memory_kind::buffer, 0);
     auto eng_gpu_usm = dnnl::engine(kind::gpu, memory_kind::usm_device, 0);
     ```
   - Introduce a global control (not attached to the engine) to select memory
     kind: `dnnl::sycl::set_memory_kind(memory_kind::buffer)` and corresponding
     environment variable duplicating the setter.

   <details><summary>What if [Appendix A](#appendix-a-wrapper-to-handle-usm-and-buffers) doesn't work</summary><p>
   It is worth mentioning that the all these options above lead to at least 2x
   number of kernels in object files, as they should be prepared to work with
   either kind of memory.

   The biggest issue with the first option is that primitives could be
   configured to use internally allocated scratchpad (which uses either buffer
   or USM). The thing is that at primitive creation time, when the scratchpad
   needs to be allocated, there is no information on what kind of memory a user
   uses. This means that the options are:
   - We switch to lazy allocation, that happens at execution, rather than at
     creation, when the memory kind is clear;
   - We implement DPC++ kernels in a way they could mix different kinds of
     memory for scratchpad and the rest inputs / outputs.

   The 3rd options is conceptually the same as the 2nd one, but makes it much
   easier to control what kind of memory is used. The only drawback (that seems
   to be quite sever) is that it is global switch. I think we should avoid
   global states for the library.

   Both these options seem quite unpleasant. The latter will lead to another 3x
   factor for the number of kernels, as now they should support the following
   combinations (say if scratchpad is allocated internally we always use USM):
   1. Input/output use USM, scratchpad USM;
   2. Input/output use buffers, scratchpad allocated internally using USM;
   3. Input/output use buffers, scratchpad provided by a user, hence buffer;

   Hence, if we decide to go with this overall option (supporting both kinds at
   run-time, but fixed for primitive run) attaching the restriction to the
   engine (or primitive attribute?) is the way to go.
   </p></details>


3. Allow mixing USM and buffers in the API in any way a user wants.
   - For instance,
     ``` cpp
     eltwise::execute({{SRC, mem(..., usm_ptr_a)}, {DST, mem(..., buffer_b)}});
     ```
   - Pros:
     - Maximum flexibility for users;
     - Scratchpad, that was created as `memory(scrachpad_desc, engine)` will
       work regardless whether it is backed up by USM or buffer.
     - No API restrictions (hence, potentially the less amount of changes to
       behavior);
   - Cons:
     - Such flexibility enlarges the changes to shoot in the foot;
     - (minor, can be fixed) Memory object creation without specifying user's
       USM or buffer makes it unclear, what to expect inside;

   <details><summary>What if [Appendix A](#appendix-a-wrapper-to-handle-usm-and-buffers) doesn't work</summary><p>
   More Cons:

   - Very difficult to implement due to SYCL kernel limitations described above;
   - Library size, and cache size (assuming we use templated approach);
   </p></details>


## Proposal and Discussion

Assuming we will develop the majority of kernels in DPC++ the only feasible
options seem to be:
1. Make a choice at the library compile time;
   - Intel binary version will be built with USM support;

2. **(Fallback recommended option)**
   Make memory kind an option for engine constructor;
   - All memory allocations through oneDNN API will comply to the engine option;

3. **(Recommended option)**
   Allow arbitrary mixing of the USM and buffers.
   - The kind of memory used during memory object could be either specified
     using a flag:
     ``` cpp
     auto mem = dnnl::sycl::make_memory(engine, md, memory_kind::buffer);
     ```
   - For default constructor pick the default option. Assuming that USM is
     generally more user-frindly than buffers, the suggestion is to default to
     USM. This is also well aligned with the Native C++ API, that works with
     pointers. Futhermore, it is suggested to use specifically device USM, as
     opposite to shared USM, as the former gives better performance and is also
     aligned with the fact library's memory objects are attached to engine
     (device).

In my opinion, since buffers are first-class citizens of DPC++ we should
properly support them. That's why option 1 doesn't seem really feasible.

If we can easily support both USM and buffer in the code
([Appendix A](#appendix-a-wrapper-to-handle-usm-and-buffers)), we should go
with option 3, as it is the most flexible, has less impact on the API, and
doesn't cost us much.

<details><summary>What if [Appendix A](#appendix-a-wrapper-to-handle-usm-and-buffers) doesn't work</summary><p>

Option 3 is not an option anymore -- need to generate to many kernels (for each
combination of USM and buffers for every kernel).

The second option looks nice. It gives the flexibility to a user, and also
somehow solves the issue of differentiating between different kinds of USM:

``` cpp
auto eng_cpu_buf = dnnl::engine(kind::cpu, memory_kind::buffer, 0);
auto cpu_mem(eng_cpu_buf, md); // use buffer

auto eng_gpu_usm = dnnl::engine(kind::gpu, memory_kind::usm_shared, 0);
auto gpu_mem(eng_gpu_usm, md); // shared USM
float *x = (float *)gpu_mem.get_data_handle();
x[0] = 1; // oK, as it is shared USM
```

However, the price is twice the number of kernels.

The usefulness of mixing the kinds of memory is also questionable.

On the other hand, option 1 means having yet another configuration, and if we
even decide to support both in the library, that would require an incompatible
changes (because before the kind of memory would have been implied).

The table of downsides:

| Built time option (option 1)                 | Engine flag (option 2 **recommended as a fallback**)                  |
| :--                                          | :--                                                                   |
| One more configuration <br> Less flexibility | Questionable usefulness <br> Slightly more complicated implementation |

We don't expect many (any?) will mix buffers and USM, so simplifying the API
accordingly is acceptable. Subjectively, asking to pass an extra argument to
engine (`memory_kind::usm_device`) is not that big of a hassle.

</p></details>


## Appendix A. Wrapper to Handle USM and Buffers.

Denis suggested the following way of encapsulating USM and buffers to handle
both types in a single kernel:

``` cpp
buffer_u8_t &dummy_buffer() {
    thread_local buffer_u8_t instance(1);
    return instance;
}

struct sycl_memory_arg_t {
    static constexpr auto rw_mode_t = cl::sycl::access::mode::read_write;
    using acc_t = cl::sycl::accessor<uint8_t, 1, rw_mode_t,
          cl::sycl::access::target::global_buffer>;

    sycl_memory_arg_t(void *usm, cl::sycl::handler &cgh)
        : usm(usm), acc(dummy_buffer().get_access<rw_mode_t>(cgh)) {}
    sycl_memory_arg_t(const acc_t &acc) : usm(nullptr), acc(acc) {}

    // std::variant would be a better option, but it is not yet supported
    void *usm;
    acc_t acc;
};

struct kernel_t {
    binary_kernel_t(sycl_memory_arg_t mem_arg) : mem_arg_(mem_arg) {}

    void operator()() {
        void *ptr = mem_arg_.usm
                  ? mem_arg_.usm
                  : (void *)mem_arg_.acc.get_pointer().get();
        // work with ptr...
    }

private:
    sycl_memory_arg_t mem_arg_;
};
```

---

EOD
