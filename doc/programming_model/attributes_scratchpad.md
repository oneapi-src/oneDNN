Primitive Attributes: Scratchpad {#dev_guide_attributes_scratchpad}
===================================================================

Some primitives might require temporary space while performing the
computations. For instance, the operations that do not have enough independent
work to utilize all cores on a system might use parallelization over the
reduction axis (e.g. k-axis in matrix-matrix multiplication). In this case
the threads compute partial results in a temporary buffer and once finished
the library reduces partial results into the final one. Another example is
a convolution implementation that uses GEMM. Before using a GEMM the source
images needs to be rearranged by so-called `im2col` transformation.
The rearrangement happens to an intermediate buffer that is then used as an
input for GEMM.

In both of these examples, the temporary memory is not required once the
computations are done. Intel MKL-DNN refers to such memory as a **scratchpad**.

@warning
    Do not confuse **scratchpad** with
    [Workspace](@ref dev_guide_inference_and_training_aspects_workspace).
    The workspace is a buffer that is shared between forward and backward
    propagation of a primitive (hence **must** be preserved between the calls)
    and is used only in training.

The amount of space required for the scratchpad depends on the primitive and the
actual implementation. The GEMM-based convolutions require a scratchpad for
the `im2col` data, while directly implemented convolutions can work with the
original data.

Both types of implementation might need extra space for the reduction in case
there are too few independent tasks. The `im2col` size is proportional to the
size of the source image multiplied by the weights spatial size. The size of a
buffer for reduction is proportional to the tensor size to be reduced (e.g.,
`diff_weights` in the case of backward by weights) multiplied by the number of
threads in the reduction groups (the upper bound is the overall number of
threads).

As you can see, the scratchpad in these cases might be significant.
By contrast, some other primitives might require very little extra space. For
instance, one of the implementation of the @ref mkldnn::sum primitive requires
temporary space only to store the pointers to data for each and every input
array (that is, the size of the scratchpad is `n * sizeof(void *)`, where `n` is
the number of summands).

Intel MKL-DNN supports two modes of dealing with scratchpads:
1. #mkldnn::scratchpad_mode::library.
   The library allocates memory for each primitive during its creation. This
   is the **default** behavior which enables user to not worry about the
   scratchpad at all. However this approach has two major downsides:
   - If primitives are cached, they may reserve a significant amount of memory.
   - Primitives are not thread safe, because simultaneous runs will make
     different threads to use the same scratchpad buffer.
2. #mkldnn::scratchpad_mode::user.
   A user provides scratchpad memory that has sufficient space at primitive
   execution (using the `MKLDNN_ARG_SCRATCHPAD` tag). This enables the user to
   reuse the memory as well as to make the primitives thread-safe. However, this
   requires a good memory manager (in terms of speed and locality) on the user's
   side and some extra boilerplate code.

@warning
    Primitives are not thread-safe by default. Users should use
    #mkldnn::scratchpad_mode::user if they want to use a single primitive from
    different threads simultaneously.

The attributes (@ref dev_guide_attributes) are used to control who provides
a scratchpad:
- C @ref mkldnn_primitive_attr_set_scratchpad_mode
- C++ @ref mkldnn::primitive_attr::set_scratchpad_mode

It is worth mentioning that all primitives support both scratchpad modes.
That is, primitive descriptor creation success or failure cannot depend on the
scratchpad mode used.

## Scratchpad Memory Engine

If the user provides scratchpad memory to a primitive, this memory must be
created using the same engine that the primitive uses.

## Examples

#### Library Manages Scratchpad

As mentioned above, this is a default behavior. We only want to highlight how a
user can query the amount of memory consumed by a primitive due to a scratchpad.

~~~cpp
// Use default attr, hence the library allocates scratchpad
mkldnn::primitive::primitive_desc op_pd(params, ...);

// Print how much memory would be hold by a primitive due to scratchpad
std::cout << "primitive will use "
          << op_pd.query_s64(mkldnn::query::memory_consumption_s64)
          << " bytes" << std::endl;

// In this case scratchpad is internal, hence user visible scratchpad memory
// descriptor should be empty:
auto zero_md = mkldnn::memory::desc();
assert(op_pd.scratchpad_desc() == zero_md);
~~~

#### User Manages Scratchpad

~~~cpp
// Create an empty (default) attributes
mkldnn::primitive_attr attr;

// Default scratchpad mode is `library`:
assert(attr.get_scratchpad_mode() == mkldnn::scratchpad_mode::library);

// Set scratchpad mode to `user`
attr.set_scratchpad_mode(mkldnn::scratchpad_mode::user);

// Create a primitive descriptor with custom attributes
mkldnn::primitive::primitive_desc op_pd(op_d, attr, engine);

// Query the scratchpad memory descriptor
mkldnn::memory::desc scratchpad_md = op_pd.scratchpad_desc();

// Note, that a primitive doesn't consume memory in this configuration:
assert(op_pd.query_s64(mkldnn::query::memory_consumption_s64) == 0);

// Create a primitive
mkldnn::primitive prim(op_pd);

// ...

// Create a scratchpad memory
// NOTE: if scratchpad is not required for a particular primitive the
//       scratchpad_md.get_size() will return 0. It is fine to have
//       scratchpad_ptr == nullptr in this case.
void *scratchpad_ptr = user_memory_manager::allocate(scratchpad_md.get_size());
// NOTE: engine here must much the engine of the primitive
mkldnn::memory scratchpad(scratchpad_md, engine, scratchpad_ptr);

// Pass a scratchpad memory to a primitive
prim.execute(stream, {
        ...,
        {MKLDNN_ARG_SCRATCHPAD, scratchpad}});
~~~
