Tensor Representation Now and Then, Here and There
==================================================

The following terms are used below (that have similar but different meanings):
1. **Tensor** is a mathematical object. Physical representation in computer's
   memory does not matter.
2. **Buffer** is a chunk of computer's memory.
3. **DNNL memory** is DNNL entity that has a logical and physical (in terms
   of DNNL) representation.
4. **Shaped buffer** is a **buffer** and a tuple `(n_0, ..., n_D)` that is
   called **shape**.

Throughout the document TensorFlow would be used as a representative of
frameworks. The activations assumed to use `NHWC` axes order and weights to
use `HWIO` axes order. For simplicity, the activations layout is always assumed
`NHWC`.


## 1. DNNL

DNNL has a notion of *logical tensors* (aka `{ndims, dims[], data_type}`) and
*physical representation* (aka memory formats and blocking descriptors).

No matter what the physical representation is (or if physical representation is
unspecified completely in case of memory format `any`), for DNNL the tensor
behind the memory as a mathematical object is the same.

DNNL **does not** impose the semantics on tensor behind DNNL memory alone
(like tensor of weights of a convolution or RNN input tensor), but only
together with an operation. That means the same memory can be passed as
source of the convolution (in which case dim #0 will represent batch, etc), and
as the weights (in which case dim #0 will represent the output channels).

What is important to notice here, is that information about tensor in encoded
in logical part of DNNL memory.

> A bit of history.
>
> In some sense, Intel MKL-DNN v0.x put a meaning into memories, since the
> formats were named. MKL-DNN memory with activations would use the memory
> format `nchw` / `nhwc`, while weights would use `oihw` / `hwio`. The logical
> description (dims) and blocking structure were indistinguishable for them.


## 2. Frameworks

Frameworks typically do not distinguish between logical and physical tensor
representation. In terms of DNNL that means that they encode the whole
information about tensor in `{ndims, dims[], data_type}` assuming the memory
format is always `abcd...`. We will call framework analogue for **DNNL memory**
a **shaped buffer**, which is a *buffer* (a chunk of memory) and its shape.

Similarly to DNNL memory the tensor represented by a *shaped buffer* is
reinterpret every time it goes from one operation to another according to the
definition of the latter operation.

To summarize there are two main differences between frameworks and DNNL in
terms of tensor representation:
1. The physical format is fixes in frameworks. This is simple and less
   error-prone.
2. Same operations in a framework and DNNL expect different logical axis order
   in tensor representation. Example: DNNL interprets the logical axes as
   `{batch, channels, height, width}` for convolution activations tensors,
   while frameworks interpret them as `{batch, height, width, channels}`. This
   difference causes a lot of confusion and brings extra complexity to
   integration code that is discussed later.


## 3. Interoperability

Vague... Let consider an example:

``` python
# Output of convolution is a shaped buffer of size (2, 3, 4, 5).
# At this very moment we can say, that
# Batch = 2, Height = 3, Width = 4, and Channels = 5.
buf = conv1(...)


# Between the convolutions the shaped buffer `buf` doesn't have semantics.
# It is simply buffer of 120 elements, with shape (2, 3, 4, 5).
print(buf.shape) # (2, 3, 4, 5)


# At the next moment, the same buffer is passed as the weights to
# another convolution.
# Here, this buffer has the following meaning:
# Height = 2, Width = 3, Input Channels = 4, and Output Channels = 5.
conv2(..., weights = buf)
```

In order to do that trick using DNNL, one needs to reinterpret the logical axes
and permute them accordingly. But the important problem here is that this
operation requires the knowledge of the semantics of the axes in the previous
operation and the semantics of the axes in the current operation to be able
to come up with the proper permutation.

``` c++
// In framework, the logical re-mapping of axes in `buf` happened
// between conv1 and conv2:
//      conv1           | conv2
//      =============== | ===============
//      Batch           | Height
//      Height          | Width
//      Width           | Input Channels
//      Output Channels | Output Channels


// Logical tensor {a, b, c, d} has the interpretation of the axes {Batch, ...}
mem1 = dnnl_conv1(...);


// Permute the logical axes {a, b, c, d} as
// {Batch, Height, Width, Output Channels} according to the table above,
// bringing the necessary changes to the blocking structure.
mem2 = mem(permute(mem1.md, {very-weird-permutation}), mem1.ptr);


// Execute conv2 with the weights
dnnl_conv2(..., weights = mem2);
```

The question is how and where the step with the permutation should take place
in the integration code?

For the where-question, the answer is definitely in the Conv2, not earlier (as
there is no enough information about permutation yet). But how would Conv2 know
what the interpretation of the axes were in the previous operation?

This requires Conv1 to append an extra piece of information to `buf` that would
map logical DNNL axes `(dim[0], dim[1], dim[2], dim[3])` to the shape of the
framework *shaped buffer*. For this particular case, the mapping will be:

| Meaning  | DNNL logical axis (activations in conv1) | TF shape index |
| -------- | ---------------------------------------- | -------------- |
| Batch    | 0                                        | 0              |
| Channels | 1                                        | 3              |
| Height   | 2                                        | 1              |
| Width    | 3                                        | 2              |

> The permutation is `(0123)->(0312)`.

The Conv2 has its own mapping:

| Meaning          | DNNL logical axis (weights in conv2) | TF shape index |
| ---------------- | ------------------------------------ | -------------- |
| Output Channels  | 0                                    | 3              |
| Input Channels   | 1                                    | 2              |
| Height           | 2                                    | 0              |
| Width            | 3                                    | 1              |

> The permutation is `(0123)->(3201)`.

Once `mem1` appears in Conv2, the integration code should compare the
DNNL-to-TF permutations in `mem1` versus the one the operation requires, and if
necessary make an axes-permutation.

```
mem1 DNNL-to-TF permutation:
    perm1 = (0123)->(0312)
Required DNNL-to-TF permutation:
    perm2 = (0123)->(3201)

Mismatch ==>
    perm = perm1 * perm2^-1 = (0123)->(2031) // !!!
    mem2 = dnnl_memory(permute_axes(mem1.md, perm), mem1.ptr)
```


## 4. Framework Memory-Related Operations

There are operations in frameworks that do not imply any meaning on the *shaped
buffers*. They just either change the shape (Flatten, NumPy reshape) or perform
transformations, like transposition.


### 4.1. (Generic) Transposition

Transposition is required in frameworks to change the order of dimensions of
the shaped buffers to give a proper semantics to a tensor inside the operation.

Example:

``` python
buf1 = [[0, 1, 2], [3, 4, 5]]   // shape: (2, 3)
buf2 = [[0, 3], [1, 4], [2, 5]] // shape: (3, 2)

buf1 + buf2 // fail, the shapes do not match

buf2t = transpose(buf2) // copy data, shape (2, 3)
buf1 + buf2t // oK, the shapes match
```

Since DNNL separates the logical and physical representation, the transposition
could merely be a change in the format with change in DNNL-to-TF permutation.
But it could be a proper transposition too:

``` cpp
// assuming buf2 has underlying DNNL memory
// in `ab` format and DNNL-to-TF permutation (10)

// buf2t = transpose(buf2)

// Option 1. W/o transposition on DNNL side
DNNL-to-TF permutation set to (01) // changed!
mem_buf2t_md_dims = {mem_buf2.md.dims[0], mem_buf2.md.dims[1]}
mem_buf2t_md = (mem_buf2_md_dims, format=ba) // change format
mem_buf2t = (mem_buf2t_md, ALLOCATE_PTR)
copy(mem_buf2t.ptr, mem_buf2.ptr)

// Option 2. W/ transposition on
DNNL-to-TF permutation set to (10) // kept as is
mem_buf2t_md_dims = {mem_buf2.md.dims[1], mem_buf2.md.dims[0]} // change here
mem_buf2t_md = (mem_buf2_md_dims, format=ab) // format kept as is
mem_buf2t = (mem_buf2t_md, ALLOCATE_PTR)
tranpose(mem_buf2t.ptr, mem_buf2.ptr) // somehow
```


### 4.2. NumPy Reshape

TBA


## 5. Conclusion

As [Interoperability section](#3-interoperability) shows the proper DNNL
integration to a framework should not only keep a DNNL memory descriptor
related to the buffer, but also a DNNL-to-framework permutation between the
logical DNNL axes and logical framework axes. This significantly increases the
complexity of the integration. And the reason for this is DNNL requirement to
propagate memory formats between the operations.

The alternative is to use plain framework layout only, allowing custom DNNL
formats appear only within one operation (where the semantics is well defined).
This might have non-negligible performance impact though.

Finally, the middle ground between these to approaches is to limit DNNL layout
propagation between operations that have clear semantics for each of the
tensors involved (like *convolution*, *pooling*, and *batch normalization*),
but EXCLUDING the operations that induce strict meaning on the input or output
tensors (such as *reshape*, *flatten*, *transposition*).
However, even this integration is fragile because it might misbehave in the
situation when user consciously gives the other meaning to one of the buffers
in the first group of the operations. For instance, there is no way to catch
the situation when a user will feed the output of one of the convolution as a
**weights** of the another convolution.

Most likely, most of the current DNNL integration follow the last middle ground
approach. Hoping (or because of ignorance) that nothing bad will happen.


---

[Return to the main RFC](README.md)
