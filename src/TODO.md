Check list
==========

API
---


[ ww35 ]

1. `const engine_t` -- to be or not to be a const...

2. stream dependency tracking

6. alpha & beta for op_desc

[ ww39 ]

2. stream caching

3. c++ api: object life time management

4. c++ api: c structure in c++

5. workspaces

8. diluted kernels for conv

9. api iface tests

11. tensor flow padding consistency

12. clone primitive (i.e. create new one) -- w/ new inputs/outputs

13. current c++ api does not propagate const in opaque types

[ ww?? ]

1. c++ api: `shared_ptr` breaks when receiving a new pointer to an
    existing primitive from the `C` land

2. c++ api: `memory` needs to be able to act as a `memory_desc` and as
   `memory_primitive_desc`

    1. do we need `mkl_dnn_primitive_get_primitive_desc(mkl_dnn_primitive_t p,
       void *const *primitive_desc)` as a user-visible function?

3. c api: relu has only one (common) data descriptor. how to deal with the case
    when src_desc ~= dst_desc with the only difference in offset_padding. I.e.
    how to deal with view?

4. review arguments order for backward primitives

Internals
---------

1. `nullptr` is of a type `std::nullptr_t`. Does this mean that `nullptr`
   requires STL?

2. get rid of stl

