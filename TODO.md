Check list
==========

API
---


[ ww35 ]

1. `const engine_t` -- to be or not to be a const...

2. stream dependency tracking

3. DONE -- memory get native handle

4. mixing lazy + non-lazy engines

5. conv w/o bias

6. alpha & beta for op_desc

7. memory_get_primitive_desc(), reorder_get_primitive_desc(), etc... vs.
   primitive_get_primitive_desc(). the same for primive_create

8. rename memory::format::n to memory::format::x

9. add primitive/memory consistency checks

10. l-r, t-b padding in convolution, pooling

11. tensor flow padding consistency

12. copy primitive (i.e. create new one) -- w/ new inputs/outputs

13. current c++ api does not propagate const in opaque types

[ ww39 ]

1. split/concat api

2. stream caching

3. c++ api: object life time managment

4. c++ api: c structure in c++

5. workspaces

6. prop_kind == forward_only


[ ww?? ]

1. c++ api: `shared_ptr` breaks when recieving a new pointer to an
existing primitive from the `C` land

2. c++ api: `memory` needs to be able to act as a `memory_desc` and as
   `memory_primitive_desc`

    1. do we need `mkl_dnn_primitive_get_primitive_desc(mkl_dnn_primitive_t p,
       void *const *primitive_desc)` as a user-visible function?


Internals
---------

1. `nullptr` is of a type `std::nullptr_t`. Does this mean that `nullptr`
   requires STL?

2. get rid of stl

