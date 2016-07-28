Check list
==========

API
---

1. `const engine_t` -- to be or not to be a const...

2. `primitive_desc_t` -- should it be `void *` or a forward declared type?

3. input/output vs. src/dst -- what to use and when

4. c++ api: `shared_ptr` breaks when recieving a new pointer to an
existing primitive from the `C` land

5. c++ api: `memory` needs to be able to act as a `memory_desc` and as
   `memory_primitive_desc`

    1. do we need `mkl_dnn_primitive_get_primitive_desc(mkl_dnn_primitive_t p,
       void *const *primitive_desc)` as a user-visible function?


C++
---

1. `nullptr` is of a type `std::nullptr_t`. Does this mean that `nullptr`
   requires STL?

