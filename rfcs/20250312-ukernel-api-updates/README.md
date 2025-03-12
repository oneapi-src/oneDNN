# Ukernel API Updates

## Motivation

ukernel API got some traction in the community. During the integration some pain
points emerged, and this document will discuss some of them and propose.

## Proposal

### Exception-free Mode

The implementation of ukernel API has its limitations, mostly related to
hardware features available. Some underlying C API calls may return unsuccessful
status, but C++ API calls will not propagate that and just throw an exception.

There's no other way to recover from the failure but to catch exceptions which
doesn't suit every user's environment. To overcome that limitation, the
following is proposed:

* `enum class pack_type` gets an additional special value `no_pack`. With that,
  `undef` value starts indicating that the kernel is unusable on the current
  system and a fallback to a different frontend is required. For example,
  querying `pack_type` on SSE41 systems will start returning this value.
  `no_pack` means that brgemm is supported but packing for a given tensor, A or
  B, is not required, and plain memory format for requested tensor configured
  through the brgemm object constructor promises to be functionally correct.
  This replaces the former `no_trans` value that was used to denote the same
  message. The difference is `no_trans` still has physical representation for
  the tensor and may confuse the user in case their tensor is originally in
  `trans` `pack_type`, or transposed memory format. The proposed value doesn't
  have that physical meaning and serves for differentiating from the unsupported
  state and is uniform across multiple backends which may have different packing
  requirements.
* `brgemm` object provides several APIs to configure it and fills internal
  object state with user choices. Then a `finalize()` call must be used to
  initiate the backend object preparation. During that preparation, underlying
  calls may return unsuccessful statuses and the C++ `finalize()` will throw an
  exception. Instead, the proposal is to start returning a `bool` value with
  `true` meaning success and `false` meaning no success instead of current
  `void`. In the latter case the fallback to a different frontend must be done
  as the brgemm object indicates it doesn't support given settings on a given
  system.

These are correspondent changes in the API and usage:
```cpp
/// dnnl_ukernel_types.h

/// Packing specification
typedef enum {
    /// Undefined pack type. (A guard value.) <-- No longer true.
    /// When returned, ukernel API can't be used.
    dnnl_pack_type_undef = 0,
    /// NEW TYPE.
    /// A value indicating the brgemm object can proceed without packing data
    /// for a requested tensor.
    dnnl_pack_type_no_pack,
    /// Plain, not transposed layout. Similar to format_tag::ab.
    dnnl_pack_type_no_trans,
    /// Plain, transposed layout. Similar to format_tag::ba.
    dnnl_pack_type_trans,
    /// Packed by 32 bits along K dimension layout.
    dnnl_pack_type_pack32,
} dnnl_pack_type_t;


/// dnnl_ukernel.hpp

/// BRGeMM ukernel
struct brgemm : public handle<dnnl_brgemm_t> {
    ...
    /// (was)
    /// Finalizes initialization of a BRGeMM ukernel object.
    ///
    /// This step must be performed prior to querying information from the
    /// object.
    void finalize() {
        dnnl_status_t status = dnnl_brgemm_finalize(get());
        if (status != dnnl_success)
            error::wrap_c_api(status, "could not finalize an object");
    }
    ...

    ...
    /// (will be)
    /// Finalizes initialization of a BRGeMM ukernel object.
    ///
    /// This step must be performed prior to querying information from the
    /// object.
    ///
    /// Returns `true` on success and `false` otherwise.
    bool finalize() {
        dnnl_status_t status = dnnl_brgemm_finalize(get());
        return status == dnnl_success;
    }
    ...
```

Usage:

```cpp
    ...

    const auto pack_type = dnnl::ukernel::brgemm::get_B_pack_type(a_dt, b_dt);

    // Fallback if received undefined value.
    if (pack_type == dnnl::ukernel::pack_type::undef) {
        printf("API can't be used\n");
        fallback_call();
    }

    // Rely on a new type.
    if (pack_type != dnnl::ukernel::pack_type::no_pack) {
        // Additional handling of transform.
        ...
        dnnl::ukernel::transform tr(..., /* allow_empty = */ true);
        if (!tr) {
            printf("API can't be used\n");
            fallback_call();
        }
    }

    ...

    dnnl::ukernel::brgemm brg(..., /* allow_empty = */ true);
    // Rare case but safer to check.
    if (!brg) {
        printf("API can't be used\n");
        fallback_call();
    }

    // Configurating `brg` object...

    // This is possible with proposed API change.
    const bool ok = brg.finalize();
    if (!ok) {
        printf("API can't be used\n");
        fallback_call();
    }

    ...
```

### Runtime Leading Dimension Support

Some applications programming model might not have all the shapes information
available at the kernel creation time. The strategy for such applications is
to generate a pre-defined set of "tiles", let's for the sake of simplicity
announce two: 16x16 and 32x32 (though there can be more), which can work with
arbitrary leading dimension values provided at runtime. The example of
application is [triton-cpu](https://github.com/triton-lang/triton-cpu), which is
currently a fork of the original triton.

The library already provides a notion of runtime dimension values through the
special `DNNL_RUNTIME_DIM_VAL` macro value. The only part that is not there is
passing leading dimensions at runtime.

The proposal is to update the `execute` with extra argument with a default empty
value:

```cpp
/// dnnl_ukernel.hpp

/// BRGeMM ukernel
struct brgemm : public handle<dnnl_brgemm_t> {
    ...

    /// Executes a BRGeMM ukernel object.
    ///
    /// @param A Base pointer to a tensor A.
    /// @param B Base pointer to a tensor B.
    /// @param A_B_offsets Vector of pairs of tensors A and B offsets for
    ///     each batch. The number of batches must coincide with the
    ///     `batch_size` value passed at object construction stage.
    /// @param C Pointer to a tensor C (accumulation buffer).
    /// @param scratchpad Pointer to a scratchpad buffer.
    /// @param actual_lds NEW ARGUMENT. Vector of actual leading dimensions for
    ///     tensors A, B, and C. Must be specified if brgemm object was created
    ///     with `DNNL_RUNTIME_DIM_VAL` instead of regular values for `lda`,
    ///     `ldb`, or `ldc`. When `get_B_pack_type` returns values other than
    ///     `no_pack`, `actual_lds[1]` for `ldb` must be used in transform
    ///     routine. If any of given values were not specified as runtime values
    ///     at kernel creation, they will be used instead.
    void execute(const void *A, const void *B,
            const std::vector<std::pair<memory::dim, memory::dim>> &A_B_offsets,
            void *C, void *scratchpad,
            const std::vector<memory::dim> &actual_lds = {}) const {
        dnnl_status_t status = dnnl_brgemm_execute(get(), A, B,
                (const dnnl_dim_t *)A_B_offsets.data(), C, scratchpad,
                static_cast<const dnnl_dim_t *>(actual_lds.data()));
        if (status != dnnl_success)
            error::wrap_c_api(
                    status, "could not execute a BRGeMM ukernel object");
    }

    /// ...
    void execute(const void *A, const void *B,
            const std::vector<std::pair<memory::dim, memory::dim>> &A_B_offsets,
            const void *C, void *D, void *scratchpad,
            const attr_params &params = default_attr_params(),
            const std::vector<memory::dim> &actual_lds = {}) const {
        dnnl_status_t status = dnnl_brgemm_execute_postops(get(), A, B,
                (const dnnl_dim_t *)A_B_offsets.data(), C, D, scratchpad,
                params.get(),
                static_cast<const dnnl_dim_t *>(actual_lds.data()));
        if (status != dnnl_success)
            error::wrap_c_api(
                    status, "could not execute a BRGeMM ukernel object");
    }

    ...
};

/// Transform ukernel
struct transform : public handle<dnnl_transform_t> {
    ...

    /// Executes a transform object.
    ///
    /// @param in Pointer to an input buffer.
    /// @param out Pointer to an output buffer.
    /// @param in_actual_ld Actual input buffer leading dimension. Must be
    ///     specified if transform was created with runtime `in_ld`.
    void execute(
            const void *in, void *out, memory::dim in_actual_ld = 0) const {
        dnnl_status_t status
                = dnnl_transform_execute(get(), in, out, in_actual_ld);
        if (status != dnnl_success)
            error::wrap_c_api(status,
                    "could not execute a BRGeMM ukernel packing B object");
    }

    ...
};
```

Usage:

```cpp
    ...

    using namespace dnnl::ukernel;

    const auto pack_type = brgemm::get_B_pack_type(a_dt, b_dt);
    const bool need_pack = pack_type != pack_type::no_pack;

    if (need_pack) {
        // Additional handling of transform.
        ...

        // `in_ld` becomes `DNNL_RUNTIME_DIM_VAL`.
        transform tr(K, N, pack_type::no_trans, DNNL_RUNTIME_DIM_VAL, out_ld,
                in_dt, out_dt, /* allow_empty = */ true);
        if (!tr) {
            printf("API can't be used\n");
            fallback_call();
        }
    }

    ...

    // Note that for `need_pack == true` the `ldb` value is fixed.
    dnnl::ukernel::brgemm brg(M, N, K, batch, DNNL_RUNTIME_DIM_VAL,
            need_pack ? transform::out_ld : DNNL_RUNTIME_DIM_VAL,
            DNNL_RUNTIME_DIM_VAL, a_dt, b_dt, c_dt, /* allow_empty = */ true);
    // Rare case but safer to check.
    if (!brg) {
        printf("API can't be used\n");
        fallback_call();
    }

    // Configurating and generating `brg` object...

    const dim_t execute_lda = LDA;
    const dim_t execute_ldb = LDB;
    const dim_t execute_ldc = LDC;

    if (need_pack) {
        tr.execute(in, tr_out, execute_ldb);
    }

    // `execute_ldb` will be ignored for packing as it requires to set fixed
    // `ldb` argument for `brg`.
    std::vector<dim_t> execute_lds = {execute_lda, execute_ldb, execute_ldc};

    brg.execute(
            A, need_pack ? tr_out : B, A_B_offsets, C, scratchpad, execute_lds);

    ...
```
