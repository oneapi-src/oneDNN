oneDNN JIT Intermediate Representation (IR)
===========================================

# Introduction

All IR objects are immutable by design and use reference counting. The base
class is `object_t` which implements intrusive reference-counting for
`object_impl_t` objects. `object_t` is a wrapper over the real implementation
in `object_impl_t`. All IR objects must have `object_impl_t` in their
inheritance hierarchy as the top-most class.

IR objects support equality comparison via `a.is_equal(b)`. `operator==()` is
reserved and overloaded for boolean comparisons. Additionally IR objects
provide `get_hash()` method to allow using them as keys for
`std::unordered_set` and `std::unordered_map` containers, see corresponding aliases:

- `object_map_t` - an unordered map with `object_t` as the key
- `object_set_t` - an unordered set with `object_t` as the key

The main IR objects are:

- Expressions: class `expr_t` (inherited from `object_t`). Examples:
    - Variables: `var_t`
    - Immediate values (constants): `bool_imm_t`, `int_imm_t` and `float_imm_t`
    - Unary/binary/ternary operations: `unary_op_t`, `binary_op_t`,
      `ternary_op_t`
- Statements: class `stmt_t` (inherited from `object_t`). Examples:
    - Let statement: `let_t`. Binds a value to a variable for the scope defined
      by the let statement.
    - Loop statement: `for_t`
    - If statement: `if_t`
    - Function call: `func_call_t`
- Functions: class `func_t` (inherited from `object_t`). A function and its
  arguments-expressions are used to construct a function call - statement
  with some side effects. Many GPU assembly constructs are represented with
  functions, for example:
    - Synchronization instructions: barrier-wait, barrier-signal, memory fences
    - FMA instructions: fma, dp4a, dpas(w)
    - Send instruction

IR expressions support operator overloading for convenience of use:

```c++
expr_t a = var_t::make(type_t::s32(), "a");
expr_t b = var_t::make(type_t::s32(), "b");
expr_t c = 2 * (a + b) - a;
expr_t c_simplified = simplify(c);
// (gdb) call c.dump()
// ((2 * (a + b)) - a)
// (gdb) call c_simplified.dump()
// (a + (b * 2))
```

# IR Kernel Generation

The generation flow consists of three main stages:

- Creating a kernel skeleton using intermediate representation (IR). The
  resulting kernel includes the high-level optimizations: loop nest,
  loads/stores, multiplications, SLM buffering.
    - After this stage the kernel is functionally correct but it needs further
      passes/optimizations to apply more low-level optimizations and to convert
      it to the form that can be lowered to assembly
- Fine-grained optimizations. This is mostly about applying low-level/local
  optimizations:
  - Transforming single SLM buffering to double/triple buffering
  - Expression simplification
  - Loop hoisting
  - Common subexpression elimination
  - Strength reduction
- Binary code generation. At this stage the kernel is fully optimized and needs
  to be translated to nGEN which is responsible for binary code generation.

## Functionality to Traverse and Modify IR

Using the IR, a kernel is just a `stmt_t`. Like most IR objects, the kernel can
contain other IR objects, in what can be considered as a tree. Some rules apply:

- Statements can include other statements, expressions and functions
- Expressions can include other expressions but can't contain statements or
  functions

`ir_visitor_t` implements generic functionality to traverse an
arbitrary IR object. Example:

```c++
// Custom visitor to count the total number of loops in the given IR object.
class loop_visitor_t : public ir_visitor_t {
public:
    void _visit(const for_t *obj) override {
        refs++;
        // To count nested loops.
        ir_visitor_t::_visit(obj);
    }
    int refs = 0;
};

// root_stmt is an IR statement
loop_visitor_t visitor;
visitor.visit(root_stmt);
```

`ir_mutator_t` is similar to the IR visitor but is used to update IR trees.

## Fine-grained optimization passes

Which IR optimization passes are used depends on the kernel being developed,
and which passes make sense for it. Some examples of existing IR passes:

- Injecting double/triple SLM buffering
    - `simple_slm_buffering_injector_t` or `unrolled_slm_buffering_injector_t`
      is used to convert single buffering to double/triple SLM buffering
      according to some rules
- Expression simplification
- Let optimization
    - Remove unused or redundant let statements 
- Loop hoisting
- Common subexpression elimination
- Strength reduction (this is only applied with unrolled SLM buffering)
- Generating proper send headers
    - Initial `send_t` function calls contain byte offsets. Messages headers
      are generated according to the message specification.
- Peephole optimizations
    - `add` + `add` -> `add3`
    - `mul` + `add` -> `mad`

## Expression Simplification

To be added.

## Binary code generation / lowering of IR to nGEN

At this step the kernel in the IR form includes all optimizations. The binary
code generator is implemented similarly to other nGEN-based kernels. The main
differences are related to IR usage. The binary generator flow steps are
described below:

- Configuring the kernel interface:
    - Use `require\*()` API to specifying/enabling kernel features, such as:
      SIMD size, SLM size, dpas, barrier, etc
- Expression binding initialization for thread grid IDs and kernel arguments
    - This is to bind IR variables for external parameters the corresponding
      registers
    - Further, any reference to such an external variable is resolved based on
      the expression binding
- Visiting the kernel IR statement. The IR tree structure is recursively
  traversed and corresponding instructions are emitted using nGEN calls.

The `ir_to_ngen_t` class is responsible for this last step, visiting the IR
objects recursively and converting them to nGEN instructions. First, we evaluate
the related conditions, values, loop bounds. During evaluation they are bound
to some registers. After that we can form the statement using proper
instructions, e.g. `cmp`/`jmpi` for the loop or `if`/`else`/`endif` for the if
statement. The body statement is visited and lowered to nGEN recursively.

The final IR is very close to assembly so the generation
process is straightforward. Some examples of how different IR objects are
handled:

- Let statement (to introduce variables and bind them to a value)
    - The register allocator is used to allocate a subregister for the variable
    - The variable is initialized either with a `mov` instruction or the value
      is evaluated in the subregister directly
    - Expression binding (`expr_binding_t`) is updated to bind the IR variable
      object to the subregister (to be able to access it in nested
      statements/expressions later)
    - The nested statement of the let statement is visited
    - The subregister is released after traversing the nested statement
- Expressions
    - Expression evaluation is handled by `expr_evaluator_t` class
    - Expression is evaluated recursively. For each expression:
        - Its operands are evaluated (and bound to subregisters if needed)
        - The expression itself is evaluated
    - Sometimes we want to compute an expression in a pre-allocated subregister
      (for example when lowering a let statement). This case is also supported
      by `expr_evaluator_t`.

Additionally, the generator implements extra logic for functionality such as:

- Instruction emulation. Some platforms don't have support for some instructions. Examples:
    - 64-bit arithmetic emulation. This is not handled by the generator and
      implemented in `gpu/jit/emulation.hpp`.
    - `add3` instruction. Emulated as two `add` instructions on older architectures.
- GRF region restrictions. Example:
    - `d` <-> `q` or `d` <-> `b` conversions require to align the smaller data
      type GRF region to match the other data type
- Direct implementation of IR functions. Examples:
    - Reorder between GRF layouts. For simplicity reorders are emitted in
      place. In the kernel IR they are represented as function calls.
    - Reduction of a GRF buffer. Similar to the GRF reorder.

# Tensor, Layout and View

Tensor, layout and view are core abstractions of the IR generator.

**Tensor** - describes a tensor with offsets (stored as IR expressions). Example:
`32x32x1x1` tensor with `[mb_idx, ic_idx, 0, 0]` offsets (activations for 2D
convolution: `N x C x H x W`).

**Layout** - describes a memory layout, contains a physical representation of a
tensor. Semantically, it's the same as the oneDNN blocked memory descriptor.
Layout properties:

- Data type
- Number of dimensions
- Offset to the start of the tensor (in elements of the data type)
    - Same as `offset0` in `dnnl_memory_desc_t`
- Layout blocks
    - Blocks are stored with their dimension index, block size and stride
        - Outer blocks and non-blocked dimensions are also fully specified with
          dedicated blocks
    - Example: `4n2c7h7w32n32c` (6 blocks) (`NChw32n32c` in oneDNN convention)

**View** - describes a "virtual" tensor (view) with its underlying tensor/layout.
Views in general don't exist in memory, instead views contain information about
how to access elements of tensors. Views help to express out-of-bound and stride
conditions. For a deeper understanding of the relationship between view and tensors:

- View tensor `V` has `m` dimensions: `V(v0, v1, ..., v(m-1))`
- Underlying tensor `T` has `n` dimensions: `T(t0, t1, ..., t(n-1))`
- Mapping from view dimensions to tensor dimensions is defined by special
  functions:
    - `t_j = F_j(v0, v1, ..., v(m-1))`
- M/N/K dimension kinds (GEMM behavior) for `V` dimensions
- Each `t_j` dimension may have an associated access mask
    - When the mask is evaluated to false, the element is assumed to be `0`

View example: 2D convolution, 3x3 filter:

- View `V` has 6 dimensions: `mb`, `ic`, `oh`, `ow`, `kh` and `kw`
- Tensor `T` has 4 dimensions: `mb`, `ic`, `ih`, `iw`
- Mapping from view to tensor:
    - `mb` is directly mapped (`t_0 = v_0`)
    - `ic` is directly mapped (`t_1 = v_1`)
    - `ih = oh * SH + kh * (DH + 1) - PH`
    - `iw = ow * SW + kw * (DW + 1) - PW`
- M/N/K dimension kinds:
    - M dimensions: `mb`, `oh`, `ow`
    - K dimensions: `ic`, `kh`, `kw`
- Access masks:
    - `mb` mask: empty
    - `ic` mask: empty
    - `ih` mask: `ih >= 0 and ih < IH`
    - `iw` mask: `iw >= 0 and iw < IW`

The view abstraction encapsulates computation semantics including
convolution-specific stride and out-of-bound conditions and M/N/K dimension
kinds. Having an outer loop nest and defined A/B/C views for the inner blocked
multiplication is enough to fully describe the convolution computation in
terms of the algorithm.

# IR Printing and Debugging

All IR objects provide:

- Overloaded `operator<<` to use with `std::ostream`
- `str()` method returning a textual representation of the object
- `dump()` method to call it under gdb to print a textual representation of
  the object:
    - `call obj.dump()` (under gdb)

All the main IR passes trace the after-pass IR statement when tracing is
enabled (controlled by `ONEDNN_VERBOSE=debuginfo`).

`ir_printer_t` class is mainly responsible for the IR printing-related logic.
