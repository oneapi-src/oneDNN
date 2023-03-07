# Graph Compiler Built-in JIT (Xbyak JIT)

Xbyak JIT is an entirely internal backend; thus, all optimization and code generation processes are built inside the project. This backend does not invoke third-party tools (gcc, llvm, etc.), resulting in faster compiling time and less dependencies.

### File Structure

```
compiler/jit/xbyak          # Interface of xbyak_jit and xbyak_jit_module
|
|-- ir                      # Xbyak IR nodes and utils
|   |-- pass                # Xbyak IR analysis and info passes
|   |-- transform           # Xbyak backend specific optimization passes 
|   |-- reg_allocation      # Register allocation infrastructures
|
|-- x86_64                  # x86_64 target, type and abi interface
|
|-- backend                 # Code generation modules
```

### Key Features

**Optimizations:** Apart from default precodegen passes, the Xbyak backend needs additional optimizations to achieve compatibility with machine-level operations and good performance. In a particular stage of these passes, the default IR is transformed to SSA form where optimizations like LICM, value numbering are applied.

**Xbyak IR:** The built-in JIT still uses internal Tensor IR to represent machine-level operations with `xbyak_intrin_node`, which is a dialect of `low_level_intrin_node`. It contains additional attributes to abstract the ISAs, instruction formats, and modifiers.

**x86_64 Target:** Available registers for different ISA extensions, ABI calling convention interfaces for different OS, and low-level data types are defined to represent target x86_64 machine in the register allocation and codegen phases.

**Register Allocation:** After optimization passes and machine-level operations are finalized, the Xbyak backend calculates the value nodes' liveness and assigns registers using an internal register allocator. The register allocation infrastructure is designed to be simple, performative, and flexible. It utilizes an interval tree as a virtual slot to represent each physical register and check liveness interference efficiently. It also uses a priority-based register allocator, utilizing the spill weights of variables, to search and assign available registers or spill variables to memory.

**Code Generation:** The final Xbyak IR is traversed and each machine-level operation is translated into corresponding instruction with appropriate operands. In this stage, function interface, memory addressing, location of values, and stack frame are carefully managed to generate the executable binary directly using the Xbyak assembler.
