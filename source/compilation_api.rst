===============
Compilation API
===============

Conceptually the compiled API is the same as the DNNL primitive building, where DNNL’s primitive building API compiles one op, LLGA compilation API compile a graph partition. The compiled object includes a handle to the compiled object and input/output placeholders.

The Compilation API compiles the partition with or without the tensor shape information, depending on the LLGA’s backend capability. Some backend may want to build a partition without shape information so that it won't cause a significant delay when an unknown shape is feed after the model is deployed.

LLGA compilation API returns an indication for compilation failure.

.. literalinclude:: code_snippets/compilation_api.cpp
    :language: cpp
