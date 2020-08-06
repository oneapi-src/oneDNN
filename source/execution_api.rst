=============
Execution API
=============

Framework passes the parameter tensors and compiled partition to Execution API for execution. The Execution API binds the tensor buffers with the input/output placeholders, submit it to device runtime for execution. Framework manages tensor allocation and execution submission. Framework may want to submit compiled partitions to different device execution abstraction to exploit the parallelism.

See ``Stream::submit()`` for details.

-----------------------
Allocator call back API
-----------------------

The compiled partitions may allocate memory during the execution, like scratchpad and workspace in the DNNL primitives. The allocated memory may have a different lifetime span, so some memory allocated is only used within the compiled partition execution, some may live until the end of the iteration, and some may persist across iterations. LLGA provides three allocation APIs for framework to pass to LLGA backend: 1) ``allocate_persistent()`` for persistent buffer which can be freed explicitly with ``deallocate_persistent()`` or framework will free all persistent buffers on its exit; 2) ``allocate_output()`` for buffer live during one iteration, typically used by allocating output tensor buffer; 3) ``allocate_temp()`` for scratch pad live only within op.

.. doxygenclass:: llga::base_allocator
   :project: oneDNN Graph Library
   :members:

