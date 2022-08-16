# RFC: v3.0 API cleanup

## Introduction
Major v3 oneDNN release is coming [soon](https://github.com/oneapi-src/oneDNN/milestone/16).
This is a perfect opportunity to provide API improvements.
To decrease maintenance pressure on the library, outdated APIs will be removed from the library which affects user integration code of oneDNN.
oneDNN v3.0 will not be API/ABI backward compatible with oneDNN v2.x.
Hence, user will need to modify their code to use oneDNN v3.0.
Below one may find a comprehensive list of targeted changes:

## List of changes
Disclaimer: header files are copied in this folder are from oneDNN v2.7 to preserve lines to refer to exact spots.

* Build:
    * Build time option `DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE` and all code supporting the `false` value will be removed.

* C/C++ types:
    * Eltwise algorithm types:
        - [`soft_relu`](dnnl.hpp#L563) will be removed. [`soft_relu_v2`](dnnl.hpp#L565) will be renamed into `soft_relu`.
            * End user must specify `alpha` value in a correspondent call.
        - [`logsigmoid`](dnnl.hpp#L567) will be removed. It can be used as current [`soft_relu_v2`](dnnl.hpp#L565) with `alpha` equal to `-1`.
            * End user must replace the algorithm with another one.
        - [`bounded_relu`](dnnl.hpp#L561) will be removed. It can be used as current [`clip`](dnnl.hpp#L584) with `alpha` equal to `0` and `beta` equal to former `alpha`.
            * End user must replace the algorithm with another one.
        - [`hardswish`](dnnl.hpp#L592) will get an extension with `alpha` and `beta` parameters and follow [`hardsigmoid`](dnnl.hpp#L594) algorithm, since in many frameworks `hardswish` is defined as `x * hardsigmoid(x)`, but current version of `hardswish` has hard coded `alpha` and `beta` values to `1/6` and `1/2` while different frameworks may utilize other values.
            * End user must specify `alpha` and `beta` values in a correspondent call.
        - [`eltwise_gelu`](dnnl.hpp#576) will be removed as an alias to `eltwise_gelu_tanh`.
            * End user must replace one type with another one.
        - C enum type algorithm values will be modified.
            * No impact for end user.
    * Memory format tags:
        - Values of certain plain tags will change to group them together.
            * No impact for end user.
    * Normalization [`use_scaleshift`](dnnl.hpp#L724) flag will be removed. This functionality is [supported](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20210223-batch-norm-scale-only) through two different flags [`use_scale`](dnnl.hpp#L740) and [`use_shift`](dnnl.hpp#L745).
        - End user must replace integration code tied to `use_scaleshift` with a new one to follow split arguments.
    * ISA enum:
        - [`avx512_mic`](dnnl.hpp#L13039) and [`avx512_mic_4ops`](dnnl.hpp#L13041) ISA values will be removed as no longer supported in the library.
            * End user must replace their code if values specified were used directly.
        - ISA enum values will be changed to better reflect an order of features supported.
            * No impact for end user.
        - ISA enum value `dnnl_cpu_isa_all` and its `cpu_isa::all` counterpart are renamed into `dnnl_cpu_isa_default` and its `cpu_isa::isa_default` counterpart.
            * End user must replace their code if values specified were used directly.
    * Propagation kind:
        - [`forward_scoring`](dnnl.hpp#507) will be removed as an alias for `forward_inference`.
            * End user must replace one type with another one.
    * Algorithm types:
        - [`pooling_avg`](dnnl.hpp#617) will be removed as an alias for `pooling_avg_exclude_padding`.
            * End user must replace one type with another one.
    * RNN direction type:
        - [`unidirectional`](dnnl.hpp#827) will be removed as an alias for `unidirectional_left2right`.
            * End user must replace one type with another one.

* Attributes:
    * Sum post-op (C API affected only):
        - [`dnnl_post_ops_append_sum`](dnnl.h#L655) and [`dnnl_post_ops_append_sum_v2`](dnnl.h#L690) will be removed. [`dnnl_post_ops_append_sum_v3`](dnnl.h#L726) will be renamed into `dnnl_post_ops_append_sum`.
            * End user must add `zero_point` and `data_type` arguments in a correspondent call.
    * Depthwise post-op:
        - [`append_dw_k3s1p1`](dnnl.hpp#L3240), [`append_dw_k3s2p1`](dnnl.hpp#L3306), [`get_params_dw_k3s1p1`](dnnl.hpp#L3262), and [`get_params_dw_k3s2p1`](dnnl.hpp#L3327) and their C API counterparts ([`dnnl_post_ops_append_dw_k3s1p1`](dnnl.h#L899), [`dnnl_post_ops_append_dw_k3s2p1`](dnnl.h#L959), [`dnnl_post_ops_get_params_dw_k3s1p1`](dnnl.h#L920), and [`dnnl_post_ops_get_params_dw_k3s2p1`](dnnl.h#L980)) will be removed.
            * End user must replace these calls with [`append_dw`](dnnl.hpp#L3144) and [`get_params_dw`](dnnl.hpp#3176) ([`dnnl_post_ops_append_dw`](dnnl.h#L837) and [`dnnl_post_ops_get_params_dw`](dnnl.h#862) in C API) with additional arguments for `kernel`, `stride`, and `padding`.

* Primitives (C++ changes only, as C changes are covered by [this RFC](https://github.com/oneapi-src/oneDNN/pull/1391)):
    * [`pooling_forward`](dnnl.hpp#L6388) and [`pooling_backward`](dnnl.hpp#L6508) will be removed. [`pooling_v2_forward`](dnnl.hpp#L12372) and [`pooling_v2_backward`](dnnl.hpp#L12502) will be renamed into `pooling_forward` and `pooling_backward`.
        * End user must update `pooling_forward` and `pooling_backward` classes with proper calls to primitive descriptor constructors or replace `pooling_v2_forward` and `pooling_v2_backward` calls.
    * [`softmax_forward`](dnnl.hpp#L6878) and [`softmax_backward`](dnnl.hpp#L6976) will be removed. [`softmax_v2_forward`](dnnl.hpp#L7094) and [`softmax_v2_backward`](dnnl.hpp#L7199) will be renamed into `softmax_forward` and `softmax_backward`.
        * End user must update `softmax_forward` and `softmax_backward` classes with proper calls to primitive descriptor constructors or replace `softmax_v2_forward` and `softmax_v2_backward` calls.
    * [`logsoftmax_forward`](dnnl.hpp#L7322) and [`logsoftmax_backward`](dnnl.hpp#L7425) will be removed. Logsoftmax functionality is [supported](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20211207-softmax-v2) through new softmax classes using a [specific algorithm](dnnl_types.h#L1587).
        * End user must replace `logsoftmax_forward` and `logsoftmax_backward` calls with future `softmax_forward` and `softmax_backward` constructors.
    * `layer_normalization_forward` [this](dnnl.hpp#L7875) and [this](dnnl.hpp#L7897) constructors and `layer_normalization_backward` [this](dnnl.hpp#L8056) and [this](dnnl.hpp#L8081) constructors relying on a single data descriptor instead of separate source and destination descriptors will be removed. Note that new constructors providing support for two memory descriptors may have same signature as former ones.
        * End user must update `layer_normalization_forward` and `layer_normalization_backward` classes with proper calls to primitive descriptor constructors.
    * [`batch_normalization_forward`](dnnl.hpp#L7561) and [`batch_normalization_backward`](dnnl.hpp#L7699) constructors relying on a single data descriptor instead of source and destination will be replaced with constructors providing separate source and destination memory descriptors.
        * End user must update calls to `batch_normalization_forward::primitive_desc` and `batch_normalization_backward::primitive_desc` constructors accordingly.
    * [`shuffle_forward`](dnnl.hpp#L11687) and [`shuffle_backward`](dnnl.hpp#L11769) constructors relying on a single data descriptor instead of source and destination will be replaced with constructors providing separate source and destination memory descriptors.
        * End user must update calls to `shuffle_forward::primitive_desc` and `shuffle_backward::primitive_desc` constructors accordingly.
    * [`eltwise_forward`](dnnl.hpp#L6656) and [`eltwise_backward`](dnnl.hpp#L6757) constructors relying on a single data descriptor instead of source and destination will be replaced with constructors providing separate source and destination memory descriptors.
        * End user must update calls to `eltwise_forward::primitive_desc` and `eltwise_backward::primitive_desc` constructors accordingly.
    * [`prelu_forward`](dnnl.hpp#L12647) and [`prelu_backward`](dnnl.hpp#L12742) constructors relying on a single data descriptor instead of source and destination will be replaced with constructors providing separate source and destination memory descriptors.
        * End user must update calls to `prelu_forward::primitive_desc` and `prelu_backward::primitive_desc` constructors accordingly.
    * [`lrn_forward`](dnnl.hpp#L6165) and [`lrn_backward`](dnnl.hpp#L6268) constructors relying on a single data descriptor instead of source and destination will be replaced with constructors providing separate source and destination memory descriptors.
        * End user must update calls to `lrn_forward::primitive_desc` and `lrn_backward::primitive_desc` constructors accordingly.
        * **NOTE!** `lrn_backward::primitive_desc` changes the order of memory descriptors provided.

* Primitive descriptors:
    * C API [`dnnl_convolution_forward_desc_init`](dnnl.h#1512), [`dnnl_convolution_backward_data_desc_init`](dnnl.h#1589), and [`dnnl_convolution_backward_weights_desc_init`](dnnl.h#1663) calls will be removed. [`dnnl_dilated_convolution_forward_desc_init`](dnnl.h#1555), [`dnnl_dilated_convolution_backward_data_desc_init`](dnnl.h#1626), and [`dnnl_dilated_convolution_backward_weights_desc_init`](dnnl.h#1704) will be renamed to `dnnl_convolution_forward_desc_init`, `dnnl_convolution_backward_data_desc_init`, and `dnnl_convolution_backward_weights_desc_init` correspondently. Note that these API calls will be updated to have two argument for forward propagation kind (engine and attributes) and three arguments for backward propagation kind (engine, attributes, and hint) according to [this RFC](https://github.com/oneapi-src/oneDNN/pull/1391).
        * End user must update calls that will be removed with dilation parameter or change the name of modified calls.
    * Sum [`primitive_desc`](dnnl.hpp#4519) constructors, Concat [`primitive_desc`](dnnl.hpp#4422) constructors, and Shuffle [`primitive_desc`](dnnl.hpp#11727) constructor will change a signature, where attributes and engine will be swapped, to align with rest primitive_desc constructors from other classes.
        * End user must update calls to mentioned constructors according to the change.
    * RNN [`constructors`](dnnl.hpp#8900) will drop `beta` and `flags` arguments as not used. C API remains as is.
        * End user must update constructor call if non-zero `alpha` argument was provided.

* Auxiliary:
    * C++ API [`void set_data_handle(void *handle, const stream &astream)`](dnnl.hpp#L2808) will be removed as stream is no longer necessary when setting data handle.
        * End user must update `set_data_handle` API to the one without `stream` input argument.
    * C++ API [`stream(const engine &aengine, flags aflags = flags::default_flags)`](dnnl.hpp#L1116) will receive `explicit` keyword to avoid implicit conversions leading to API misuse.
