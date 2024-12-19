# RFC for Auto-Tuning API

## Introduction & Motivation

Auto-tuning is a feature supported as a mode in PyTorch's [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html) function 
and in TensorFlow via [XLA](https://github.com/sourcecode369/tensorflow-1/blob/9f446aba8aaeb2b3c4c6e5ba1ab4cf31494b8a64/tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.cc#L279).
While the median model level improvement is generally modest, there are cases where there are 
large speedups. This has been observed on Intel hardware (see [Using tuning to evaluate the level of oneDNN performance](https://github.com/intel-innersource/libraries.performance.math.onednn/pull/5931)) 
and externally on other [hardware](https://mangpo.net/papers/xla-autotuning-pact2021.pdf).

The goal is to allow end users to optionally try auto-tuning in the case where out of the
box performance is insufficient.

## Proposal
OneDNN should implement auto tuning as a feature that can be exposed to end users of frameworks.
Major requirements for framework integration:
1) No changes to primitive API.
2) Primitive and kernel cache states should not be affected.
3) No regressions after tuning.
4) Simple knob to enable/disable tuning.

### Option 1 - Tune During Primitive Execution- (Recommended)
Tuning happens during one call of the execute function on the primitive. 
Subsequent execute calls on the same primitive will not result in re-tuning the primitive.  
Tuned results will be stored in a primitive implementation specific lookup table that will be referenced  
when the primitive is (re)created. (Some gpu implementations such as conv and batchnorm already use lookup tables.)  
Tuning will happen under a cold cache mode and will be limited to max_nconfigs.  
Primitive cache entry for the primitive(s) being tuned will be updated to point to the tuned implementation when tuning is complete.  
Kernel cache entries will be unmodified for now, but can be modified if we want to enable tuning of resuable kernels later.  
If the user wants to persist the tuned configs between sessions, the lookup tables can optionally be written to files.  
If an implementation does not support tuning, the tuning process will be skipped.
In tune mode, there is no guarantee of correctness.

***Option 1.1 - As a Primtive Attr***
```c
primitive_attr_t attr;
primtive_desc_t prim_desc;
attr.tunable=true;
create_primtive_desc(prim_desc, attr);
create_primitive(prim, prim_desc); 
execute(prim);  //tuning happens here
attr.tunable=false;
create_primtive_desc(prim_desc, attr);
create_primitive(prim, prim_desc);  //primitive is created with config selected from tuning
execute(prim); //normal execution
```
***Option 1.2 - As a Global Variable***
```c
create_primitive(prim); 
dnnl_set_tune(true);   
execute(prim);  //tuning happens here
dnnl_set_tune(false); 
create_primitive(prim); //primitive is created with config selected from tuning
execute(prim); //normal execution
```
***Option 1.3 - As a Primitive Attr Whose Default Value can be set by a Global Variable***
```c
set_tune_mode(true);
primitive_attr_t attr;
primtive_desc_t prim_desc;
create_primtive_desc(prim_desc, attr);
create_primitive(prim, prim_desc); 
execute(prim);  //tuning happens here
set_tune_mode(false);
primitive_attr_t new_attr; //attr must be recreated or tunable field manually set to false 
create_primtive_desc(prim_desc, new_attr);
create_primitive(prim, prim_desc);  //primitive is created with config selected from tuning
execute(prim); //normal execution
```


### Option 2 -Tune During Primtive Creation
Unlike the first option, primitive does not have to be recreated afterward.
However, oneDNN will have to allocate and initialize all memory needed for execution internally during creation.
This adds additional complexity to the implementation, potentially high memory consumption
and need for an optimized data filling routine.

Since frameworks seem ok with first option, would recommend Option 1.

***Option 2.1 - As a Global Variable***
```c
dnnl_set_tune(true);   
create_primitive(prim); // tuning happens here
dnnl_set_tune(false); 
execute(prim); //normal execution with tuned implementation
```
***Option 2.2 - As a Primtive Attr***


### Implementation Details
The following structure will be added to primitive_attr_t.
```c
struct tune_info_t {
    void set_tune_iter(int i); // set configuration to try
    void set_tune_profile(int i, double time); //record min time for ith configuration 
    enum tune_status_t { searching /*default*/, finalize }; 
    int iter = -1; //which configuration to try
    std::vector<double> iter_profile_time; // measured time for ith configuration
    int max_iters = 1; //max number of iters to try obtained by querying implementation
    tune_status_t tune_status = searching; //search or finalize status
};
```
During the primitive execute call, will query the implementation for the number of configs it has via
`query::tune_nconfigs`.  For each config it will create the primitive, execute it 5 times, and record the min
time in the tune_info_t structure. In the case where number of configurations to try is greater than 40 it will stop.
It will then recreate the primitive with tune_status set to finalize. During this call the config with the best 
performance will be stored in a lookup table managed by the primitive and primitive cache will be updated to point
to this implementation.

### Additional Considerations
***Tuning across different implementations:*** This can be tricky for nested primitives as primitive_desc_iterator only 
iterates through outermost implementations. Nested implementations may use scratchpad allocated buffers or take 
different arguments than the outermost primitive. One solution to dispatch correctly between implementations after 
tuning would be to use lookup tables to decide whether to return unimplemented or not. That would imply 
all implementations for a particular primtive will need to generate keys in the same way for their lookup tables. 
Given that currently the most relevant case for this is GeMM based primitives and that dispatching logic between 
the two implementations seems to work well, would recommend this issue be addressed later if the need arises.

***Multi-threaded behavior:*** Since most of the tuning time will be spent creating primitives, threading can 
likely  reduce tuning time. Each primtive can be tuned in a different thread. In that scenario,
the implementation should be thread-safe. Lookup tables should be thread-safe and performance profiling should
be done in a thread-safe way.

***Dynamic Tuning:*** Currently tuning happens in a predetermined way; configs are pregenerated and executed blindly.
Implementations can dynamically adjust which configurations to try next by looking at the iter_profile_time vector which
shows times for previously executed configs. However, implementation will be responsible for maintaining mapping of iter
number to actual configuation tried between primitive creation calls. The primitive_attr struct is const so implementation can't
write back into this structure.

***Performance Measurement:*** Performance is measured with the profiling api. To simulate cold cache mode a reorder is done
between each execution to wipe the cache. This implementation should closely replicate the behavior of benchdnn; there
are memory bound cases that are highly sensitive to cache behavior. If the performance measurement is inaccurate while
tuning, this can result in regressions.



### API

```c
/// include/oneapi/dnnl/dnnl.h

/// Enable/disable tuning. All primitives executed when "true"
/// will be tuned (if underlying implementation supports tuning). Tuning must 
/// be disabled by setting to "false" and primitives recreated in order for tuned implementations 
/// to take effect.
///
/// @param int Set/Unset tuning status.
/// @returns #dnnl_success on success or a status describing the error
///  otherwise.
dnnl_status_t DNNL_API dnnl_set_tune(int tune);
```

```c++
/// include/oneapi/dnnl/dnnl.hpp:
inline void set_tune(bool tune) {
    error::wrap_c_api(dnnl_set_tune((int)tune), "could not set tune status");
}

```

### Performance Example (PVC)
`./benchdnn --conv --engine=gpu --mb=1 --dt=s8 --mode=f --cold-cache=all  --batch=shapes_resnet_50_v1_5` 

Speedup 1.2x  

Total time Before .54 ms

Total time After  .45 ms  
