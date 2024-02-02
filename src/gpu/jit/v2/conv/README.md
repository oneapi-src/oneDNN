Shapeless GPU Convolution
===========================================

This is a new convolution implementation for GPU which aims to solve two issues:

- Long kernel creation time of the existing JIT convolution in `gpu/jit/conv`
	- This implementation relies on reusable kernels which, once created, can be reused for shapes with different sizes
- Challenging kernel configuration management. JIT kernels are highly configurable which makes their setup very challenging.
	- This is resolved with more control over configurability (offer a limited set of kernels to select between them) and proper performance modeling

### How to build and test

```bash
# 1. Build with OpenCL GPU runtime
cmake . -Bbuild -DONEDNN_GPU_RUNTIME=OCL -DONEDNN_DEV_MODE=ON -DDNNL_GPU_CONV_PLANNER=ON -DONEDNN_BUILD_GRAPH=OFF
make -C build -j `nproc` benchdnn gpu_conv_planner

# 2. Test
export enable_conv_v2=1
./build/tests/benchdnn/benchdnn -v5 --conv --dir=FWD_I --batch=shapes_resnet_50_v1_5
...
perf,gpu,jit:ir_v2,"resnet_50_v1_5:res2a_branch2b*3",--mode=F --conv --engine=gpu --dir=FWD_I mb128ic64ih56oc64oh56kh3ph1n"resnet_50_v1_5:res2a_branch2b*3",28.8946,87.1855,1.42384,20293.4,1.42887,20222
run: --mode=F --conv --engine=gpu --dir=FWD_I mb128ic256ih56oc64oh56kh1ph0n"resnet_50_v1_5:res2b_branch2a*2"
...
```

Look for `jit:ir_v2` implementation name.

### How to debug

- Use `export ONEDNN_VERBOSE=debuginfo=255` to enable debugging output
- Use `call obj.dump()` under gdb to inspect object contents

### How to update plan registry (auto-search)

Auto-search uses a list of hardcoded recipes to generate kernel descriptors. In
event of changes in the kernel generation or of adding new features, use the
snippet below to overwrite the kernel registry in oneDNN.

```bash
export enable_conv_v2=1
export ONEDNN_GPU_CONV_PLAN_REGISTRY_PATH=plan_registry_data.bin
./build/src/gpu/jit/v2/conv/planner/gpu_conv_planner --auto-search
cp ${ONEDNN_GPU_CONV_PLAN_REGISTRY_PATH}.cpp /path/to/onednn/src/gpu/jit/v2/conv/plan_registry_data.cpp
```
