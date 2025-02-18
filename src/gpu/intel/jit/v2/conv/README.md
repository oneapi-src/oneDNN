Shapeless GPU Convolution
===========================================

This is a new convolution implementation for GPU which aims to solve two issues:

- Long kernel creation time of the existing JIT convolution in `gpu/jit/conv`
	- This implementation relies on reusable kernels which, once created, can be reused for shapes with different sizes
- Challenging kernel configuration management. JIT kernels are highly configurable which makes their setup very challenging.
	- This is resolved with more control over configurability (offer a limited set of kernels to select between them) and proper performance modeling

### How to build and test

```bash
# 1. Build with OpenCL GPU runtime with experimental support to enable v2 convolution
cmake . -Bbuild -DONEDNN_GPU_RUNTIME=OCL -DONEDNN_EXPERIMENTAL=ON -DONEDNN_BUILD_GRAPH=OFF
make -C build -j `nproc` benchdnn gpu_conv_planner

# 2. Test
./build/tests/benchdnn/benchdnn -v5 --engine=gpu --mode=F --conv --impl=v2 --dir=FWD_I --batch=shapes_resnet_50_v1_5
...
run: --mode=F --conv --engine=gpu --dir=FWD_I ic64ih56oc64oh56kh3ph1n"resnet_50_v1_5:res2a_branch2b*3"
perf,gpu,jit:ir_v2,"resnet_50_v1_5:res2a_branch2b*3",--mode=F --conv --engine=gpu --dir=FWD_I ic64ih56oc64oh56kh3ph1n"resnet_50_v1_5:res2a_branch2b*3",0.451478,155.925,0.10656,4236.84,0.107055,4217.25

# 3. Set kernel descriptor from environment
export desc="--prop fwd --src axb:f32 --wei axcb:f32 --dst axb:f32 --hw xehpc --fma mad --simd 16 --regs 128 --iter ic16mb16oc32 --tg ow4oc4 --loop-desc kw,kh,kd,ic --load a:2d,b:2d --store c:2d"
./build/tests/benchdnn/benchdnn -v5 --engine=gpu --mode=F --conv --impl=v2 --dir=FWD_I --dt=f32 mb128ic256ih56oc64oh56kh1ph0
...
perf,gpu,jit:ir_v2,,--mode=F --conv --engine=gpu --dir=FWD_I mb128ic256ih56oc64oh56kh1ph0,13.1533,158.426,1.124,11702.3,1.13858,11552.4
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
export ONEDNN_GPU_CONV_PLAN_REGISTRY_PATH=plan_registry_data.txt
./build/src/gpu/intel/jit/v2/conv/planner/gpu_conv_planner --auto-search
cp ${ONEDNN_GPU_CONV_PLAN_REGISTRY_PATH}.cpp /path/to/onednn/src/gpu/intel/jit/v2/conv/plan_registry_data.cpp
```
