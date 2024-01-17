# Graph API: Add Python binding

## Motivation

There are three major motivations for adding python binding of oneDNN graph API
in oneDNN repository.

- It is easier to construct tests, examples, and bug reproducers with python
  API. It helps to avoid boilerplate C++ code which requires explicit types and
  declarations. It also helps to interoperate with other python packages. For
  example, we can perform numeric correctness check with numpy or deep learning
  frameworks easily with python code.
- It is required by integrating oneDNN graph into PyTorch 2.0 torch.compile API.
  Having the python binding inside oneDNN repository will largely reduce the
  amount of integration code as well as the maintenance effort in PyTorch.
  Otherwise, the framework developers will have to implement the binding on the
  framework side (e.g., [PyTorch#105883](https://github.com/pytorch/pytorch/pull/105883)).
- In addition to C/C++ API, it provides another API option (python API) for users
  to call oneDNN, catering to the preferences of the deep learning community.

Though it brings benefits, adding a python binding of graph API in oneDNN
repository also introduces new challenges to oneDNN project. The design options
and challenges will be discussed in the following sections.

## Implementation

A PoC for the python binding code can be found in [the attachment
of this RFC](./src/binding.cpp).

In this PoC, C++ APIs are exposed through python binding directly without
extra wrappers or abstractions. With thin and simple python binding layer,
we can reduce the potential bugs in the binding code and python code,
simplify the validation effort, and provide a similar programming
model as the C++ API. Another reason of keeping python API low level is that it
already meets the current requirements of PyTorch integration. However,
this design consideration also makes the python API less pythonic and less
ease-of-use. High level python wrapper or modules can be added when the request
emerges.

With python binding implemented, the python code example for creating op and
logical tensors is as follows:

```python
from dnnl_graph import *

# init op
matmul0 = op(0, op.MatMul, "matmul0")
matmul0.set_attr(op.transpose_a, False)
matmul0.set_attr(op.transpose_b, True)

lt0 = logical_tensor(0, logical_tensor.f32, [64, 1280], logical_tensor.strided, logical_tensor.variable)
lt1 = logical_tensor(1, logical_tensor.f32, [1280, 1280], logical_tensor.strided, logical_tensor.constant)
lt2 = logical_tensor(2, logical_tensor.f32, [64, 1280], logical_tensor.strided, logical_tensor.variable)

matmul0.add_inputs([lt0, lt1])
matmul0.add_output(lt2)

# init graph
g = graph(engine.cpu)
g.add_op(matmul0, True)

# finalize graph
g.finalize()

# get partitions
p = g.get_partitions(partition.fusion)
```

### Limitation

There are some limitations of the above implementation though.

Regarding tensor memory:
- Passing tensor memory between numpy and oneDNN library through graph python
  API without copy can be achieved with
  [pybind11](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#).
- But passing tensor memory between frameworks and oneDNN library may need more
  investigation. In PyTorch, [torch.Tensor.data_ptr()](https://pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html)
  can return the memory buffer address as a python integer number to
  the library through python API. This solution has been validated with a set of models.
  In TensorFlow, in C++ API, [tensorflow::TensorBuffer](https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor-buffer)
  has data() method which returns pointer to the memory buffer, while in python API,
  [tf.Tensor](https://www.tensorflow.org/api_docs/python/tf/Tensor)
  does not allow to get raw data pointer. 
- [dlpack](https://github.com/dmlc/dlpack) is another option for exchanging tensor memory between frameworks, which has been
  widely adopted in [NumPy](https://numpy.org/doc/stable/release/1.22.0-notes.html#add-nep-47-compatible-dlpack-support),
  [PyTorch](https://pytorch.org/docs/stable/dlpack.html),
  [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/experimental/dlpack/from_dlpack),
  [TVM](https://tvm.apache.org/docs/reference/api/python/contrib.html#module-tvm.contrib.dlpack), etc.
  To interact with dlpack, oneDNN graph's [Tensor](https://oneapi-src.github.io/oneDNN/class_dnnl_graph_tensor.html#details-classdnnl-1-1graph-1-1tensor)
  needs to map from/to dlpack’s [DLManagedTensor](https://github.com/dmlc/dlpack/blob/master/include/dlpack/dlpack.h#L157-L170),
  which will require API changes in oneDNN graph API.

Regarding runtime objects:

- Custom thread pool interface is not exposed and runtime threading object is
  not supported on python API. The reason is that we have not found a way on the
  framework side to extract runtime threading object, wrap it into a python
  object, and pass it to the library through python API.
- Similarly, allocator API is not exposed at this moment due to the same reason.
  The API is not required by the current PyTorch inductor integration.
- On GPU side, SYCL runtime objects (queue, device, and context) are not covered.
  The reason is that it will bring framework header dependencies in order to share
  the SYCL runtime objects between the framework and the library.

## Source code placement

As mentioned above, there are some limitations of putting the source code of graph
API python binding inside oneDNN repo. So to ensure a comprehensive discussion, we
present three viable options for managing the source code.

- Option 1: put the source code of graph API python binding inside oneDNN repo.
  - Pros: single source of code makes it easy for code management and framework integration.
  - Cons: there is currently uncertainty regarding the use of the python API as an
    integration point for other frameworks beyond PyTorch, which limits its potential
    use cases. Also, an integration into oneDNN will bring framework header dependencies
    for GPU support.

- Option 2: create a separate repository to host the source code of graph API python
  binding.
  - Pros: as a separate repository, it’s more flexible and easier to extend, and it
    will have less impact to oneDNN.
  - Cons: starting a new repository can be expensive and require additional effort
    for code management and alignment between repositories (the new repository,
    oneDNN, and frameworks). Framework needs to incorporate the separate repository
    into its 3rd party in order to use the python binding.

- Option 3: put the source code of graph API python binding as part of PyTorch.
  - Pros: currently the main request of graph API python binding is from PyTorch,
    PyTorch team already has the implementation in the PR. This option also has
    less impact to oneDNN, and there's no need to start a new repository.
  - Cons: more code to upstream to PyTorch, which leads to concerns from the community.

As the request from PyTorch is to support option 1, below discusses more details
about option 1.

### Folder structure
As the python binding is focusing on oneDNN graph API, it is
suggested to place the binding implementation into `src/graph/` folder.
A new folder `src/graph/python/` will be created for the purpose.

Test cases implemented with python API will be placed in `tests/python/`.

### Dependency of pybind11

As indicated in the implementation, the graph API's python binding will require
the introduction of a new dependency on pybind11. [Pybind11](https://github.com/pybind/pybind11)
is widely used in the deep learning framework community for exposing C++ APIs
and types through python binding and achieving the inter-operability between
C++ and python. It has been adopted in PyTorch, TensorFlow, and MLIR.
More information about pybind11 can be found at the
[documentation](https://pybind11.readthedocs.io/en/stable/index.html).

Pybind11 can be utilized in one of three ways: by adding it as a git submodule of
oneDNN, copying its source code into the repository, or using cmake to find it
in the environment.

To reduce the burden of source code maintenance, it's suggested to utilize cmake
to find pybind11 in the environment, instead of incorporating it as a git submodule
or copying its source code into the oneDNN repository.

### Build option

Graph python API is an optional feature for oneDNN users. A new cmake option
will be defined and exposed to control the feature at the library build stage.
The new option will take effect only when the existing ONEDNN_BUILD_GRAPH option
is enabled.

With `ONEDNN_BUILD_GRAPH_PYTHON_BINDING=ON`, the binding code will be built and
a .so file will be generated in the build directory of oneDNN.

| CMake option                      | Value               | Description |
| ---                               | ---                 | ---         |
| ONEDNN_BUILD_GRAPH_PYTHON_BINDING | ON / OFF (default)  | Enable / disable graph API python binding. |

```bash
cmake -DONEDNN_BUILD_GRAPH=ON -DONEDNN_BUILD_GRAPH_PYTHON_BINDING=ON ..
make -j
```

### Installation

After building finished, an auto-generated file `setup.py` will be generated
under `${ONEDNN_ROOT}/dnnl_graph` folder.

Users will need to build and install the python package wheel by themselves
by executing following commands:

```bash
cd ${ONEDNN_ROOT}/dnnl_graph

# or directly install via `python setup.py install`
python setup.py bdist_wheel

# The generated wheel file will be located at dist/ folder
cd dist
pip install dnnl_graph-3.2.0-cp310-cp310-linux_x86_64.whl
```

After the installation, one can import the package in the corresponding python
interpreter by:

```python
# or from dnnl_graph import *
import dnnl_graph
```

### Versioning

Graph API python binding shares the same versioning schema of oneDNN project and
maintains backward compatibility in the minor versions under the same major
version. The python API works with the library C++ implementation under the same
commit. Adding a new C++ API may require to change the python binding code to
expose the same functionality through python API.

### Validation

- As mentioned in the motivation, python API as a testing and validation utility
  for the library, it does not require additional validation for the utility
  itself.
- Because python API is also used as the integration point for PyTorch, we need
  to add basic API tests and backward compatibility tests in oneDNN library.
- As the python binding layer is very thin by design, we can continue to rely on
  the existing benchdnn graph test suites for the library performance and
  correctness validation.

### Documentation

Graph python API will not be part of oneDNN specification. If we only take the
python binding as a testing and validation utility, we will only need to provide
documents along with the source code. But if we also take the python binding as
the integration point for frameworks, we will need to provide documents via
oneDNN document webpage.

## Decision

Option 1 will bring PyTorch header dependencies for GPU support. With upstreaming
GPU support this part can’t be ignored. Putting it into oneDNN will make it hard
to keep oneDNN and PyTorch aligned. For example, if new PyTorch requires new
bindings or headers, oneDNN will have to make a new release with updated bindings
for PyTorch.

The decision is to go with option 2 or 3, depending on PyTorch team's preference.

(EOD)
