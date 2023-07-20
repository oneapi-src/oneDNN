# Graph API: Add Python binding

## Motivation

There are two major motivations for adding a python binding of oneDNN graph API
in oneDNN repository.

- It is easier to construct tests, examples, and bug reproducers with python
  API. It helps to avoid boilerplate C++ code which requires explicit types and
  declarations. It also helps to interoperate with other python packages. For
  example, we can perform numeric correctness check with numpy or deep learning
  frameworks easily with python code.
- It is required by integrating oneDNN graph into PyTorch 2.0 torch.compile API.
  Having the python binding inside oneDNN repository will largely reduce the
  amount of integration code as well as the maintenance effort in PyTorch.
  Otherwise, the framework developers will have to the binding on the framework
  side (e.g., [PyTorch#105883](https://github.com/pytorch/pytorch/pull/105883)).

Though it brings benefits, adding a python binding of graph API in oneDNN
repository also introduces new challenges to oneDNN project. The design options
and challenges will be discussed in the following sections.

## Pybind11

[Pybind11](https://github.com/pybind/pybind11) is widely used in the deep
learning framework community for exposing C++ APIs and types through python
binding and achieving the inter-operability between C++ and python. Pybind11 has
been adopted in PyTorch, TensorFlow, and MLIR. More information about pybind11
can be found at the
[documentation](https://pybind11.readthedocs.io/en/stable/index.html).

It is recommended to follow the convention in the deep learning community and
use pybind11 for the python binding of graph API. With that, it requires to
include pybind11 into oneDNN repository and build system. We can achieve that by
either adding pybind11 as a git submodule of oneDNN or copying pybind11 source
code into oneDNN repository. According to the existing maintenance methodology
for gtest and xbyak source code in oneDNN, it is recommended to copy pybind11
source code and maintain it inside oneDNN repository.

[License of pybind11](https://github.com/pybind/pybind11/blob/master/LICENSE)
allows it to be modified and redistributed.

## Source code placement

As the proposal is focusing on the python binding of oneDNN graph API, it is
suggested to place pybind11 source code and the binding implementation into
src/graph/ folder. A new folder src/graph/python/ will be created for the
purpose.

Test cases implemented with python API will be placed in tests/python/.

## Implementation

A PoC for the python binding code with pybind11 can be found in [the attachment
of this RFC](./src/binding.cpp).

## Build option

Graph python API is an optional feature for oneDNN users. A new cmake option
will be defined and exposed to control the feature at the library build stage.
The new option will take effect only when the existing ONEDNN_BUILD_GRAPH option
is enabled.

With ONEDNN_BUILD_GRAPH_PYTHON_BINDING=ON, the binding code will be built and a
python package wheel will be generated in the build directory of oneDNN.

| CMake option                      | Value               | Description |
| ---                               | ---                 | ---         |
| ONEDNN_BUILD_GRAPH_PYTHON_BINDING | ON / OFF (default)  | Enable / disable graph API python binding. |

## Installation

The users will need to install the python package wheel by themselves by
executing, for example:

```bash
# The name dnng is just for demonstration and can be changed in implementation.
 pip install dnng-0.2.0-cp37-cp37m-linux_x86_64.whl
```

After the installation, one can import the package in the corresponding python
interpreter by:

```python
import dnng
# or,
from dnng import *
```

## Low level python API

As demonstrated in the implementation, C++ APIs will be exposed through python
binding directly without extra wrappers or abstractions. With thin and simple
python binding layer, we can reduce the potential bugs in the binding code and
python code, simplify the validation effort, and provide a similar programming
model as the C++ API. Another reason of keeping python API low level is that it
already meets the current requirements of PyTorch integration. But it also makes
the python API less pythonic and less easy-of-use. High level python wrapper or
modules can be added when the request emerges.

The python code example for creating op and logical tensors as follows:

```python
from dnng import *

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

## Runtime objects

- Custom thread pool interface is not exposed and runtime threading object is
  not supported on python API. The reason is that we have not found a way on the
  framework side to extract runtime threading object, wrap it into a python
  object, and pass it to the library through python API.
- Similarly, allocator API is not exposed at this moment due to the same reason.
  The API is not required by the current PyTorch inductor integration.
- Passing tensor memory between numpy and oneDNN library through graph python
  API without copy can be achieved with
  [pybind11](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#).
  But passing tensor memory between frameworks and oneDNN library may need more
  investigation. Currently, a workaround is developed on PyTorch side to pass
  the memory buffer address as a python integer number into the library through
  python API. The workaround has been validated with a set of models.
- On GPU side, for testing purpose, we can rely on [dpctl
  package](https://github.com/IntelPython/dpctl) to create and manage SYCL
  runtime objects (queue, device, and context) and pass them to the library
  through python API. It may require the frameworks to also integrate dpctl and
  share the SYCL runtime objects between the framework and the library.

## Versioning

Graph API python binding shares the same versioning schema of oneDNN project and
maintains backward compatibility in the minor versions under the same major
version. The python API works with the library C++ implementation under the same
commit. Adding a new C++ API may require to change the python binding code to
expose the same functionality through python API.

## Validation

- As mentioned in the motivation, python API as a testing and validation utility
  for the library, it does not require additional validation for the utility
  itself.
- Because python API is also used as the integration point for PyTorch, we need
  to add basic API tests and backward compatibility tests in oneDNN library.
- As the python binding layer is very thin by design, we can continue to rely on
  the existing benchdnn graph test suites for the library performance and
  correctness validation.

## Documentation

Graph python API will not be part of oneDNN specification. If we only take the
python binding as a testing and validation utility, we will only need to provide
documents along with the source code. But if we also take the python binding as
the integration point for frameworks, we will need to provide documents via
oneDNN document webpage.

## Feature status

We do see the value of exposing graph API python binding as a utility for
library validation. But due to uncertainty of using the python API as an
integration point for other frameworks besides PyTorch and the limitations of
exposing runtime objects through python API as mentioned above, it's proposed to
add the graph API python binding as an experimental feature for oneDNN firstly.
We will track the python API usage in both library validation and frameworks
integration and review its maturity in the future before converting it as a
production feature of the library.

(EOD)
