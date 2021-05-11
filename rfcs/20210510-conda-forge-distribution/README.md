# Conda-forge distribution

## Introduction

Conda is a python package manager that widely used by ML FWKs users

## Conda channels
- default. This is a channel managed by Anaconda core team.
  oneDNN is available as part of Intel oneAPI [distribution](https://anaconda.org/intel/repo).
- forge. Conda forge is a community led Conda channel that consists of build
  infrastructure and distributions for the Conda package manager. Recently oneDNN became
  available in [Conda-forge](https://github.com/conda-forge/onednn-feedstock) channel.

The main differences between Conda and Conda-forge:
- Conda-forge has broader architecture support;
- Conda-forge has more packages available;
- To provide binary compatibility between packages in Conda-forge all package
  dependencies should be packaged by Conda-forge, meaning there is no way to
  distribute oneDNN built by Intel C++ Compiler without having this compiler
  in Conda-forge;
- And [more](https://conda-forge.org/docs/user/introduction.html#why-conda-forge).

## oneDNN usage cases. Summary

|          | Runtime package        | Build time package     | Full bundle  |
| :------- | :--------------------- | :--------------------- | :----------- |
| Scenario | Used as a dependency for binary distributions | Used as a dependency for source code distributions that are dependent on oneDNN. | Covers remaining cases |
| Channel  | **conda**, pip, OS repos | **conda**, pip, OS repos | Intel oneAPI |
| Library  | x                      | x                      | x            |
| License  | x                      | x                      | x            |
| Headers  |                        | x                      | x            |
| Examples |                        |                        | x            |
| Doc      |                        |                        | x            |

## oneDNN Conda distributions. As of today

| Package prefix   | Distribution channel | CPU arch | GPU arch | configurations | compiler | dependencies |
| :--------------- | :------------------- | :------- | :------- | :------------- | :------- | :----------- |
| onednn           | Conda-forge          | Intel64 / AMD64 , arm64 (Apple), aarch64, ppc64le | | gomp, tbb | gcc, vc | gomp, tbb |
| onednn(-devel)   | Conda                | Intel64 / AMD64 | Intel Graphics | iomp, gomp, tbb, dpcpp | icc, gcc, vc | iomp, gomp, tbb, vcomp, dpcpp |

## oneDNN Conda distributions. Proposal

| Package prefix   | Distribution channel | CPU arch | GPU arch | configurations | compiler | dependencies |
| :--------------- | :------------------- | :------- | :------- | :------------- | :------- | :----------- |
| onednn           | Conda-forge          | Intel64 / AMD64 , arm64 (Apple), aarch64, ppc64le | | gomp, tbb, **tp** | gcc, vc | gomp, tbb, **vcomp** |
| onednn(-devel)   | Conda                | Intel64 / AMD64 | Intel Graphics | iomp, gomp, tbb, dpcpp | icc, gcc, vc | iomp, gomp, tbb, vcomp, dpcpp |

## oneDNN Conda distributions. Proposal. Details

- Add threadpool configuration. This configuration will be useful for Conda-forge TensorFlow package;
- Add OpenMP configuration on Windows.
