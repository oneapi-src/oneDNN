# Build oneDNN for PR linter checks.

set -o errexit -o pipefail -o noclobber

if [ -n "$CLANG_VERSION" ]; then
  export CC=clang-$CLANG_VERSION
  export CXX=clang++-$CLANG_VERSION
else
  export CC=clang
  export CXX=clang++
fi

if [[ "$ONEDNN_ACTION" == "configure" ]]; then
    if [[ "$GITHUB_JOB" == "pr-clang-tidy" ]]; then
      set -x
      cmake \
          -Bbuild -S. \
          -DCMAKE_BUILD_TYPE=debug \
          -DONEDNN_BUILD_GRAPH=ON \
          -DDNNL_EXPERIMENTAL=ON \
          -DDNNL_EXPERIMENTAL_SPARSE=ON \
          -DDNNL_EXPERIMENTAL_PROFILING=ON \
          -DDNNL_EXPERIMENTAL_UKERNEL=ON \
          -DONEDNN_EXPERIMENTAL_LOGGING=ON \
          -DDNNL_USE_CLANG_TIDY=CHECK \
          -DDNNL_CPU_RUNTIME=OMP \
          -DDNNL_GPU_RUNTIME=OCL \
          -DDNNL_WERROR=ON \
          -DDNNL_BUILD_FOR_CI=ON \
          -DCMAKE_C_FLAGS="-I/usr/lib/llvm-12/lib/clang/12.0.1/include/" \
          -DCMAKE_CXX_FLAGS="-I/usr/lib/llvm-12/lib/clang/12.0.1/include/"

      set +x
    elif [[ "$GITHUB_JOB" == "pr-format-tags" ]]; then
      set -x
      cmake -B../build -S. -DONEDNN_BUILD_GRAPH=OFF -DDNNL_EXPERIMENTAL_SPARSE=ON
      set +x
    else
      echo "Unknown linter job: $GITHUB_JOB"
      exit 1
    fi
elif [[ "$ONEDNN_ACTION" == "build" ]]; then
    set -x
    cmake --build build -j4
    set +x
else
    echo "Unknown action: $ONEDNN_ACTION"
    exit 1
fi
