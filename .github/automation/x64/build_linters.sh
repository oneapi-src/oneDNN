# Build oneDNN for PR linter checks.

set -o errexit -o pipefail -o noclobber

export CC=clang
export CXX=clang++

if [[ "$ONEDNN_ACTION" == "configure" ]]; then
    if [[ "$GITHUB_JOB" == "pr-clang-tidy" ]]; then
      set -x
      cmake \
          -Bbuild -S. \
          -DCMAKE_BUILD_TYPE=debug \
          -DONEDNN_BUILD_GRAPH=ON \
          -DDNNL_EXPERIMENTAL=ON \
          -DDNNL_EXPERIMENTAL_PROFILING=ON \
          -DDNNL_EXPERIMENTAL_UKERNEL=ON \
          -DONEDNN_EXPERIMENTAL_LOGGING=ON \
          -DDNNL_CPU_RUNTIME=OMP \
          -DDNNL_GPU_RUNTIME=OCL \
          -DDNNL_WERROR=ON \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      set +x
    elif [[ "$GITHUB_JOB" == "pr-format-tags" ]]; then
      set -x
      cmake -B../build -S. -DONEDNN_BUILD_GRAPH=OFF
      set +x
    else
      echo "Unknown linter job: $GITHUB_JOB"
      exit 1
    fi
elif [[ "$ONEDNN_ACTION" == "build" ]]; then
    set -x
    cmake --build build -j`nproc`
    set +x
else
    echo "Unknown action: $ONEDNN_ACTION"
    exit 1
fi
