# CMake generated Testfile for 
# Source directory: /home/shreyas/G/shr-fuj/oneDNN_open_source/tests
# Build directory: /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(api-c "api-c")
set_tests_properties(api-c PROPERTIES  _BACKTRACE_TRIPLES "/home/shreyas/G/shr-fuj/oneDNN_open_source/cmake/utils.cmake;40;add_test;/home/shreyas/G/shr-fuj/oneDNN_open_source/cmake/utils.cmake;52;add_dnnl_test;/home/shreyas/G/shr-fuj/oneDNN_open_source/tests/CMakeLists.txt;73;register_exe;/home/shreyas/G/shr-fuj/oneDNN_open_source/tests/CMakeLists.txt;0;")
add_test(test_c_symbols-c "test_c_symbols-c")
set_tests_properties(test_c_symbols-c PROPERTIES  _BACKTRACE_TRIPLES "/home/shreyas/G/shr-fuj/oneDNN_open_source/cmake/utils.cmake;40;add_test;/home/shreyas/G/shr-fuj/oneDNN_open_source/cmake/utils.cmake;52;add_dnnl_test;/home/shreyas/G/shr-fuj/oneDNN_open_source/tests/CMakeLists.txt;84;register_exe;/home/shreyas/G/shr-fuj/oneDNN_open_source/tests/CMakeLists.txt;0;")
subdirs("gtests")
subdirs("benchdnn")
subdirs("noexcept")
