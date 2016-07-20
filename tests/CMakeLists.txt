include_directories({CMAKE_SOURCE_DIR}/api)

add_executable(api-check-c api_check.c)
target_link_libraries(api-check-c ${LIB_NAME})
add_test(api-check-c api-check-c)

# add_executable(simple_convolution-c simple_convolution.c)
# set_property(TARGET simple_convolution-c PROPERTY C_STANDARD 99)
# target_link_libraries(simple_convolution-c ${LIB_NAME})
# add_test(simple_convolution-c simple_convolution-c)

add_executable(only_convolution-c only_convolution.c)
set_property(TARGET only_convolution-c PROPERTY C_STANDARD 99)
target_link_libraries(only_convolution-c ${LIB_NAME})
add_test(only_convolution-c only_convolution-c)

add_executable(only_convolution-cxx only_convolution.cpp)
set_property(TARGET only_convolution-cxx PROPERTY CXX_STANDARD 11)
set_property(TARGET only_convolution-cxx PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(only_convolution-cxx ${LIB_NAME})
add_test(only_convolution-cxx only_convolution-cpp)

add_executable(api_check-cxx api_check.cpp)
set_property(TARGET api_check-cxx PROPERTY CXX_STANDARD 11)
set_property(TARGET api_check-cxx PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(api_check-cxx ${LIB_NAME})
add_test(api_check-cxx api_check-cxx)
