# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shreyas/G/shr-fuj/oneDNN_open_source

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il

# Include any dependencies generated for this target.
include tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/progress.make

# Include the compile flags for this target's objects.
include tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/flags.make

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.o: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/flags.make
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.o: ../tests/gtests/graph/api/test_cpp_api_graph.cpp
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.o: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.o -MF CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.o.d -o CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/api/test_cpp_api_graph.cpp

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/api/test_cpp_api_graph.cpp > CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.i

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/api/test_cpp_api_graph.cpp -o CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.s

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.o: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/flags.make
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.o: ../tests/gtests/graph/api/api_test_main.cpp
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.o: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.o -MF CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.o.d -o CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/api/api_test_main.cpp

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/api/api_test_main.cpp > CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.i

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/api/api_test_main.cpp -o CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.s

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.o: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/flags.make
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.o: ../tests/gtests/graph/api/test_api_common.cpp
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.o: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.o -MF CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.o.d -o CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/api/test_api_common.cpp

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/api/test_api_common.cpp > CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.i

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/api/test_api_common.cpp -o CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.s

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.o: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/flags.make
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.o: ../tests/gtests/graph/test_allocator.cpp
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.o: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.o -MF CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.o.d -o CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/test_allocator.cpp

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/test_allocator.cpp > CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.i

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/test_allocator.cpp -o CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.s

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.o: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/flags.make
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.o: ../tests/test_thread.cpp
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.o: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.o -MF CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.o.d -o CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/test_thread.cpp

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/test_thread.cpp > CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.i

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/test_thread.cpp -o CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.s

# Object files for target test_graph_cpp_api_graph
test_graph_cpp_api_graph_OBJECTS = \
"CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.o" \
"CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.o" \
"CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.o" \
"CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.o" \
"CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.o"

# External object files for target test_graph_cpp_api_graph
test_graph_cpp_api_graph_EXTERNAL_OBJECTS =

tests/gtests/graph/api/test_graph_cpp_api_graph: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_cpp_api_graph.cpp.o
tests/gtests/graph/api/test_graph_cpp_api_graph: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/api_test_main.cpp.o
tests/gtests/graph/api/test_graph_cpp_api_graph: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/test_api_common.cpp.o
tests/gtests/graph/api/test_graph_cpp_api_graph: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/test_allocator.cpp.o
tests/gtests/graph/api/test_graph_cpp_api_graph: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/__/__/__/test_thread.cpp.o
tests/gtests/graph/api/test_graph_cpp_api_graph: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/build.make
tests/gtests/graph/api/test_graph_cpp_api_graph: tests/gtests/gtest/libdnnl_gtest.a
tests/gtests/graph/api/test_graph_cpp_api_graph: src/libdnnl.so.3.5
tests/gtests/graph/api/test_graph_cpp_api_graph: tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable test_graph_cpp_api_graph"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_graph_cpp_api_graph.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/build: tests/gtests/graph/api/test_graph_cpp_api_graph
.PHONY : tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/build

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/clean:
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api && $(CMAKE_COMMAND) -P CMakeFiles/test_graph_cpp_api_graph.dir/cmake_clean.cmake
.PHONY : tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/clean

tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/depend:
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shreyas/G/shr-fuj/oneDNN_open_source /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/graph/api /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/gtests/graph/api/CMakeFiles/test_graph_cpp_api_graph.dir/depend

