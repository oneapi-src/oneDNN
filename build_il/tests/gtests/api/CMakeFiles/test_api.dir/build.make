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
include tests/gtests/api/CMakeFiles/test_api.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/gtests/api/CMakeFiles/test_api.dir/progress.make

# Include the compile flags for this target's objects.
include tests/gtests/api/CMakeFiles/test_api.dir/flags.make

tests/gtests/api/CMakeFiles/test_api.dir/test_engine.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/test_engine.cpp.o: ../tests/gtests/api/test_engine.cpp
tests/gtests/api/CMakeFiles/test_api.dir/test_engine.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/test_engine.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/test_engine.cpp.o -MF CMakeFiles/test_api.dir/test_engine.cpp.o.d -o CMakeFiles/test_api.dir/test_engine.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_engine.cpp

tests/gtests/api/CMakeFiles/test_api.dir/test_engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/test_engine.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_engine.cpp > CMakeFiles/test_api.dir/test_engine.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/test_engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/test_engine.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_engine.cpp -o CMakeFiles/test_api.dir/test_engine.cpp.s

tests/gtests/api/CMakeFiles/test_api.dir/test_memory.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/test_memory.cpp.o: ../tests/gtests/api/test_memory.cpp
tests/gtests/api/CMakeFiles/test_api.dir/test_memory.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/test_memory.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/test_memory.cpp.o -MF CMakeFiles/test_api.dir/test_memory.cpp.o.d -o CMakeFiles/test_api.dir/test_memory.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory.cpp

tests/gtests/api/CMakeFiles/test_api.dir/test_memory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/test_memory.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory.cpp > CMakeFiles/test_api.dir/test_memory.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/test_memory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/test_memory.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory.cpp -o CMakeFiles/test_api.dir/test_memory.cpp.s

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_creation.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/test_memory_creation.cpp.o: ../tests/gtests/api/test_memory_creation.cpp
tests/gtests/api/CMakeFiles/test_api.dir/test_memory_creation.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/test_memory_creation.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/test_memory_creation.cpp.o -MF CMakeFiles/test_api.dir/test_memory_creation.cpp.o.d -o CMakeFiles/test_api.dir/test_memory_creation.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_creation.cpp

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_creation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/test_memory_creation.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_creation.cpp > CMakeFiles/test_api.dir/test_memory_creation.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_creation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/test_memory_creation.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_creation.cpp -o CMakeFiles/test_api.dir/test_memory_creation.cpp.s

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc.cpp.o: ../tests/gtests/api/test_memory_desc.cpp
tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc.cpp.o -MF CMakeFiles/test_api.dir/test_memory_desc.cpp.o.d -o CMakeFiles/test_api.dir/test_memory_desc.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_desc.cpp

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/test_memory_desc.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_desc.cpp > CMakeFiles/test_api.dir/test_memory_desc.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/test_memory_desc.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_desc.cpp -o CMakeFiles/test_api.dir/test_memory_desc.cpp.s

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.o: ../tests/gtests/api/test_memory_desc_ops.cpp
tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.o -MF CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.o.d -o CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_desc_ops.cpp

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_desc_ops.cpp > CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_desc_ops.cpp -o CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.s

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_map.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/test_memory_map.cpp.o: ../tests/gtests/api/test_memory_map.cpp
tests/gtests/api/CMakeFiles/test_api.dir/test_memory_map.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/test_memory_map.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/test_memory_map.cpp.o -MF CMakeFiles/test_api.dir/test_memory_map.cpp.o.d -o CMakeFiles/test_api.dir/test_memory_map.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_map.cpp

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_map.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/test_memory_map.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_map.cpp > CMakeFiles/test_api.dir/test_memory_map.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/test_memory_map.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/test_memory_map.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_memory_map.cpp -o CMakeFiles/test_api.dir/test_memory_map.cpp.s

tests/gtests/api/CMakeFiles/test_api.dir/test_namespace.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/test_namespace.cpp.o: ../tests/gtests/api/test_namespace.cpp
tests/gtests/api/CMakeFiles/test_api.dir/test_namespace.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/test_namespace.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/test_namespace.cpp.o -MF CMakeFiles/test_api.dir/test_namespace.cpp.o.d -o CMakeFiles/test_api.dir/test_namespace.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_namespace.cpp

tests/gtests/api/CMakeFiles/test_api.dir/test_namespace.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/test_namespace.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_namespace.cpp > CMakeFiles/test_api.dir/test_namespace.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/test_namespace.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/test_namespace.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_namespace.cpp -o CMakeFiles/test_api.dir/test_namespace.cpp.s

tests/gtests/api/CMakeFiles/test_api.dir/test_stream.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/test_stream.cpp.o: ../tests/gtests/api/test_stream.cpp
tests/gtests/api/CMakeFiles/test_api.dir/test_stream.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/test_stream.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/test_stream.cpp.o -MF CMakeFiles/test_api.dir/test_stream.cpp.o.d -o CMakeFiles/test_api.dir/test_stream.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_stream.cpp

tests/gtests/api/CMakeFiles/test_api.dir/test_stream.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/test_stream.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_stream.cpp > CMakeFiles/test_api.dir/test_stream.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/test_stream.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/test_stream.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_stream.cpp -o CMakeFiles/test_api.dir/test_stream.cpp.s

tests/gtests/api/CMakeFiles/test_api.dir/test_submemory.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/test_submemory.cpp.o: ../tests/gtests/api/test_submemory.cpp
tests/gtests/api/CMakeFiles/test_api.dir/test_submemory.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/test_submemory.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/test_submemory.cpp.o -MF CMakeFiles/test_api.dir/test_submemory.cpp.o.d -o CMakeFiles/test_api.dir/test_submemory.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_submemory.cpp

tests/gtests/api/CMakeFiles/test_api.dir/test_submemory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/test_submemory.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_submemory.cpp > CMakeFiles/test_api.dir/test_submemory.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/test_submemory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/test_submemory.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api/test_submemory.cpp -o CMakeFiles/test_api.dir/test_submemory.cpp.s

tests/gtests/api/CMakeFiles/test_api.dir/__/main.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/__/main.cpp.o: ../tests/gtests/main.cpp
tests/gtests/api/CMakeFiles/test_api.dir/__/main.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/__/main.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/__/main.cpp.o -MF CMakeFiles/test_api.dir/__/main.cpp.o.d -o CMakeFiles/test_api.dir/__/main.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/main.cpp

tests/gtests/api/CMakeFiles/test_api.dir/__/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/__/main.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/main.cpp > CMakeFiles/test_api.dir/__/main.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/__/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/__/main.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/main.cpp -o CMakeFiles/test_api.dir/__/main.cpp.s

tests/gtests/api/CMakeFiles/test_api.dir/__/__/test_thread.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/flags.make
tests/gtests/api/CMakeFiles/test_api.dir/__/__/test_thread.cpp.o: ../tests/test_thread.cpp
tests/gtests/api/CMakeFiles/test_api.dir/__/__/test_thread.cpp.o: tests/gtests/api/CMakeFiles/test_api.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object tests/gtests/api/CMakeFiles/test_api.dir/__/__/test_thread.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/api/CMakeFiles/test_api.dir/__/__/test_thread.cpp.o -MF CMakeFiles/test_api.dir/__/__/test_thread.cpp.o.d -o CMakeFiles/test_api.dir/__/__/test_thread.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/test_thread.cpp

tests/gtests/api/CMakeFiles/test_api.dir/__/__/test_thread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_api.dir/__/__/test_thread.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/test_thread.cpp > CMakeFiles/test_api.dir/__/__/test_thread.cpp.i

tests/gtests/api/CMakeFiles/test_api.dir/__/__/test_thread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_api.dir/__/__/test_thread.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/test_thread.cpp -o CMakeFiles/test_api.dir/__/__/test_thread.cpp.s

# Object files for target test_api
test_api_OBJECTS = \
"CMakeFiles/test_api.dir/test_engine.cpp.o" \
"CMakeFiles/test_api.dir/test_memory.cpp.o" \
"CMakeFiles/test_api.dir/test_memory_creation.cpp.o" \
"CMakeFiles/test_api.dir/test_memory_desc.cpp.o" \
"CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.o" \
"CMakeFiles/test_api.dir/test_memory_map.cpp.o" \
"CMakeFiles/test_api.dir/test_namespace.cpp.o" \
"CMakeFiles/test_api.dir/test_stream.cpp.o" \
"CMakeFiles/test_api.dir/test_submemory.cpp.o" \
"CMakeFiles/test_api.dir/__/main.cpp.o" \
"CMakeFiles/test_api.dir/__/__/test_thread.cpp.o"

# External object files for target test_api
test_api_EXTERNAL_OBJECTS =

tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/test_engine.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/test_memory.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/test_memory_creation.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/test_memory_desc_ops.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/test_memory_map.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/test_namespace.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/test_stream.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/test_submemory.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/__/main.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/__/__/test_thread.cpp.o
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/build.make
tests/gtests/api/test_api: src/libdnnl.so.3.5
tests/gtests/api/test_api: tests/gtests/gtest/libdnnl_gtest.a
tests/gtests/api/test_api: tests/gtests/api/CMakeFiles/test_api.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CXX executable test_api"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_api.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/gtests/api/CMakeFiles/test_api.dir/build: tests/gtests/api/test_api
.PHONY : tests/gtests/api/CMakeFiles/test_api.dir/build

tests/gtests/api/CMakeFiles/test_api.dir/clean:
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api && $(CMAKE_COMMAND) -P CMakeFiles/test_api.dir/cmake_clean.cmake
.PHONY : tests/gtests/api/CMakeFiles/test_api.dir/clean

tests/gtests/api/CMakeFiles/test_api.dir/depend:
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shreyas/G/shr-fuj/oneDNN_open_source /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/api /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/api/CMakeFiles/test_api.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/gtests/api/CMakeFiles/test_api.dir/depend

