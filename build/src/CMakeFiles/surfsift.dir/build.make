# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/li/Desktop/feature_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/li/Desktop/feature_test/build

# Include any dependencies generated for this target.
include src/CMakeFiles/surfsift.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/surfsift.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/surfsift.dir/flags.make

src/CMakeFiles/surfsift.dir/surf_sift.cpp.o: src/CMakeFiles/surfsift.dir/flags.make
src/CMakeFiles/surfsift.dir/surf_sift.cpp.o: ../src/surf_sift.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/li/Desktop/feature_test/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/surfsift.dir/surf_sift.cpp.o"
	cd /home/li/Desktop/feature_test/build/src && g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/surfsift.dir/surf_sift.cpp.o -c /home/li/Desktop/feature_test/src/surf_sift.cpp

src/CMakeFiles/surfsift.dir/surf_sift.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/surfsift.dir/surf_sift.cpp.i"
	cd /home/li/Desktop/feature_test/build/src && g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/li/Desktop/feature_test/src/surf_sift.cpp > CMakeFiles/surfsift.dir/surf_sift.cpp.i

src/CMakeFiles/surfsift.dir/surf_sift.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/surfsift.dir/surf_sift.cpp.s"
	cd /home/li/Desktop/feature_test/build/src && g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/li/Desktop/feature_test/src/surf_sift.cpp -o CMakeFiles/surfsift.dir/surf_sift.cpp.s

src/CMakeFiles/surfsift.dir/surf_sift.cpp.o.requires:
.PHONY : src/CMakeFiles/surfsift.dir/surf_sift.cpp.o.requires

src/CMakeFiles/surfsift.dir/surf_sift.cpp.o.provides: src/CMakeFiles/surfsift.dir/surf_sift.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/surfsift.dir/build.make src/CMakeFiles/surfsift.dir/surf_sift.cpp.o.provides.build
.PHONY : src/CMakeFiles/surfsift.dir/surf_sift.cpp.o.provides

src/CMakeFiles/surfsift.dir/surf_sift.cpp.o.provides.build: src/CMakeFiles/surfsift.dir/surf_sift.cpp.o

# Object files for target surfsift
surfsift_OBJECTS = \
"CMakeFiles/surfsift.dir/surf_sift.cpp.o"

# External object files for target surfsift
surfsift_EXTERNAL_OBJECTS =

../bin/surfsift: src/CMakeFiles/surfsift.dir/surf_sift.cpp.o
../bin/surfsift: src/CMakeFiles/surfsift.dir/build.make
../bin/surfsift: /home/li/anaconda2/lib/libopencv_videostab.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_ts.a
../bin/surfsift: /home/li/anaconda2/lib/libopencv_superres.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_stitching.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_contrib.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_nonfree.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_gpu.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_photo.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_objdetect.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_legacy.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_video.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_ml.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_calib3d.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_features2d.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_highgui.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_imgproc.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_flann.so.2.4.11
../bin/surfsift: /home/li/anaconda2/lib/libopencv_core.so.2.4.11
../bin/surfsift: src/CMakeFiles/surfsift.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../../bin/surfsift"
	cd /home/li/Desktop/feature_test/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/surfsift.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/surfsift.dir/build: ../bin/surfsift
.PHONY : src/CMakeFiles/surfsift.dir/build

src/CMakeFiles/surfsift.dir/requires: src/CMakeFiles/surfsift.dir/surf_sift.cpp.o.requires
.PHONY : src/CMakeFiles/surfsift.dir/requires

src/CMakeFiles/surfsift.dir/clean:
	cd /home/li/Desktop/feature_test/build/src && $(CMAKE_COMMAND) -P CMakeFiles/surfsift.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/surfsift.dir/clean

src/CMakeFiles/surfsift.dir/depend:
	cd /home/li/Desktop/feature_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/li/Desktop/feature_test /home/li/Desktop/feature_test/src /home/li/Desktop/feature_test/build /home/li/Desktop/feature_test/build/src /home/li/Desktop/feature_test/build/src/CMakeFiles/surfsift.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/surfsift.dir/depend

