# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.26

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = E:\cmake-3.26.0-rc1-windows-x86_64\bin\cmake.exe

# The command to remove a file.
RM = E:\cmake-3.26.0-rc1-windows-x86_64\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\PCA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\PCA\build

# Include any dependencies generated for this target.
include CMakeFiles/pca.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pca.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pca.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pca.dir/flags.make

CMakeFiles/pca.dir/src/pca.cpp.obj: CMakeFiles/pca.dir/flags.make
CMakeFiles/pca.dir/src/pca.cpp.obj: CMakeFiles/pca.dir/includes_CXX.rsp
CMakeFiles/pca.dir/src/pca.cpp.obj: D:/PCA/src/pca.cpp
CMakeFiles/pca.dir/src/pca.cpp.obj: CMakeFiles/pca.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\PCA\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pca.dir/src/pca.cpp.obj"
	E:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pca.dir/src/pca.cpp.obj -MF CMakeFiles\pca.dir\src\pca.cpp.obj.d -o CMakeFiles\pca.dir\src\pca.cpp.obj -c D:\PCA\src\pca.cpp

CMakeFiles/pca.dir/src/pca.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pca.dir/src/pca.cpp.i"
	E:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\PCA\src\pca.cpp > CMakeFiles\pca.dir\src\pca.cpp.i

CMakeFiles/pca.dir/src/pca.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pca.dir/src/pca.cpp.s"
	E:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\PCA\src\pca.cpp -o CMakeFiles\pca.dir\src\pca.cpp.s

# Object files for target pca
pca_OBJECTS = \
"CMakeFiles/pca.dir/src/pca.cpp.obj"

# External object files for target pca
pca_EXTERNAL_OBJECTS =

pca.exe: CMakeFiles/pca.dir/src/pca.cpp.obj
pca.exe: CMakeFiles/pca.dir/build.make
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_gapi455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_highgui455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_ml455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_objdetect455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_photo455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_stitching455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_video455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_videoio455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_imgcodecs455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_dnn455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_calib3d455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_features2d455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_flann455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_imgproc455.dll.a
pca.exe: E:/opencv-4.5.5/opencv/build/x64/MinGW/lib/libopencv_core455.dll.a
pca.exe: CMakeFiles/pca.dir/linkLibs.rsp
pca.exe: CMakeFiles/pca.dir/objects1.rsp
pca.exe: CMakeFiles/pca.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\PCA\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pca.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\pca.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pca.dir/build: pca.exe
.PHONY : CMakeFiles/pca.dir/build

CMakeFiles/pca.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\pca.dir\cmake_clean.cmake
.PHONY : CMakeFiles/pca.dir/clean

CMakeFiles/pca.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\PCA D:\PCA D:\PCA\build D:\PCA\build D:\PCA\build\CMakeFiles\pca.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pca.dir/depend

