cmake_minimum_required(VERSION 3.22)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(Renderer VERSION 0.1)

add_library(TMD_compile_flags INTERFACE)
target_compile_features(TMD_compile_flags INTERFACE cxx_std_17)
target_compile_options(
	TMD_compile_flags INTERFACE
	"$<$<CONFIG:DEBUG>:-O0;-ggdb;-Wall>"
	"$<$<CONFIG:RELEASE>:-O3>"
)
target_link_options(
	TMD_compile_flags INTERFACE
	"$<$<CONFIG:DEBUG>:-g>"
)

add_subdirectory(TMD)
add_subdirectory(GUI)

if (BUILD_TESTS)
	add_subdirectory(tests)
endif()