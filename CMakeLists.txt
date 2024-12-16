cmake_minimum_required(VERSION 3.22)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(TMD VERSION 0.1)

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

if (BUILD_TEST)
	set(IMPLEMENT_SLOW_EVALUATION true)
endif()

add_subdirectory(3rdParty)
add_subdirectory(TMD)

if (BUILD_TEST)
	add_subdirectory(test)
endif()