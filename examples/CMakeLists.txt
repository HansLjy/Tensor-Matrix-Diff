cmake_minimum_required(VERSION 3.22)

set(EXAMPLE_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(APPEND EXAMPLE_OUTPUT_DIR "outputs")
configure_file(ExampleConfig.hpp.in ExampleConfig.hpp)

add_executable(gaussian gaussian.cc)
target_link_libraries(gaussian PRIVATE TMD Numerics)
target_include_directories(gaussian PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
